"""Deep induction circuit analysis.

Specialized tools for analyzing induction circuits: full path tracing through
embedding -> previous-token head -> induction head, matching score matrices,
copy circuit verification, prefix search behavior, and circuit completeness testing.

References:
    Olsson et al. (2022) "In-context Learning and Induction Heads"
    Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"
"""

import jax
import jax.numpy as jnp
import numpy as np


def induction_circuit_path_tracing(model, tokens):
    """Trace the full induction circuit path through the model.

    Identifies candidate previous-token heads (layer 0) and induction heads
    (later layers), then traces how information flows through the circuit:
    token[i] -> prev-token head copies to position[i+1] -> induction head
    matches pattern and copies token prediction.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len] with repeated patterns.

    Returns:
        dict with:
            prev_token_scores: [n_layers, n_heads] how much each head attends to prev token
            induction_scores: [n_layers, n_heads] how much each head does induction
            circuit_paths: list of (prev_head, ind_head, path_strength) tuples
            best_circuit: (prev_layer, prev_head, ind_layer, ind_head, strength)
            path_strength_matrix: [n_total_heads, n_total_heads] composition strengths
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    # Score each head for previous-token attention
    prev_token_scores = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is None:
            continue
        pat = np.array(pattern)  # [n_heads, seq, seq]
        for head in range(n_heads):
            # How much does position i attend to position i-1?
            score = 0.0
            count = 0
            for pos in range(1, seq_len):
                score += pat[head, pos, pos - 1]
                count += 1
            prev_token_scores[layer, head] = score / max(count, 1)

    # Score each head for induction behavior
    # Induction: if token[i] == token[j] for j < i, head at position i attends to j+1
    tokens_np = np.array(tokens)
    induction_scores = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is None:
            continue
        pat = np.array(pattern)
        for head in range(n_heads):
            score = 0.0
            count = 0
            for pos in range(2, seq_len):
                for prev in range(pos - 1):
                    if tokens_np[prev] == tokens_np[pos - 1] and prev + 1 < pos:
                        score += pat[head, pos, prev + 1]
                        count += 1
            induction_scores[layer, head] = score / max(count, 1)

    # Find circuit paths: prev-token head -> induction head
    total_heads = n_layers * n_heads
    path_matrix = np.zeros((total_heads, total_heads))
    circuit_paths = []

    for pl in range(n_layers):
        for ph in range(n_heads):
            for il in range(pl + 1, n_layers):
                for ih in range(n_heads):
                    strength = prev_token_scores[pl, ph] * induction_scores[il, ih]
                    src_idx = pl * n_heads + ph
                    dst_idx = il * n_heads + ih
                    path_matrix[src_idx, dst_idx] = strength
                    if strength > 0.01:
                        circuit_paths.append((
                            (pl, ph), (il, ih), float(strength)
                        ))

    circuit_paths.sort(key=lambda x: -x[2])

    # Best circuit
    best = None
    best_strength = 0.0
    for (pl, ph), (il, ih), s in circuit_paths:
        if s > best_strength:
            best = (pl, ph, il, ih, s)
            best_strength = s
    if best is None:
        best = (0, 0, min(1, n_layers - 1), 0, 0.0)

    return {
        "prev_token_scores": prev_token_scores,
        "induction_scores": induction_scores,
        "circuit_paths": circuit_paths[:20],
        "best_circuit": best,
        "path_strength_matrix": path_matrix,
    }


def matching_score_matrix(model, tokens):
    """Compute the QK matching score matrix for induction behavior.

    For each head, compute how well the QK circuit matches tokens that appeared
    previously in the same context — the core mechanism of induction heads.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            qk_match_scores: [n_layers, n_heads] matching score per head
            token_match_matrix: [n_layers, n_heads, seq_len, seq_len] per-position scores
            best_matching_heads: list of (layer, head, score)
            mean_match_by_layer: [n_layers] average matching score per layer
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    tokens_np = np.array(tokens)
    qk_scores = np.zeros((n_layers, n_heads))
    token_match = np.zeros((n_layers, n_heads, seq_len, seq_len))

    for layer in range(n_layers):
        pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is None:
            continue
        pat = np.array(pattern)

        for head in range(n_heads):
            total_score = 0.0
            count = 0
            for qi in range(1, seq_len):
                for ki in range(qi):
                    # Does this head attend to positions where matching occurs?
                    if tokens_np[ki] == tokens_np[qi - 1] and ki + 1 <= qi:
                        # Induction pattern: attend to ki+1 when token[ki]==token[qi-1]
                        if ki + 1 < seq_len and ki + 1 <= qi:
                            token_match[layer, head, qi, ki + 1] = pat[head, qi, ki + 1]
                            total_score += pat[head, qi, ki + 1]
                            count += 1

            qk_scores[layer, head] = total_score / max(count, 1)

    # Best matching heads
    flat = [(int(l), int(h), float(qk_scores[l, h]))
            for l in range(n_layers) for h in range(n_heads)]
    best = sorted(flat, key=lambda x: -x[2])[:10]

    mean_by_layer = np.mean(qk_scores, axis=1)

    return {
        "qk_match_scores": qk_scores,
        "token_match_matrix": token_match,
        "best_matching_heads": best,
        "mean_match_by_layer": mean_by_layer,
    }


def copy_circuit_verification(model, tokens, layer, head):
    """Verify whether a specific head implements a copy circuit.

    Checks if the OV circuit of the head copies token identity from source
    to destination by projecting through the unembedding.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        layer: Layer of the head to verify.
        head: Head index to verify.

    Returns:
        dict with:
            copy_score: float (how well the head copies tokens)
            top_copied_tokens: list of (src_pos, dst_pos, token, logit_boost)
            ov_eigenvalues: top eigenvalues of the OV circuit
            token_copy_accuracy: fraction of positions where copied token matches
            ov_trace: float (trace of OV projected through unembedding)
    """
    from irtk.hook_points import HookState

    n_heads = model.cfg.n_heads

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
    v = cache.get(f"blocks.{layer}.attn.hook_v")

    seq_len = len(tokens)
    tokens_np = np.array(tokens)

    W_O = np.array(model.blocks[layer].attn.W_O)  # [n_heads, d_head, d_model]
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]

    if pattern is None or v is None:
        return {
            "copy_score": 0.0,
            "top_copied_tokens": [],
            "ov_eigenvalues": np.array([]),
            "token_copy_accuracy": 0.0,
            "ov_trace": 0.0,
        }

    pat = np.array(pattern)
    v_arr = np.array(v)

    # For each destination position, compute which token the head promotes
    copy_hits = 0
    total = 0
    top_copied = []

    for dst in range(1, seq_len):
        # Weighted sum of values
        weighted_v = np.zeros(v_arr.shape[-1])
        for src in range(dst + 1):
            weighted_v += pat[head, dst, src] * v_arr[src, head]

        # Project through OV and unembedding
        output = weighted_v @ W_O[head]  # [d_model]
        logits = output @ W_U  # [d_vocab]

        predicted_token = int(np.argmax(logits))

        # Check if it copies the most-attended-to source's token
        most_attended_src = int(np.argmax(pat[head, dst, :dst + 1]))
        src_token = int(tokens_np[most_attended_src])

        if predicted_token == src_token:
            copy_hits += 1

        top_copied.append((
            most_attended_src, dst, src_token,
            float(logits[src_token])
        ))
        total += 1

    copy_accuracy = copy_hits / max(total, 1)

    # OV circuit eigenvalues
    W_V = np.array(model.blocks[layer].attn.W_V)  # [n_heads, d_model, d_head]
    # OV circuit: W_V[h] @ W_O[h] = [d_model, d_head] @ [d_head, d_model] = [d_model, d_model]
    ov_proj = W_O[head] @ W_U  # [d_head, d_vocab]
    sv = np.linalg.svd(ov_proj, compute_uv=False)
    top_sv = sv[:min(5, len(sv))]

    # Copy score: average logit boost for source token
    copy_score = float(np.mean([t[3] for t in top_copied])) if top_copied else 0.0

    # OV trace: W_V @ W_O = [d_model, d_head] @ [d_head, d_model] = [d_model, d_model]
    ov_full = W_V[head] @ W_O[head]
    ov_trace = float(np.trace(ov_full))

    top_copied.sort(key=lambda x: -x[3])

    return {
        "copy_score": copy_score,
        "top_copied_tokens": top_copied[:10],
        "ov_eigenvalues": top_sv,
        "token_copy_accuracy": copy_accuracy,
        "ov_trace": ov_trace,
    }


def prefix_search_analysis(model, tokens):
    """Analyze how the model searches for matching prefixes.

    For induction to work, the model needs to find positions where the current
    prefix has appeared before. This function measures how well each head
    performs this prefix matching at different offsets.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len] (should contain repeated subsequences).

    Returns:
        dict with:
            prefix_match_scores: [n_layers, n_heads] how well each head matches prefixes
            offset_profiles: dict of head -> [max_offset] attention to prefix matches
            best_prefix_heads: list of (layer, head, score)
            prefix_length_sensitivity: [n_layers, n_heads] sensitivity to prefix length
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    tokens_np = np.array(tokens)
    prefix_scores = np.zeros((n_layers, n_heads))
    prefix_sensitivity = np.zeros((n_layers, n_heads))
    offset_profiles = {}

    max_offset = min(5, seq_len // 2)

    for layer in range(n_layers):
        pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is None:
            continue
        pat = np.array(pattern)

        for head in range(n_heads):
            offsets = np.zeros(max_offset)
            offset_counts = np.zeros(max_offset)

            for pos in range(2, seq_len):
                # Check for prefix matches of various lengths
                for prev_start in range(pos - 1):
                    match_len = 0
                    while (match_len < max_offset and
                           pos - 1 - match_len >= 0 and
                           prev_start + match_len < pos - 1 and
                           tokens_np[prev_start + match_len] == tokens_np[pos - 1 - match_len + match_len]):
                        # Simpler: check if token at prev_start matches token at pos-1
                        if match_len == 0 and tokens_np[prev_start] == tokens_np[pos - 1]:
                            match_len = 1
                        else:
                            break

                    if match_len > 0 and prev_start + match_len < seq_len:
                        target = prev_start + match_len
                        if target <= pos:
                            attn = pat[head, pos, target]
                            for o in range(match_len):
                                if o < max_offset:
                                    offsets[o] += attn
                                    offset_counts[o] += 1

            # Normalize
            for o in range(max_offset):
                if offset_counts[o] > 0:
                    offsets[o] /= offset_counts[o]

            prefix_scores[layer, head] = float(np.mean(offsets))
            prefix_sensitivity[layer, head] = float(np.std(offsets)) if max_offset > 1 else 0.0
            offset_profiles[(layer, head)] = offsets.copy()

    flat = [(int(l), int(h), float(prefix_scores[l, h]))
            for l in range(n_layers) for h in range(n_heads)]
    best = sorted(flat, key=lambda x: -x[2])[:10]

    return {
        "prefix_match_scores": prefix_scores,
        "offset_profiles": offset_profiles,
        "best_prefix_heads": best,
        "prefix_length_sensitivity": prefix_sensitivity,
    }


def induction_circuit_completeness(model, tokens, metric_fn):
    """Test how complete the induction circuit is by ablating components.

    Ablates candidate previous-token heads and induction heads to measure
    how much of the model's induction behavior depends on the identified circuit.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function mapping logits to scalar.

    Returns:
        dict with:
            base_metric: float
            circuit_heads: list of identified circuit heads
            ablation_effects: dict of head -> metric change when ablated
            circuit_faithfulness: float (fraction of metric explained by circuit)
            redundancy_score: float (overlap between circuit and non-circuit)
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Get base metric
    base_logits = np.array(model(tokens))
    base_metric = float(metric_fn(base_logits))

    # First identify induction heads
    path_result = induction_circuit_path_tracing(model, tokens)
    prev_scores = path_result["prev_token_scores"]
    ind_scores = path_result["induction_scores"]

    # Identify circuit heads (threshold: above mean)
    circuit_heads = []
    for l in range(n_layers):
        for h in range(n_heads):
            if prev_scores[l, h] > np.mean(prev_scores) + 0.5 * np.std(prev_scores):
                circuit_heads.append(("prev_token", l, h))
            if ind_scores[l, h] > np.mean(ind_scores) + 0.5 * np.std(ind_scores):
                circuit_heads.append(("induction", l, h))

    # Ablate each circuit head
    ablation_effects = {}
    for role, layer, head in circuit_heads:
        hook_key = f"blocks.{layer}.hook_z"
        h = head
        def make_hook(h_idx):
            def hook_fn(x, name):
                return x.at[:, h_idx, :].set(0.0)
            return hook_fn
        state = HookState(hook_fns={hook_key: make_hook(h)}, cache={})
        abl_logits = np.array(model(tokens, hook_state=state))
        effect = base_metric - float(metric_fn(abl_logits))
        ablation_effects[(role, layer, head)] = effect

    # Ablate all circuit heads simultaneously
    hook_fns = {}
    for role, layer, head in circuit_heads:
        hook_key = f"blocks.{layer}.hook_z"
        h = head
        if hook_key not in hook_fns:
            heads_to_ablate = [hd for (_, l, hd) in circuit_heads if l == layer]
            def make_multi_hook(heads_list):
                def hook_fn(x, name):
                    for hh in heads_list:
                        x = x.at[:, hh, :].set(0.0)
                    return x
                return hook_fn
            hook_fns[hook_key] = make_multi_hook(heads_to_ablate)

    if hook_fns:
        state = HookState(hook_fns=hook_fns, cache={})
        full_abl_logits = np.array(model(tokens, hook_state=state))
        full_abl_metric = float(metric_fn(full_abl_logits))
        circuit_effect = base_metric - full_abl_metric
    else:
        circuit_effect = 0.0

    # Faithfulness: how much of the metric change is explained by the circuit
    total_individual = sum(abs(v) for v in ablation_effects.values())
    faithfulness = abs(circuit_effect) / (abs(base_metric) + 1e-10)

    # Redundancy: ratio of total individual effects to combined effect
    redundancy = 1.0 - abs(circuit_effect) / (total_individual + 1e-10) if total_individual > 0 else 0.0

    return {
        "base_metric": base_metric,
        "circuit_heads": circuit_heads,
        "ablation_effects": ablation_effects,
        "circuit_faithfulness": faithfulness,
        "redundancy_score": float(np.clip(redundancy, 0, 1)),
    }
