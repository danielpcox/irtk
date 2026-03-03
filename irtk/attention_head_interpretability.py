"""Attention head interpretability tools.

Per-head interpretability: function classification heuristics, entropy-behavior
mapping, QK/OV readable summaries, importance vs interpretability tradeoff,
and head summary cards.

These tools help build intuition about what individual attention heads do,
complementing quantitative analysis with qualitative interpretability.
"""

import jax
import jax.numpy as jnp
import numpy as np


def head_function_classification(model, tokens):
    """Classify each head's function using behavioral heuristics.

    Applies a battery of tests to categorize heads into functional types:
    previous-token, induction, positional, content-based, inhibition, etc.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            classifications: dict of (layer, head) -> primary function label
            scores: dict of (layer, head) -> dict of function -> score
            function_counts: dict of function -> count
            confidence: dict of (layer, head) -> float (confidence in classification)
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    tokens_np = np.array(tokens)
    classifications = {}
    all_scores = {}
    confidences = {}

    for layer in range(n_layers):
        pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is None:
            continue
        pat = np.array(pattern)

        for head in range(n_heads):
            scores = {}

            # 1. Previous-token score
            prev_score = 0.0
            for pos in range(1, seq_len):
                prev_score += pat[head, pos, pos - 1]
            scores["previous_token"] = prev_score / max(seq_len - 1, 1)

            # 2. Positional (diagonal) score
            diag_score = 0.0
            for pos in range(seq_len):
                diag_score += pat[head, pos, pos]
            scores["self_attention"] = diag_score / max(seq_len, 1)

            # 3. BOS/first token attention
            bos_score = 0.0
            for pos in range(1, seq_len):
                bos_score += pat[head, pos, 0]
            scores["bos_attention"] = bos_score / max(seq_len - 1, 1)

            # 4. Induction score
            ind_score = 0.0
            ind_count = 0
            for pos in range(2, seq_len):
                for prev in range(pos - 1):
                    if tokens_np[prev] == tokens_np[pos - 1] and prev + 1 < pos:
                        ind_score += pat[head, pos, prev + 1]
                        ind_count += 1
            scores["induction"] = ind_score / max(ind_count, 1)

            # 5. Entropy (spread of attention)
            entropy = 0.0
            for pos in range(seq_len):
                p = pat[head, pos, :pos + 1]
                p = p[p > 1e-10]
                entropy -= float(np.sum(p * np.log(p)))
            scores["high_entropy"] = entropy / max(seq_len, 1)

            # 6. Local window attention (within 3 positions)
            local_score = 0.0
            for pos in range(seq_len):
                for offset in range(max(0, pos - 2), pos + 1):
                    local_score += pat[head, pos, offset]
            scores["local_window"] = local_score / max(seq_len, 1)

            # Classify based on highest score
            # Normalize by expected baselines
            func_scores = {
                "previous_token": scores["previous_token"],
                "self_attention": scores["self_attention"],
                "bos_attention": scores["bos_attention"],
                "induction": scores["induction"],
                "local_window": scores["local_window"] - scores["self_attention"] - scores["previous_token"],
                "distributed": scores["high_entropy"],
            }

            best_func = max(func_scores, key=func_scores.get)
            best_score = func_scores[best_func]
            second_best = sorted(func_scores.values())[-2]

            classifications[(layer, head)] = best_func
            all_scores[(layer, head)] = scores
            confidences[(layer, head)] = float(best_score - second_best) if best_score > 0 else 0.0

    # Count functions
    func_counts = {}
    for func in classifications.values():
        func_counts[func] = func_counts.get(func, 0) + 1

    return {
        "classifications": classifications,
        "scores": all_scores,
        "function_counts": func_counts,
        "confidence": confidences,
    }


def entropy_behavior_mapping(model, tokens):
    """Map attention entropy to behavioral characteristics for each head.

    Heads with low entropy focus on few positions (specialized), while
    high-entropy heads distribute attention broadly (aggregating).

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            head_entropy: [n_layers, n_heads] mean entropy per head
            entropy_by_position: [n_layers, n_heads, seq_len] entropy at each position
            entropy_categories: dict of (layer, head) -> "focused"/"moderate"/"diffuse"
            focus_positions: dict of (layer, head) -> list of most-attended positions
            entropy_variance: [n_layers, n_heads] variance of entropy across positions
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    head_entropy = np.zeros((n_layers, n_heads))
    entropy_by_pos = np.zeros((n_layers, n_heads, seq_len))
    categories = {}
    focus_positions = {}
    entropy_var = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is None:
            continue
        pat = np.array(pattern)

        for head in range(n_heads):
            entropies = []
            for pos in range(seq_len):
                p = pat[head, pos, :pos + 1]
                p = p[p > 1e-10]
                h = -float(np.sum(p * np.log(p + 1e-10)))
                entropy_by_pos[layer, head, pos] = h
                entropies.append(h)

            mean_h = float(np.mean(entropies))
            head_entropy[layer, head] = mean_h
            entropy_var[layer, head] = float(np.var(entropies))

            # Categorize
            max_possible = np.log(seq_len)
            ratio = mean_h / (max_possible + 1e-10)
            if ratio < 0.3:
                categories[(layer, head)] = "focused"
            elif ratio < 0.6:
                categories[(layer, head)] = "moderate"
            else:
                categories[(layer, head)] = "diffuse"

            # Find focus positions (where does the head attend most on average?)
            avg_attn = np.mean(pat[head], axis=0)  # average over query positions
            top_pos = np.argsort(-avg_attn)[:3]
            focus_positions[(layer, head)] = [int(p) for p in top_pos]

    return {
        "head_entropy": head_entropy,
        "entropy_by_position": entropy_by_pos,
        "entropy_categories": categories,
        "focus_positions": focus_positions,
        "entropy_variance": entropy_var,
    }


def qk_ov_summary(model, layer, head, top_k=5):
    """Generate readable summaries of a head's QK and OV circuits.

    The QK circuit determines "what to attend to" and the OV circuit
    determines "what to write to the residual stream."

    Args:
        model: HookedTransformer model.
        layer: Layer of the head.
        head: Head index.
        top_k: Number of top interactions to show.

    Returns:
        dict with:
            qk_top_interactions: list of (query_dir, key_dir, score) — top QK alignments
            ov_top_mappings: list of (input_dir, output_dir, score) — top OV mappings
            qk_rank: effective rank of QK circuit
            ov_rank: effective rank of OV circuit
            qk_singular_values: top singular values of W_Q^T W_K
            ov_singular_values: top singular values of W_O W_V
    """
    W_Q = np.array(model.blocks[layer].attn.W_Q)  # [n_heads, d_model, d_head]
    W_K = np.array(model.blocks[layer].attn.W_K)
    W_V = np.array(model.blocks[layer].attn.W_V)
    W_O = np.array(model.blocks[layer].attn.W_O)  # [n_heads, d_head, d_model]

    d_model = W_Q.shape[1]
    d_head = W_Q.shape[2]

    # QK circuit: W_Q[h]^T @ W_K[h] gives [d_head, d_head]
    # Full QK: x_q @ W_Q @ W_K^T @ x_k^T
    qk_matrix = W_Q[head] @ W_K[head].T  # [d_model, d_model]
    U_qk, S_qk, Vt_qk = np.linalg.svd(qk_matrix, full_matrices=False)

    # OV circuit: W_V[h] @ W_O[h] gives [d_model, d_model]
    ov_matrix = W_V[head] @ W_O[head]  # [d_model, d_model]
    U_ov, S_ov, Vt_ov = np.linalg.svd(ov_matrix, full_matrices=False)

    # Top QK interactions (singular vector pairs)
    qk_interactions = []
    for i in range(min(top_k, len(S_qk))):
        qk_interactions.append((
            U_qk[:, i].tolist()[:3],  # first 3 dims of query direction
            Vt_qk[i, :].tolist()[:3],  # first 3 dims of key direction
            float(S_qk[i])
        ))

    # Top OV mappings
    ov_mappings = []
    for i in range(min(top_k, len(S_ov))):
        ov_mappings.append((
            Vt_ov[i, :].tolist()[:3],  # input direction
            U_ov[:, i].tolist()[:3],  # output direction
            float(S_ov[i])
        ))

    # Effective rank (based on singular value distribution)
    def eff_rank(sv):
        sv = sv[sv > 1e-10]
        if len(sv) == 0:
            return 0.0
        p = sv / np.sum(sv)
        return float(np.exp(-np.sum(p * np.log(p + 1e-10))))

    return {
        "qk_top_interactions": qk_interactions,
        "ov_top_mappings": ov_mappings,
        "qk_rank": eff_rank(S_qk),
        "ov_rank": eff_rank(S_ov),
        "qk_singular_values": S_qk[:top_k].tolist(),
        "ov_singular_values": S_ov[:top_k].tolist(),
    }


def importance_interpretability_tradeoff(model, tokens, metric_fn):
    """Analyze the tradeoff between head importance and interpretability.

    Important heads (high metric effect when ablated) may or may not be
    interpretable (clear, focused attention patterns). This function
    quantifies both dimensions.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function mapping logits to scalar.

    Returns:
        dict with:
            importance: [n_layers, n_heads] metric effect of ablation
            interpretability: [n_layers, n_heads] pattern clarity score
            tradeoff_correlation: float (correlation between importance and interpretability)
            important_uninterpretable: list of (layer, head) high importance, low clarity
            unimportant_interpretable: list of (layer, head) low importance, high clarity
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    # Get base metric
    base_logits = np.array(model(tokens))
    base_metric = float(metric_fn(base_logits))

    # Importance via ablation
    importance = np.zeros((n_layers, n_heads))

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    for layer in range(n_layers):
        for head in range(n_heads):
            hook_key = f"blocks.{layer}.hook_z"
            h = head
            def make_hook(h_idx):
                def hook_fn(x, name):
                    return x.at[:, h_idx, :].set(0.0)
                return hook_fn
            state = HookState(hook_fns={hook_key: make_hook(h)}, cache={})
            abl_logits = np.array(model(tokens, hook_state=state))
            importance[layer, head] = abs(base_metric - float(metric_fn(abl_logits)))

    # Interpretability via pattern clarity
    interpretability = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is None:
            continue
        pat = np.array(pattern)

        for head in range(n_heads):
            # Clarity = 1 - normalized entropy (low entropy = high clarity)
            total_clarity = 0.0
            for pos in range(seq_len):
                p = pat[head, pos, :pos + 1]
                p = p[p > 1e-10]
                h = -float(np.sum(p * np.log(p + 1e-10)))
                max_h = np.log(pos + 1) if pos > 0 else 1.0
                total_clarity += 1.0 - h / (max_h + 1e-10)
            interpretability[layer, head] = total_clarity / max(seq_len, 1)

    # Correlation
    imp_flat = importance.flatten()
    int_flat = interpretability.flatten()
    if np.std(imp_flat) > 1e-10 and np.std(int_flat) > 1e-10:
        corr = float(np.corrcoef(imp_flat, int_flat)[0, 1])
    else:
        corr = 0.0

    # Find tradeoff quadrants
    imp_thresh = np.median(imp_flat)
    int_thresh = np.median(int_flat)

    important_uninterpretable = []
    unimportant_interpretable = []

    for l in range(n_layers):
        for h in range(n_heads):
            if importance[l, h] > imp_thresh and interpretability[l, h] < int_thresh:
                important_uninterpretable.append((l, h))
            if importance[l, h] < imp_thresh and interpretability[l, h] > int_thresh:
                unimportant_interpretable.append((l, h))

    return {
        "importance": importance,
        "interpretability": interpretability,
        "tradeoff_correlation": corr,
        "important_uninterpretable": important_uninterpretable,
        "unimportant_interpretable": unimportant_interpretable,
    }


def head_summary_card(model, tokens, layer, head, metric_fn=None):
    """Generate a comprehensive summary card for a single attention head.

    Combines multiple analyses into one digestible summary.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        layer: Layer index.
        head: Head index.
        metric_fn: Optional metric function for importance scoring.

    Returns:
        dict with:
            identity: (layer, head) tuple
            function_type: str (classification label)
            entropy_category: str ("focused"/"moderate"/"diffuse")
            mean_entropy: float
            top_attended_positions: list of int
            qk_rank: float
            ov_rank: float
            importance: float (if metric_fn provided)
            clarity: float (pattern clarity score)
            top_qk_sv: list of float
            top_ov_sv: list of float
    """
    from irtk.hook_points import HookState

    seq_len = len(tokens)

    # Get classification
    class_result = head_function_classification(model, tokens)
    func_type = class_result["classifications"].get((layer, head), "unknown")

    # Get entropy info
    entropy_result = entropy_behavior_mapping(model, tokens)
    entropy_cat = entropy_result["entropy_categories"].get((layer, head), "unknown")
    mean_ent = float(entropy_result["head_entropy"][layer, head])
    focus = entropy_result["focus_positions"].get((layer, head), [])

    # Get QK/OV summary
    circuit_result = qk_ov_summary(model, layer, head)

    # Clarity
    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    clarity = 0.0
    pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
    if pattern is not None:
        pat = np.array(pattern)
        for pos in range(seq_len):
            p = pat[head, pos, :pos + 1]
            p = p[p > 1e-10]
            h = -float(np.sum(p * np.log(p + 1e-10)))
            max_h = np.log(pos + 1) if pos > 0 else 1.0
            clarity += 1.0 - h / (max_h + 1e-10)
        clarity /= max(seq_len, 1)

    # Importance
    importance = 0.0
    if metric_fn is not None:
        base_logits = np.array(model(tokens))
        base_metric = float(metric_fn(base_logits))
        h = head
        hook_key = f"blocks.{layer}.hook_z"
        def make_hook(h_idx):
            def hook_fn(x, name):
                return x.at[:, h_idx, :].set(0.0)
            return hook_fn
        state = HookState(hook_fns={hook_key: make_hook(h)}, cache={})
        abl_logits = np.array(model(tokens, hook_state=state))
        importance = abs(base_metric - float(metric_fn(abl_logits)))

    return {
        "identity": (layer, head),
        "function_type": func_type,
        "entropy_category": entropy_cat,
        "mean_entropy": mean_ent,
        "top_attended_positions": focus,
        "qk_rank": circuit_result["qk_rank"],
        "ov_rank": circuit_result["ov_rank"],
        "importance": importance,
        "clarity": clarity,
        "top_qk_sv": circuit_result["qk_singular_values"],
        "top_ov_sv": circuit_result["ov_singular_values"],
    }
