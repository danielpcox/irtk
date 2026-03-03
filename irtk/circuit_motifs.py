"""Common circuit motif detection and analysis.

Detect and characterize recurring circuit patterns in transformers:
skip trigram circuits, negative name movers, backup/redundancy circuits,
signal boosting patterns, and automated motif cataloging.

References:
    Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"
    Wang et al. (2023) "Interpretability in the Wild" (IOI circuit)
"""

import jax
import jax.numpy as jnp
import numpy as np


def skip_trigram_detection(model, tokens, metric_fn):
    """Detect skip-trigram circuit patterns.

    A skip trigram circuit is one where the model predicts a token based
    on a pattern like A...B -> C, skipping intermediate tokens. This tests
    for attention heads that implement this pattern.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function mapping logits to scalar.

    Returns:
        dict with:
            skip_scores: [n_layers, n_heads] how much each head skips positions
            long_range_heads: list of (layer, head, avg_distance)
            skip_attention_profile: [n_layers, n_heads] mean distance attended to
            direct_vs_skip: [n_layers, n_heads] ratio of local to skip attention
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    skip_scores = np.zeros((n_layers, n_heads))
    mean_dist = np.zeros((n_layers, n_heads))
    direct_vs_skip = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is None:
            continue
        pat = np.array(pattern)

        for head in range(n_heads):
            total_dist = 0.0
            local_attn = 0.0
            skip_attn = 0.0
            count = 0

            for pos in range(1, seq_len):
                for src in range(pos):
                    dist = pos - src
                    attn = pat[head, pos, src]
                    total_dist += dist * attn
                    if dist <= 2:
                        local_attn += attn
                    else:
                        skip_attn += attn
                    count += 1

            mean_dist[layer, head] = total_dist / max(seq_len - 1, 1)
            skip_scores[layer, head] = skip_attn / max(seq_len - 1, 1)
            total = local_attn + skip_attn
            if total > 1e-10:
                direct_vs_skip[layer, head] = local_attn / total
            else:
                direct_vs_skip[layer, head] = 0.5

    # Find long-range heads
    flat = [(int(l), int(h), float(mean_dist[l, h]))
            for l in range(n_layers) for h in range(n_heads)]
    long_range = sorted(flat, key=lambda x: -x[2])[:10]

    return {
        "skip_scores": skip_scores,
        "long_range_heads": long_range,
        "skip_attention_profile": mean_dist,
        "direct_vs_skip": direct_vs_skip,
    }


def negative_mover_detection(model, tokens, pos=-1, top_k=5):
    """Detect negative name mover heads.

    Negative movers are heads whose direct logit effect suppresses
    the correct answer, often discovered in IOI circuits. They attend
    to duplicate tokens and write negative contributions.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Target position to analyze.
        top_k: Top results per head.

    Returns:
        dict with:
            head_logit_effects: dict of (layer, head) -> [d_vocab] logit contributions
            negative_heads: list of (layer, head, suppression_score) top negative movers
            positive_heads: list of (layer, head, boost_score) top positive movers
            suppression_per_token: dict of (layer, head) -> list of (token, logit) most suppressed
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]

    head_effects = {}
    neg_scores = []
    pos_scores = []
    suppression = {}

    for layer in range(n_layers):
        z = cache.get(f"blocks.{layer}.attn.hook_z")
        if z is None:
            continue
        z_arr = np.array(z)
        W_O = np.array(model.blocks[layer].attn.W_O)

        for head in range(n_heads):
            output = z_arr[pos, head] @ W_O[head]  # [d_model]
            logits = output @ W_U  # [d_vocab]
            head_effects[(layer, head)] = logits

            mean_logit = float(np.mean(logits))
            neg_scores.append((layer, head, float(np.min(logits))))
            pos_scores.append((layer, head, float(np.max(logits))))

            # Most suppressed tokens
            bottom = np.argsort(logits)[:top_k]
            suppression[(layer, head)] = [(int(t), float(logits[t])) for t in bottom]

    neg_heads = sorted(neg_scores, key=lambda x: x[2])[:10]
    pos_heads = sorted(pos_scores, key=lambda x: -x[2])[:10]

    return {
        "head_logit_effects": head_effects,
        "negative_heads": neg_heads,
        "positive_heads": pos_heads,
        "suppression_per_token": suppression,
    }


def backup_circuit_detection(model, tokens, metric_fn):
    """Detect backup/redundancy circuits in the model.

    Tests whether ablating one head causes another to compensate,
    indicating backup circuit patterns.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function mapping logits to scalar.

    Returns:
        dict with:
            single_ablation_effects: [n_layers, n_heads] effect of ablating each head
            compensation_matrix: [total_heads, total_heads] how much head j compensates for head i
            backup_pairs: list of ((layer_i, head_i), (layer_j, head_j), compensation)
            redundancy_score: float (overall circuit redundancy)
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    total = n_layers * n_heads

    base_logits = np.array(model(tokens))
    base_metric = float(metric_fn(base_logits))

    # Single ablation effects
    single_effects = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        for head in range(n_heads):
            h = head
            def make_hook(h_idx):
                def hook_fn(x, name):
                    return x.at[:, h_idx, :].set(0.0)
                return hook_fn
            state = HookState(hook_fns={f"blocks.{layer}.hook_z": make_hook(h)}, cache={})
            abl_logits = np.array(model(tokens, hook_state=state))
            single_effects[layer, head] = base_metric - float(metric_fn(abl_logits))

    # Find top important heads for double ablation test
    flat_effects = [(l, h, abs(single_effects[l, h]))
                    for l in range(n_layers) for h in range(n_heads)]
    flat_effects.sort(key=lambda x: -x[2])
    top_heads = flat_effects[:min(6, total)]  # Test top 6

    # Double ablation to find compensation
    compensation_matrix = np.zeros((total, total))
    backup_pairs = []

    for li, hi, _ in top_heads:
        idx_i = li * n_heads + hi
        for lj, hj, _ in top_heads:
            if (li, hi) == (lj, hj):
                continue
            idx_j = lj * n_heads + hj

            # Ablate both
            hook_fns = {}
            for (l, h) in [(li, hi), (lj, hj)]:
                key = f"blocks.{l}.hook_z"
                if key in hook_fns:
                    prev = hook_fns[key]
                    hh = h
                    def make_combined(prev_fn, h_idx):
                        def hook_fn(x, name):
                            x = prev_fn(x, name)
                            return x.at[:, h_idx, :].set(0.0)
                        return hook_fn
                    hook_fns[key] = make_combined(prev, hh)
                else:
                    hh = h
                    def make_single(h_idx):
                        def hook_fn(x, name):
                            return x.at[:, h_idx, :].set(0.0)
                        return hook_fn
                    hook_fns[key] = make_single(hh)

            state = HookState(hook_fns=hook_fns, cache={})
            double_logits = np.array(model(tokens, hook_state=state))
            double_effect = base_metric - float(metric_fn(double_logits))

            # Compensation: if double effect < sum of singles, there's compensation
            expected = single_effects[li, hi] + single_effects[lj, hj]
            compensation = expected - double_effect
            compensation_matrix[idx_i, idx_j] = compensation

            if compensation > 0.01:
                backup_pairs.append(((li, hi), (lj, hj), float(compensation)))

    backup_pairs.sort(key=lambda x: -x[2])

    # Overall redundancy
    total_single = float(np.sum(np.abs(single_effects)))
    if total_single > 1e-10:
        redundancy = float(np.sum(np.maximum(compensation_matrix, 0))) / total_single
    else:
        redundancy = 0.0

    return {
        "single_ablation_effects": single_effects,
        "compensation_matrix": compensation_matrix,
        "backup_pairs": backup_pairs[:20],
        "redundancy_score": float(np.clip(redundancy, 0, 1)),
    }


def signal_boosting_detection(model, tokens, metric_fn):
    """Detect signal boosting patterns across layers.

    Signal boosting occurs when later layers amplify signals written
    by earlier layers, creating a progressive refinement pattern.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function mapping logits to scalar.

    Returns:
        dict with:
            layer_contributions: [n_layers] metric contribution per layer
            cumulative_signal: [n_layers] cumulative metric building through layers
            boosting_layers: list of int (layers that amplify previous signal)
            attenuation_layers: list of int (layers that reduce signal)
            signal_trajectory: [n_layers] signal strength at each layer
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    base_logits = np.array(model(tokens))
    base_metric = float(metric_fn(base_logits))

    # Measure metric using logit lens at each layer
    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') and model.unembed.b_U is not None else np.zeros(W_U.shape[1])

    signal_trajectory = np.zeros(n_layers)
    layer_contributions = np.zeros(n_layers)

    for layer in range(n_layers):
        resid = cache.get(f"blocks.{layer}.hook_resid_post")
        if resid is None:
            continue
        resid_arr = np.array(resid)
        logits = resid_arr @ W_U + b_U
        signal_trajectory[layer] = float(metric_fn(logits))

    # Layer contributions (difference from previous)
    for layer in range(n_layers):
        if layer == 0:
            # Get pre-layer residual
            resid_pre = cache.get("blocks.0.hook_resid_pre")
            if resid_pre is not None:
                pre_logits = np.array(resid_pre) @ W_U + b_U
                layer_contributions[0] = signal_trajectory[0] - float(metric_fn(pre_logits))
            else:
                layer_contributions[0] = signal_trajectory[0]
        else:
            layer_contributions[layer] = signal_trajectory[layer] - signal_trajectory[layer - 1]

    # Identify boosting and attenuation
    boosting = []
    attenuation = []
    for layer in range(1, n_layers):
        if layer_contributions[layer] > 0 and signal_trajectory[layer - 1] > 0:
            boosting.append(layer)
        elif layer_contributions[layer] < 0 and signal_trajectory[layer - 1] > 0:
            attenuation.append(layer)

    cumulative = np.cumsum(layer_contributions)

    return {
        "layer_contributions": layer_contributions,
        "cumulative_signal": cumulative,
        "boosting_layers": boosting,
        "attenuation_layers": attenuation,
        "signal_trajectory": signal_trajectory,
    }


def motif_catalog(model, tokens, metric_fn):
    """Generate a catalog of detected circuit motifs in the model.

    Runs multiple motif detectors and summarizes findings into a
    comprehensive catalog.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function mapping logits to scalar.

    Returns:
        dict with:
            motifs_found: list of dict (each with type, components, strength)
            total_motifs: int
            dominant_motif: str (most prevalent motif type)
            component_participation: dict of (layer, head) -> list of motif types
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    motifs = []
    participation = {}

    # Check for skip trigram patterns
    skip_result = skip_trigram_detection(model, tokens, metric_fn)
    for l, h, dist in skip_result["long_range_heads"][:3]:
        if dist > 2.0:
            motifs.append({
                "type": "skip_trigram",
                "components": [(l, h)],
                "strength": dist,
                "description": f"L{l}H{h} attends avg distance {dist:.1f}",
            })
            key = (l, h)
            participation.setdefault(key, []).append("skip_trigram")

    # Check for negative movers
    neg_result = negative_mover_detection(model, tokens)
    for l, h, score in neg_result["negative_heads"][:3]:
        if score < -0.01:
            motifs.append({
                "type": "negative_mover",
                "components": [(l, h)],
                "strength": abs(score),
                "description": f"L{l}H{h} suppresses with score {score:.4f}",
            })
            key = (l, h)
            participation.setdefault(key, []).append("negative_mover")

    # Check for signal boosting
    boost_result = signal_boosting_detection(model, tokens, metric_fn)
    if boost_result["boosting_layers"]:
        motifs.append({
            "type": "signal_boosting",
            "components": [(l, -1) for l in boost_result["boosting_layers"]],
            "strength": float(np.max(np.abs(boost_result["layer_contributions"]))),
            "description": f"Signal boosted at layers {boost_result['boosting_layers']}",
        })

    # Count motif types
    type_counts = {}
    for m in motifs:
        type_counts[m["type"]] = type_counts.get(m["type"], 0) + 1

    dominant = max(type_counts, key=type_counts.get) if type_counts else "none"

    return {
        "motifs_found": motifs,
        "total_motifs": len(motifs),
        "dominant_motif": dominant,
        "component_participation": participation,
    }
