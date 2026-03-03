"""Attention sparsity analysis.

Analyzes how sparse or concentrated attention patterns are: entropy-based
sparsity, attention mass distribution, sparse vs dense heads, effective
attention window.

References:
    Correia et al. (2019) "Adaptively Sparse Transformers"
    Shi et al. (2021) "Sparseformer: Sparse Attention for Visual Transformers"
"""

import jax
import jax.numpy as jnp
import numpy as np


def attention_entropy_profile(model, tokens):
    """Compute attention entropy for each head at each position.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            entropy: array [n_layers, n_heads, seq_len] of attention entropy
            mean_entropy: array [n_layers, n_heads] mean over positions
            sparsest_head: tuple (layer, head) with lowest mean entropy
            densest_head: tuple (layer, head) with highest mean entropy
            max_possible_entropy: float, log(seq_len)
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)

    entropy = np.zeros((n_layers, n_heads, seq_len))

    for layer in range(n_layers):
        pattern = cache_state.cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is not None:
            p = np.array(pattern)  # [n_heads, seq_len, seq_len]
            for h in range(n_heads):
                for q in range(seq_len):
                    probs = p[h, q, :q+1]  # causal: only attend to past
                    probs = probs + 1e-10
                    entropy[layer, h, q] = -float(np.sum(probs * np.log(probs)))

    mean_ent = np.mean(entropy, axis=2)
    sparsest = np.unravel_index(np.argmin(mean_ent), mean_ent.shape)
    densest = np.unravel_index(np.argmax(mean_ent), mean_ent.shape)

    return {
        "entropy": entropy,
        "mean_entropy": mean_ent,
        "sparsest_head": (int(sparsest[0]), int(sparsest[1])),
        "densest_head": (int(densest[0]), int(densest[1])),
        "max_possible_entropy": float(np.log(seq_len)),
    }


def attention_mass_distribution(model, tokens, top_k=3):
    """Analyze how attention mass is distributed.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        top_k: Number of top positions to measure mass concentration.

    Returns:
        dict with:
            top_k_mass: [n_layers, n_heads] fraction of mass in top-k positions
            gini_coefficient: [n_layers, n_heads] attention inequality
            max_attention: [n_layers, n_heads] maximum single attention weight
            effective_window: [n_layers, n_heads] effective number of attended positions
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)

    top_k_mass = np.zeros((n_layers, n_heads))
    gini = np.zeros((n_layers, n_heads))
    max_attn = np.zeros((n_layers, n_heads))
    eff_window = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        pattern = cache_state.cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is not None:
            p = np.array(pattern)
            for h in range(n_heads):
                # Average over query positions
                for q in range(seq_len):
                    row = p[h, q, :q+1]
                    sorted_row = np.sort(row)[::-1]
                    k = min(top_k, len(sorted_row))
                    top_k_mass[layer, h] += np.sum(sorted_row[:k])
                    max_attn[layer, h] += sorted_row[0]

                    # Gini coefficient
                    n = len(row)
                    if n > 1:
                        sorted_asc = np.sort(row)
                        index = np.arange(1, n + 1)
                        gini[layer, h] += float((2 * np.sum(index * sorted_asc) - (n + 1) * np.sum(sorted_asc)) / (n * np.sum(sorted_asc) + 1e-10))

                    # Effective window (exp of entropy)
                    probs = row + 1e-10
                    ent = -np.sum(probs * np.log(probs))
                    eff_window[layer, h] += np.exp(ent)

                top_k_mass[layer, h] /= seq_len
                max_attn[layer, h] /= seq_len
                gini[layer, h] /= seq_len
                eff_window[layer, h] /= seq_len

    return {
        "top_k_mass": top_k_mass,
        "gini_coefficient": gini,
        "max_attention": max_attn,
        "effective_window": eff_window,
    }


def sparse_vs_dense_heads(model, tokens, sparsity_threshold=0.7):
    """Classify heads as sparse or dense based on attention patterns.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        sparsity_threshold: Top-1 mass threshold for "sparse" classification.

    Returns:
        dict with:
            is_sparse: [n_layers, n_heads] boolean
            sparsity_scores: [n_layers, n_heads] float (higher = sparser)
            n_sparse: int
            n_dense: int
            sparse_heads: list of (layer, head) tuples
            dense_heads: list of (layer, head) tuples
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)

    sparsity = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        pattern = cache_state.cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is not None:
            p = np.array(pattern)
            for h in range(n_heads):
                total_max = 0.0
                for q in range(seq_len):
                    total_max += np.max(p[h, q, :q+1])
                sparsity[layer, h] = total_max / seq_len

    is_sparse = sparsity >= sparsity_threshold
    sparse_heads = [(int(l), int(h)) for l in range(n_layers) for h in range(n_heads) if is_sparse[l, h]]
    dense_heads = [(int(l), int(h)) for l in range(n_layers) for h in range(n_heads) if not is_sparse[l, h]]

    return {
        "is_sparse": is_sparse,
        "sparsity_scores": sparsity,
        "n_sparse": int(np.sum(is_sparse)),
        "n_dense": int(np.sum(~is_sparse)),
        "sparse_heads": sparse_heads,
        "dense_heads": dense_heads,
    }


def attention_window_analysis(model, tokens):
    """Analyze the effective attention window of each head.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            mean_distance: [n_layers, n_heads] mean attention distance
            median_distance: [n_layers, n_heads] median attention distance
            window_90: [n_layers, n_heads] distance covering 90% of attention mass
            local_fraction: [n_layers, n_heads] fraction attending to ±2 positions
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)

    mean_dist = np.zeros((n_layers, n_heads))
    median_dist = np.zeros((n_layers, n_heads))
    window_90 = np.zeros((n_layers, n_heads))
    local_frac = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        pattern = cache_state.cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is not None:
            p = np.array(pattern)
            for h in range(n_heads):
                total_mean = 0.0
                total_median = 0.0
                total_w90 = 0.0
                total_local = 0.0

                for q in range(seq_len):
                    row = p[h, q, :q+1]
                    distances = np.arange(q, -1, -1, dtype=float)  # distance from query

                    # Mean distance
                    total_mean += float(np.sum(row * distances))

                    # Median distance (weighted)
                    cum = np.cumsum(np.sort(row)[::-1])
                    sorted_dist = distances[np.argsort(row)[::-1]]
                    med_idx = np.searchsorted(cum, 0.5)
                    med_idx = min(med_idx, len(sorted_dist) - 1)
                    total_median += sorted_dist[med_idx]

                    # 90% window
                    w90_idx = np.searchsorted(cum, 0.9)
                    w90_idx = min(w90_idx, len(sorted_dist) - 1)
                    total_w90 += w90_idx + 1

                    # Local fraction (distance <= 2)
                    local_mask = distances <= 2
                    total_local += float(np.sum(row[local_mask]))

                mean_dist[layer, h] = total_mean / seq_len
                median_dist[layer, h] = total_median / seq_len
                window_90[layer, h] = total_w90 / seq_len
                local_frac[layer, h] = total_local / seq_len

    return {
        "mean_distance": mean_dist,
        "median_distance": median_dist,
        "window_90": window_90,
        "local_fraction": local_frac,
    }


def attention_pattern_stability(model, tokens_list):
    """Measure how stable attention patterns are across different inputs.

    Args:
        model: HookedTransformer model.
        tokens_list: List of input token arrays.

    Returns:
        dict with:
            pattern_variance: [n_layers, n_heads] variance across inputs
            stable_heads: list of (layer, head) with low variance
            unstable_heads: list with high variance
            mean_pattern_norm: [n_layers, n_heads]
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Collect patterns across inputs
    all_norms = []
    for tokens in tokens_list:
        cache_state = HookState(hook_fns={}, cache={})
        model(tokens, hook_state=cache_state)

        norms = np.zeros((n_layers, n_heads))
        for layer in range(n_layers):
            pattern = cache_state.cache.get(f"blocks.{layer}.attn.hook_pattern")
            if pattern is not None:
                p = np.array(pattern)
                for h in range(n_heads):
                    norms[layer, h] = float(np.linalg.norm(p[h]))
        all_norms.append(norms)

    all_norms = np.array(all_norms)
    variance = np.var(all_norms, axis=0)
    mean_norm = np.mean(all_norms, axis=0)

    threshold = np.mean(variance)
    stable = [(int(l), int(h)) for l in range(n_layers) for h in range(n_heads) if variance[l, h] < threshold]
    unstable = [(int(l), int(h)) for l in range(n_layers) for h in range(n_heads) if variance[l, h] >= threshold]

    return {
        "pattern_variance": variance,
        "stable_heads": stable,
        "unstable_heads": unstable,
        "mean_pattern_norm": mean_norm,
    }
