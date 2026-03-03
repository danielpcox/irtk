"""Representation bottleneck analysis.

Analyzes information bottlenecks in the residual stream: layer compression,
representational capacity, effective dimensionality changes, and redundancy.

References:
    Shwartz-Ziv & Tishby (2017) "Opening the Black Box of Deep Neural Networks via Information"
    Saxe et al. (2019) "On the Information Bottleneck Theory of Deep Learning"
"""

import jax
import jax.numpy as jnp
import numpy as np


def layer_compression_analysis(model, tokens_list, pos=-1):
    """Measure representational compression at each layer.

    Uses PCA to estimate effective dimensionality at each layer,
    revealing where the model compresses or expands representations.

    Args:
        model: HookedTransformer model.
        tokens_list: List of token arrays for sampling activations.
        pos: Position to analyze.

    Returns:
        dict with:
            effective_dims: array [n_layers+1] of effective dimensionality
            compression_ratio: array [n_layers] of dim[l]/dim[l-1]
            bottleneck_layer: int, layer with minimum dimensionality
            expansion_layer: int, layer with maximum dimensionality increase
            total_compression: float, ratio of final to initial dim
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    # Collect activations at each layer
    layer_acts = {l: [] for l in range(n_layers + 1)}

    for tokens in tokens_list:
        state = HookState(hook_fns={}, cache={})
        model(tokens, hook_state=state)
        cache = state.cache

        for layer in range(n_layers + 1):
            if layer == 0:
                key = "blocks.0.hook_resid_pre"
            else:
                key = f"blocks.{layer - 1}.hook_resid_post"
            resid = cache.get(key)
            if resid is not None:
                layer_acts[layer].append(np.array(resid[pos]))

    effective_dims = np.zeros(n_layers + 1)

    for layer in range(n_layers + 1):
        acts = layer_acts[layer]
        if len(acts) < 2:
            effective_dims[layer] = float(model.cfg.d_model)
            continue

        matrix = np.stack(acts)
        centered = matrix - np.mean(matrix, axis=0, keepdims=True)
        S = np.linalg.svd(centered, compute_uv=False)
        S2 = S ** 2
        total = np.sum(S2) + 1e-10
        probs = S2 / total
        probs = probs[probs > 1e-12]
        effective_dims[layer] = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))

    # Compression ratios
    compression = np.zeros(n_layers)
    for l in range(n_layers):
        if effective_dims[l] > 1e-10:
            compression[l] = effective_dims[l + 1] / effective_dims[l]

    bottleneck = int(np.argmin(effective_dims))
    expansion_ratios = np.diff(effective_dims)
    expansion = int(np.argmax(expansion_ratios)) if len(expansion_ratios) > 0 else 0

    total_comp = effective_dims[-1] / (effective_dims[0] + 1e-10)

    return {
        "effective_dims": effective_dims,
        "compression_ratio": compression,
        "bottleneck_layer": bottleneck,
        "expansion_layer": expansion,
        "total_compression": float(total_comp),
    }


def representational_capacity(model, tokens, pos=-1):
    """Estimate representational capacity at each layer.

    Measures how much of the available d_model space is actively used
    by computing the fraction of significant singular values.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.

    Returns:
        dict with:
            utilization: array [n_layers+1] of fraction of d_model used
            top_sv_fraction: array [n_layers+1] of top singular value / total
            capacity_bits: array [n_layers+1] of log2 of effective dim
            most_utilized_layer: int
            least_utilized_layer: int
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    seq_len = len(tokens)

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    utilization = np.zeros(n_layers + 1)
    top_sv_frac = np.zeros(n_layers + 1)
    capacity = np.zeros(n_layers + 1)

    for layer in range(n_layers + 1):
        if layer == 0:
            key = "blocks.0.hook_resid_pre"
        else:
            key = f"blocks.{layer - 1}.hook_resid_post"
        resid = cache.get(key)
        if resid is None:
            continue

        # Use all positions as samples
        matrix = np.array(resid)  # [seq_len, d_model]
        centered = matrix - np.mean(matrix, axis=0, keepdims=True)

        if seq_len < 2:
            utilization[layer] = 1.0
            capacity[layer] = np.log2(d_model)
            continue

        S = np.linalg.svd(centered, compute_uv=False)
        S2 = S ** 2
        total = np.sum(S2) + 1e-10

        # Effective rank
        probs = S2 / total
        probs = probs[probs > 1e-12]
        eff_rank = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))
        utilization[layer] = eff_rank / d_model
        top_sv_frac[layer] = float(S2[0] / total)
        capacity[layer] = float(np.log2(max(1, eff_rank)))

    most_util = int(np.argmax(utilization))
    least_util = int(np.argmin(utilization))

    return {
        "utilization": utilization,
        "top_sv_fraction": top_sv_frac,
        "capacity_bits": capacity,
        "most_utilized_layer": most_util,
        "least_utilized_layer": least_util,
    }


def redundancy_analysis(model, tokens, pos=-1):
    """Detect redundancy between consecutive layer representations.

    Measures how much information at each layer is already present
    in the previous layer.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.

    Returns:
        dict with:
            cosine_similarities: array [n_layers] of similarity between consecutive layers
            residual_norms: array [n_layers] of norm added by each layer
            relative_change: array [n_layers] of ||delta|| / ||resid||
            most_redundant_layer: int
            most_transformative_layer: int
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    states = []
    for layer in range(n_layers + 1):
        if layer == 0:
            key = "blocks.0.hook_resid_pre"
        else:
            key = f"blocks.{layer - 1}.hook_resid_post"
        resid = cache.get(key)
        if resid is not None:
            states.append(np.array(resid[pos]))
        else:
            states.append(np.zeros(model.cfg.d_model))

    cos_sims = np.zeros(n_layers)
    resid_norms = np.zeros(n_layers)
    rel_change = np.zeros(n_layers)

    for l in range(n_layers):
        a = states[l]
        b = states[l + 1]
        na = np.linalg.norm(a) + 1e-10
        nb = np.linalg.norm(b) + 1e-10
        cos_sims[l] = float(np.dot(a, b) / (na * nb))
        delta = b - a
        resid_norms[l] = float(np.linalg.norm(delta))
        rel_change[l] = float(np.linalg.norm(delta) / nb)

    most_redundant = int(np.argmax(cos_sims))
    most_transform = int(np.argmax(rel_change))

    return {
        "cosine_similarities": cos_sims,
        "residual_norms": resid_norms,
        "relative_change": rel_change,
        "most_redundant_layer": most_redundant,
        "most_transformative_layer": most_transform,
    }


def information_flow_bottleneck(model, tokens, pos=-1):
    """Identify information flow bottlenecks via attention and MLP analysis.

    Measures how much information passes through attention vs MLP at each
    layer and identifies bottleneck points.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.

    Returns:
        dict with:
            attn_info_fraction: array [n_layers] of attention contribution fraction
            mlp_info_fraction: array [n_layers] of MLP contribution fraction
            attn_norms: array [n_layers]
            mlp_norms: array [n_layers]
            bottleneck_layer: int, layer with smallest total throughput
            dominant_pathway: str, "attention" or "mlp"
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    attn_norms = np.zeros(n_layers)
    mlp_norms = np.zeros(n_layers)

    for layer in range(n_layers):
        attn_out = cache.get(f"blocks.{layer}.hook_attn_out")
        mlp_out = cache.get(f"blocks.{layer}.hook_mlp_out")
        if attn_out is not None:
            attn_norms[layer] = float(np.linalg.norm(np.array(attn_out[pos])))
        if mlp_out is not None:
            mlp_norms[layer] = float(np.linalg.norm(np.array(mlp_out[pos])))

    total = attn_norms + mlp_norms + 1e-10
    attn_frac = attn_norms / total
    mlp_frac = mlp_norms / total

    bottleneck = int(np.argmin(total))
    dominant = "attention" if np.sum(attn_norms) > np.sum(mlp_norms) else "mlp"

    return {
        "attn_info_fraction": attn_frac,
        "mlp_info_fraction": mlp_frac,
        "attn_norms": attn_norms,
        "mlp_norms": mlp_norms,
        "bottleneck_layer": bottleneck,
        "dominant_pathway": dominant,
    }


def cross_position_redundancy(model, tokens):
    """Measure redundancy across positions in the residual stream.

    Analyzes how similar representations are across positions at each layer.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            mean_pairwise_similarity: array [n_layers+1]
            position_effective_rank: array [n_layers+1]
            most_redundant_layer: int
            most_diverse_layer: int
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    mean_sim = np.zeros(n_layers + 1)
    eff_rank = np.zeros(n_layers + 1)

    for layer in range(n_layers + 1):
        if layer == 0:
            key = "blocks.0.hook_resid_pre"
        else:
            key = f"blocks.{layer - 1}.hook_resid_post"
        resid = cache.get(key)
        if resid is None:
            continue

        matrix = np.array(resid)  # [seq_len, d_model]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        normed = matrix / norms
        sim = normed @ normed.T

        seq_len = matrix.shape[0]
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        mean_sim[layer] = float(np.mean(sim[mask])) if np.any(mask) else 0.0

        # Position effective rank
        centered = matrix - np.mean(matrix, axis=0, keepdims=True)
        S = np.linalg.svd(centered, compute_uv=False)
        S2 = S ** 2
        total = np.sum(S2) + 1e-10
        probs = S2 / total
        probs = probs[probs > 1e-12]
        eff_rank[layer] = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))

    most_redundant = int(np.argmax(mean_sim))
    most_diverse = int(np.argmin(mean_sim))

    return {
        "mean_pairwise_similarity": mean_sim,
        "position_effective_rank": eff_rank,
        "most_redundant_layer": most_redundant,
        "most_diverse_layer": most_diverse,
    }
