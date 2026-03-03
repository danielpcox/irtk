"""Activation subspace analysis.

Analyze the subspace structure of activations: which directions are used,
how subspaces overlap across layers, and projection analysis.
"""

import jax
import jax.numpy as jnp
from irtk.hook_points import HookState


def _run_and_cache(model, tokens):
    """Run model and return activation cache."""
    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    return hook_state.cache


def activation_pca(model, tokens, layer=0, n_components=5):
    """PCA of residual stream activations at a given layer.

    Returns:
        dict with:
        - explained_variance: fraction explained by each component
        - cumulative_variance: cumulative explained variance
        - effective_dimensionality: number of components for 90% variance
        - principal_components: top-n_components directions
    """
    cache = _run_and_cache(model, tokens)

    key = f'blocks.{layer}.hook_resid_post'
    if key not in cache:
        return {'explained_variance': [], 'cumulative_variance': [], 'effective_dimensionality': 0}

    resid = cache[key]  # [seq, d_model]
    resid_centered = resid - jnp.mean(resid, axis=0, keepdims=True)

    # SVD for PCA
    U, S, Vt = jnp.linalg.svd(resid_centered, full_matrices=False)

    total_var = jnp.sum(S ** 2)
    var_explained = S ** 2 / jnp.maximum(total_var, 1e-10)

    cumulative = jnp.cumsum(var_explained)
    eff_dim = int(jnp.searchsorted(cumulative, 0.9)) + 1

    n = min(n_components, len(S))
    return {
        'explained_variance': [float(v) for v in var_explained[:n]],
        'cumulative_variance': [float(c) for c in cumulative[:n]],
        'effective_dimensionality': eff_dim,
        'total_dimensions': len(S),
        'principal_components': Vt[:n],  # [n_components, d_model]
    }


def subspace_overlap(model, tokens):
    """Measure overlap between activation subspaces at different layers.

    Uses principal angle analysis (cosine of principal angles via SVD).

    Returns:
        dict with per_pair list containing:
        - layer_i, layer_j: the two layers
        - overlap: mean cosine of top principal angles
        - max_overlap: maximum principal angle cosine
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers

    # Get top-k principal directions at each layer
    k = min(5, len(tokens))
    layer_dirs = {}

    for l in range(n_layers):
        key = f'blocks.{l}.hook_resid_post'
        if key not in cache:
            continue
        resid = cache[key]
        resid_c = resid - jnp.mean(resid, axis=0, keepdims=True)
        U, S, Vt = jnp.linalg.svd(resid_c, full_matrices=False)
        layer_dirs[l] = Vt[:k]  # [k, d_model]

    pairs = []
    layers = sorted(layer_dirs.keys())
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            li, lj = layers[i], layers[j]
            # Principal angles via SVD of Q1^T @ Q2
            cross = layer_dirs[li] @ layer_dirs[lj].T  # [k, k]
            svs = jnp.linalg.svd(cross, compute_uv=False)

            overlap = float(jnp.mean(svs))
            max_overlap = float(jnp.max(svs))

            pairs.append({
                'layer_i': li,
                'layer_j': lj,
                'overlap': overlap,
                'max_overlap': max_overlap,
            })

    return {'per_pair': pairs}


def projection_analysis(model, tokens, direction, layer=None):
    """Project activations onto a specific direction and track across layers.

    Args:
        direction: [d_model] vector to project onto
        layer: if None, track across all layers

    Returns:
        dict with per_layer list containing:
        - layer: layer index
        - mean_projection: mean projection across positions
        - std_projection: std of projection
        - projections: per-position projections
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers

    # Normalize direction
    direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-10)

    results = []
    layers = [layer] if layer is not None else list(range(n_layers))

    for l in layers:
        key = f'blocks.{l}.hook_resid_post'
        if key not in cache:
            continue

        resid = cache[key]  # [seq, d_model]
        projs = resid @ direction  # [seq]

        results.append({
            'layer': l,
            'mean_projection': float(jnp.mean(projs)),
            'std_projection': float(jnp.std(projs)),
            'max_projection': float(jnp.max(projs)),
            'min_projection': float(jnp.min(projs)),
            'projections': [float(p) for p in projs],
        })

    return {'per_layer': results, 'direction_norm': float(jnp.linalg.norm(direction))}


def null_space_analysis(model, tokens, layer=0):
    """Analyze the null space of activations at a layer.

    The null space contains directions not used by the model at this layer.

    Returns:
        dict with:
        - utilized_dims: number of significantly used dimensions
        - null_dims: number of near-null dimensions
        - utilization_fraction: fraction of d_model actively used
        - singular_values: all singular values
    """
    cache = _run_and_cache(model, tokens)

    key = f'blocks.{layer}.hook_resid_post'
    if key not in cache:
        return {'utilized_dims': 0, 'null_dims': 0, 'utilization_fraction': 0.0}

    resid = cache[key]  # [seq, d_model]
    resid_c = resid - jnp.mean(resid, axis=0, keepdims=True)

    S = jnp.linalg.svd(resid_c, compute_uv=False)

    # Threshold for "active" dimension
    threshold = float(S[0]) * 0.01  # 1% of max
    utilized = int(jnp.sum(S > threshold))
    null_dims = len(S) - utilized

    return {
        'utilized_dims': utilized,
        'null_dims': null_dims,
        'utilization_fraction': utilized / len(S),
        'singular_values': [float(s) for s in S],
        'max_singular_value': float(S[0]),
        'min_singular_value': float(S[-1]),
    }


def component_subspace_analysis(model, tokens, layer=0):
    """Analyze how attention and MLP outputs occupy subspaces.

    Returns:
        dict with:
        - attn_rank: effective rank of attention output
        - mlp_rank: effective rank of MLP output
        - shared_subspace: overlap between attn and MLP subspaces
        - orthogonal_fraction: fraction of MLP output orthogonal to attn
    """
    cache = _run_and_cache(model, tokens)

    attn_key = f'blocks.{layer}.hook_attn_out'
    mlp_key = f'blocks.{layer}.hook_mlp_out'

    if attn_key not in cache or mlp_key not in cache:
        return {'attn_rank': 0, 'mlp_rank': 0, 'shared_subspace': 0.0}

    attn = cache[attn_key]  # [seq, d_model]
    mlp = cache[mlp_key]    # [seq, d_model]

    def _effective_rank(X):
        S = jnp.linalg.svd(X, compute_uv=False)
        S_norm = S / jnp.maximum(jnp.sum(S), 1e-10)
        S_norm = jnp.maximum(S_norm, 1e-10)
        entropy = -jnp.sum(S_norm * jnp.log(S_norm))
        return float(jnp.exp(entropy))

    attn_rank = _effective_rank(attn)
    mlp_rank = _effective_rank(mlp)

    # Subspace overlap: project MLP onto attn's subspace
    k = min(5, len(tokens))
    Ua, Sa, Vta = jnp.linalg.svd(attn, full_matrices=False)
    Um, Sm, Vtm = jnp.linalg.svd(mlp, full_matrices=False)

    # Top-k directions
    attn_dirs = Vta[:k]  # [k, d_model]
    mlp_dirs = Vtm[:k]   # [k, d_model]

    cross = attn_dirs @ mlp_dirs.T
    svs = jnp.linalg.svd(cross, compute_uv=False)
    shared = float(jnp.mean(svs))

    # Orthogonal fraction: how much of MLP is not in attn's subspace
    # Project each MLP position onto attn subspace directions and measure residual
    # attn_dirs: [k, d_model], mlp: [seq, d_model]
    proj_coeffs = mlp @ attn_dirs.T  # [seq, k]
    mlp_proj = proj_coeffs @ attn_dirs  # [seq, d_model] - projection onto attn subspace
    residual = mlp - mlp_proj
    residual_norm = float(jnp.linalg.norm(residual))
    total_norm = float(jnp.linalg.norm(mlp))
    ortho_frac = residual_norm / max(total_norm, 1e-10)

    return {
        'attn_rank': attn_rank,
        'mlp_rank': mlp_rank,
        'shared_subspace': shared,
        'orthogonal_fraction': ortho_frac,
    }
