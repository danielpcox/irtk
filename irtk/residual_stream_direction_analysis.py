"""Residual stream direction analysis.

Analyze important directions in the residual stream: unembed-aligned
directions, maximally active dimensions, feature-carrying directions,
and directional composition.
"""

import jax
import jax.numpy as jnp


def unembed_aligned_directions(model, tokens, layer, top_k=5):
    """Find directions in the residual stream most aligned with unembedding.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index
        top_k: number of top directions

    Returns:
        dict with top unembed-aligned directions and their effects.
    """
    _, cache = model.run_with_cache(tokens)
    resid = cache[f'blocks.{layer}.hook_resid_post']  # [seq, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    # SVD of residual to get principal directions
    U, S, Vt = jnp.linalg.svd(resid, full_matrices=False)
    n = min(top_k, Vt.shape[0])

    results = []
    for i in range(n):
        direction = Vt[i]  # [d_model]
        # Project onto unembed to see what tokens this direction promotes
        logit_effect = direction @ W_U  # [d_vocab]
        top_promoted = jnp.argsort(-logit_effect)[:3]
        top_demoted = jnp.argsort(logit_effect)[:3]

        results.append({
            'component': i,
            'singular_value': float(S[i]),
            'variance_explained': float(S[i] ** 2 / jnp.sum(S ** 2)),
            'top_promoted': [{'token': int(t), 'logit': float(logit_effect[t])} for t in top_promoted],
            'top_demoted': [{'token': int(t), 'logit': float(logit_effect[t])} for t in top_demoted],
        })

    return {'layer': layer, 'directions': results}


def maximally_active_dimensions(model, tokens, layer, top_k=5):
    """Find the most active dimensions in the residual stream.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index
        top_k: number of top dimensions

    Returns:
        dict with top active dimensions and their statistics.
    """
    _, cache = model.run_with_cache(tokens)
    resid = cache[f'blocks.{layer}.hook_resid_post']  # [seq, d_model]

    # Variance per dimension
    dim_variance = jnp.var(resid, axis=0)  # [d_model]
    dim_mean_abs = jnp.mean(jnp.abs(resid), axis=0)

    top_by_variance = jnp.argsort(-dim_variance)[:top_k]
    top_by_magnitude = jnp.argsort(-dim_mean_abs)[:top_k]

    W_U = model.unembed.W_U

    variance_dims = []
    for d in top_by_variance:
        d = int(d)
        logit_weight = W_U[d]  # [d_vocab]
        top_token = int(jnp.argmax(jnp.abs(logit_weight)))
        variance_dims.append({
            'dimension': d,
            'variance': float(dim_variance[d]),
            'mean_abs': float(dim_mean_abs[d]),
            'top_logit_token': top_token,
            'top_logit_weight': float(logit_weight[top_token]),
        })

    return {
        'layer': layer,
        'top_by_variance': variance_dims,
        'total_variance': float(jnp.sum(dim_variance)),
        'top_k_variance_share': float(jnp.sum(dim_variance[top_by_variance]) / jnp.sum(dim_variance)),
    }


def direction_contribution_tracking(model, tokens, direction):
    """Track which components contribute to a specific direction across layers.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        direction: [d_model] direction to track

    Returns:
        dict with per-layer, per-component contributions.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-10)

    layers = []
    for l in range(n_layers):
        attn_out = cache[f'blocks.{l}.hook_attn_out']  # [seq, d_model]
        mlp_out = cache[f'blocks.{l}.hook_mlp_out']

        attn_proj = float(jnp.mean(jnp.sum(attn_out * direction, axis=-1)))
        mlp_proj = float(jnp.mean(jnp.sum(mlp_out * direction, axis=-1)))

        # Per-head breakdown
        z = cache[f'blocks.{l}.attn.hook_z']
        n_heads = z.shape[1]
        head_projs = []
        for h in range(n_heads):
            z_h = z[:, h, :]
            W_O_h = model.blocks[l].attn.W_O[h]
            output = z_h @ W_O_h
            proj = float(jnp.mean(jnp.sum(output * direction, axis=-1)))
            head_projs.append({'head': h, 'projection': proj})

        layers.append({
            'layer': l,
            'attn_contribution': attn_proj,
            'mlp_contribution': mlp_proj,
            'total_contribution': attn_proj + mlp_proj,
            'per_head': head_projs,
        })

    return {
        'layers': layers,
        'cumulative': sum(l['total_contribution'] for l in layers),
    }


def residual_direction_diversity(model, tokens):
    """Measure how diverse the residual stream directions are per layer.

    Returns:
        dict with per-layer direction diversity metrics.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    results = []
    for l in range(n_layers):
        resid = cache[f'blocks.{l}.hook_resid_post']  # [seq, d_model]
        # Normalize per position
        norms = jnp.linalg.norm(resid, axis=-1, keepdims=True)
        normalized = resid / jnp.maximum(norms, 1e-10)

        # Pairwise cosine similarity
        cos_matrix = normalized @ normalized.T
        seq_len = cos_matrix.shape[0]
        mask = 1.0 - jnp.eye(seq_len)
        mean_cos = float(jnp.sum(cos_matrix * mask) / jnp.maximum(jnp.sum(mask), 1e-10))

        # Effective dimensionality via SVD
        S = jnp.linalg.svd(resid, compute_uv=False)
        S_norm = S / jnp.maximum(jnp.sum(S), 1e-10)
        S_safe = jnp.maximum(S_norm, 1e-10)
        entropy = -float(jnp.sum(S_safe * jnp.log(S_safe)))
        eff_dim = float(jnp.exp(jnp.array(entropy)))

        results.append({
            'layer': l,
            'mean_pairwise_cosine': mean_cos,
            'direction_diversity': 1.0 - mean_cos,
            'effective_dimensionality': eff_dim,
        })

    return {
        'per_layer': results,
        'mean_diversity': float(jnp.mean(jnp.array([r['direction_diversity'] for r in results]))),
    }


def important_direction_overlap(model, tokens):
    """Measure overlap of principal directions across layers.

    Returns:
        dict with principal direction overlap between consecutive layers.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_components = 5

    results = []
    for l in range(n_layers - 1):
        resid_curr = cache[f'blocks.{l}.hook_resid_post']
        resid_next = cache[f'blocks.{l+1}.hook_resid_post']

        U_c, S_c, Vt_c = jnp.linalg.svd(resid_curr, full_matrices=False)
        U_n, S_n, Vt_n = jnp.linalg.svd(resid_next, full_matrices=False)

        k = min(n_components, Vt_c.shape[0], Vt_n.shape[0])
        Va = Vt_c[:k]
        Vb = Vt_n[:k]
        overlap = Va @ Vb.T
        S_overlap = jnp.linalg.svd(overlap, compute_uv=False)

        results.append({
            'from_layer': l,
            'to_layer': l + 1,
            'mean_overlap': float(jnp.mean(S_overlap)),
            'min_overlap': float(jnp.min(S_overlap)),
            'max_overlap': float(jnp.max(S_overlap)),
        })

    return {
        'transitions': results,
        'mean_overlap': float(jnp.mean(jnp.array([r['mean_overlap'] for r in results])))
        if results else 0.0,
    }
