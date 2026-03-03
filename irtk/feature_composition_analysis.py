"""Feature composition analysis.

Analyze how features compose across layers: feature interaction,
amplification, cancellation, and compositional structure.
"""

import jax
import jax.numpy as jnp


def feature_amplification(model, tokens, direction):
    """Track how a feature direction is amplified or suppressed across layers.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        direction: [d_model] feature direction to track

    Returns:
        dict with per-layer projection of residual onto direction.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-10)

    stages = []
    for l in range(n_layers):
        resid_pre = cache[f'blocks.{l}.hook_resid_pre']  # [seq, d_model]
        resid_post = cache[f'blocks.{l}.hook_resid_post']

        proj_pre = jnp.sum(resid_pre * direction, axis=-1)  # [seq]
        proj_post = jnp.sum(resid_post * direction, axis=-1)

        attn_out = cache[f'blocks.{l}.hook_attn_out']
        mlp_out = cache[f'blocks.{l}.hook_mlp_out']
        attn_proj = jnp.sum(attn_out * direction, axis=-1)
        mlp_proj = jnp.sum(mlp_out * direction, axis=-1)

        stages.append({
            'layer': l,
            'pre_projection': float(jnp.mean(proj_pre)),
            'post_projection': float(jnp.mean(proj_post)),
            'attn_contribution': float(jnp.mean(attn_proj)),
            'mlp_contribution': float(jnp.mean(mlp_proj)),
            'amplification': float(jnp.mean(jnp.abs(proj_post)) /
                                   jnp.maximum(jnp.mean(jnp.abs(proj_pre)), 1e-10)),
        })

    return {
        'stages': stages,
        'total_amplification': stages[-1]['post_projection'] /
                               max(abs(stages[0]['pre_projection']), 1e-10)
        if stages else 1.0,
    }


def feature_cancellation(model, tokens):
    """Detect feature cancellation: directions where attn and MLP oppose each other.

    Args:
        model: HookedTransformer
        tokens: input token IDs

    Returns:
        dict with per-layer cancellation analysis.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    results = []
    for l in range(n_layers):
        attn_out = cache[f'blocks.{l}.hook_attn_out']  # [seq, d_model]
        mlp_out = cache[f'blocks.{l}.hook_mlp_out']

        # Cosine between attn and MLP outputs (per position)
        attn_norm = jnp.linalg.norm(attn_out, axis=-1, keepdims=True)
        mlp_norm = jnp.linalg.norm(mlp_out, axis=-1, keepdims=True)
        cosines = jnp.sum(attn_out * mlp_out, axis=-1) / jnp.maximum(
            attn_norm.squeeze(-1) * mlp_norm.squeeze(-1), 1e-10)

        # Cancellation when cosine < 0
        mean_cos = float(jnp.mean(cosines))
        cancellation_fraction = float(jnp.mean(cosines < 0))

        # Magnitude of cancellation
        combined = attn_out + mlp_out
        combined_norm = float(jnp.mean(jnp.linalg.norm(combined, axis=-1)))
        sum_norms = float(jnp.mean(attn_norm.squeeze(-1) + mlp_norm.squeeze(-1)))
        cancellation_ratio = 1.0 - combined_norm / max(sum_norms, 1e-10)

        results.append({
            'layer': l,
            'mean_cosine': mean_cos,
            'cancellation_fraction': cancellation_fraction,
            'cancellation_ratio': cancellation_ratio,
        })

    return {
        'per_layer': results,
        'mean_cancellation': float(jnp.mean(jnp.array([r['cancellation_ratio'] for r in results]))),
    }


def cross_layer_feature_interaction(model, tokens, layer_a, layer_b):
    """Analyze how features from one layer interact with another.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer_a: source layer
        layer_b: target layer

    Returns:
        dict with feature interaction scores.
    """
    _, cache = model.run_with_cache(tokens)

    # Get dominant directions at each layer
    resid_a = cache[f'blocks.{layer_a}.hook_resid_post']  # [seq, d_model]
    resid_b = cache[f'blocks.{layer_b}.hook_resid_post']

    # SVD to find principal directions
    U_a, S_a, Vt_a = jnp.linalg.svd(resid_a, full_matrices=False)
    U_b, S_b, Vt_b = jnp.linalg.svd(resid_b, full_matrices=False)

    # Subspace overlap via principal angles
    n_components = min(5, min(Vt_a.shape[0], Vt_b.shape[0]))
    Va = Vt_a[:n_components]  # [k, d_model]
    Vb = Vt_b[:n_components]
    overlap = Va @ Vb.T  # [k, k]
    S_overlap = jnp.linalg.svd(overlap, compute_uv=False)
    principal_angles = [float(jnp.arccos(jnp.clip(s, -1.0, 1.0))) for s in S_overlap]

    # Cosine between mean residuals
    mean_a = jnp.mean(resid_a, axis=0)
    mean_b = jnp.mean(resid_b, axis=0)
    mean_cos = float(jnp.sum(mean_a * mean_b) /
                     jnp.maximum(jnp.linalg.norm(mean_a) * jnp.linalg.norm(mean_b), 1e-10))

    return {
        'layer_a': layer_a,
        'layer_b': layer_b,
        'principal_angles': principal_angles,
        'mean_angle': float(jnp.mean(jnp.array(principal_angles))),
        'subspace_overlap': float(jnp.mean(S_overlap)),
        'mean_direction_cosine': mean_cos,
    }


def component_feature_alignment(model, tokens, layer, direction):
    """Measure how well each component at a layer aligns with a feature direction.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index
        direction: [d_model] feature direction

    Returns:
        dict with per-component alignment to the direction.
    """
    _, cache = model.run_with_cache(tokens)
    direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-10)

    attn_out = cache[f'blocks.{layer}.hook_attn_out']  # [seq, d_model]
    mlp_out = cache[f'blocks.{layer}.hook_mlp_out']

    # Per-head alignment
    z = cache[f'blocks.{layer}.attn.hook_z']  # [seq, n_heads, d_head]
    n_heads = z.shape[1]

    head_alignments = []
    for h in range(n_heads):
        z_h = z[:, h, :]
        W_O_h = model.blocks[layer].attn.W_O[h]
        output = z_h @ W_O_h  # [seq, d_model]
        proj = jnp.sum(output * direction, axis=-1)  # [seq]
        head_alignments.append({
            'head': h,
            'mean_projection': float(jnp.mean(proj)),
            'abs_projection': float(jnp.mean(jnp.abs(proj))),
        })

    # MLP alignment
    mlp_proj = jnp.sum(mlp_out * direction, axis=-1)

    # Total attn alignment
    attn_proj = jnp.sum(attn_out * direction, axis=-1)

    return {
        'layer': layer,
        'per_head': head_alignments,
        'attn_mean_projection': float(jnp.mean(attn_proj)),
        'mlp_mean_projection': float(jnp.mean(mlp_proj)),
        'dominant_component': 'attn' if abs(float(jnp.mean(attn_proj))) > abs(float(jnp.mean(mlp_proj))) else 'mlp',
    }


def feature_composition_scores(model, tokens):
    """Compute composition scores between features across layers.

    Measures how much each layer builds on vs overwrites previous layers.

    Returns:
        dict with per-layer composition analysis.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    results = []
    for l in range(n_layers):
        resid_pre = cache[f'blocks.{l}.hook_resid_pre']  # [seq, d_model]
        resid_post = cache[f'blocks.{l}.hook_resid_post']
        delta = resid_post - resid_pre  # layer's total contribution

        # How much of the output is "new" vs "preserved"?
        pre_norm = float(jnp.mean(jnp.linalg.norm(resid_pre, axis=-1)))
        delta_norm = float(jnp.mean(jnp.linalg.norm(delta, axis=-1)))
        post_norm = float(jnp.mean(jnp.linalg.norm(resid_post, axis=-1)))

        # Cosine between delta and pre (positive = reinforcing, negative = correcting)
        cos = jnp.sum(resid_pre * delta, axis=-1) / jnp.maximum(
            jnp.linalg.norm(resid_pre, axis=-1) * jnp.linalg.norm(delta, axis=-1), 1e-10)
        mean_cos = float(jnp.mean(cos))

        results.append({
            'layer': l,
            'pre_norm': pre_norm,
            'delta_norm': delta_norm,
            'post_norm': post_norm,
            'delta_to_pre_ratio': delta_norm / max(pre_norm, 1e-10),
            'reinforcement_score': mean_cos,
            'is_reinforcing': mean_cos > 0,
        })

    return {
        'per_layer': results,
        'mean_reinforcement': float(jnp.mean(jnp.array([r['reinforcement_score'] for r in results]))),
    }
