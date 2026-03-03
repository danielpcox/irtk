"""Logit gradient analysis: gradient-based attribution of logits.

Analyze how logits depend on activations at each layer using gradients,
Jacobian structure, and sensitivity measures.
"""

import jax
import jax.numpy as jnp
from irtk.hook_points import HookState


def _run_and_cache(model, tokens):
    """Run model and return activation cache."""
    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    return hook_state.cache


def logit_sensitivity_profile(model, tokens, pos=-1, target_token=None):
    """Measure how sensitive each logit is to perturbations at each layer.

    Uses gradient norms to quantify sensitivity.

    Returns:
        dict with per_layer list containing:
        - layer: layer index
        - sensitivity: gradient norm (how much logit changes per unit perturbation)
        - relative_sensitivity: normalized across layers
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers

    if target_token is None:
        logits = model(tokens)
        target_token = int(jnp.argmax(logits[pos]))

    sensitivities = []
    for l in range(n_layers):
        key = f'blocks.{l}.hook_resid_post'
        if key not in cache:
            continue

        resid = cache[key]  # [seq, d_model]

        def logit_from_resid(r):
            # Project through remaining layers approximately via unembed
            return jnp.dot(r[pos], model.unembed.W_U[:, target_token]) + model.unembed.b_U[target_token]

        grad = jax.grad(logit_from_resid)(resid)  # [seq, d_model]
        grad_norm = float(jnp.linalg.norm(grad[pos]))
        sensitivities.append({'layer': l, 'sensitivity': grad_norm})

    # Normalize
    max_sens = max(s['sensitivity'] for s in sensitivities) if sensitivities else 1.0
    for s in sensitivities:
        s['relative_sensitivity'] = s['sensitivity'] / max(max_sens, 1e-10)

    return {
        'target_token': target_token,
        'pos': pos,
        'per_layer': sensitivities,
    }


def logit_jacobian_structure(model, tokens, pos=-1, top_k=5):
    """Analyze the Jacobian from final residual to logits.

    Returns:
        dict with:
        - effective_rank: how many dimensions of residual matter for logits
        - top_singular_values: largest singular values of Jacobian
        - concentration: fraction of variance in top-k directions
    """
    # The Jacobian of logits w.r.t. final residual is approximately W_U
    # (exact if no final layernorm, approximate otherwise)
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    # SVD of W_U
    U, S, Vt = jnp.linalg.svd(W_U, full_matrices=False)

    # Effective rank
    S_normalized = S / jnp.sum(S)
    S_normalized = jnp.maximum(S_normalized, 1e-10)
    entropy = -jnp.sum(S_normalized * jnp.log(S_normalized))
    eff_rank = float(jnp.exp(entropy))

    # Top-k concentration
    total_var = float(jnp.sum(S ** 2))
    topk_var = float(jnp.sum(S[:top_k] ** 2))
    concentration = topk_var / max(total_var, 1e-10)

    return {
        'effective_rank': eff_rank,
        'top_singular_values': [float(s) for s in S[:top_k]],
        'concentration': concentration,
        'total_singular_values': len(S),
        'condition_number': float(S[0] / jnp.maximum(S[-1], 1e-10)),
    }


def per_dimension_logit_impact(model, tokens, pos=-1, target_token=None, top_k=5):
    """Which dimensions of the residual stream most affect the target logit?

    Returns:
        dict with:
        - top_dimensions: list of (dim_index, impact) for most impactful dims
        - total_impact: sum of all dimension impacts
        - concentration: fraction of impact in top-k dimensions
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers

    if target_token is None:
        logits = model(tokens)
        target_token = int(jnp.argmax(logits[pos]))

    # Get final residual
    key = f'blocks.{n_layers - 1}.hook_resid_post'
    resid = cache[key]  # [seq, d_model]

    # The logit for target_token = resid[pos] @ W_U[:, target_token] + b_U[target_token]
    W_col = model.unembed.W_U[:, target_token]  # [d_model]

    # Per-dimension contribution = resid[pos, d] * W_U[d, target_token]
    contributions = resid[pos] * W_col  # [d_model]

    # Sort by absolute impact
    abs_contrib = jnp.abs(contributions)
    top_indices = jnp.argsort(-abs_contrib)[:top_k]

    total = float(jnp.sum(abs_contrib))
    topk_sum = float(jnp.sum(abs_contrib[top_indices]))

    top_dims = []
    for idx in top_indices:
        top_dims.append({
            'dimension': int(idx),
            'impact': float(contributions[idx]),
            'abs_impact': float(abs_contrib[idx]),
        })

    return {
        'target_token': target_token,
        'top_dimensions': top_dims,
        'total_impact': total,
        'concentration': topk_sum / max(total, 1e-10),
    }


def gradient_alignment_across_layers(model, tokens, pos=-1, target_token=None):
    """How aligned are the gradient directions at different layers?

    If gradients point in similar directions, layers are optimizing
    the same logit in a coordinated way.

    Returns:
        dict with:
        - per_layer: list with gradient info
        - mean_alignment: mean pairwise cosine of gradient directions
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers

    if target_token is None:
        logits = model(tokens)
        target_token = int(jnp.argmax(logits[pos]))

    # Compute gradient direction at each layer
    grads = []
    for l in range(n_layers):
        key = f'blocks.{l}.hook_resid_post'
        if key not in cache:
            continue

        resid = cache[key]

        def logit_fn(r):
            return jnp.dot(r[pos], model.unembed.W_U[:, target_token]) + model.unembed.b_U[target_token]

        g = jax.grad(logit_fn)(resid)
        grads.append(g[pos])  # [d_model]

    # Compute pairwise alignment
    results = []
    for i, g in enumerate(grads):
        norm = float(jnp.linalg.norm(g))
        results.append({
            'layer': i,
            'gradient_norm': norm,
        })

    # Pairwise cosines
    alignments = []
    for i in range(len(grads)):
        for j in range(i + 1, len(grads)):
            ni = jnp.linalg.norm(grads[i])
            nj = jnp.linalg.norm(grads[j])
            cos = float(jnp.dot(grads[i], grads[j]) / jnp.maximum(ni * nj, 1e-10))
            alignments.append(cos)

    mean_align = sum(alignments) / max(len(alignments), 1) if alignments else 0.0

    return {
        'target_token': target_token,
        'per_layer': results,
        'mean_alignment': mean_align,
        'pairwise_alignments': alignments,
    }


def logit_curvature(model, tokens, pos=-1, target_token=None, epsilon=0.01):
    """Estimate curvature of the logit landscape around current activations.

    Higher curvature = more nonlinear, small perturbations matter more.

    Returns:
        dict with per_layer curvature estimates.
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers

    if target_token is None:
        logits = model(tokens)
        target_token = int(jnp.argmax(logits[pos]))

    results = []
    for l in range(n_layers):
        key = f'blocks.{l}.hook_resid_post'
        if key not in cache:
            continue

        resid = cache[key]  # [seq, d_model]

        def logit_fn(r):
            return jnp.dot(r[pos], model.unembed.W_U[:, target_token]) + model.unembed.b_U[target_token]

        # Base logit
        base_logit = float(logit_fn(resid))

        # Perturb in random directions and measure deviation from linearity
        key_rng = jax.random.PRNGKey(l)
        direction = jax.random.normal(key_rng, resid.shape)
        direction = direction / jnp.linalg.norm(direction)

        # f(x + eps*d) and f(x - eps*d)
        plus = float(logit_fn(resid + epsilon * direction))
        minus = float(logit_fn(resid - epsilon * direction))

        # Second derivative estimate: (f(x+h) - 2f(x) + f(x-h)) / h^2
        curvature = abs((plus - 2 * base_logit + minus) / (epsilon ** 2))

        results.append({
            'layer': l,
            'curvature': curvature,
            'base_logit': base_logit,
        })

    return {
        'target_token': target_token,
        'per_layer': results,
    }
