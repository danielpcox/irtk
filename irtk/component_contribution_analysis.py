"""Component contribution analysis.

Measure how each model component (attention heads, MLPs, embeddings)
contributes to the final output, layer-by-layer.
"""

import jax
import jax.numpy as jnp
from irtk.hook_points import HookState


def _run_and_cache(model, tokens):
    """Run model and return activation cache."""
    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    return hook_state.cache


def cumulative_component_contribution(model, tokens, pos=-1, target_token=None):
    """Track cumulative contribution of each component to the target logit.

    Returns:
        dict with per_component list containing:
        - component: name (e.g., 'embed', 'L0_attn', 'L0_mlp')
        - logit_contribution: how much this component adds to target logit
        - cumulative_logit: running total of logit
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers

    if target_token is None:
        logits = model(tokens)
        target_token = int(jnp.argmax(logits[pos]))

    W_U_col = model.unembed.W_U[:, target_token]  # [d_model]
    b_U = float(model.unembed.b_U[target_token])

    results = []
    cumulative = b_U  # start with bias

    # Embedding contribution
    embed_key = 'blocks.0.hook_resid_pre'
    if embed_key in cache:
        embed_vec = cache[embed_key][pos]  # [d_model]
        embed_logit = float(jnp.dot(embed_vec, W_U_col))
        cumulative += embed_logit
        results.append({
            'component': 'embed',
            'logit_contribution': embed_logit,
            'cumulative_logit': cumulative,
        })

    # Per-layer attention and MLP
    for l in range(n_layers):
        attn_key = f'blocks.{l}.hook_attn_out'
        mlp_key = f'blocks.{l}.hook_mlp_out'

        if attn_key in cache:
            attn_out = cache[attn_key][pos]
            attn_logit = float(jnp.dot(attn_out, W_U_col))
            cumulative += attn_logit
            results.append({
                'component': f'L{l}_attn',
                'logit_contribution': attn_logit,
                'cumulative_logit': cumulative,
            })

        if mlp_key in cache:
            mlp_out = cache[mlp_key][pos]
            mlp_logit = float(jnp.dot(mlp_out, W_U_col))
            cumulative += mlp_logit
            results.append({
                'component': f'L{l}_mlp',
                'logit_contribution': mlp_logit,
                'cumulative_logit': cumulative,
            })

    return {
        'target_token': target_token,
        'per_component': results,
        'final_logit': cumulative,
    }


def component_norm_contribution(model, tokens, pos=-1):
    """Measure norm contribution of each component to the residual.

    Returns:
        dict with per_component list containing:
        - component: name
        - output_norm: norm of the component's output
        - fraction_of_residual: fraction of final residual norm
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers

    # Final residual norm
    final_key = f'blocks.{n_layers - 1}.hook_resid_post'
    if final_key not in cache:
        return {'per_component': []}
    final_norm = float(jnp.linalg.norm(cache[final_key][pos]))

    results = []

    # Embedding
    embed_key = 'blocks.0.hook_resid_pre'
    if embed_key in cache:
        enorm = float(jnp.linalg.norm(cache[embed_key][pos]))
        results.append({
            'component': 'embed',
            'output_norm': enorm,
            'fraction_of_residual': enorm / max(final_norm, 1e-10),
        })

    for l in range(n_layers):
        for comp, name in [('hook_attn_out', f'L{l}_attn'), ('hook_mlp_out', f'L{l}_mlp')]:
            key = f'blocks.{l}.{comp}'
            if key in cache:
                cnorm = float(jnp.linalg.norm(cache[key][pos]))
                results.append({
                    'component': name,
                    'output_norm': cnorm,
                    'fraction_of_residual': cnorm / max(final_norm, 1e-10),
                })

    return {'per_component': results, 'final_residual_norm': final_norm}


def component_direction_alignment(model, tokens, pos=-1, target_token=None):
    """How aligned is each component's output with the prediction direction?

    Returns:
        dict with per_component cosine alignment with the unembed direction.
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers

    if target_token is None:
        logits = model(tokens)
        target_token = int(jnp.argmax(logits[pos]))

    W_U_col = model.unembed.W_U[:, target_token]
    wu_norm = jnp.linalg.norm(W_U_col)

    results = []

    embed_key = 'blocks.0.hook_resid_pre'
    if embed_key in cache:
        v = cache[embed_key][pos]
        cos = float(jnp.dot(v, W_U_col) / jnp.maximum(jnp.linalg.norm(v) * wu_norm, 1e-10))
        results.append({'component': 'embed', 'alignment': cos})

    for l in range(n_layers):
        for comp, name in [('hook_attn_out', f'L{l}_attn'), ('hook_mlp_out', f'L{l}_mlp')]:
            key = f'blocks.{l}.{comp}'
            if key in cache:
                v = cache[key][pos]
                cos = float(jnp.dot(v, W_U_col) / jnp.maximum(jnp.linalg.norm(v) * wu_norm, 1e-10))
                results.append({'component': name, 'alignment': cos})

    return {
        'target_token': target_token,
        'per_component': results,
    }


def component_interference(model, tokens, pos=-1):
    """Measure constructive vs destructive interference between components.

    Returns:
        dict with per_pair list showing which components reinforce or cancel.
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers

    # Collect all component outputs
    components = {}
    embed_key = 'blocks.0.hook_resid_pre'
    if embed_key in cache:
        components['embed'] = cache[embed_key][pos]

    for l in range(n_layers):
        for comp, name in [('hook_attn_out', f'L{l}_attn'), ('hook_mlp_out', f'L{l}_mlp')]:
            key = f'blocks.{l}.{comp}'
            if key in cache:
                components[name] = cache[key][pos]

    pairs = []
    names = list(components.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            vi = components[names[i]]
            vj = components[names[j]]
            ni = jnp.linalg.norm(vi)
            nj = jnp.linalg.norm(vj)
            cos = float(jnp.dot(vi, vj) / jnp.maximum(ni * nj, 1e-10))

            pairs.append({
                'component_i': names[i],
                'component_j': names[j],
                'cosine': cos,
                'type': 'constructive' if cos > 0.1 else ('destructive' if cos < -0.1 else 'orthogonal'),
            })

    return {'per_pair': pairs}


def component_importance_ranking(model, tokens, pos=-1, target_token=None):
    """Rank components by their importance to the prediction.

    Combines logit contribution magnitude and alignment.

    Returns:
        dict with ranked list of components.
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers

    if target_token is None:
        logits = model(tokens)
        target_token = int(jnp.argmax(logits[pos]))

    W_U_col = model.unembed.W_U[:, target_token]

    components = []

    embed_key = 'blocks.0.hook_resid_pre'
    if embed_key in cache:
        v = cache[embed_key][pos]
        logit = float(jnp.dot(v, W_U_col))
        components.append({'component': 'embed', 'logit_contribution': logit, 'abs_contribution': abs(logit)})

    for l in range(n_layers):
        for comp, name in [('hook_attn_out', f'L{l}_attn'), ('hook_mlp_out', f'L{l}_mlp')]:
            key = f'blocks.{l}.{comp}'
            if key in cache:
                v = cache[key][pos]
                logit = float(jnp.dot(v, W_U_col))
                components.append({'component': name, 'logit_contribution': logit, 'abs_contribution': abs(logit)})

    # Sort by absolute contribution
    components.sort(key=lambda x: -x['abs_contribution'])

    # Add rank
    for i, c in enumerate(components):
        c['rank'] = i + 1

    return {
        'target_token': target_token,
        'ranked_components': components,
    }
