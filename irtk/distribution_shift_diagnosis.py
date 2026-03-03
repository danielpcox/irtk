"""Distribution shift diagnosis.

Analyze how model internals change between different inputs: activation
divergence, component vulnerability, feature stability, layer-wise
adaptation difficulty, and prediction robustness.
"""

import jax
import jax.numpy as jnp


def activation_divergence_profile(model, tokens_a, tokens_b):
    """Measure activation divergence at every hook point between two inputs.

    Args:
        model: HookedTransformer
        tokens_a: first input token IDs
        tokens_b: second input token IDs

    Returns:
        dict with per-hook divergence scores.
    """
    _, cache_a = model.run_with_cache(tokens_a)
    _, cache_b = model.run_with_cache(tokens_b)
    n_layers = model.cfg.n_layers

    per_hook = []
    for l in range(n_layers):
        for name_suffix in ['hook_resid_pre', 'hook_resid_post', 'hook_attn_out', 'hook_mlp_out']:
            hook_name = f'blocks.{l}.{name_suffix}'
            if hook_name in cache_a and hook_name in cache_b:
                a = cache_a[hook_name]
                b = cache_b[hook_name]
                # Use mean over min shared positions
                min_len = min(a.shape[0], b.shape[0])
                a_trunc = a[:min_len]
                b_trunc = b[:min_len]
                l2_dist = float(jnp.mean(jnp.linalg.norm(a_trunc - b_trunc, axis=-1)))
                cosine = float(jnp.mean(jnp.sum(a_trunc * b_trunc, axis=-1) /
                    (jnp.linalg.norm(a_trunc, axis=-1) * jnp.linalg.norm(b_trunc, axis=-1) + 1e-10)))
                per_hook.append({
                    'hook': hook_name,
                    'layer': l,
                    'l2_distance': l2_dist,
                    'cosine_similarity': cosine,
                })

    per_hook.sort(key=lambda h: -h['l2_distance'])
    return {
        'per_hook': per_hook,
        'most_divergent': per_hook[0] if per_hook else None,
        'mean_divergence': float(jnp.mean(jnp.array([h['l2_distance'] for h in per_hook]))) if per_hook else 0.0,
    }


def component_vulnerability(model, tokens_a, tokens_b):
    """Identify which components (heads, MLPs) are most affected by input shift.

    Args:
        model: HookedTransformer
        tokens_a: baseline input
        tokens_b: shifted input

    Returns:
        dict with per-component vulnerability scores.
    """
    _, cache_a = model.run_with_cache(tokens_a)
    _, cache_b = model.run_with_cache(tokens_b)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    results = []
    min_len = min(len(tokens_a), len(tokens_b))

    for l in range(n_layers):
        # Attention heads
        z_a = cache_a[f'blocks.{l}.attn.hook_z'][:min_len]
        z_b = cache_b[f'blocks.{l}.attn.hook_z'][:min_len]
        for h in range(n_heads):
            diff = float(jnp.mean(jnp.linalg.norm(z_a[:, h, :] - z_b[:, h, :], axis=-1)))
            base_norm = float(jnp.mean(jnp.linalg.norm(z_a[:, h, :], axis=-1)))
            results.append({
                'component': f'L{l}H{h}',
                'layer': l,
                'type': 'head',
                'head': h,
                'absolute_change': diff,
                'relative_change': diff / max(base_norm, 1e-10),
            })

        # MLP
        mlp_a = cache_a[f'blocks.{l}.hook_mlp_out'][:min_len]
        mlp_b = cache_b[f'blocks.{l}.hook_mlp_out'][:min_len]
        mlp_diff = float(jnp.mean(jnp.linalg.norm(mlp_a - mlp_b, axis=-1)))
        mlp_base = float(jnp.mean(jnp.linalg.norm(mlp_a, axis=-1)))
        results.append({
            'component': f'MLP{l}',
            'layer': l,
            'type': 'mlp',
            'head': None,
            'absolute_change': mlp_diff,
            'relative_change': mlp_diff / max(mlp_base, 1e-10),
        })

    results.sort(key=lambda r: -r['relative_change'])
    return {
        'per_component': results,
        'most_vulnerable': results[0] if results else None,
    }


def feature_stability(model, tokens_a, tokens_b, direction):
    """Track how a specific feature direction responds to input shift across layers.

    Args:
        model: HookedTransformer
        tokens_a: baseline input
        tokens_b: shifted input
        direction: feature direction vector [d_model]

    Returns:
        dict with per-layer feature stability.
    """
    _, cache_a = model.run_with_cache(tokens_a)
    _, cache_b = model.run_with_cache(tokens_b)
    n_layers = model.cfg.n_layers
    direction = direction / (jnp.linalg.norm(direction) + 1e-10)

    per_layer = []
    for l in range(n_layers):
        resid_a = cache_a[f'blocks.{l}.hook_resid_post']
        resid_b = cache_b[f'blocks.{l}.hook_resid_post']

        proj_a = float(jnp.mean(resid_a @ direction))
        proj_b = float(jnp.mean(resid_b @ direction))
        diff = abs(proj_a - proj_b)

        per_layer.append({
            'layer': l,
            'projection_a': proj_a,
            'projection_b': proj_b,
            'projection_diff': diff,
            'stable': diff < 0.1 * max(abs(proj_a), abs(proj_b), 1e-10),
        })

    n_stable = sum(1 for p in per_layer if p['stable'])
    return {
        'per_layer': per_layer,
        'stability_fraction': n_stable / max(n_layers, 1),
        'most_unstable_layer': max(per_layer, key=lambda p: p['projection_diff'])['layer'] if per_layer else 0,
    }


def layer_adaptation_difficulty(model, tokens_a, tokens_b):
    """Measure how much each layer's output changes between inputs (adaptation difficulty).

    Args:
        model: HookedTransformer
        tokens_a: baseline input
        tokens_b: shifted input

    Returns:
        dict with per-layer adaptation difficulty.
    """
    _, cache_a = model.run_with_cache(tokens_a)
    _, cache_b = model.run_with_cache(tokens_b)
    n_layers = model.cfg.n_layers
    min_len = min(len(tokens_a), len(tokens_b))

    per_layer = []
    for l in range(n_layers):
        resid_a = cache_a[f'blocks.{l}.hook_resid_post'][:min_len]
        resid_b = cache_b[f'blocks.{l}.hook_resid_post'][:min_len]

        l2 = float(jnp.mean(jnp.linalg.norm(resid_a - resid_b, axis=-1)))
        cos = float(jnp.mean(jnp.sum(resid_a * resid_b, axis=-1) /
            (jnp.linalg.norm(resid_a, axis=-1) * jnp.linalg.norm(resid_b, axis=-1) + 1e-10)))

        per_layer.append({
            'layer': l,
            'l2_distance': l2,
            'cosine_similarity': cos,
            'difficulty': l2 * (1.0 - cos),
        })

    per_layer_sorted = sorted(per_layer, key=lambda p: -p['difficulty'])
    return {
        'per_layer': per_layer,
        'hardest_layer': per_layer_sorted[0]['layer'] if per_layer_sorted else 0,
        'mean_difficulty': float(jnp.mean(jnp.array([p['difficulty'] for p in per_layer]))) if per_layer else 0.0,
    }


def prediction_robustness(model, tokens_a, tokens_b):
    """Compare model predictions between two inputs: agreement, rank changes, KL divergence.

    Args:
        model: HookedTransformer
        tokens_a: baseline input
        tokens_b: shifted input

    Returns:
        dict with prediction comparison.
    """
    logits_a = model(tokens_a)
    logits_b = model(tokens_b)
    min_len = min(logits_a.shape[0], logits_b.shape[0])

    probs_a = jax.nn.softmax(logits_a[:min_len])
    probs_b = jax.nn.softmax(logits_b[:min_len])

    per_position = []
    for pos in range(min_len):
        top_a = int(jnp.argmax(probs_a[pos]))
        top_b = int(jnp.argmax(probs_b[pos]))
        kl = float(jnp.sum(probs_a[pos] * jnp.log(probs_a[pos] / (probs_b[pos] + 1e-10) + 1e-10)))
        agree = top_a == top_b

        per_position.append({
            'position': pos,
            'top_token_a': top_a,
            'top_token_b': top_b,
            'agree': bool(agree),
            'kl_divergence': max(kl, 0.0),
        })

    agreement_rate = sum(1 for p in per_position if p['agree']) / max(min_len, 1)
    mean_kl = float(jnp.mean(jnp.array([p['kl_divergence'] for p in per_position])))
    return {
        'per_position': per_position,
        'agreement_rate': agreement_rate,
        'mean_kl_divergence': mean_kl,
    }
