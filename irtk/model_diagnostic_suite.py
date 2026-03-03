"""Model diagnostic suite: quick health checks and summary reports."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def model_health_check(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Run a quick health check on the model's computations.

    Checks for NaN/Inf, extreme values, degenerate attention patterns.
    """
    logits, cache = model.run_with_cache(tokens)

    issues = []
    n_layers = model.cfg.n_layers

    # Check logits
    if jnp.any(jnp.isnan(logits)):
        issues.append('NaN in final logits')
    if jnp.any(jnp.isinf(logits)):
        issues.append('Inf in final logits')
    max_logit = float(jnp.max(jnp.abs(logits)))

    # Check each layer
    layer_stats = []
    for layer in range(n_layers):
        resid_key = f'blocks.{layer}.hook_resid_post'
        if resid_key in cache:
            resid = cache[resid_key]
            has_nan = bool(jnp.any(jnp.isnan(resid)))
            max_val = float(jnp.max(jnp.abs(resid)))
            mean_norm = float(jnp.mean(jnp.linalg.norm(resid, axis=-1)))

            if has_nan:
                issues.append(f'NaN in layer {layer} residual')

            layer_stats.append({
                'layer': layer,
                'has_nan': has_nan,
                'max_value': max_val,
                'mean_norm': mean_norm,
            })

        # Check attention patterns
        pattern_key = f'blocks.{layer}.attn.hook_pattern'
        if pattern_key in cache:
            pattern = cache[pattern_key]
            if jnp.any(jnp.isnan(pattern)):
                issues.append(f'NaN in layer {layer} attention')

    return {
        'is_healthy': len(issues) == 0,
        'issues': issues,
        'max_logit_magnitude': max_logit,
        'layer_stats': layer_stats,
        'n_issues': len(issues),
    }


def computation_budget_profile(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Profile how computation is distributed across layers.

    Shows the fraction of total norm change contributed by each component.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    total_change = 0.0
    per_component = []

    for layer in range(n_layers):
        attn_key = f'blocks.{layer}.hook_attn_out'
        mlp_key = f'blocks.{layer}.hook_mlp_out'

        attn_change = float(jnp.mean(jnp.linalg.norm(cache[attn_key], axis=-1)))
        mlp_change = float(jnp.mean(jnp.linalg.norm(cache[mlp_key], axis=-1)))

        per_component.append({
            'name': f'L{layer}_attn',
            'layer': layer,
            'type': 'attention',
            'magnitude': attn_change,
        })
        per_component.append({
            'name': f'L{layer}_mlp',
            'layer': layer,
            'type': 'mlp',
            'magnitude': mlp_change,
        })
        total_change += attn_change + mlp_change

    # Add fractions
    for c in per_component:
        c['fraction'] = c['magnitude'] / (total_change + 1e-10)

    per_component.sort(key=lambda x: x['magnitude'], reverse=True)

    return {
        'per_component': per_component,
        'total_computation': total_change,
        'top_component': per_component[0]['name'] if per_component else '',
    }


def prediction_quality_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Summarize prediction quality across all positions.

    Includes entropy, confidence, and prediction consistency.
    """
    logits, _ = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]

    probs = jax.nn.softmax(logits, axis=-1)
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    per_position = []
    for pos in range(seq_len):
        p = probs[pos]
        lp = log_probs[pos]
        entropy = float(-jnp.sum(p * lp))
        top_token = int(jnp.argmax(p))
        confidence = float(p[top_token])
        top2 = jnp.sort(p)[-2:][::-1]
        margin = float(top2[0] - top2[1])

        per_position.append({
            'position': pos,
            'entropy': entropy,
            'top_token': top_token,
            'confidence': confidence,
            'margin': margin,
        })

    mean_entropy = sum(p['entropy'] for p in per_position) / len(per_position)
    mean_confidence = sum(p['confidence'] for p in per_position) / len(per_position)

    return {
        'per_position': per_position,
        'mean_entropy': mean_entropy,
        'mean_confidence': mean_confidence,
        'most_confident_position': max(per_position, key=lambda x: x['confidence'])['position'],
        'least_confident_position': min(per_position, key=lambda x: x['confidence'])['position'],
    }


def attention_health_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Summarize attention pattern health across all heads.

    Checks for degenerate patterns, extreme sparsity, attention sinks.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    per_head = []
    for layer in range(n_layers):
        pattern_key = f'blocks.{layer}.attn.hook_pattern'
        patterns = cache[pattern_key]  # [n_heads, seq, seq]

        for head in range(n_heads):
            pattern = patterns[head]
            entropy = float(jnp.mean(-jnp.sum(pattern * jnp.log(pattern + 1e-10), axis=-1)))
            max_attn = float(jnp.mean(jnp.max(pattern, axis=-1)))
            bos_weight = float(jnp.mean(pattern[:, 0]))

            is_degenerate = entropy < 0.1
            is_bos_sink = bos_weight > 0.8

            per_head.append({
                'layer': layer,
                'head': head,
                'mean_entropy': entropy,
                'mean_max_attention': max_attn,
                'bos_weight': bos_weight,
                'is_degenerate': is_degenerate,
                'is_bos_sink': is_bos_sink,
            })

    n_degenerate = sum(1 for h in per_head if h['is_degenerate'])
    n_bos_sinks = sum(1 for h in per_head if h['is_bos_sink'])

    return {
        'per_head': per_head,
        'n_degenerate': n_degenerate,
        'n_bos_sinks': n_bos_sinks,
        'n_healthy': len(per_head) - n_degenerate,
    }


def residual_stream_health(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Check residual stream health: norm growth, variance, stability.

    Healthy models should have controlled norm growth without explosion.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    prev_norm = None

    for layer in range(n_layers):
        resid_key = f'blocks.{layer}.hook_resid_post'
        if resid_key not in cache:
            continue
        resid = cache[resid_key]

        mean_norm = float(jnp.mean(jnp.linalg.norm(resid, axis=-1)))
        std_norm = float(jnp.std(jnp.linalg.norm(resid, axis=-1)))
        max_val = float(jnp.max(jnp.abs(resid)))

        growth = mean_norm / prev_norm if prev_norm is not None and prev_norm > 0 else 1.0

        per_layer.append({
            'layer': layer,
            'mean_norm': mean_norm,
            'std_norm': std_norm,
            'max_value': max_val,
            'growth_rate': growth,
            'is_exploding': growth > 5.0,
        })
        prev_norm = mean_norm

    max_growth = max(p['growth_rate'] for p in per_layer) if per_layer else 1.0
    is_stable = max_growth < 3.0

    return {
        'per_layer': per_layer,
        'max_growth_rate': max_growth,
        'is_stable': is_stable,
        'final_norm': per_layer[-1]['mean_norm'] if per_layer else 0.0,
    }
