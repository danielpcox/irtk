"""Attention head importance.

Comprehensive head importance scoring: knockout effects, gradient norms,
output magnitude, logit attribution, and composite ranking.
"""

import jax
import jax.numpy as jnp


def head_knockout_importance(model, tokens):
    """Measure head importance by zeroing each head's output.

    Returns:
        dict with per-head knockout effects.
    """
    clean_logits = model(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    results = []
    for l in range(n_layers):
        for h in range(n_heads):
            def make_hook(target_h):
                def hook_fn(x, name):
                    # x: [seq, n_heads, d_head]
                    return x.at[:, target_h, :].set(0.0)
                return hook_fn

            hook_name = f'blocks.{l}.attn.hook_z'
            mod_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, make_hook(h))])
            diff = mod_logits - clean_logits
            max_change = float(jnp.max(jnp.abs(diff)))
            mean_change = float(jnp.mean(jnp.abs(diff)))

            results.append({
                'layer': l,
                'head': h,
                'max_logit_change': max_change,
                'mean_logit_change': mean_change,
            })

    results.sort(key=lambda r: -r['max_logit_change'])
    return {
        'per_head': results,
        'most_important': results[0] if results else None,
        'least_important': results[-1] if results else None,
    }


def head_output_magnitude(model, tokens):
    """Rank heads by their output magnitude in the residual stream.

    Returns:
        dict with per-head output norms.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    results = []
    for l in range(n_layers):
        z = cache[f'blocks.{l}.attn.hook_z']
        for h in range(n_heads):
            z_h = z[:, h, :]
            W_O_h = model.blocks[l].attn.W_O[h]
            output = z_h @ W_O_h
            norm = float(jnp.mean(jnp.linalg.norm(output, axis=-1)))
            results.append({'layer': l, 'head': h, 'output_norm': norm})

    results.sort(key=lambda r: -r['output_norm'])
    return {'per_head': results}


def head_logit_attribution(model, tokens, position=-1):
    """Attribute logits to specific heads.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        position: position to analyze

    Returns:
        dict with per-head logit attribution.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    W_U = model.unembed.W_U
    pos = position if position >= 0 else len(tokens) - 1

    # Target token: argmax of final logits
    logits = model(tokens)
    target = int(jnp.argmax(logits[pos]))

    results = []
    for l in range(n_layers):
        z = cache[f'blocks.{l}.attn.hook_z']
        for h in range(n_heads):
            z_h = z[pos, h, :]  # [d_head]
            W_O_h = model.blocks[l].attn.W_O[h]
            output = z_h @ W_O_h  # [d_model]
            logit_contrib = float(output @ W_U[:, target])
            results.append({
                'layer': l,
                'head': h,
                'logit_contribution': logit_contrib,
                'abs_contribution': abs(logit_contrib),
            })

    results.sort(key=lambda r: -r['abs_contribution'])
    return {
        'target_token': target,
        'position': pos,
        'per_head': results,
    }


def composite_importance_ranking(model, tokens):
    """Compute composite importance scores combining multiple metrics.

    Returns:
        dict with composite rankings.
    """
    _, cache = model.run_with_cache(tokens)
    clean_logits = model(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    W_U = model.unembed.W_U
    pos = len(tokens) - 1
    target = int(jnp.argmax(clean_logits[pos]))

    results = []
    for l in range(n_layers):
        z = cache[f'blocks.{l}.attn.hook_z']
        for h in range(n_heads):
            z_h = z[:, h, :]
            W_O_h = model.blocks[l].attn.W_O[h]
            output = z_h @ W_O_h

            # Output norm
            norm = float(jnp.mean(jnp.linalg.norm(output, axis=-1)))

            # Logit attribution
            logit_attr = abs(float(jnp.mean(output, axis=0) @ W_U[:, target]))

            # Composite: geometric mean of norm and logit attribution
            composite = float(jnp.sqrt(jnp.array(norm * logit_attr + 1e-10)))

            results.append({
                'layer': l,
                'head': h,
                'output_norm': norm,
                'logit_attribution': logit_attr,
                'composite_score': composite,
            })

    results.sort(key=lambda r: -r['composite_score'])
    return {
        'per_head': results,
        'most_important': results[0] if results else None,
    }


def head_importance_by_position(model, tokens):
    """Measure head importance separately for each position.

    Returns:
        dict with per-position, per-head importance.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    per_position = []
    for pos in range(seq_len):
        head_scores = []
        for l in range(n_layers):
            z = cache[f'blocks.{l}.attn.hook_z']
            for h in range(n_heads):
                z_h = z[pos, h, :]
                W_O_h = model.blocks[l].attn.W_O[h]
                output_norm = float(jnp.linalg.norm(z_h @ W_O_h))
                head_scores.append({
                    'layer': l,
                    'head': h,
                    'output_norm': output_norm,
                })

        head_scores.sort(key=lambda h: -h['output_norm'])
        per_position.append({
            'position': pos,
            'top_head': head_scores[0] if head_scores else None,
            'head_scores': head_scores,
        })

    return {'per_position': per_position}
