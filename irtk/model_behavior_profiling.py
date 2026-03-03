"""Model behavior profiling.

Profile overall model behavior: how confident it is, how distributed
predictions are, attention pattern statistics, and computation budget.
"""

import jax
import jax.numpy as jnp
from irtk.hook_points import HookState


def _run_and_cache(model, tokens):
    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    return hook_state.cache


def prediction_confidence_profile(model, tokens):
    """Profile how confident the model is at each position.

    Returns:
        dict with per_position confidence metrics.
    """
    logits = model(tokens)  # [seq, d_vocab]
    seq_len = logits.shape[0]

    results = []
    for p in range(seq_len):
        probs = jax.nn.softmax(logits[p])
        top_prob = float(jnp.max(probs))
        top_token = int(jnp.argmax(probs))

        # Entropy
        probs_safe = jnp.maximum(probs, 1e-10)
        entropy = -float(jnp.sum(probs * jnp.log(probs_safe)))

        # Top-5 probability mass
        top5 = jnp.sort(probs)[-5:]
        top5_mass = float(jnp.sum(top5))

        results.append({
            'position': p,
            'top_token': top_token,
            'top_probability': top_prob,
            'entropy': entropy,
            'top5_mass': top5_mass,
        })

    mean_confidence = sum(r['top_probability'] for r in results) / len(results)
    mean_entropy = sum(r['entropy'] for r in results) / len(results)

    return {
        'per_position': results,
        'mean_confidence': mean_confidence,
        'mean_entropy': mean_entropy,
    }


def attention_pattern_profile(model, tokens):
    """Profile attention pattern statistics across the model.

    Returns:
        dict with per_layer attention statistics.
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers

    results = []
    for l in range(n_layers):
        pattern_key = f'blocks.{l}.attn.hook_pattern'
        if pattern_key not in cache:
            continue

        pattern = cache[pattern_key]  # [n_heads, seq_q, seq_k]
        n_heads = pattern.shape[0]

        per_head = []
        for h in range(n_heads):
            p = pattern[h]  # [seq_q, seq_k]

            # Mean entropy across query positions
            p_safe = jnp.maximum(p, 1e-10)
            entropies = -jnp.sum(p * jnp.log(p_safe), axis=-1)
            mean_entropy = float(jnp.mean(entropies))

            # Sparsity: mean max attention weight
            max_weights = jnp.max(p, axis=-1)
            mean_max = float(jnp.mean(max_weights))

            per_head.append({
                'head': h,
                'mean_entropy': mean_entropy,
                'mean_max_weight': mean_max,
                'is_sparse': mean_max > 0.5,
            })

        mean_layer_entropy = sum(h['mean_entropy'] for h in per_head) / len(per_head)
        n_sparse = sum(1 for h in per_head if h['is_sparse'])

        results.append({
            'layer': l,
            'per_head': per_head,
            'mean_entropy': mean_layer_entropy,
            'n_sparse_heads': n_sparse,
        })

    return {'per_layer': results}


def computation_budget_profile(model, tokens, pos=-1):
    """Where does the model spend its "computation budget"?

    Measures relative contribution magnitudes of different components.

    Returns:
        dict with budget allocation per component.
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)
    if pos < 0:
        pos = seq_len + pos

    budget = []

    for l in range(n_layers):
        attn_key = f'blocks.{l}.hook_attn_out'
        mlp_key = f'blocks.{l}.hook_mlp_out'

        attn_norm = 0.0
        mlp_norm = 0.0

        if attn_key in cache:
            attn_norm = float(jnp.linalg.norm(cache[attn_key][pos]))
        if mlp_key in cache:
            mlp_norm = float(jnp.linalg.norm(cache[mlp_key][pos]))

        budget.append({
            'layer': l,
            'attn_budget': attn_norm,
            'mlp_budget': mlp_norm,
            'total_budget': attn_norm + mlp_norm,
            'attn_fraction': attn_norm / max(attn_norm + mlp_norm, 1e-10),
        })

    total_budget = sum(b['total_budget'] for b in budget)
    for b in budget:
        b['fraction_of_total'] = b['total_budget'] / max(total_budget, 1e-10)

    return {
        'per_layer': budget,
        'total_computation': total_budget,
    }


def position_difficulty_profile(model, tokens):
    """Which positions are "harder" for the model to predict?

    Uses entropy and confidence as difficulty proxies.

    Returns:
        dict with per_position difficulty metrics.
    """
    logits = model(tokens)
    seq_len = logits.shape[0]

    results = []
    for p in range(seq_len):
        probs = jax.nn.softmax(logits[p])
        top_prob = float(jnp.max(probs))

        probs_safe = jnp.maximum(probs, 1e-10)
        entropy = -float(jnp.sum(probs * jnp.log(probs_safe)))
        max_entropy = float(jnp.log(jnp.array(probs.shape[0], dtype=jnp.float32)))
        normalized_entropy = entropy / max(max_entropy, 1e-10)

        # Top-2 gap (how decisive)
        sorted_probs = jnp.sort(probs)
        gap = float(sorted_probs[-1] - sorted_probs[-2])

        results.append({
            'position': p,
            'difficulty': normalized_entropy,  # higher = harder
            'confidence': top_prob,
            'top2_gap': gap,
        })

    # Rank by difficulty
    results.sort(key=lambda x: -x['difficulty'])
    for i, r in enumerate(results):
        r['difficulty_rank'] = i + 1

    return {'per_position': results}


def model_summary_stats(model, tokens):
    """Compute overall summary statistics for model behavior on input.

    Returns:
        dict with high-level summary metrics.
    """
    cache = _run_and_cache(model, tokens)
    logits = model(tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)

    # Prediction stats
    probs = jax.nn.softmax(logits, axis=-1)
    mean_top_prob = float(jnp.mean(jnp.max(probs, axis=-1)))

    probs_safe = jnp.maximum(probs, 1e-10)
    entropies = -jnp.sum(probs * jnp.log(probs_safe), axis=-1)
    mean_entropy = float(jnp.mean(entropies))

    # Residual growth
    embed_key = 'blocks.0.hook_resid_pre'
    final_key = f'blocks.{n_layers - 1}.hook_resid_post'
    growth = 1.0
    if embed_key in cache and final_key in cache:
        embed_norm = float(jnp.mean(jnp.linalg.norm(cache[embed_key], axis=-1)))
        final_norm = float(jnp.mean(jnp.linalg.norm(cache[final_key], axis=-1)))
        growth = final_norm / max(embed_norm, 1e-10)

    # Attention statistics
    total_sparse = 0
    total_heads = 0
    for l in range(n_layers):
        pattern_key = f'blocks.{l}.attn.hook_pattern'
        if pattern_key in cache:
            p = cache[pattern_key]
            max_weights = jnp.max(p, axis=-1)  # [n_heads, seq_q]
            mean_max = float(jnp.mean(max_weights))
            n_heads = p.shape[0]
            total_heads += n_heads
            total_sparse += int(jnp.sum(jnp.mean(jnp.max(p, axis=-1), axis=-1) > 0.5))

    return {
        'n_layers': n_layers,
        'seq_len': seq_len,
        'mean_top_probability': mean_top_prob,
        'mean_entropy': mean_entropy,
        'residual_growth': growth,
        'sparse_head_fraction': total_sparse / max(total_heads, 1) if total_heads > 0 else 0.0,
        'total_heads': total_heads,
    }
