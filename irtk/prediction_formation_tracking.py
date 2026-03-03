"""Prediction formation tracking.

Track how the model's prediction forms step by step: which components
push toward the final answer, when commitment happens, and what
alternatives are considered.
"""

import jax
import jax.numpy as jnp
from irtk.hook_points import HookState


def _run_and_cache(model, tokens):
    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    return hook_state.cache


def prediction_timeline(model, tokens, pos=-1, top_k=3):
    """Track the top predictions at each stage through the model.

    Returns:
        dict with per_stage list of top predictions.
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)
    if pos < 0:
        pos = seq_len + pos

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    stages = []

    # Embedding
    embed_key = 'blocks.0.hook_resid_pre'
    if embed_key in cache:
        logits = cache[embed_key][pos] @ W_U + b_U
        probs = jax.nn.softmax(logits)
        top_indices = jnp.argsort(-logits)[:top_k]
        tops = [{'token': int(t), 'logit': float(logits[t]), 'prob': float(probs[t])} for t in top_indices]
        stages.append({'stage': 'embed', 'top_predictions': tops})

    # After each layer
    for l in range(n_layers):
        key = f'blocks.{l}.hook_resid_post'
        if key not in cache:
            continue
        logits = cache[key][pos] @ W_U + b_U
        probs = jax.nn.softmax(logits)
        top_indices = jnp.argsort(-logits)[:top_k]
        tops = [{'token': int(t), 'logit': float(logits[t]), 'prob': float(probs[t])} for t in top_indices]
        stages.append({'stage': f'L{l}', 'top_predictions': tops})

    return {'pos': pos, 'stages': stages}


def commitment_point(model, tokens, pos=-1):
    """Find when the model commits to its final prediction.

    Returns:
        dict with:
        - commit_layer: first layer where final prediction becomes top-1
        - final_token: the final predicted token
        - confidence_trajectory: confidence at each layer
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)
    if pos < 0:
        pos = seq_len + pos

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    # Final prediction
    final_logits = model(tokens)
    final_token = int(jnp.argmax(final_logits[pos]))

    trajectory = []
    commit_layer = n_layers  # default: never committed early

    for l in range(n_layers):
        key = f'blocks.{l}.hook_resid_post'
        if key not in cache:
            continue

        logits = cache[key][pos] @ W_U + b_U
        probs = jax.nn.softmax(logits)
        top = int(jnp.argmax(logits))
        confidence = float(probs[final_token])

        is_committed = top == final_token
        trajectory.append({
            'layer': l,
            'top_token': top,
            'is_committed': is_committed,
            'final_token_confidence': confidence,
        })

        if is_committed and commit_layer == n_layers:
            commit_layer = l

    return {
        'final_token': final_token,
        'commit_layer': commit_layer,
        'trajectory': trajectory,
    }


def prediction_drivers(model, tokens, pos=-1, target_token=None):
    """Identify which components drive the final prediction.

    Returns:
        dict with per_component logit contributions sorted by impact.
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)
    if pos < 0:
        pos = seq_len + pos

    if target_token is None:
        logits = model(tokens)
        target_token = int(jnp.argmax(logits[pos]))

    W_U_col = model.unembed.W_U[:, target_token]

    drivers = []

    # Embedding
    embed_key = 'blocks.0.hook_resid_pre'
    if embed_key in cache:
        logit = float(jnp.dot(cache[embed_key][pos], W_U_col))
        drivers.append({'component': 'embed', 'logit': logit, 'abs_logit': abs(logit)})

    for l in range(n_layers):
        for comp, name in [('hook_attn_out', f'L{l}_attn'), ('hook_mlp_out', f'L{l}_mlp')]:
            key = f'blocks.{l}.{comp}'
            if key in cache:
                logit = float(jnp.dot(cache[key][pos], W_U_col))
                drivers.append({'component': name, 'logit': logit, 'abs_logit': abs(logit)})

    drivers.sort(key=lambda x: -x['abs_logit'])
    return {'target_token': target_token, 'drivers': drivers}


def alternative_prediction_analysis(model, tokens, pos=-1, top_k=3):
    """Track how alternative predictions rise and fall through layers.

    Returns:
        dict with per_token trajectories for top final predictions.
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)
    if pos < 0:
        pos = seq_len + pos

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    # Get final top-k
    final_logits = model(tokens)
    top_tokens = jnp.argsort(-final_logits[pos])[:top_k]

    trajectories = {int(t): [] for t in top_tokens}

    for l in range(n_layers):
        key = f'blocks.{l}.hook_resid_post'
        if key not in cache:
            continue

        logits = cache[key][pos] @ W_U + b_U

        for t in top_tokens:
            t_int = int(t)
            rank = int(jnp.sum(logits > logits[t_int]))
            trajectories[t_int].append({
                'layer': l,
                'logit': float(logits[t_int]),
                'rank': rank,
            })

    per_token = []
    for t_int, traj in trajectories.items():
        per_token.append({
            'token': t_int,
            'trajectory': traj,
            'final_rank': traj[-1]['rank'] if traj else -1,
        })

    return {'pos': pos, 'per_token': per_token}


def prediction_stability(model, tokens, pos=-1):
    """How stable is the prediction across layers?

    Returns:
        dict with stability metrics.
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)
    if pos < 0:
        pos = seq_len + pos

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    predictions = []
    for l in range(n_layers):
        key = f'blocks.{l}.hook_resid_post'
        if key not in cache:
            continue
        logits = cache[key][pos] @ W_U + b_U
        predictions.append(int(jnp.argmax(logits)))

    # Count changes
    n_changes = sum(1 for i in range(1, len(predictions)) if predictions[i] != predictions[i-1])
    stability = 1.0 - n_changes / max(len(predictions) - 1, 1)

    # Longest streak
    max_streak = 1
    current_streak = 1
    for i in range(1, len(predictions)):
        if predictions[i] == predictions[i-1]:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 1

    return {
        'predictions': predictions,
        'n_changes': n_changes,
        'stability': stability,
        'longest_streak': max_streak,
        'final_prediction': predictions[-1] if predictions else -1,
    }
