"""Token prediction dynamics: how predictions evolve through the model.

Track how the model's predicted token changes layer by layer,
including rank changes, confidence evolution, and prediction stability.
"""

import jax
import jax.numpy as jnp


def prediction_trajectory(model, tokens, position=-1):
    """Track the predicted token at each layer.

    Projects the residual stream through the unembedding at each layer.

    Returns:
        dict with 'per_layer' list of prediction dicts, 'n_changes' int.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    b_U = model.unembed.b_U  # [d_vocab]
    n_layers = len(model.blocks)
    per_layer = []
    prev_pred = None
    n_changes = 0
    for layer in range(n_layers):
        resid = cache[("resid_post", layer)]  # [seq, d_model]
        h = resid[position]  # [d_model]
        layer_logits = h @ W_U + b_U  # [d_vocab]
        pred_token = int(jnp.argmax(layer_logits))
        pred_logit = float(layer_logits[pred_token])
        probs = jax.nn.softmax(layer_logits)
        pred_prob = float(probs[pred_token])
        changed = prev_pred is not None and pred_token != prev_pred
        if changed:
            n_changes += 1
        per_layer.append({
            "layer": layer,
            "predicted_token": pred_token,
            "predicted_logit": pred_logit,
            "predicted_prob": pred_prob,
            "changed_from_previous": changed,
        })
        prev_pred = pred_token
    return {"per_layer": per_layer, "n_changes": n_changes}


def prediction_confidence_evolution(model, tokens, position=-1):
    """How does prediction confidence (max prob) evolve through layers?

    Returns:
        dict with 'per_layer' list with entropy and max_prob, 'final_entropy'.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        resid = cache[("resid_post", layer)]
        h = resid[position]
        layer_logits = h @ W_U + b_U
        log_probs = jax.nn.log_softmax(layer_logits)
        probs = jnp.exp(log_probs)
        entropy = float(-jnp.sum(probs * log_probs))
        max_prob = float(jnp.max(probs))
        per_layer.append({
            "layer": layer,
            "entropy": entropy,
            "max_prob": max_prob,
        })
    return {
        "per_layer": per_layer,
        "final_entropy": per_layer[-1]["entropy"],
    }


def prediction_rank_tracking(model, tokens, position=-1, track_tokens=None):
    """Track how specific token ranks change through layers.

    If track_tokens is None, tracks the final predicted token.

    Returns:
        dict with 'tracked_tokens' list, 'per_layer' list of rank dicts.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    n_layers = len(model.blocks)
    if track_tokens is None:
        final_resid = cache[("resid_post", n_layers - 1)]
        final_logits = final_resid[position] @ W_U + b_U
        track_tokens = [int(jnp.argmax(final_logits))]
    per_layer = []
    for layer in range(n_layers):
        resid = cache[("resid_post", layer)]
        h = resid[position]
        layer_logits = h @ W_U + b_U
        ranks_sorted = jnp.argsort(-layer_logits)
        rank_map = {}
        for tok in track_tokens:
            rank = int(jnp.argmax(ranks_sorted == tok))
            rank_map[int(tok)] = rank
        per_layer.append({
            "layer": layer,
            "ranks": rank_map,
        })
    return {
        "tracked_tokens": [int(t) for t in track_tokens],
        "per_layer": per_layer,
    }


def prediction_stability(model, tokens, position=-1):
    """How stable is the prediction across layers?

    Measures cosine similarity of logit vectors between consecutive layers.

    Returns:
        dict with 'per_transition' list, 'mean_stability'.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    n_layers = len(model.blocks)
    layer_logits_list = []
    for layer in range(n_layers):
        resid = cache[("resid_post", layer)]
        h = resid[position]
        layer_logits_list.append(h @ W_U + b_U)
    per_transition = []
    for i in range(1, n_layers):
        prev = layer_logits_list[i - 1]
        curr = layer_logits_list[i]
        cos = float(jnp.dot(prev, curr) / (jnp.linalg.norm(prev) * jnp.linalg.norm(curr) + 1e-10))
        per_transition.append({
            "from_layer": i - 1,
            "to_layer": i,
            "cosine_similarity": cos,
        })
    mean_stab = sum(p["cosine_similarity"] for p in per_transition) / len(per_transition) if per_transition else 1.0
    return {
        "per_transition": per_transition,
        "mean_stability": mean_stab,
    }


def token_prediction_dynamics_summary(model, tokens, position=-1):
    """Summary of prediction dynamics across the model.

    Returns:
        dict with 'n_changes', 'final_entropy', 'mean_stability',
        'per_layer' predictions.
    """
    traj = prediction_trajectory(model, tokens, position=position)
    conf = prediction_confidence_evolution(model, tokens, position=position)
    stab = prediction_stability(model, tokens, position=position)
    return {
        "n_changes": traj["n_changes"],
        "final_entropy": conf["final_entropy"],
        "mean_stability": stab["mean_stability"],
        "per_layer": traj["per_layer"],
    }
