"""Training signal analysis: analyze loss landscape and training signal distribution.

Compute per-token losses, gradient statistics, loss attribution to components,
and training signal concentration patterns.
"""

import jax
import jax.numpy as jnp


def per_token_loss_analysis(model, tokens):
    """Compute cross-entropy loss at each token position.

    Shows which positions the model finds hardest to predict.
    """
    logits = model(tokens)  # [seq, vocab]
    seq_len = tokens.shape[0]

    # Target tokens are shifted: position i predicts position i+1
    per_position = []
    for i in range(seq_len - 1):
        target = int(tokens[i + 1])
        log_probs = jax.nn.log_softmax(logits[i])
        loss = -float(log_probs[target])
        prob = float(jnp.exp(log_probs[target]))
        rank = int(jnp.sum(logits[i] > logits[i, target]))

        per_position.append({
            "position": i,
            "target_token": target,
            "loss": loss,
            "probability": prob,
            "rank": rank,
        })

    mean_loss = sum(p["loss"] for p in per_position) / max(len(per_position), 1)
    max_loss_pos = max(per_position, key=lambda p: p["loss"])["position"] if per_position else 0

    return {
        "per_position": per_position,
        "mean_loss": mean_loss,
        "max_loss_position": max_loss_pos,
        "n_correct": sum(1 for p in per_position if p["rank"] == 0),
    }


def loss_component_attribution(model, tokens, position=-1):
    """Attribute loss to attention and MLP components via their logit contributions.

    Measures how much each component helps or hurts the prediction.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]
    if position < 0:
        position = seq_len + position
    if position >= seq_len - 1:
        position = seq_len - 2

    target = int(tokens[position + 1])
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    per_layer = []
    for layer in range(n_layers):
        attn_key = f"blocks.{layer}.hook_attn_out"
        mlp_key = f"blocks.{layer}.hook_mlp_out"

        attn_logit = 0.0
        mlp_logit = 0.0
        if attn_key in cache:
            attn_out = cache[attn_key][position]  # [d_model]
            attn_logits = attn_out @ W_U  # [vocab]
            attn_logit = float(attn_logits[target])
        if mlp_key in cache:
            mlp_out = cache[mlp_key][position]
            mlp_logits = mlp_out @ W_U
            mlp_logit = float(mlp_logits[target])

        per_layer.append({
            "layer": layer,
            "attn_target_logit": attn_logit,
            "mlp_target_logit": mlp_logit,
            "total_contribution": attn_logit + mlp_logit,
            "helps_prediction": (attn_logit + mlp_logit) > 0,
        })

    return {
        "per_layer": per_layer,
        "target_token": target,
        "position": position,
        "total_attn": sum(p["attn_target_logit"] for p in per_layer),
        "total_mlp": sum(p["mlp_target_logit"] for p in per_layer),
    }


def prediction_entropy_profile(model, tokens):
    """Compute prediction entropy at each position.

    Low entropy = confident prediction; high entropy = uncertain.
    """
    logits = model(tokens)  # [seq, vocab]
    seq_len = tokens.shape[0]

    per_position = []
    for i in range(seq_len):
        probs = jax.nn.softmax(logits[i])
        entropy = -float(jnp.sum(probs * jnp.log(probs + 1e-10)))
        max_prob = float(jnp.max(probs))
        top_token = int(jnp.argmax(logits[i]))

        per_position.append({
            "position": i,
            "entropy": entropy,
            "max_prob": max_prob,
            "top_prediction": top_token,
            "is_confident": max_prob > 0.5,
        })

    mean_entropy = sum(p["entropy"] for p in per_position) / max(len(per_position), 1)

    return {
        "per_position": per_position,
        "mean_entropy": mean_entropy,
        "n_confident": sum(1 for p in per_position if p["is_confident"]),
    }


def loss_concentration_analysis(model, tokens):
    """Analyze how concentrated the loss is across positions.

    If loss is concentrated, a few positions drive most of the training signal.
    """
    result = per_token_loss_analysis(model, tokens)
    losses = [p["loss"] for p in result["per_position"]]
    if not losses:
        return {"gini": 0.0, "top_k_fraction": 0.0, "per_position": []}

    losses_arr = jnp.array(losses)
    total_loss = float(jnp.sum(losses_arr))

    # Sort losses descending
    sorted_losses = jnp.sort(losses_arr)[::-1]
    cumulative = jnp.cumsum(sorted_losses) / (total_loss + 1e-10)

    # Top-k concentration (how much loss comes from top 20%)
    k = max(1, len(losses) // 5)
    top_k_frac = float(jnp.sum(sorted_losses[:k]) / (total_loss + 1e-10))

    # Gini coefficient
    n = len(losses)
    if n > 1 and total_loss > 0:
        sorted_asc = jnp.sort(losses_arr)
        indices = jnp.arange(1, n + 1)
        gini = float((2.0 * jnp.sum(indices * sorted_asc) - (n + 1) * total_loss) / (n * total_loss))
    else:
        gini = 0.0

    return {
        "gini": gini,
        "top_20pct_loss_fraction": top_k_frac,
        "is_concentrated": top_k_frac > 0.5,
        "total_loss": total_loss,
        "n_positions": len(losses),
    }


def training_signal_summary(model, tokens):
    """Cross-position summary of training signal quality.

    Combines loss, entropy, and concentration information.
    """
    loss_result = per_token_loss_analysis(model, tokens)
    entropy_result = prediction_entropy_profile(model, tokens)

    per_position = []
    n = min(len(loss_result["per_position"]), len(entropy_result["per_position"]))
    for i in range(n):
        lp = loss_result["per_position"][i]
        ep = entropy_result["per_position"][i]
        per_position.append({
            "position": i,
            "loss": lp["loss"],
            "rank": lp["rank"],
            "entropy": ep["entropy"],
            "max_prob": ep["max_prob"],
        })

    return {
        "per_position": per_position,
        "mean_loss": loss_result["mean_loss"],
        "mean_entropy": entropy_result["mean_entropy"],
        "n_correct": loss_result["n_correct"],
        "n_confident": entropy_result["n_confident"],
    }
