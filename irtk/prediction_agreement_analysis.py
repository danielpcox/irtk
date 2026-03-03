"""Prediction agreement analysis: consistency between different prediction methods."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def logit_lens_agreement(model: HookedTransformer, tokens: jnp.ndarray,
                          position: int = -1) -> dict:
    """Compare logit lens predictions across layers.

    Shows how early different layers agree on the final prediction
    and when the model "commits" to its answer.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    b_U = model.unembed.b_U  # [d_vocab]

    final_resid = cache[("resid_post", model.cfg.n_layers - 1)]
    final_logits = final_resid[position] @ W_U + b_U
    final_pred = int(jnp.argmax(final_logits))

    per_layer = []
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)]
        logits = resid[position] @ W_U + b_U
        pred = int(jnp.argmax(logits))
        target_logit = float(logits[final_pred])
        target_rank = int(jnp.sum(logits > logits[final_pred]))
        per_layer.append({
            "layer": layer,
            "prediction": pred,
            "agrees_with_final": pred == final_pred,
            "final_token_logit": target_logit,
            "final_token_rank": target_rank,
        })
    # Find commit layer: first layer where all subsequent layers agree
    commit_layer = model.cfg.n_layers - 1
    for i in range(model.cfg.n_layers - 1, -1, -1):
        if not per_layer[i]["agrees_with_final"]:
            commit_layer = i + 1 if i + 1 < model.cfg.n_layers else i
            break
    else:
        commit_layer = 0
    return {
        "final_prediction": final_pred,
        "commit_layer": commit_layer,
        "per_layer": per_layer,
        "agreement_fraction": sum(1 for p in per_layer if p["agrees_with_final"]) / len(per_layer),
    }


def head_prediction_agreement(model: HookedTransformer, tokens: jnp.ndarray,
                               layer: int = -1, position: int = -1) -> dict:
    """Compare predictions implied by individual attention heads.

    Each head's output projected through W_U gives an implied prediction.
    Agreement indicates redundancy; disagreement indicates competition.
    """
    if layer < 0:
        layer = model.cfg.n_layers + layer
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    z = cache[("z", layer)]  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

    predictions = []
    for head in range(model.cfg.n_heads):
        head_out = z[position, head, :] @ W_O[head]  # [d_model]
        head_logits = head_out @ W_U + b_U
        pred = int(jnp.argmax(head_logits))
        max_logit = float(jnp.max(head_logits))
        predictions.append({
            "head": head,
            "prediction": pred,
            "max_logit": max_logit,
        })
    # Count unique predictions
    preds = [p["prediction"] for p in predictions]
    unique_preds = len(set(preds))
    # Most common prediction
    from collections import Counter
    counts = Counter(preds)
    most_common = counts.most_common(1)[0]

    return {
        "layer": layer,
        "position": int(position % tokens.shape[0]),
        "per_head": predictions,
        "n_unique_predictions": unique_preds,
        "most_common_prediction": most_common[0],
        "most_common_count": most_common[1],
        "agreement_fraction": most_common[1] / model.cfg.n_heads,
    }


def component_prediction_comparison(model: HookedTransformer, tokens: jnp.ndarray,
                                     position: int = -1) -> dict:
    """Compare predictions from attention vs MLP at each layer.

    Shows whether attention and MLP agree or compete on the prediction
    at each layer.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    per_layer = []
    for layer in range(model.cfg.n_layers):
        attn_out = cache[("attn_out", layer)]
        mlp_out = cache[("mlp_out", layer)]

        attn_logits = attn_out[position] @ W_U + b_U
        mlp_logits = mlp_out[position] @ W_U + b_U

        attn_pred = int(jnp.argmax(attn_logits))
        mlp_pred = int(jnp.argmax(mlp_logits))

        # Correlation between logit vectors
        a_centered = attn_logits - jnp.mean(attn_logits)
        m_centered = mlp_logits - jnp.mean(mlp_logits)
        a_norm = jnp.sqrt(jnp.sum(a_centered ** 2)).clip(1e-8)
        m_norm = jnp.sqrt(jnp.sum(m_centered ** 2)).clip(1e-8)
        correlation = float(jnp.sum(a_centered * m_centered) / (a_norm * m_norm))

        per_layer.append({
            "layer": layer,
            "attn_prediction": attn_pred,
            "mlp_prediction": mlp_pred,
            "agree": attn_pred == mlp_pred,
            "logit_correlation": correlation,
        })
    n_agree = sum(1 for p in per_layer if p["agree"])
    return {
        "per_layer": per_layer,
        "agreement_fraction": n_agree / len(per_layer) if per_layer else 0,
        "mean_correlation": sum(p["logit_correlation"] for p in per_layer) / len(per_layer) if per_layer else 0,
    }


def prediction_stability(model: HookedTransformer, tokens: jnp.ndarray,
                          position: int = -1) -> dict:
    """Track how stable the top prediction is across layers.

    Measures the logit gap between top-1 and top-2 predictions at
    each layer to show confidence evolution.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    per_layer = []
    prev_pred = None
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)]
        logits = resid[position] @ W_U + b_U
        sorted_logits = jnp.sort(logits)[::-1]
        pred = int(jnp.argmax(logits))
        gap = float(sorted_logits[0] - sorted_logits[1])
        changed = pred != prev_pred if prev_pred is not None else False
        per_layer.append({
            "layer": layer,
            "prediction": pred,
            "logit_gap": gap,
            "changed_from_previous": changed,
        })
        prev_pred = pred
    changes = sum(1 for p in per_layer if p["changed_from_previous"])
    return {
        "per_layer": per_layer,
        "total_changes": changes,
        "is_stable": changes <= 1,
    }


def prediction_agreement_summary(model: HookedTransformer, tokens: jnp.ndarray,
                                  position: int = -1) -> dict:
    """Combined prediction agreement analysis."""
    lens = logit_lens_agreement(model, tokens, position)
    stability = prediction_stability(model, tokens, position)
    comp = component_prediction_comparison(model, tokens, position)
    return {
        "final_prediction": lens["final_prediction"],
        "commit_layer": lens["commit_layer"],
        "layer_agreement": lens["agreement_fraction"],
        "total_changes": stability["total_changes"],
        "is_stable": stability["is_stable"],
        "attn_mlp_agreement": comp["agreement_fraction"],
        "attn_mlp_correlation": comp["mean_correlation"],
    }
