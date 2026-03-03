"""Layer skip analysis: what happens when individual layers are bypassed."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def layer_skip_logit_impact(model: HookedTransformer, tokens: jnp.ndarray,
                             position: int = -1) -> dict:
    """Measure how skipping each layer affects the output logits.

    Compares full-model logits with logits when each layer's
    contribution (attn + mlp) is zeroed out.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    final_resid = cache[("resid_post", model.cfg.n_layers - 1)]
    full_logits = final_resid[position] @ W_U + b_U
    full_pred = int(jnp.argmax(full_logits))
    full_max_logit = float(jnp.max(full_logits))

    per_layer = []
    for layer in range(model.cfg.n_layers):
        attn_out = cache[("attn_out", layer)]
        mlp_out = cache[("mlp_out", layer)]
        layer_contribution = attn_out[position] + mlp_out[position]
        skip_resid = final_resid[position] - layer_contribution
        skip_logits = skip_resid @ W_U + b_U
        skip_pred = int(jnp.argmax(skip_logits))
        logit_change = float(jnp.mean((skip_logits - full_logits) ** 2))
        target_logit_change = float(skip_logits[full_pred] - full_logits[full_pred])

        per_layer.append({
            "layer": layer,
            "prediction_changes": skip_pred != full_pred,
            "skip_prediction": skip_pred,
            "mse_logit_change": logit_change,
            "target_logit_change": target_logit_change,
        })

    most_critical = max(per_layer, key=lambda p: p["mse_logit_change"])
    return {
        "full_prediction": full_pred,
        "per_layer": per_layer,
        "most_critical_layer": most_critical["layer"],
    }


def layer_residual_contribution(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Measure each layer's contribution norm relative to the residual.

    Shows how much each layer adds to the growing residual stream.
    """
    _, cache = model.run_with_cache(tokens)
    per_layer = []
    for layer in range(model.cfg.n_layers):
        attn_out = cache[("attn_out", layer)]
        mlp_out = cache[("mlp_out", layer)]
        resid_pre = cache[("resid_pre", layer)]

        contrib = attn_out + mlp_out
        contrib_norm = float(jnp.mean(jnp.sqrt(jnp.sum(contrib ** 2, axis=-1))))
        resid_norm = float(jnp.mean(jnp.sqrt(jnp.sum(resid_pre ** 2, axis=-1))))
        ratio = contrib_norm / max(resid_norm, 1e-8)

        per_layer.append({
            "layer": layer,
            "contribution_norm": contrib_norm,
            "residual_norm": resid_norm,
            "relative_contribution": ratio,
        })
    contributions = [p["contribution_norm"] for p in per_layer]
    return {
        "per_layer": per_layer,
        "max_contribution_layer": int(jnp.argmax(jnp.array(contributions))),
        "mean_contribution": sum(contributions) / len(contributions),
    }


def layer_redundancy_analysis(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Check layer redundancy by measuring similarity of adjacent layer outputs.

    Highly similar adjacent layers suggest redundancy.
    """
    _, cache = model.run_with_cache(tokens)
    pairs = []
    for layer in range(model.cfg.n_layers - 1):
        r1 = cache[("resid_post", layer)]  # [seq, d_model]
        r2 = cache[("resid_post", layer + 1)]  # [seq, d_model]
        # Mean cosine similarity
        n1 = jnp.sqrt(jnp.sum(r1 ** 2, axis=-1, keepdims=True)).clip(1e-8)
        n2 = jnp.sqrt(jnp.sum(r2 ** 2, axis=-1, keepdims=True)).clip(1e-8)
        cos = jnp.mean(jnp.sum((r1 / n1) * (r2 / n2), axis=-1))
        pairs.append({
            "layers": (layer, layer + 1),
            "similarity": float(cos),
            "is_redundant": float(cos) > 0.95,
        })
    n_redundant = sum(1 for p in pairs if p["is_redundant"])
    return {
        "pairs": pairs,
        "n_redundant_pairs": n_redundant,
        "most_similar_pair": max(pairs, key=lambda p: p["similarity"])["layers"] if pairs else None,
    }


def layer_cumulative_effect(model: HookedTransformer, tokens: jnp.ndarray,
                             position: int = -1) -> dict:
    """Track how the prediction builds up cumulatively through layers.

    At each layer, project the residual to logit space to see the
    running prediction trajectory.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    per_layer = []
    prev_pred = None
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)]
        logits = resid[position] @ W_U + b_U
        pred = int(jnp.argmax(logits))
        max_logit = float(jnp.max(logits))
        entropy = float(-jnp.sum(jnp.exp(logits - jnp.max(logits)) /
                         jnp.sum(jnp.exp(logits - jnp.max(logits))) *
                         (logits - jnp.max(logits) -
                          jnp.log(jnp.sum(jnp.exp(logits - jnp.max(logits)))))))
        per_layer.append({
            "layer": layer,
            "prediction": pred,
            "max_logit": max_logit,
            "entropy": entropy,
            "changed": pred != prev_pred if prev_pred is not None else False,
        })
        prev_pred = pred
    return {
        "per_layer": per_layer,
        "final_prediction": per_layer[-1]["prediction"],
        "n_changes": sum(1 for p in per_layer if p["changed"]),
    }


def layer_skip_summary(model: HookedTransformer, tokens: jnp.ndarray,
                        position: int = -1) -> dict:
    """Combined layer skip analysis summary."""
    impact = layer_skip_logit_impact(model, tokens, position)
    contrib = layer_residual_contribution(model, tokens)
    return {
        "most_critical_layer": impact["most_critical_layer"],
        "per_layer": [
            {
                "layer": layer,
                "prediction_changes_on_skip": impact["per_layer"][layer]["prediction_changes"],
                "mse_impact": impact["per_layer"][layer]["mse_logit_change"],
                "contribution_norm": contrib["per_layer"][layer]["contribution_norm"],
                "relative_contribution": contrib["per_layer"][layer]["relative_contribution"],
            }
            for layer in range(model.cfg.n_layers)
        ],
    }
