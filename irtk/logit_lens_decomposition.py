"""Logit lens decomposition: decompose per-layer predictions into component contributions."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def per_layer_logit_lens(model: HookedTransformer, tokens: jnp.ndarray,
                            position: int = -1, top_k: int = 5) -> dict:
    """Logit lens at each layer: what the model would predict if decoding now.

    Shows how predictions form through the network.
    """
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    W_U = model.unembed.W_U  # [d_model, d_vocab]
    b_U = model.unembed.b_U  # [d_vocab]

    per_layer = []
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)][position]  # [d_model]
        logits = resid @ W_U + b_U  # [d_vocab]
        probs = jnp.exp(logits - jnp.max(logits))
        probs = probs / jnp.sum(probs)

        top_indices = jnp.argsort(-logits)[:top_k]
        top_tokens = [int(idx) for idx in top_indices]
        top_probs = [float(probs[idx]) for idx in top_indices]

        per_layer.append({
            "layer": layer,
            "top_token": top_tokens[0],
            "top_prob": top_probs[0],
            "top_k_tokens": top_tokens,
            "top_k_probs": top_probs,
            "entropy": float(-jnp.sum(probs * jnp.log(probs.clip(1e-10)))),
        })
    return {
        "position": position,
        "per_layer": per_layer,
    }


def component_logit_contribution(model: HookedTransformer, tokens: jnp.ndarray,
                                    layer: int = 0, position: int = -1,
                                    top_k: int = 5) -> dict:
    """Attention and MLP contributions to the logit lens at a specific layer.

    Decomposes the prediction change from this layer into attn and MLP parts.
    """
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    attn_out = cache[("attn_out", layer)][position]  # [d_model]
    mlp_out = cache[("mlp_out", layer)][position]  # [d_model]

    attn_logits = attn_out @ W_U  # [d_vocab]
    mlp_logits = mlp_out @ W_U  # [d_vocab]

    # Full logits at this layer
    resid = cache[("resid_post", layer)][position]
    full_logits = resid @ W_U + b_U

    top_indices = jnp.argsort(-full_logits)[:top_k]

    per_token = []
    for idx in top_indices:
        idx_int = int(idx)
        per_token.append({
            "token": idx_int,
            "attn_logit": float(attn_logits[idx_int]),
            "mlp_logit": float(mlp_logits[idx_int]),
            "total_logit": float(full_logits[idx_int]),
        })

    attn_norm = float(jnp.sqrt(jnp.sum(attn_logits ** 2)))
    mlp_norm = float(jnp.sqrt(jnp.sum(mlp_logits ** 2)))
    total = attn_norm + mlp_norm

    return {
        "layer": layer,
        "position": position,
        "per_token": per_token,
        "attn_fraction": attn_norm / max(total, 1e-8),
        "mlp_fraction": mlp_norm / max(total, 1e-8),
    }


def prediction_change_attribution(model: HookedTransformer, tokens: jnp.ndarray,
                                     position: int = -1) -> dict:
    """Attribute the change in top prediction from each layer to attn vs MLP.

    Shows which components drive prediction shifts.
    """
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    per_layer = []
    prev_top = None
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)][position]
        logits = resid @ W_U + b_U
        top = int(jnp.argmax(logits))

        attn_out = cache[("attn_out", layer)][position]
        mlp_out = cache[("mlp_out", layer)][position]
        attn_logit_change = float(attn_out @ W_U[:, top])
        mlp_logit_change = float(mlp_out @ W_U[:, top])

        per_layer.append({
            "layer": layer,
            "top_token": top,
            "changed_prediction": top != prev_top if prev_top is not None else True,
            "attn_logit_for_top": attn_logit_change,
            "mlp_logit_for_top": mlp_logit_change,
        })
        prev_top = top

    n_changes = sum(1 for p in per_layer if p["changed_prediction"])
    return {
        "position": position,
        "per_layer": per_layer,
        "n_prediction_changes": n_changes,
        "final_token": per_layer[-1]["top_token"] if per_layer else -1,
    }


def logit_lens_entropy_trajectory(model: HookedTransformer, tokens: jnp.ndarray,
                                     position: int = -1) -> dict:
    """Entropy of logit lens predictions through layers.

    Decreasing entropy = predictions becoming sharper.
    """
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    per_layer = []
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)][position]
        logits = resid @ W_U + b_U
        probs = jnp.exp(logits - jnp.max(logits))
        probs = probs / jnp.sum(probs)
        entropy = float(-jnp.sum(probs * jnp.log(probs.clip(1e-10))))
        top_prob = float(jnp.max(probs))
        per_layer.append({
            "layer": layer,
            "entropy": entropy,
            "top_prob": top_prob,
        })

    entropies = [p["entropy"] for p in per_layer]
    is_sharpening = len(entropies) > 1 and entropies[-1] < entropies[0]
    return {
        "position": position,
        "per_layer": per_layer,
        "is_sharpening": is_sharpening,
        "entropy_reduction": entropies[0] - entropies[-1] if len(entropies) > 1 else 0,
    }


def logit_lens_decomposition_summary(model: HookedTransformer, tokens: jnp.ndarray,
                                        position: int = -1) -> dict:
    """Combined logit lens decomposition summary."""
    lens = per_layer_logit_lens(model, tokens, position)
    entropy = logit_lens_entropy_trajectory(model, tokens, position)
    change = prediction_change_attribution(model, tokens, position)
    return {
        "position": lens["position"],
        "final_token": change["final_token"],
        "n_prediction_changes": change["n_prediction_changes"],
        "is_sharpening": entropy["is_sharpening"],
        "entropy_reduction": entropy["entropy_reduction"],
    }
