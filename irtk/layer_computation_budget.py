"""Layer computation budget: how much computation each layer uses."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def layer_norm_budget(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Norm of each layer's total update (attn + MLP) to the residual stream."""
    _, cache = model.run_with_cache(tokens)

    per_layer = []
    total_budget = 0.0
    for layer in range(model.cfg.n_layers):
        attn = cache[("attn_out", layer)]  # [seq, d_model]
        mlp = cache[("mlp_out", layer)]    # [seq, d_model]
        combined = attn + mlp

        attn_norm = float(jnp.mean(jnp.sqrt(jnp.sum(attn ** 2, axis=-1))))
        mlp_norm = float(jnp.mean(jnp.sqrt(jnp.sum(mlp ** 2, axis=-1))))
        total_norm = float(jnp.mean(jnp.sqrt(jnp.sum(combined ** 2, axis=-1))))

        per_layer.append({
            "layer": layer,
            "attn_norm": attn_norm,
            "mlp_norm": mlp_norm,
            "total_norm": total_norm,
        })
        total_budget += total_norm

    for p in per_layer:
        p["fraction"] = p["total_norm"] / max(total_budget, 1e-8)

    return {
        "per_layer": per_layer,
        "total_budget": total_budget,
    }


def layer_information_gain(model: HookedTransformer, tokens: jnp.ndarray,
                              position: int = -1) -> dict:
    """Information gain per layer: entropy reduction in predictions."""
    import jax
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    per_layer = []
    prev_entropy = None
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)][position]
        logits = resid @ W_U + b_U
        probs = jnp.exp(jax.nn.log_softmax(logits))
        entropy = float(-jnp.sum(probs * jnp.log(probs.clip(1e-10))))

        info_gain = (prev_entropy - entropy) if prev_entropy is not None else 0.0
        per_layer.append({
            "layer": layer,
            "entropy": entropy,
            "info_gain": info_gain,
            "is_informative": info_gain > 0.01,
        })
        prev_entropy = entropy

    return {
        "position": position,
        "per_layer": per_layer,
        "most_informative": max(range(len(per_layer)),
                               key=lambda i: per_layer[i]["info_gain"]),
    }


def component_balance(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Balance between attention and MLP computation per layer."""
    _, cache = model.run_with_cache(tokens)

    per_layer = []
    for layer in range(model.cfg.n_layers):
        attn = cache[("attn_out", layer)]
        mlp = cache[("mlp_out", layer)]

        attn_norm = float(jnp.mean(jnp.sqrt(jnp.sum(attn ** 2, axis=-1))))
        mlp_norm = float(jnp.mean(jnp.sqrt(jnp.sum(mlp ** 2, axis=-1))))
        total = attn_norm + mlp_norm

        per_layer.append({
            "layer": layer,
            "attn_fraction": attn_norm / max(total, 1e-8),
            "mlp_fraction": mlp_norm / max(total, 1e-8),
            "dominant": "attn" if attn_norm > mlp_norm else "mlp",
        })

    return {"per_layer": per_layer}


def residual_growth_budget(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How much does the residual stream grow at each layer?"""
    _, cache = model.run_with_cache(tokens)

    per_layer = []
    for layer in range(model.cfg.n_layers):
        pre = cache[("resid_pre", layer)]
        post = cache[("resid_post", layer)]

        pre_norm = float(jnp.mean(jnp.sqrt(jnp.sum(pre ** 2, axis=-1))))
        post_norm = float(jnp.mean(jnp.sqrt(jnp.sum(post ** 2, axis=-1))))
        growth = post_norm - pre_norm

        per_layer.append({
            "layer": layer,
            "pre_norm": pre_norm,
            "post_norm": post_norm,
            "growth": growth,
            "growth_rate": post_norm / max(pre_norm, 1e-8),
        })

    return {"per_layer": per_layer}


def computation_budget_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Combined computation budget summary."""
    norm = layer_norm_budget(model, tokens)
    bal = component_balance(model, tokens)
    per_layer = []
    for i in range(model.cfg.n_layers):
        per_layer.append({
            "layer": i,
            "total_norm": norm["per_layer"][i]["total_norm"],
            "fraction": norm["per_layer"][i]["fraction"],
            "dominant": bal["per_layer"][i]["dominant"],
        })
    return {"per_layer": per_layer, "total_budget": norm["total_budget"]}
