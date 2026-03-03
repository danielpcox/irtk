"""Layer function characterization: what each layer does."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def layer_effect_on_logits(model: HookedTransformer, tokens: jnp.ndarray,
                            position: int = -1) -> dict:
    """Characterize each layer's effect on the logit distribution.

    Shows how each layer changes the prediction landscape.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    per_layer = []
    for layer in range(model.cfg.n_layers):
        resid_pre = cache[("resid_pre", layer)]
        resid_post = cache[("resid_post", layer)]
        pre_logits = resid_pre[position] @ W_U + b_U
        post_logits = resid_post[position] @ W_U + b_U
        pre_pred = int(jnp.argmax(pre_logits))
        post_pred = int(jnp.argmax(post_logits))
        logit_mse = float(jnp.mean((post_logits - pre_logits) ** 2))
        # Entropy change
        pre_probs = jnp.exp(pre_logits - jnp.max(pre_logits))
        pre_probs = pre_probs / jnp.sum(pre_probs)
        post_probs = jnp.exp(post_logits - jnp.max(post_logits))
        post_probs = post_probs / jnp.sum(post_probs)
        pre_ent = float(-jnp.sum(pre_probs * jnp.log(pre_probs.clip(1e-10))))
        post_ent = float(-jnp.sum(post_probs * jnp.log(post_probs.clip(1e-10))))
        per_layer.append({
            "layer": layer,
            "changes_prediction": pre_pred != post_pred,
            "logit_mse": logit_mse,
            "entropy_change": post_ent - pre_ent,
            "pre_entropy": pre_ent,
            "post_entropy": post_ent,
        })
    return {"per_layer": per_layer}


def layer_attn_mlp_decomposition(model: HookedTransformer, tokens: jnp.ndarray,
                                   position: int = -1) -> dict:
    """Decompose each layer's contribution into attention and MLP parts."""
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U

    per_layer = []
    for layer in range(model.cfg.n_layers):
        attn_out = cache[("attn_out", layer)]
        mlp_out = cache[("mlp_out", layer)]

        attn_logits = attn_out[position] @ W_U
        mlp_logits = mlp_out[position] @ W_U

        attn_impact = float(jnp.std(attn_logits))
        mlp_impact = float(jnp.std(mlp_logits))

        per_layer.append({
            "layer": layer,
            "attn_logit_std": attn_impact,
            "mlp_logit_std": mlp_impact,
            "attn_fraction": attn_impact / max(attn_impact + mlp_impact, 1e-8),
        })
    return {"per_layer": per_layer}


def layer_information_change(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Track how much new information each layer adds (vs amplifying existing).

    Uses cosine distance between pre and post residuals.
    """
    _, cache = model.run_with_cache(tokens)
    per_layer = []
    for layer in range(model.cfg.n_layers):
        pre = cache[("resid_pre", layer)]  # [seq, d_model]
        post = cache[("resid_post", layer)]  # [seq, d_model]
        pre_n = jnp.sqrt(jnp.sum(pre ** 2, axis=-1, keepdims=True)).clip(1e-8)
        post_n = jnp.sqrt(jnp.sum(post ** 2, axis=-1, keepdims=True)).clip(1e-8)
        cos = jnp.mean(jnp.sum((pre / pre_n) * (post / post_n), axis=-1))
        angular_change = float(1.0 - cos)
        norm_change = float(jnp.mean(post_n) / jnp.mean(pre_n))
        per_layer.append({
            "layer": layer,
            "angular_change": angular_change,
            "norm_change": norm_change,
            "is_high_change": angular_change > 0.1,
        })
    return {"per_layer": per_layer}


def layer_role_classification(model: HookedTransformer, tokens: jnp.ndarray,
                               position: int = -1) -> dict:
    """Classify each layer's primary role: refining, redirecting, or amplifying.

    - Refining: small angular change, entropy decreases
    - Redirecting: large angular change
    - Amplifying: small angular change, norm increases
    """
    effect = layer_effect_on_logits(model, tokens, position)
    info = layer_information_change(model, tokens)

    per_layer = []
    for layer in range(model.cfg.n_layers):
        e = effect["per_layer"][layer]
        i = info["per_layer"][layer]
        if i["angular_change"] > 0.1:
            role = "redirecting"
        elif e["entropy_change"] < -0.1:
            role = "refining"
        elif i["norm_change"] > 1.1:
            role = "amplifying"
        else:
            role = "maintaining"
        per_layer.append({
            "layer": layer,
            "role": role,
            "angular_change": i["angular_change"],
            "entropy_change": e["entropy_change"],
            "norm_change": i["norm_change"],
        })
    return {"per_layer": per_layer}


def layer_characterization_summary(model: HookedTransformer, tokens: jnp.ndarray,
                                     position: int = -1) -> dict:
    """Combined layer characterization."""
    roles = layer_role_classification(model, tokens, position)
    decomp = layer_attn_mlp_decomposition(model, tokens, position)
    per_layer = []
    for layer in range(model.cfg.n_layers):
        r = roles["per_layer"][layer]
        d = decomp["per_layer"][layer]
        per_layer.append({
            "layer": layer,
            "role": r["role"],
            "angular_change": r["angular_change"],
            "entropy_change": r["entropy_change"],
            "attn_fraction": d["attn_fraction"],
        })
    return {"per_layer": per_layer}
