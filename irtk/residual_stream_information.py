"""Residual stream information: information-theoretic analysis."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def residual_prediction_entropy(model: HookedTransformer, tokens: jnp.ndarray,
                                   position: int = -1) -> dict:
    """Entropy of predictions decoded from the residual stream at each layer.

    Tracks how prediction certainty builds through the network.
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
        per_layer.append({
            "layer": layer,
            "entropy": entropy,
        })

    entropies = [p["entropy"] for p in per_layer]
    return {
        "position": position,
        "per_layer": per_layer,
        "entropy_range": max(entropies) - min(entropies) if entropies else 0,
        "final_entropy": entropies[-1] if entropies else 0,
    }


def information_added_per_layer(model: HookedTransformer, tokens: jnp.ndarray,
                                   position: int = -1) -> dict:
    """Information added by each layer (entropy reduction of predictions).

    Positive = layer adds useful information (reduces entropy).
    """
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
        probs = jnp.exp(logits - jnp.max(logits))
        probs = probs / jnp.sum(probs)
        entropy = float(-jnp.sum(probs * jnp.log(probs.clip(1e-10))))

        info_added = (prev_entropy - entropy) if prev_entropy is not None else 0
        per_layer.append({
            "layer": layer,
            "entropy": entropy,
            "info_added": info_added,
            "is_informative": info_added > 0.1,
        })
        prev_entropy = entropy

    return {
        "position": position,
        "per_layer": per_layer,
        "total_info": sum(max(p["info_added"], 0) for p in per_layer),
    }


def residual_kl_from_final(model: HookedTransformer, tokens: jnp.ndarray,
                              position: int = -1) -> dict:
    """KL divergence of each layer's predictions from the final prediction.

    Shows how quickly predictions converge to the final answer.
    """
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    # Final layer distribution
    final_resid = cache[("resid_post", model.cfg.n_layers - 1)][position]
    final_logits = final_resid @ W_U + b_U
    final_probs = jnp.exp(final_logits - jnp.max(final_logits))
    final_probs = final_probs / jnp.sum(final_probs)

    per_layer = []
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)][position]
        logits = resid @ W_U + b_U
        probs = jnp.exp(logits - jnp.max(logits))
        probs = probs / jnp.sum(probs)

        kl = float(jnp.sum(final_probs * jnp.log((final_probs / probs.clip(1e-10)).clip(1e-10))))
        per_layer.append({
            "layer": layer,
            "kl_from_final": max(kl, 0),
        })

    kls = [p["kl_from_final"] for p in per_layer]
    # Find convergence layer (first layer where KL < 0.1)
    conv_layer = model.cfg.n_layers - 1
    for p in per_layer:
        if p["kl_from_final"] < 0.1:
            conv_layer = p["layer"]
            break

    return {
        "position": position,
        "per_layer": per_layer,
        "convergence_layer": conv_layer,
        "initial_kl": kls[0] if kls else 0,
    }


def residual_mutual_information_proxy(model: HookedTransformer, tokens: jnp.ndarray,
                                         layer: int = 0) -> dict:
    """Proxy for mutual information between residual stream and output.

    Uses the explained variance in logit space as an approximation.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    resid = cache[("resid_post", layer)]  # [seq, d_model]
    logits = resid @ W_U  # [seq, d_vocab]

    # Variance of logits across positions
    logit_var = float(jnp.mean(jnp.var(logits, axis=0)))
    # Mean logit magnitude
    logit_mean_mag = float(jnp.mean(jnp.abs(logits)))

    return {
        "layer": layer,
        "logit_variance": logit_var,
        "logit_mean_magnitude": logit_mean_mag,
        "information_proxy": logit_var,
    }


def residual_information_summary(model: HookedTransformer, tokens: jnp.ndarray,
                                    position: int = -1) -> dict:
    """Combined residual stream information analysis."""
    entropy = residual_prediction_entropy(model, tokens, position)
    info = information_added_per_layer(model, tokens, position)
    kl = residual_kl_from_final(model, tokens, position)
    return {
        "position": entropy["position"],
        "final_entropy": entropy["final_entropy"],
        "total_info_added": info["total_info"],
        "convergence_layer": kl["convergence_layer"],
        "initial_kl": kl["initial_kl"],
    }
