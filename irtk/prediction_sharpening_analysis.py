"""Prediction sharpening analysis: how predictions get sharper through layers."""

import jax
import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def sharpening_trajectory(model: HookedTransformer, tokens: jnp.ndarray,
                             position: int = -1) -> dict:
    """Track prediction sharpening (entropy decrease) through layers.

    Measures how the probability distribution concentrates over layers.
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
        probs = jnp.exp(jax.nn.log_softmax(logits))
        entropy = float(-jnp.sum(probs * jnp.log(probs.clip(1e-10))))
        top_prob = float(jnp.max(probs))
        per_layer.append({
            "layer": layer,
            "entropy": entropy,
            "top_prob": top_prob,
            "top_token": int(jnp.argmax(probs)),
        })

    entropies = [p["entropy"] for p in per_layer]
    return {
        "position": position,
        "per_layer": per_layer,
        "initial_entropy": entropies[0],
        "final_entropy": entropies[-1],
        "total_sharpening": entropies[0] - entropies[-1],
    }


def component_sharpening_contribution(model: HookedTransformer, tokens: jnp.ndarray,
                                         layer: int = 0, position: int = -1) -> dict:
    """How much does each component (attn/MLP) sharpen predictions at a layer?"""
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    resid_pre = cache[("resid_pre", layer)][position]
    resid_mid = cache[("resid_mid", layer)][position]
    resid_post = cache[("resid_post", layer)][position]

    def get_entropy(resid):
        logits = resid @ W_U + b_U
        probs = jnp.exp(jax.nn.log_softmax(logits))
        return float(-jnp.sum(probs * jnp.log(probs.clip(1e-10))))

    h_pre = get_entropy(resid_pre)
    h_mid = get_entropy(resid_mid)
    h_post = get_entropy(resid_post)

    return {
        "layer": layer,
        "position": position,
        "entropy_pre": h_pre,
        "entropy_mid": h_mid,
        "entropy_post": h_post,
        "attn_sharpening": h_pre - h_mid,
        "mlp_sharpening": h_mid - h_post,
        "total_sharpening": h_pre - h_post,
    }


def top_k_probability_evolution(model: HookedTransformer, tokens: jnp.ndarray,
                                   position: int = -1, top_k: int = 5) -> dict:
    """Track how top-k token probabilities evolve through layers."""
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    # Get final top tokens
    final_resid = cache[("resid_post", model.cfg.n_layers - 1)][position]
    final_logits = final_resid @ W_U + b_U
    final_probs = jnp.exp(jax.nn.log_softmax(final_logits))
    top_tokens = jnp.argsort(-final_probs)[:top_k]
    top_token_list = [int(t) for t in top_tokens]

    per_layer = []
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)][position]
        logits = resid @ W_U + b_U
        probs = jnp.exp(jax.nn.log_softmax(logits))
        token_probs = {int(t): float(probs[t]) for t in top_tokens}
        per_layer.append({
            "layer": layer,
            "token_probs": token_probs,
            "top_k_mass": sum(token_probs.values()),
        })

    return {
        "position": position,
        "tracked_tokens": top_token_list,
        "per_layer": per_layer,
    }


def sharpening_rate(model: HookedTransformer, tokens: jnp.ndarray,
                       position: int = -1) -> dict:
    """Rate of entropy decrease per layer."""
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    entropies = []
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)][position]
        logits = resid @ W_U + b_U
        probs = jnp.exp(jax.nn.log_softmax(logits))
        entropy = float(-jnp.sum(probs * jnp.log(probs.clip(1e-10))))
        entropies.append(entropy)

    rates = []
    for i in range(len(entropies) - 1):
        rates.append({
            "layers": (i, i + 1),
            "rate": entropies[i] - entropies[i + 1],
            "is_sharpening": entropies[i] > entropies[i + 1],
        })

    max_rate_idx = max(range(len(rates)), key=lambda i: rates[i]["rate"]) if rates else 0
    return {
        "position": position,
        "per_transition": rates,
        "fastest_sharpening_layer": rates[max_rate_idx]["layers"][1] if rates else 0,
        "max_rate": rates[max_rate_idx]["rate"] if rates else 0.0,
    }


def prediction_sharpening_summary(model: HookedTransformer, tokens: jnp.ndarray,
                                     position: int = -1) -> dict:
    """Combined sharpening analysis summary."""
    traj = sharpening_trajectory(model, tokens, position)
    rate = sharpening_rate(model, tokens, position)
    return {
        "position": traj["position"],
        "total_sharpening": traj["total_sharpening"],
        "initial_entropy": traj["initial_entropy"],
        "final_entropy": traj["final_entropy"],
        "fastest_sharpening_layer": rate["fastest_sharpening_layer"],
        "max_rate": rate["max_rate"],
    }
