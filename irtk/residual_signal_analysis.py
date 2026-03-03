"""Residual signal analysis: signal vs noise, interference, and coherence."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def signal_noise_ratio(model: HookedTransformer, tokens: jnp.ndarray,
                         position: int = -1) -> dict:
    """Signal-to-noise ratio in the residual stream at each layer.

    Signal = component along the final prediction direction.
    Noise = component orthogonal to it.
    """
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    # Final residual as the "signal direction"
    final_resid = cache[("resid_post", model.cfg.n_layers - 1)][position]
    signal_dir = final_resid / jnp.sqrt(jnp.sum(final_resid ** 2)).clip(1e-8)

    per_layer = []
    for layer in range(model.cfg.n_layers):
        resid = cache[("resid_post", layer)][position]
        resid_norm = float(jnp.sqrt(jnp.sum(resid ** 2)))
        signal_component = float(jnp.sum(resid * signal_dir))
        noise_component = float(jnp.sqrt(max(resid_norm ** 2 - signal_component ** 2, 0)))
        snr = abs(signal_component) / max(noise_component, 1e-8)

        per_layer.append({
            "layer": layer,
            "signal": abs(signal_component),
            "noise": noise_component,
            "snr": snr,
            "signal_fraction": abs(signal_component) / max(resid_norm, 1e-8),
        })
    return {
        "position": position,
        "per_layer": per_layer,
        "final_snr": per_layer[-1]["snr"] if per_layer else 0,
    }


def component_interference(model: HookedTransformer, tokens: jnp.ndarray,
                              layer: int = 0, position: int = -1) -> dict:
    """Interference between attention and MLP outputs.

    Negative cosine = cancellation (destructive interference).
    """
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    attn_out = cache[("attn_out", layer)][position]
    mlp_out = cache[("mlp_out", layer)][position]

    a_norm = jnp.sqrt(jnp.sum(attn_out ** 2)).clip(1e-8)
    m_norm = jnp.sqrt(jnp.sum(mlp_out ** 2)).clip(1e-8)
    cosine = float(jnp.sum(attn_out * mlp_out) / (a_norm * m_norm))

    combined = attn_out + mlp_out
    combined_norm = float(jnp.sqrt(jnp.sum(combined ** 2)))
    expected_norm = float(a_norm) + float(m_norm)
    cancellation = 1.0 - combined_norm / max(expected_norm, 1e-8)

    return {
        "layer": layer,
        "position": position,
        "cosine": cosine,
        "attn_norm": float(a_norm),
        "mlp_norm": float(m_norm),
        "combined_norm": combined_norm,
        "cancellation": cancellation,
        "has_interference": cosine < -0.1,
    }


def residual_coherence(model: HookedTransformer, tokens: jnp.ndarray,
                         position: int = -1) -> dict:
    """How coherent the residual stream updates are across layers.

    High coherence = updates point in consistent directions.
    """
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    updates = []
    for layer in range(model.cfg.n_layers):
        pre = cache[("resid_pre", layer)][position]
        post = cache[("resid_post", layer)][position]
        update = post - pre
        updates.append(update)

    per_pair = []
    for i in range(len(updates) - 1):
        u1 = updates[i]
        u2 = updates[i + 1]
        n1 = jnp.sqrt(jnp.sum(u1 ** 2)).clip(1e-8)
        n2 = jnp.sqrt(jnp.sum(u2 ** 2)).clip(1e-8)
        cos = float(jnp.sum(u1 * u2) / (n1 * n2))
        per_pair.append({
            "layer_pair": (i, i + 1),
            "cosine": cos,
        })

    cosines = [p["cosine"] for p in per_pair]
    mean_coherence = sum(cosines) / max(len(cosines), 1)
    return {
        "position": position,
        "per_pair": per_pair,
        "mean_coherence": mean_coherence,
        "is_coherent": mean_coherence > 0.3,
    }


def update_orthogonality(model: HookedTransformer, tokens: jnp.ndarray,
                            position: int = -1) -> dict:
    """How orthogonal the layer updates are to each other.

    Highly orthogonal updates = each layer adds independent information.
    """
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    updates = []
    for layer in range(model.cfg.n_layers):
        pre = cache[("resid_pre", layer)][position]
        post = cache[("resid_post", layer)][position]
        updates.append(post - pre)

    n = len(updates)
    if n < 2:
        return {"position": position, "mean_orthogonality": 1.0, "pairs": []}

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            n1 = jnp.sqrt(jnp.sum(updates[i] ** 2)).clip(1e-8)
            n2 = jnp.sqrt(jnp.sum(updates[j] ** 2)).clip(1e-8)
            cos = float(jnp.sum(updates[i] * updates[j]) / (n1 * n2))
            pairs.append({
                "layers": (i, j),
                "cosine": cos,
            })

    mean_abs_cos = sum(abs(p["cosine"]) for p in pairs) / len(pairs)
    return {
        "position": position,
        "pairs": pairs,
        "mean_abs_cosine": mean_abs_cos,
        "mean_orthogonality": 1.0 - mean_abs_cos,
    }


def residual_signal_summary(model: HookedTransformer, tokens: jnp.ndarray,
                               position: int = -1) -> dict:
    """Combined residual signal analysis."""
    snr = signal_noise_ratio(model, tokens, position)
    coh = residual_coherence(model, tokens, position)
    orth = update_orthogonality(model, tokens, position)
    return {
        "position": snr["position"],
        "final_snr": snr["final_snr"],
        "mean_coherence": coh["mean_coherence"],
        "is_coherent": coh["is_coherent"],
        "mean_orthogonality": orth["mean_orthogonality"],
    }
