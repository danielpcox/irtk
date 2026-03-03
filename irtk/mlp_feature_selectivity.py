"""MLP feature selectivity: what inputs neurons respond to."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def neuron_activation_selectivity(model: HookedTransformer, tokens: jnp.ndarray,
                                     layer: int = 0, top_k: int = 10) -> dict:
    """Selectivity of neurons: how many positions activate each neuron.

    Highly selective = fires for few positions; broadly active = fires everywhere.
    """
    _, cache = model.run_with_cache(tokens)
    post = cache[("post", layer)]  # [seq, d_mlp]
    seq_len = post.shape[0]

    active_count = jnp.sum((post > 0).astype(jnp.float32), axis=0)  # [d_mlp]
    selectivity = 1.0 - active_count / seq_len

    top_selective = jnp.argsort(-selectivity)[:top_k]
    top_broad = jnp.argsort(selectivity)[:top_k]

    selective = []
    for idx in top_selective:
        idx_int = int(idx)
        selective.append({
            "neuron": idx_int,
            "selectivity": float(selectivity[idx_int]),
            "active_positions": int(active_count[idx_int]),
        })

    broad = []
    for idx in top_broad:
        idx_int = int(idx)
        broad.append({
            "neuron": idx_int,
            "selectivity": float(selectivity[idx_int]),
            "active_positions": int(active_count[idx_int]),
        })

    return {
        "layer": layer,
        "most_selective": selective,
        "most_broad": broad,
        "mean_selectivity": float(jnp.mean(selectivity)),
    }


def neuron_peak_response(model: HookedTransformer, tokens: jnp.ndarray,
                            layer: int = 0, top_k: int = 10) -> dict:
    """Peak activation magnitude per neuron.

    Neurons with high peak responses may encode specific features strongly.
    """
    _, cache = model.run_with_cache(tokens)
    post = cache[("post", layer)]  # [seq, d_mlp]

    peak_activations = jnp.max(post, axis=0)  # [d_mlp]
    mean_activations = jnp.mean(post, axis=0)  # [d_mlp]

    top_indices = jnp.argsort(-peak_activations)[:top_k]
    top_neurons = []
    for idx in top_indices:
        idx_int = int(idx)
        top_neurons.append({
            "neuron": idx_int,
            "peak_activation": float(peak_activations[idx_int]),
            "mean_activation": float(mean_activations[idx_int]),
            "peak_mean_ratio": float(peak_activations[idx_int] / max(abs(float(mean_activations[idx_int])), 1e-8)),
        })
    return {
        "layer": layer,
        "top_neurons": top_neurons,
        "mean_peak": float(jnp.mean(peak_activations)),
        "max_peak": float(jnp.max(peak_activations)),
    }


def neuron_position_preference(model: HookedTransformer, tokens: jnp.ndarray,
                                  layer: int = 0, neuron: int = 0) -> dict:
    """Which positions does a specific neuron prefer?"""
    _, cache = model.run_with_cache(tokens)
    post = cache[("post", layer)]  # [seq, d_mlp]

    activations = post[:, neuron]  # [seq]
    per_position = []
    for pos in range(len(tokens)):
        per_position.append({
            "position": pos,
            "activation": float(activations[pos]),
            "is_active": float(activations[pos]) > 0,
        })

    peak_pos = int(jnp.argmax(activations))
    return {
        "layer": layer,
        "neuron": neuron,
        "per_position": per_position,
        "peak_position": peak_pos,
        "n_active_positions": sum(1 for p in per_position if p["is_active"]),
    }


def neuron_output_direction(model: HookedTransformer, layer: int = 0,
                               top_k: int = 5) -> dict:
    """Output direction (W_out column) for each neuron.

    Shows what vocabulary items each neuron promotes/suppresses.
    """
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    # Neuron output direction projected through unembedding
    neuron_logits = W_out @ W_U  # [d_mlp, d_vocab]

    per_neuron = []
    for neuron in range(min(top_k, W_out.shape[0])):
        logits = neuron_logits[neuron]
        top_idx = jnp.argsort(-logits)[:5]
        promoted = [(int(idx), float(logits[idx])) for idx in top_idx]
        bot_idx = jnp.argsort(logits)[:5]
        suppressed = [(int(idx), float(logits[idx])) for idx in bot_idx]

        per_neuron.append({
            "neuron": neuron,
            "promoted": promoted,
            "suppressed": suppressed,
        })
    return {
        "layer": layer,
        "per_neuron": per_neuron,
    }


def feature_selectivity_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer feature selectivity summary."""
    per_layer = []
    for layer in range(model.cfg.n_layers):
        sel = neuron_activation_selectivity(model, tokens, layer)
        peak = neuron_peak_response(model, tokens, layer)
        per_layer.append({
            "layer": layer,
            "mean_selectivity": sel["mean_selectivity"],
            "mean_peak": peak["mean_peak"],
        })
    return {"per_layer": per_layer}
