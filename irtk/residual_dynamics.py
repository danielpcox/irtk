"""Residual stream dynamics analysis.

Tracks how the residual stream evolves through the model: drift analysis,
signal vs noise decomposition, directional alignment tracking, and
information flow patterns.

References:
    Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"
    Veit et al. (2016) "Residual Networks Behave Like Ensembles of Shallow Networks"
"""

import jax
import jax.numpy as jnp
import numpy as np


def residual_drift_analysis(model, tokens, pos=-1):
    """Measure how the residual stream direction changes through layers.

    Drift = cosine distance between consecutive residual stream states.
    High drift means the residual stream direction is changing rapidly.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position to analyze.

    Returns:
        dict with:
            cosine_drift: array [n_layers] of cosine distance between consecutive states
            cumulative_drift: array [n_layers] of cosine distance from initial embedding
            max_drift_layer: int, layer with largest directional change
            total_drift: float, cosine distance from embedding to final state
            norm_trajectory: array [n_layers+1] of residual stream norms
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    states = []
    norms = []

    for layer in range(n_layers + 1):
        if layer == 0:
            key = "blocks.0.hook_resid_pre"
        else:
            key = f"blocks.{layer - 1}.hook_resid_post"
        resid = cache.get(key)
        if resid is not None:
            vec = np.array(resid[pos])
            states.append(vec)
            norms.append(float(np.linalg.norm(vec)))
        else:
            states.append(np.zeros(model.cfg.d_model))
            norms.append(0.0)

    cosine_drift = np.zeros(n_layers)
    cumulative_drift = np.zeros(n_layers)
    initial = states[0]
    initial_norm = np.linalg.norm(initial) + 1e-10

    for l in range(n_layers):
        prev_norm = np.linalg.norm(states[l]) + 1e-10
        curr_norm = np.linalg.norm(states[l + 1]) + 1e-10
        cos_sim = np.dot(states[l], states[l + 1]) / (prev_norm * curr_norm)
        cosine_drift[l] = 1.0 - float(cos_sim)

        cos_from_init = np.dot(initial, states[l + 1]) / (initial_norm * curr_norm)
        cumulative_drift[l] = 1.0 - float(cos_from_init)

    total = cumulative_drift[-1] if len(cumulative_drift) > 0 else 0.0
    max_layer = int(np.argmax(cosine_drift))

    return {
        "cosine_drift": cosine_drift,
        "cumulative_drift": cumulative_drift,
        "max_drift_layer": max_layer,
        "total_drift": float(total),
        "norm_trajectory": np.array(norms),
    }


def signal_noise_decomposition(model, tokens, target_token=None, pos=-1):
    """Decompose residual stream into signal (prediction-relevant) and noise.

    Signal = component aligned with the unembedding direction of the target.
    Noise = orthogonal component.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        target_token: Token to measure signal for (None = top predicted).
        pos: Position.

    Returns:
        dict with:
            signal_norms: array [n_layers+1] of signal component norms
            noise_norms: array [n_layers+1] of noise component norms
            signal_fraction: array [n_layers+1] of signal/(signal+noise) ratio
            snr_trajectory: array [n_layers+1] of signal-to-noise ratio
            signal_growth_rate: array [n_layers] of per-layer signal growth
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    hook_state = HookState(hook_fns={}, cache={})
    logits = model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    if target_token is None:
        target_token = int(jnp.argmax(logits[pos]))

    unembed_dir = np.array(model.unembed.W_U[:, target_token])
    unembed_dir = unembed_dir / (np.linalg.norm(unembed_dir) + 1e-10)

    signal = np.zeros(n_layers + 1)
    noise = np.zeros(n_layers + 1)

    for layer in range(n_layers + 1):
        if layer == 0:
            key = "blocks.0.hook_resid_pre"
        else:
            key = f"blocks.{layer - 1}.hook_resid_post"
        resid = cache.get(key)
        if resid is not None:
            vec = np.array(resid[pos])
            proj = np.dot(vec, unembed_dir)
            signal[layer] = abs(proj)
            noise[layer] = np.sqrt(max(0, np.dot(vec, vec) - proj ** 2))

    total = signal + noise + 1e-10
    frac = signal / total
    snr = signal / (noise + 1e-10)

    growth = np.zeros(n_layers)
    for l in range(n_layers):
        if signal[l] > 1e-10:
            growth[l] = (signal[l + 1] - signal[l]) / signal[l]

    return {
        "signal_norms": signal,
        "noise_norms": noise,
        "signal_fraction": frac,
        "snr_trajectory": snr,
        "signal_growth_rate": growth,
    }


def residual_projection_tracking(model, tokens, directions, pos=-1):
    """Track projections of the residual stream onto specified directions through layers.

    Useful for tracking how specific features (e.g., concept directions) build up.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        directions: array [n_dirs, d_model] of unit vectors to project onto.
        pos: Position.

    Returns:
        dict with:
            projections: array [n_dirs, n_layers+1] of projection magnitudes per direction per layer
            max_projection_layer: array [n_dirs] of layer where each direction peaks
            emergence_layer: array [n_dirs] of layer where projection first exceeds 50% of max
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    directions = np.array(directions)
    n_dirs = directions.shape[0]

    # Normalize directions
    dir_norms = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-10
    directions = directions / dir_norms

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    projections = np.zeros((n_dirs, n_layers + 1))

    for layer in range(n_layers + 1):
        if layer == 0:
            key = "blocks.0.hook_resid_pre"
        else:
            key = f"blocks.{layer - 1}.hook_resid_post"
        resid = cache.get(key)
        if resid is not None:
            vec = np.array(resid[pos])
            for d in range(n_dirs):
                projections[d, layer] = np.dot(vec, directions[d])

    max_layer = np.argmax(np.abs(projections), axis=1).astype(int)
    emergence = np.zeros(n_dirs, dtype=int)
    for d in range(n_dirs):
        max_proj = np.max(np.abs(projections[d]))
        if max_proj > 1e-10:
            threshold = 0.5 * max_proj
            for l in range(n_layers + 1):
                if abs(projections[d, l]) >= threshold:
                    emergence[d] = l
                    break

    return {
        "projections": projections,
        "max_projection_layer": max_layer,
        "emergence_layer": emergence,
    }


def attention_vs_mlp_contribution_ratio(model, tokens, pos=-1):
    """Track the relative contribution of attention vs MLP at each layer.

    Measures how the balance between attention and MLP changes through the model.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.

    Returns:
        dict with:
            attn_norms: array [n_layers] of attention output norms
            mlp_norms: array [n_layers] of MLP output norms
            attn_ratio: array [n_layers] of attention fraction
            attn_dominant_layers: list of layers where attention dominates
            mlp_dominant_layers: list of layers where MLP dominates
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    attn_norms = np.zeros(n_layers)
    mlp_norms = np.zeros(n_layers)

    for layer in range(n_layers):
        attn_out = cache.get(f"blocks.{layer}.hook_attn_out")
        mlp_out = cache.get(f"blocks.{layer}.hook_mlp_out")
        if attn_out is not None:
            attn_norms[layer] = float(np.linalg.norm(np.array(attn_out[pos])))
        if mlp_out is not None:
            mlp_norms[layer] = float(np.linalg.norm(np.array(mlp_out[pos])))

    total = attn_norms + mlp_norms + 1e-10
    attn_ratio = attn_norms / total

    attn_dom = [l for l in range(n_layers) if attn_ratio[l] > 0.5]
    mlp_dom = [l for l in range(n_layers) if attn_ratio[l] <= 0.5]

    return {
        "attn_norms": attn_norms,
        "mlp_norms": mlp_norms,
        "attn_ratio": attn_ratio,
        "attn_dominant_layers": attn_dom,
        "mlp_dominant_layers": mlp_dom,
    }


def residual_stream_bottleneck(model, tokens, pos=-1, n_components=5):
    """Analyze the effective dimensionality of the residual stream at each layer.

    Measures how many dimensions are actively used at each layer, detecting
    potential information bottlenecks.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.
        n_components: Number of SVD components for analysis.

    Returns:
        dict with:
            effective_dims: array [n_layers+1] of effective dimensionality
            top_sv_fraction: array [n_layers+1] of fraction explained by top component
            bottleneck_layer: int, layer with lowest effective dimensionality
            expansion_layer: int, layer with highest effective dimensionality
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    eff_dims = np.zeros(n_layers + 1)
    top_sv_frac = np.zeros(n_layers + 1)

    for layer in range(n_layers + 1):
        if layer == 0:
            key = "blocks.0.hook_resid_pre"
        else:
            key = f"blocks.{layer - 1}.hook_resid_post"
        resid = cache.get(key)
        if resid is not None:
            # Use all positions' residual streams
            matrix = np.array(resid)  # [seq, d_model]
            S = np.linalg.svd(matrix, compute_uv=False)
            S2 = S ** 2
            total = np.sum(S2)
            if total > 1e-10:
                probs = S2 / total
                probs = probs[probs > 1e-12]
                eff_dims[layer] = np.exp(-np.sum(probs * np.log(probs + 1e-12)))
                top_sv_frac[layer] = S2[0] / total
            else:
                eff_dims[layer] = 0
                top_sv_frac[layer] = 0

    bottleneck = int(np.argmin(eff_dims[eff_dims > 0])) if np.any(eff_dims > 0) else 0
    expansion = int(np.argmax(eff_dims))

    return {
        "effective_dims": eff_dims,
        "top_sv_fraction": top_sv_frac,
        "bottleneck_layer": bottleneck,
        "expansion_layer": expansion,
    }
