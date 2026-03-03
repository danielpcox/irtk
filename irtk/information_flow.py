"""Information-theoretic analysis of transformer layers.

Measure mutual information, entropy, and information bottleneck
properties to understand how task-relevant information transforms
through layers.

References:
    - Shwartz-Ziv & Tishby (2017) "Opening the Black Box of DNNs"
    - Voita et al. (2019) "The Bottom-up Evolution of Representations"
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def layer_entropy(
    model: HookedTransformer,
    token_sequences: list,
    n_bins: int = 50,
) -> dict:
    """Compute activation entropy at each layer.

    Discretizes activations into bins and computes Shannon entropy,
    measuring the information content per layer.

    Args:
        model: HookedTransformer.
        token_sequences: Test inputs.
        n_bins: Number of histogram bins.

    Returns:
        Dict with:
        - "layer_entropies": [n_layers] entropy per layer
        - "embedding_entropy": entropy of the embedding layer
        - "entropy_trend": "increasing", "decreasing", or "non-monotonic"
        - "max_entropy_layer": layer with highest entropy
    """
    n_layers = model.cfg.n_layers
    entropies = np.zeros(n_layers)
    embed_entropies = []

    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)

        # Embedding entropy
        if "hook_embed" in cache.cache_dict:
            emb = np.array(cache.cache_dict["hook_embed"]).flatten()
            h, _ = np.histogram(emb, bins=n_bins, density=True)
            h = h[h > 0]
            h = h / h.sum()
            embed_entropies.append(float(-np.sum(h * np.log(h + 1e-10))))

        for layer in range(n_layers):
            hook = f"blocks.{layer}.hook_resid_post"
            if hook in cache.cache_dict:
                acts = np.array(cache.cache_dict[hook]).flatten()
                h, _ = np.histogram(acts, bins=n_bins, density=True)
                h = h[h > 0]
                h = h / h.sum()
                entropies[layer] += float(-np.sum(h * np.log(h + 1e-10)))

    n_inputs = max(len(token_sequences), 1)
    entropies /= n_inputs

    # Determine trend
    if n_layers >= 2:
        diffs = np.diff(entropies)
        if np.all(diffs >= -0.01):
            trend = "increasing"
        elif np.all(diffs <= 0.01):
            trend = "decreasing"
        else:
            trend = "non-monotonic"
    else:
        trend = "flat"

    return {
        "layer_entropies": entropies,
        "embedding_entropy": float(np.mean(embed_entropies)) if embed_entropies else 0.0,
        "entropy_trend": trend,
        "max_entropy_layer": int(np.argmax(entropies)),
    }


def mutual_information_estimate(
    model: HookedTransformer,
    token_sequences: list,
    layer: int,
    n_bins: int = 30,
) -> dict:
    """Estimate mutual information between input embeddings and layer activations.

    Uses binned joint and marginal histograms as an approximation.

    Args:
        model: HookedTransformer.
        token_sequences: Test inputs.
        layer: Layer to measure MI at.
        n_bins: Number of bins for discretization.

    Returns:
        Dict with:
        - "mutual_information": estimated MI in nats
        - "input_entropy": entropy of input embeddings
        - "layer_entropy": entropy of layer activations
        - "normalized_mi": MI / min(H_input, H_layer)
    """
    input_vals = []
    layer_vals = []

    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)

        if "hook_embed" in cache.cache_dict:
            emb = np.array(cache.cache_dict["hook_embed"]).flatten()
            input_vals.extend(emb.tolist())

        hook = f"blocks.{layer}.hook_resid_post"
        if hook in cache.cache_dict:
            acts = np.array(cache.cache_dict[hook]).flatten()
            layer_vals.extend(acts.tolist())

    if not input_vals or not layer_vals:
        return {"mutual_information": 0.0, "input_entropy": 0.0,
                "layer_entropy": 0.0, "normalized_mi": 0.0}

    # Marginal entropies
    min_len = min(len(input_vals), len(layer_vals))
    x = np.array(input_vals[:min_len])
    y = np.array(layer_vals[:min_len])

    hx, _ = np.histogram(x, bins=n_bins, density=True)
    hx = hx[hx > 0]; hx = hx / hx.sum()
    H_x = float(-np.sum(hx * np.log(hx + 1e-10)))

    hy, _ = np.histogram(y, bins=n_bins, density=True)
    hy = hy[hy > 0]; hy = hy / hy.sum()
    H_y = float(-np.sum(hy * np.log(hy + 1e-10)))

    # Joint entropy
    hxy, _, _ = np.histogram2d(x, y, bins=n_bins, density=True)
    hxy = hxy[hxy > 0]; hxy = hxy / hxy.sum()
    H_xy = float(-np.sum(hxy * np.log(hxy + 1e-10)))

    mi = max(0.0, H_x + H_y - H_xy)
    norm_mi = mi / max(min(H_x, H_y), 1e-10)

    return {
        "mutual_information": mi,
        "input_entropy": H_x,
        "layer_entropy": H_y,
        "normalized_mi": float(norm_mi),
    }


def compression_analysis(
    model: HookedTransformer,
    token_sequences: list,
    n_bins: int = 30,
) -> dict:
    """Identify compression and fitting phases across layers.

    Measures how information compresses (lower dimensionality / entropy)
    or expands (higher dimensionality) at each layer.

    Args:
        model: HookedTransformer.
        token_sequences: Test inputs.
        n_bins: Number of bins.

    Returns:
        Dict with:
        - "effective_dimensions": [n_layers] effective dimensionality per layer
        - "compression_ratios": [n_layers] ratio to max possible
        - "compression_phase": first layer where compression begins
        - "fitting_phase": last layer where expansion occurs
    """
    n_layers = model.cfg.n_layers
    eff_dims = np.zeros(n_layers)

    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)

        for layer in range(n_layers):
            hook = f"blocks.{layer}.hook_resid_post"
            if hook in cache.cache_dict:
                acts = np.array(cache.cache_dict[hook])
                if acts.ndim > 1:
                    # Effective dimension: participation ratio of singular values
                    flat = acts.reshape(-1, acts.shape[-1])
                    s = np.linalg.svd(flat, compute_uv=False)
                    s = s / max(s.sum(), 1e-10)
                    pr = (s.sum() ** 2) / max((s ** 2).sum(), 1e-10)
                    eff_dims[layer] += float(pr)

    n_inputs = max(len(token_sequences), 1)
    eff_dims /= n_inputs

    d_model = model.cfg.d_model
    comp_ratios = eff_dims / max(d_model, 1)

    # Find phases
    compression_start = n_layers
    fitting_end = 0
    for i in range(1, n_layers):
        if eff_dims[i] < eff_dims[i - 1] * 0.95:
            compression_start = min(compression_start, i)
        if eff_dims[i] > eff_dims[i - 1] * 1.05:
            fitting_end = max(fitting_end, i)

    return {
        "effective_dimensions": eff_dims,
        "compression_ratios": comp_ratios,
        "compression_phase": compression_start,
        "fitting_phase": fitting_end,
    }


def information_flow_by_position(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    source_pos: int,
    target_pos: int,
) -> dict:
    """Track information flow from source to target position through layers.

    Measures how much the source position's representation influences
    the target position at each layer via attention patterns.

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        source_pos: Source token position.
        target_pos: Target token position.

    Returns:
        Dict with:
        - "attention_to_source": [n_layers, n_heads] attention from target to source
        - "layer_influence": [n_layers] aggregated influence per layer
        - "peak_layer": layer with strongest source->target flow
        - "total_flow": sum of layer influences
    """
    tokens = jnp.array(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    _, cache = model.run_with_cache(tokens)

    attn_scores = np.zeros((n_layers, n_heads))
    layer_inf = np.zeros(n_layers)

    for layer in range(n_layers):
        hook = f"blocks.{layer}.attn.hook_pattern"
        if hook in cache.cache_dict:
            pat = np.array(cache.cache_dict[hook])
            # pat shape: [n_heads, seq, seq] or [seq, seq]
            if pat.ndim == 3:
                for h in range(min(n_heads, pat.shape[0])):
                    if target_pos < pat.shape[1] and source_pos < pat.shape[2]:
                        attn_scores[layer, h] = float(pat[h, target_pos, source_pos])
            elif pat.ndim == 2:
                if target_pos < pat.shape[0] and source_pos < pat.shape[1]:
                    attn_scores[layer, 0] = float(pat[target_pos, source_pos])

            layer_inf[layer] = float(np.mean(attn_scores[layer]))

    peak = int(np.argmax(layer_inf))
    total = float(np.sum(layer_inf))

    return {
        "attention_to_source": attn_scores,
        "layer_influence": layer_inf,
        "peak_layer": peak,
        "total_flow": total,
    }


def information_bottleneck_curve(
    model: HookedTransformer,
    token_sequences: list,
    metric_fn: Callable,
    n_bins: int = 30,
) -> dict:
    """Compute the information plane trajectory across layers.

    For each layer, estimates I(X; T) (compression) and I(T; Y)
    (prediction quality), where T is the layer representation.

    Args:
        model: HookedTransformer.
        token_sequences: Test inputs.
        metric_fn: Function from logits -> float (proxy for Y).
        n_bins: Number of bins.

    Returns:
        Dict with:
        - "compression": [n_layers] I(X; T) estimate per layer
        - "prediction": [n_layers] correlation between T and metric
        - "ib_trade_off": [n_layers] prediction / (compression + eps)
        - "optimal_layer": layer with best trade-off
    """
    n_layers = model.cfg.n_layers

    # Collect activations and metrics
    all_embeds = []
    all_layer_acts = [[] for _ in range(n_layers)]
    all_metrics = []

    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        logits = model(tokens)
        all_metrics.append(float(metric_fn(logits)))

        _, cache = model.run_with_cache(tokens)

        if "hook_embed" in cache.cache_dict:
            emb = np.array(cache.cache_dict["hook_embed"]).flatten()
            all_embeds.append(emb)

        for layer in range(n_layers):
            hook = f"blocks.{layer}.hook_resid_post"
            if hook in cache.cache_dict:
                acts = np.array(cache.cache_dict[hook]).flatten()
                all_layer_acts[layer].append(acts)

    if not all_embeds or not all_metrics:
        return {"compression": np.zeros(n_layers), "prediction": np.zeros(n_layers),
                "ib_trade_off": np.zeros(n_layers), "optimal_layer": 0}

    metrics_arr = np.array(all_metrics)

    compression = np.zeros(n_layers)
    prediction = np.zeros(n_layers)

    for layer in range(n_layers):
        if not all_layer_acts[layer]:
            continue

        # Prediction: correlation of layer norm with metric
        norms = np.array([np.linalg.norm(a) for a in all_layer_acts[layer]])
        if len(norms) > 1 and np.std(norms) > 1e-10 and np.std(metrics_arr) > 1e-10:
            prediction[layer] = abs(float(np.corrcoef(norms, metrics_arr[:len(norms)])[0, 1]))

        # Compression: effective dimensionality ratio
        if len(all_layer_acts[layer]) > 1:
            stacked = np.array([a[:min(len(a) for a in all_layer_acts[layer])]
                               for a in all_layer_acts[layer]])
            s = np.linalg.svd(stacked, compute_uv=False)
            s = s / max(s.sum(), 1e-10)
            pr = (s.sum() ** 2) / max((s ** 2).sum(), 1e-10)
            compression[layer] = float(pr / max(stacked.shape[1], 1))

    trade_off = prediction / np.maximum(compression, 1e-10)
    optimal = int(np.argmax(trade_off))

    return {
        "compression": compression,
        "prediction": prediction,
        "ib_trade_off": trade_off,
        "optimal_layer": optimal,
    }
