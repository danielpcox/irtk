"""Layer communication analysis.

Analyze information passing between layers: message norms, channel
utilization, bandwidth analysis, and layer bypass detection.

References:
    Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"
    Veit et al. (2016) "Residual Networks Behave Like Ensembles of Relatively Shallow Networks"
"""

import jax
import jax.numpy as jnp
import numpy as np


def layer_message_norms(model, tokens, pos=-1):
    """Measure the norm of messages (contributions) each layer sends.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position to analyze.

    Returns:
        dict with:
            attn_message_norms: [n_layers] norm of attention output
            mlp_message_norms: [n_layers] norm of MLP output
            total_message_norms: [n_layers] combined message norm
            residual_norms: [n_layers+1] residual stream norm
            message_to_residual_ratio: [n_layers] how much each layer changes the stream
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    attn_norms = np.zeros(n_layers)
    mlp_norms = np.zeros(n_layers)
    resid_norms = np.zeros(n_layers + 1)

    r = cache.get("blocks.0.hook_resid_pre")
    if r is not None:
        resid_norms[0] = float(np.linalg.norm(np.array(r[pos])))

    for layer in range(n_layers):
        attn = cache.get(f"blocks.{layer}.hook_attn_out")
        if attn is not None:
            attn_norms[layer] = float(np.linalg.norm(np.array(attn[pos])))

        mlp = cache.get(f"blocks.{layer}.hook_mlp_out")
        if mlp is not None:
            mlp_norms[layer] = float(np.linalg.norm(np.array(mlp[pos])))

        r = cache.get(f"blocks.{layer}.hook_resid_post")
        if r is not None:
            resid_norms[layer + 1] = float(np.linalg.norm(np.array(r[pos])))

    total = attn_norms + mlp_norms
    ratio = total / (resid_norms[:-1] + 1e-10)

    return {
        "attn_message_norms": attn_norms,
        "mlp_message_norms": mlp_norms,
        "total_message_norms": total,
        "residual_norms": resid_norms,
        "message_to_residual_ratio": ratio,
    }


def channel_utilization(model, tokens, pos=-1):
    """Analyze how efficiently the d_model dimensions are used per layer.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.

    Returns:
        dict with:
            effective_dims: [n_layers+1] effective dimensionality per layer
            utilization: [n_layers+1] fraction of dims used (effective/total)
            top_dim_fraction: [n_layers+1] fraction of norm in top 10% dims
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    eff_dims = np.zeros(n_layers + 1)
    utilization = np.zeros(n_layers + 1)
    top_frac = np.zeros(n_layers + 1)

    for layer in range(n_layers + 1):
        if layer == 0:
            key = "blocks.0.hook_resid_pre"
        else:
            key = f"blocks.{layer - 1}.hook_resid_post"

        r = cache.get(key)
        if r is not None:
            vec = np.array(r[pos])
            abs_vec = np.abs(vec)
            total_norm = np.sum(abs_vec**2) + 1e-10

            # Effective dimensionality via participation ratio
            pr = total_norm**2 / (np.sum(abs_vec**4) + 1e-10)
            eff_dims[layer] = float(pr)
            utilization[layer] = float(pr / d_model)

            # Top 10% dim fraction
            sorted_sq = np.sort(abs_vec**2)[::-1]
            top_n = max(1, d_model // 10)
            top_frac[layer] = float(np.sum(sorted_sq[:top_n]) / total_norm)

    return {
        "effective_dims": eff_dims,
        "utilization": utilization,
        "top_dim_fraction": top_frac,
    }


def bandwidth_analysis(model, tokens, pos=-1):
    """Analyze the information bandwidth between layers.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.

    Returns:
        dict with:
            layer_deltas: [n_layers, d_model] what each layer adds
            delta_ranks: [n_layers] effective rank of delta
            bandwidth: [n_layers] information bandwidth (norm * rank proxy)
            bottleneck_layer: int, layer with lowest bandwidth
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    deltas = np.zeros((n_layers, d_model))
    delta_ranks = np.zeros(n_layers)
    bandwidth = np.zeros(n_layers)

    for layer in range(n_layers):
        attn = cache.get(f"blocks.{layer}.hook_attn_out")
        mlp = cache.get(f"blocks.{layer}.hook_mlp_out")
        delta = np.zeros(d_model)
        if attn is not None:
            delta += np.array(attn[pos])
        if mlp is not None:
            delta += np.array(mlp[pos])
        deltas[layer] = delta

        # Effective rank via participation ratio on absolute values
        abs_d = np.abs(delta)
        total = np.sum(abs_d**2) + 1e-10
        pr = total**2 / (np.sum(abs_d**4) + 1e-10)
        delta_ranks[layer] = float(pr)
        bandwidth[layer] = float(np.linalg.norm(delta) * pr / d_model)

    bottleneck = int(np.argmin(bandwidth))

    return {
        "layer_deltas": deltas,
        "delta_ranks": delta_ranks,
        "bandwidth": bandwidth,
        "bottleneck_layer": bottleneck,
    }


def layer_bypass_detection(model, tokens, metric_fn):
    """Detect layers that act as near-identity (bypass/skip).

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function from logits -> scalar.

    Returns:
        dict with:
            skip_similarity: [n_layers] how much each layer acts like identity
            metric_impact: [n_layers] metric change when skipping
            is_bypass: [n_layers] boolean (low impact AND low delta)
            n_bypass_layers: int
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    baseline = metric_fn(model(tokens))

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    skip_sim = np.zeros(n_layers)
    metric_impact = np.zeros(n_layers)

    for layer in range(n_layers):
        # Skip similarity: ratio of delta norm to residual norm
        attn = cache.get(f"blocks.{layer}.hook_attn_out")
        mlp = cache.get(f"blocks.{layer}.hook_mlp_out")
        pre_key = "blocks.0.hook_resid_pre" if layer == 0 else f"blocks.{layer - 1}.hook_resid_post"
        resid = cache.get(pre_key)

        if resid is not None:
            resid_norm = float(np.linalg.norm(np.array(resid[-1]))) + 1e-10
            delta_norm = 0.0
            if attn is not None:
                delta_norm += float(np.linalg.norm(np.array(attn[-1])))
            if mlp is not None:
                delta_norm += float(np.linalg.norm(np.array(mlp[-1])))
            skip_sim[layer] = 1.0 - min(1.0, delta_norm / resid_norm)

        # Metric impact when skipping
        hooks = {
            f"blocks.{layer}.hook_attn_out": lambda x, n: jnp.zeros_like(x),
            f"blocks.{layer}.hook_mlp_out": lambda x, n: jnp.zeros_like(x),
        }
        state = HookState(hook_fns=hooks, cache={})
        logits = model(tokens, hook_state=state)
        metric_impact[layer] = abs(baseline - metric_fn(logits))

    # Bypass: high skip similarity AND low metric impact
    threshold_sim = 0.8
    threshold_impact = abs(baseline) * 0.05 if abs(baseline) > 1e-10 else 0.05
    is_bypass = (skip_sim > threshold_sim) & (metric_impact < threshold_impact)

    return {
        "skip_similarity": skip_sim,
        "metric_impact": metric_impact,
        "is_bypass": is_bypass,
        "n_bypass_layers": int(np.sum(is_bypass)),
    }


def inter_layer_alignment(model, tokens, pos=-1):
    """Measure alignment between consecutive layer outputs.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.

    Returns:
        dict with:
            cosine_alignment: [n_layers] cosine between consecutive residuals
            attn_mlp_alignment: [n_layers] cosine between attn and mlp outputs
            mean_alignment: float
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    cos_align = np.zeros(n_layers)
    attn_mlp_align = np.zeros(n_layers)

    residuals = []
    r = cache.get("blocks.0.hook_resid_pre")
    residuals.append(np.array(r[pos]) if r is not None else np.zeros(model.cfg.d_model))
    for layer in range(n_layers):
        r = cache.get(f"blocks.{layer}.hook_resid_post")
        residuals.append(np.array(r[pos]) if r is not None else np.zeros(model.cfg.d_model))

    for layer in range(n_layers):
        a = residuals[layer]
        b = residuals[layer + 1]
        na = np.linalg.norm(a) + 1e-10
        nb = np.linalg.norm(b) + 1e-10
        cos_align[layer] = float(np.dot(a, b) / (na * nb))

        attn = cache.get(f"blocks.{layer}.hook_attn_out")
        mlp = cache.get(f"blocks.{layer}.hook_mlp_out")
        if attn is not None and mlp is not None:
            av = np.array(attn[pos])
            mv = np.array(mlp[pos])
            na = np.linalg.norm(av) + 1e-10
            nm = np.linalg.norm(mv) + 1e-10
            attn_mlp_align[layer] = float(np.dot(av, mv) / (na * nm))

    return {
        "cosine_alignment": cos_align,
        "attn_mlp_alignment": attn_mlp_align,
        "mean_alignment": float(np.mean(cos_align)),
    }
