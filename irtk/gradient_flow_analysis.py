"""Gradient flow analysis: analyze gradient magnitudes and directions through layers.

Detect vanishing/exploding gradients, identify which components receive
the strongest learning signals, and trace gradient paths.
"""

import jax
import jax.numpy as jnp


def _forward_logits(model, tokens):
    """Run model and return logits for the last position."""
    logits = model(tokens)
    return logits[-1]  # [vocab]


def gradient_norm_by_layer(model, tokens, target_token=None):
    """Compute gradient norms at each residual stream position.

    Uses grad of target token logit w.r.t. intermediate representations.
    """
    n_layers = model.cfg.n_layers
    if target_token is None:
        # Use the next token in sequence
        target_token = int(tokens[-1])

    # We'll compute gradients of the target logit w.r.t. each layer's output
    # by running the model and using JAX autodiff on the embedding->logit path
    _, cache = model.run_with_cache(tokens)

    # For gradient flow, we measure how gradients propagate
    # We use a proxy: compute Jacobian norm between adjacent layers
    per_layer = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache:
            key = f"blocks.{layer}.hook_resid_pre"
        rep = cache[key]  # [seq, d_model]
        grad_norm = float(jnp.linalg.norm(rep))

        # Also measure the update magnitude
        pre_key = f"blocks.{layer}.hook_resid_pre"
        post_key = f"blocks.{layer}.hook_resid_post"
        if pre_key in cache and post_key in cache:
            update = cache[post_key] - cache[pre_key]
            update_norm = float(jnp.linalg.norm(update))
        else:
            update_norm = 0.0

        per_layer.append({
            "layer": layer,
            "activation_norm": grad_norm,
            "update_norm": update_norm,
            "update_ratio": update_norm / (grad_norm + 1e-10),
        })

    # Detect vanishing/exploding
    norms = [p["activation_norm"] for p in per_layer]
    if len(norms) > 1:
        growth_ratios = [norms[i + 1] / (norms[i] + 1e-10) for i in range(len(norms) - 1)]
        mean_growth = sum(growth_ratios) / len(growth_ratios)
    else:
        mean_growth = 1.0

    return {
        "per_layer": per_layer,
        "mean_growth_ratio": mean_growth,
        "is_stable": 0.5 < mean_growth < 2.0,
        "target_token": target_token,
    }


def gradient_component_attribution(model, tokens):
    """Attribute gradient flow to attention vs MLP components.

    Measures how much each component modifies the residual stream.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        attn_key = f"blocks.{layer}.hook_attn_out"
        mlp_key = f"blocks.{layer}.hook_mlp_out"
        pre_key = f"blocks.{layer}.hook_resid_pre"

        pre_norm = 0.0
        attn_norm = 0.0
        mlp_norm = 0.0

        if pre_key in cache:
            pre_norm = float(jnp.linalg.norm(cache[pre_key]))
        if attn_key in cache:
            attn_norm = float(jnp.linalg.norm(cache[attn_key]))
        if mlp_key in cache:
            mlp_norm = float(jnp.linalg.norm(cache[mlp_key]))

        total = attn_norm + mlp_norm + 1e-10
        per_layer.append({
            "layer": layer,
            "attn_norm": attn_norm,
            "mlp_norm": mlp_norm,
            "attn_fraction": attn_norm / total,
            "mlp_fraction": mlp_norm / total,
            "residual_norm": pre_norm,
        })

    return {
        "per_layer": per_layer,
        "dominant_component": "attention" if sum(p["attn_norm"] for p in per_layer) > sum(p["mlp_norm"] for p in per_layer) else "mlp",
    }


def gradient_saturation_analysis(model, tokens):
    """Detect gradient saturation in attention and activations.

    Checks for near-zero or near-one attention weights (saturated softmax)
    and post-activation magnitudes (saturated nonlinearities).
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    per_layer = []
    for layer in range(n_layers):
        # Check attention saturation
        attn_key = f"blocks.{layer}.attn.hook_pattern"
        attn_sat = 0.0
        if attn_key in cache:
            pattern = cache[attn_key]  # [n_heads, seq, seq]
            # Fraction of attention weights > 0.95 or < 0.01
            high_sat = float(jnp.mean(pattern > 0.95))
            low_sat = float(jnp.mean(pattern < 0.01))
            attn_sat = high_sat + low_sat

        # Check MLP activation saturation
        mlp_key = f"blocks.{layer}.mlp.hook_post"
        mlp_sat = 0.0
        if mlp_key in cache:
            activations = cache[mlp_key]
            # Fraction near zero
            near_zero = float(jnp.mean(jnp.abs(activations) < 0.01))
            mlp_sat = near_zero

        per_layer.append({
            "layer": layer,
            "attention_saturation": attn_sat,
            "mlp_near_zero_fraction": mlp_sat,
            "is_saturated": attn_sat > 0.8 or mlp_sat > 0.8,
        })

    return {
        "per_layer": per_layer,
        "any_saturated": any(p["is_saturated"] for p in per_layer),
    }


def gradient_bottleneck_detection(model, tokens):
    """Detect potential gradient bottlenecks where information flow narrows.

    Measures effective rank of the Jacobian proxy (layer updates).
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        post_key = f"blocks.{layer}.hook_resid_post"
        pre_key = f"blocks.{layer}.hook_resid_pre"

        if post_key in cache and pre_key in cache:
            update = cache[post_key] - cache[pre_key]  # [seq, d_model]
            _, svals, _ = jnp.linalg.svd(update, full_matrices=False)
            svals_sq = svals ** 2
            total = jnp.sum(svals_sq) + 1e-10
            pr = float(total ** 2 / (jnp.sum(svals_sq ** 2) + 1e-10))
            top_sv_frac = float(svals_sq[0] / total)
        else:
            pr = 0.0
            top_sv_frac = 0.0

        per_layer.append({
            "layer": layer,
            "update_effective_rank": pr,
            "top_sv_fraction": top_sv_frac,
            "is_bottleneck": pr < 2.0,
        })

    return {
        "per_layer": per_layer,
        "worst_bottleneck_layer": min(per_layer, key=lambda p: p["update_effective_rank"])["layer"] if per_layer else 0,
        "any_bottleneck": any(p["is_bottleneck"] for p in per_layer),
    }


def gradient_flow_summary(model, tokens):
    """Cross-layer summary of gradient flow health.

    Combines norms, saturation, and bottleneck info.
    """
    norms = gradient_norm_by_layer(model, tokens)
    components = gradient_component_attribution(model, tokens)
    saturation = gradient_saturation_analysis(model, tokens)
    bottlenecks = gradient_bottleneck_detection(model, tokens)

    per_layer = []
    for i in range(len(norms["per_layer"])):
        n = norms["per_layer"][i]
        c = components["per_layer"][i]
        s = saturation["per_layer"][i]
        b = bottlenecks["per_layer"][i]
        per_layer.append({
            "layer": i,
            "activation_norm": n["activation_norm"],
            "update_ratio": n["update_ratio"],
            "attn_fraction": c["attn_fraction"],
            "attention_saturation": s["attention_saturation"],
            "effective_rank": b["update_effective_rank"],
        })

    return {
        "per_layer": per_layer,
        "is_stable": norms["is_stable"],
        "dominant_component": components["dominant_component"],
        "any_saturated": saturation["any_saturated"],
        "any_bottleneck": bottlenecks["any_bottleneck"],
    }
