"""Model internals dashboard for mechanistic interpretability.

Comprehensive summary tools for understanding a model's internal state:
per-layer statistics, head classification, MLP utilization, residual
stream health, and bottleneck detection.

Designed for quick model audits before deeper mechanistic analysis.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional


def layer_statistics(
    model,
    tokens,
    layers: Optional[list] = None,
) -> dict:
    """Compute comprehensive per-layer statistics.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        layers: Layers to analyze. Defaults to all.

    Returns:
        Dict with per_layer stats (residual norms, attn entropy,
        MLP activation stats), summary statistics.
    """
    from irtk.hook_points import HookState

    if layers is None:
        layers = list(range(model.cfg.n_layers))

    # Collect all needed activations
    cache = {}
    hook_state = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hook_state)

    per_layer = []
    for l in layers:
        stats = {"layer": l}

        # Residual stream norms
        resid_key = f"blocks.{l}.hook_resid_post"
        if resid_key in cache:
            resid = np.array(cache[resid_key])
            stats["resid_norm_mean"] = float(np.mean(np.linalg.norm(resid, axis=-1)))
            stats["resid_norm_std"] = float(np.std(np.linalg.norm(resid, axis=-1)))

        # Attention pattern entropy
        pattern_key = f"blocks.{l}.attn.hook_pattern"
        if pattern_key in cache:
            pattern = np.array(cache[pattern_key])  # [n_heads, seq, seq]
            entropy = -np.sum(pattern * np.log(pattern + 1e-10), axis=-1)  # [n_heads, seq]
            stats["attn_entropy_mean"] = float(np.mean(entropy))
            stats["attn_entropy_per_head"] = [float(np.mean(entropy[h])) for h in range(pattern.shape[0])]

        # MLP activation norms
        mlp_key = f"blocks.{l}.mlp.hook_post"
        if mlp_key in cache:
            mlp_act = np.array(cache[mlp_key])
            stats["mlp_activation_mean"] = float(np.mean(np.abs(mlp_act)))
            stats["mlp_sparsity"] = float(np.mean(np.abs(mlp_act) < 0.01))

        # Attention output norm
        attn_key = f"blocks.{l}.hook_attn_out"
        if attn_key in cache:
            attn_out = np.array(cache[attn_key])
            stats["attn_output_norm"] = float(np.mean(np.linalg.norm(attn_out, axis=-1)))

        per_layer.append(stats)

    # Summary
    resid_norms = [s.get("resid_norm_mean", 0) for s in per_layer]
    summary = {
        "max_resid_layer": int(np.argmax(resid_norms)) if resid_norms else 0,
        "resid_growth": float(resid_norms[-1] / max(resid_norms[0], 1e-10)) if len(resid_norms) > 1 else 1.0,
        "n_layers": len(layers),
    }

    return {
        "per_layer": per_layer,
        "summary": summary,
    }


def head_classification_summary(
    model,
    tokens,
    top_k: int = 5,
) -> dict:
    """Classify each attention head by behavior pattern.

    Categories: previous-token, induction, positional, content-based.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        top_k: Number of top heads to highlight per category.

    Returns:
        Dict with classifications, category_counts, top_heads_per_category.
    """
    from irtk.hook_points import HookState

    cache = {}
    hook_state = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hook_state)

    seq_len = len(tokens)
    classifications = {}
    scores = {"previous_token": [], "diagonal": [], "first_token": [], "uniform": []}

    for l in range(model.cfg.n_layers):
        pattern_key = f"blocks.{l}.attn.hook_pattern"
        if pattern_key not in cache:
            continue
        pattern = np.array(cache[pattern_key])  # [n_heads, seq, seq]

        for h in range(model.cfg.n_heads):
            p = pattern[h]  # [seq, seq]
            head_id = (l, h)

            # Previous token score: avg attention to position i-1
            prev_score = 0.0
            if seq_len > 1:
                prev_vals = [p[i, i - 1] for i in range(1, seq_len)]
                prev_score = float(np.mean(prev_vals))

            # Diagonal (self-attention) score
            diag_score = float(np.mean(np.diag(p)[:seq_len]))

            # First token score
            first_score = float(np.mean(p[:, 0]))

            # Uniform score (high entropy = uniform)
            entropy = -np.sum(p * np.log(p + 1e-10), axis=-1)
            max_entropy = np.log(np.arange(1, seq_len + 1) + 1e-10)
            uniform_score = float(np.mean(entropy / (max_entropy + 1e-10)))

            # Classify by highest score
            head_scores = {
                "previous_token": prev_score,
                "diagonal": diag_score,
                "first_token": first_score,
                "uniform": uniform_score,
            }
            category = max(head_scores, key=head_scores.get)
            classifications[f"L{l}H{h}"] = {
                "category": category,
                "scores": head_scores,
            }

            for cat in scores:
                scores[cat].append((head_id, head_scores[cat]))

    # Top heads per category
    top_per_cat = {}
    for cat in scores:
        sorted_heads = sorted(scores[cat], key=lambda x: -x[1])
        top_per_cat[cat] = [(f"L{l}H{h}", float(s)) for (l, h), s in sorted_heads[:top_k]]

    # Category counts
    cat_counts = {}
    for info in classifications.values():
        cat = info["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    return {
        "classifications": classifications,
        "category_counts": cat_counts,
        "top_heads_per_category": top_per_cat,
        "n_heads_total": model.cfg.n_layers * model.cfg.n_heads,
    }


def mlp_utilization(
    model,
    tokens,
) -> dict:
    """Analyze MLP neuron utilization across layers.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.

    Returns:
        Dict with per-layer dead_fraction, mean_activation, top_neurons,
        activation_sparsity.
    """
    from irtk.hook_points import HookState

    cache = {}
    hook_state = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hook_state)

    per_layer = []
    for l in range(model.cfg.n_layers):
        mlp_key = f"blocks.{l}.mlp.hook_post"
        if mlp_key not in cache:
            per_layer.append({
                "layer": l,
                "dead_fraction": 0.0,
                "mean_activation": 0.0,
                "sparsity": 0.0,
            })
            continue

        act = np.array(cache[mlp_key])  # [seq_len, d_mlp]
        abs_act = np.abs(act)

        # Dead neurons: near-zero across all positions
        max_per_neuron = np.max(abs_act, axis=0)
        dead_fraction = float(np.mean(max_per_neuron < 0.01))

        # Mean activation
        mean_act = float(np.mean(abs_act))

        # Sparsity (fraction near zero)
        sparsity = float(np.mean(abs_act < 0.01))

        # Top active neurons
        mean_per_neuron = np.mean(abs_act, axis=0)
        top_idx = np.argsort(mean_per_neuron)[::-1][:5]
        top_neurons = [(int(idx), float(mean_per_neuron[idx])) for idx in top_idx]

        per_layer.append({
            "layer": l,
            "dead_fraction": dead_fraction,
            "mean_activation": mean_act,
            "sparsity": sparsity,
            "top_neurons": top_neurons,
        })

    # Summary
    dead_fracs = [s["dead_fraction"] for s in per_layer]
    sparsities = [s["sparsity"] for s in per_layer]

    return {
        "per_layer": per_layer,
        "overall_dead_fraction": float(np.mean(dead_fracs)),
        "overall_sparsity": float(np.mean(sparsities)),
        "n_layers": model.cfg.n_layers,
    }


def residual_stream_health(
    model,
    tokens,
) -> dict:
    """Check residual stream health across layers.

    Monitors for issues like norm explosion, rank collapse, or
    excessive component dominance.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.

    Returns:
        Dict with norm_trajectory, rank_trajectory, component_balance,
        health_warnings.
    """
    from irtk.hook_points import HookState

    cache = {}
    hook_state = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hook_state)

    norm_trajectory = []
    rank_trajectory = []
    attn_norms = []
    mlp_norms = []
    warnings = []

    for l in range(model.cfg.n_layers):
        # Residual norms
        resid_key = f"blocks.{l}.hook_resid_post"
        if resid_key in cache:
            resid = np.array(cache[resid_key])
            norm = float(np.mean(np.linalg.norm(resid, axis=-1)))
            norm_trajectory.append(norm)

            # Effective rank
            svs = np.linalg.svd(resid, compute_uv=False)
            svs_norm = svs / (np.sum(svs) + 1e-10)
            eff_rank = float(np.exp(-np.sum(svs_norm * np.log(svs_norm + 1e-10))))
            rank_trajectory.append(eff_rank)

        # Component norms
        attn_key = f"blocks.{l}.hook_attn_out"
        if attn_key in cache:
            attn_out = np.array(cache[attn_key])
            attn_norms.append(float(np.mean(np.linalg.norm(attn_out, axis=-1))))

        mlp_key = f"blocks.{l}.hook_mlp_out"
        if mlp_key in cache:
            mlp_out = np.array(cache[mlp_key])
            mlp_norms.append(float(np.mean(np.linalg.norm(mlp_out, axis=-1))))

    # Health checks
    if len(norm_trajectory) > 1:
        growth = norm_trajectory[-1] / max(norm_trajectory[0], 1e-10)
        if growth > 10:
            warnings.append(f"Norm explosion: {growth:.1f}x growth")
        if growth < 0.1:
            warnings.append(f"Norm collapse: {growth:.3f}x")

    if len(rank_trajectory) > 1:
        rank_drop = rank_trajectory[-1] / max(rank_trajectory[0], 1e-10)
        if rank_drop < 0.5:
            warnings.append(f"Rank collapse: {rank_drop:.2f}x")

    # Component balance
    balance = {}
    if attn_norms and mlp_norms:
        attn_total = sum(attn_norms)
        mlp_total = sum(mlp_norms)
        total = attn_total + mlp_total + 1e-10
        balance["attn_fraction"] = float(attn_total / total)
        balance["mlp_fraction"] = float(mlp_total / total)

    return {
        "norm_trajectory": jnp.array(norm_trajectory) if norm_trajectory else jnp.array([]),
        "rank_trajectory": jnp.array(rank_trajectory) if rank_trajectory else jnp.array([]),
        "attn_norms": jnp.array(attn_norms) if attn_norms else jnp.array([]),
        "mlp_norms": jnp.array(mlp_norms) if mlp_norms else jnp.array([]),
        "component_balance": balance,
        "health_warnings": warnings,
    }


def bottleneck_detection(
    model,
    tokens,
    metric_fn: Optional[Callable] = None,
) -> dict:
    """Detect information bottlenecks in the model.

    Identifies layers where the residual stream loses effective rank
    or where ablation has disproportionate impact.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        metric_fn: Optional fn(logits, tokens) -> scalar. If provided,
            measures layer-by-layer ablation impact.

    Returns:
        Dict with bottleneck_layers, rank_profile, ablation_profile.
    """
    from irtk.hook_points import HookState

    cache = {}
    hook_state = HookState(hook_fns={}, cache=cache)
    logits = model(tokens, hook_state=hook_state)

    # Rank profile
    rank_profile = []
    for l in range(model.cfg.n_layers):
        resid_key = f"blocks.{l}.hook_resid_post"
        if resid_key in cache:
            resid = np.array(cache[resid_key])
            svs = np.linalg.svd(resid, compute_uv=False)
            svs_norm = svs / (np.sum(svs) + 1e-10)
            eff_rank = float(np.exp(-np.sum(svs_norm * np.log(svs_norm + 1e-10))))
            rank_profile.append(eff_rank)
        else:
            rank_profile.append(0.0)

    # Find bottleneck layers (local minima in rank)
    bottleneck_layers = []
    for i in range(1, len(rank_profile) - 1):
        if rank_profile[i] < rank_profile[i - 1] and rank_profile[i] < rank_profile[i + 1]:
            bottleneck_layers.append(i)
    # Also check if last layer is minimum
    if len(rank_profile) > 1 and rank_profile[-1] < rank_profile[-2]:
        bottleneck_layers.append(len(rank_profile) - 1)

    # Ablation profile (if metric provided)
    ablation_profile = []
    if metric_fn is not None:
        base_score = float(metric_fn(logits, tokens))

        for l in range(model.cfg.n_layers):
            # Zero out layer's attention output
            hook_name = f"blocks.{l}.hook_attn_out"

            def zero_hook(x, name):
                return jnp.zeros_like(x)

            hs = HookState(hook_fns={hook_name: zero_hook}, cache=None)
            abl_logits = model(tokens, hook_state=hs)
            abl_score = float(metric_fn(abl_logits, tokens))
            ablation_profile.append(base_score - abl_score)

    return {
        "bottleneck_layers": bottleneck_layers,
        "rank_profile": jnp.array(rank_profile),
        "ablation_profile": jnp.array(ablation_profile) if ablation_profile else jnp.array([]),
        "min_rank_layer": int(np.argmin(rank_profile)) if rank_profile else 0,
        "n_layers": model.cfg.n_layers,
    }
