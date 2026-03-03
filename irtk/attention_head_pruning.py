"""Attention head pruning: impact of removing individual attention heads."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def head_ablation_impact(model: HookedTransformer, tokens: jnp.ndarray,
                          layer: int = 0, position: int = -1) -> dict:
    """Measure logit impact of ablating each head in a layer.

    Removes each head's contribution and measures the change in logits.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

    final_resid = cache[("resid_post", model.cfg.n_layers - 1)]
    full_logits = final_resid[position] @ W_U + b_U
    full_pred = int(jnp.argmax(full_logits))

    z = cache[("z", layer)]  # [seq, n_heads, d_head]

    per_head = []
    for head in range(model.cfg.n_heads):
        head_out = z[position, head, :] @ W_O[head]  # [d_model]
        ablated_resid = final_resid[position] - head_out
        ablated_logits = ablated_resid @ W_U + b_U
        ablated_pred = int(jnp.argmax(ablated_logits))
        mse = float(jnp.mean((ablated_logits - full_logits) ** 2))
        target_change = float(ablated_logits[full_pred] - full_logits[full_pred])
        per_head.append({
            "head": int(head),
            "prediction_changes": ablated_pred != full_pred,
            "ablated_prediction": ablated_pred,
            "mse_change": mse,
            "target_logit_change": target_change,
        })
    per_head.sort(key=lambda h: h["mse_change"], reverse=True)
    return {
        "layer": layer,
        "full_prediction": full_pred,
        "per_head": per_head,
        "most_impactful_head": per_head[0]["head"],
    }


def head_importance_ranking(model: HookedTransformer, tokens: jnp.ndarray,
                             position: int = -1) -> dict:
    """Rank all heads across all layers by ablation impact."""
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    final_resid = cache[("resid_post", model.cfg.n_layers - 1)]
    full_logits = final_resid[position] @ W_U + b_U
    full_pred = int(jnp.argmax(full_logits))

    all_heads = []
    for layer in range(model.cfg.n_layers):
        z = cache[("z", layer)]
        W_O = model.blocks[layer].attn.W_O
        for head in range(model.cfg.n_heads):
            head_out = z[position, head, :] @ W_O[head]
            ablated_resid = final_resid[position] - head_out
            ablated_logits = ablated_resid @ W_U + b_U
            mse = float(jnp.mean((ablated_logits - full_logits) ** 2))
            all_heads.append({
                "layer": layer,
                "head": int(head),
                "importance": mse,
                "output_norm": float(jnp.sqrt(jnp.sum(head_out ** 2))),
            })
    all_heads.sort(key=lambda h: h["importance"], reverse=True)
    return {
        "ranking": all_heads,
        "most_important": f"L{all_heads[0]['layer']}H{all_heads[0]['head']}",
        "least_important": f"L{all_heads[-1]['layer']}H{all_heads[-1]['head']}",
    }


def head_pruning_tolerance(model: HookedTransformer, tokens: jnp.ndarray,
                            position: int = -1) -> dict:
    """How many heads can be pruned before the prediction changes?

    Greedily prunes from least impactful to most impactful.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U
    final_resid = cache[("resid_post", model.cfg.n_layers - 1)]
    full_logits = final_resid[position] @ W_U + b_U
    full_pred = int(jnp.argmax(full_logits))

    # Collect all head outputs
    head_outputs = []
    for layer in range(model.cfg.n_layers):
        z = cache[("z", layer)]
        W_O = model.blocks[layer].attn.W_O
        for head in range(model.cfg.n_heads):
            out = z[position, head, :] @ W_O[head]
            norm = float(jnp.sqrt(jnp.sum(out ** 2)))
            head_outputs.append({
                "layer": layer, "head": int(head),
                "output": out, "norm": norm,
            })
    # Sort by norm (prune smallest first)
    head_outputs.sort(key=lambda h: h["norm"])

    pruned = 0
    current_resid = final_resid[position]
    for h in head_outputs:
        test_resid = current_resid - h["output"]
        test_logits = test_resid @ W_U + b_U
        test_pred = int(jnp.argmax(test_logits))
        if test_pred != full_pred:
            break
        current_resid = test_resid
        pruned += 1

    total = model.cfg.n_layers * model.cfg.n_heads
    return {
        "full_prediction": full_pred,
        "heads_prunable": pruned,
        "total_heads": total,
        "pruning_tolerance": pruned / total,
    }


def head_output_norm_distribution(model: HookedTransformer, tokens: jnp.ndarray,
                                   position: int = -1) -> dict:
    """Distribution of head output norms to identify dead or dominant heads."""
    _, cache = model.run_with_cache(tokens)
    all_norms = []
    for layer in range(model.cfg.n_layers):
        z = cache[("z", layer)]
        W_O = model.blocks[layer].attn.W_O
        for head in range(model.cfg.n_heads):
            out = z[position, head, :] @ W_O[head]
            norm = float(jnp.sqrt(jnp.sum(out ** 2)))
            all_norms.append({
                "layer": layer, "head": int(head), "output_norm": norm,
            })
    norms = [h["output_norm"] for h in all_norms]
    mean_norm = sum(norms) / len(norms)
    threshold = mean_norm * 0.1
    n_near_dead = sum(1 for n in norms if n < threshold)
    return {
        "per_head": all_norms,
        "mean_norm": mean_norm,
        "max_norm": max(norms),
        "min_norm": min(norms),
        "n_near_dead": n_near_dead,
    }


def head_pruning_summary(model: HookedTransformer, tokens: jnp.ndarray,
                          position: int = -1) -> dict:
    """Combined head pruning analysis."""
    ranking = head_importance_ranking(model, tokens, position)
    tolerance = head_pruning_tolerance(model, tokens, position)
    norms = head_output_norm_distribution(model, tokens, position)
    return {
        "most_important": ranking["most_important"],
        "least_important": ranking["least_important"],
        "heads_prunable": tolerance["heads_prunable"],
        "total_heads": tolerance["total_heads"],
        "pruning_tolerance": tolerance["pruning_tolerance"],
        "n_near_dead": norms["n_near_dead"],
        "mean_output_norm": norms["mean_norm"],
    }
