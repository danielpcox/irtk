"""MLP specialization analysis: how MLP layers specialize for different functions.

Detect neuron specialization, cross-layer division of labor,
input selectivity, and output targeting patterns.
"""

import jax
import jax.numpy as jnp


def mlp_input_selectivity(model, tokens, layer=0, top_k=10):
    """Measure how selective each MLP neuron is to specific input patterns.

    Selective neurons fire for few input positions; non-selective fire broadly.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)

    pre_key = f"blocks.{layer}.mlp.hook_pre"
    post_key = f"blocks.{layer}.mlp.hook_post"

    if post_key not in cache:
        return {"per_neuron": [], "layer": layer, "mean_selectivity": 0.0}

    activations = cache[post_key]  # [seq, d_mlp]
    d_mlp = activations.shape[-1]

    per_neuron = []
    for n in range(min(top_k, d_mlp)):
        acts = activations[:, n]  # [seq]
        max_act = float(jnp.max(jnp.abs(acts)))
        if max_act < 1e-10:
            selectivity = 0.0
        else:
            # Selectivity: 1 - (entropy / max_entropy)
            abs_acts = jnp.abs(acts) / (jnp.sum(jnp.abs(acts)) + 1e-10)
            entropy = -float(jnp.sum(abs_acts * jnp.log(abs_acts + 1e-10)))
            max_entropy = float(jnp.log(seq_len))
            selectivity = 1.0 - entropy / (max_entropy + 1e-10)

        top_pos = int(jnp.argmax(jnp.abs(acts)))
        per_neuron.append({
            "neuron": n,
            "selectivity": selectivity,
            "max_activation": max_act,
            "top_position": top_pos,
            "top_token_id": int(tokens[top_pos]),
        })

    per_neuron.sort(key=lambda x: x["selectivity"], reverse=True)
    mean_sel = sum(p["selectivity"] for p in per_neuron) / max(len(per_neuron), 1)

    return {
        "per_neuron": per_neuron,
        "layer": layer,
        "mean_selectivity": mean_sel,
        "is_selective": mean_sel > 0.5,
    }


def mlp_output_targeting(model, tokens, layer=0, top_k=10):
    """Analyze which vocabulary tokens each MLP neuron promotes.

    Projects neuron output directions through the unembedding matrix.
    """
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    W_U = model.unembed.W_U  # [d_model, vocab]
    b_U = model.unembed.b_U

    d_mlp = W_out.shape[0]
    per_neuron = []
    for n in range(min(top_k, d_mlp)):
        out_dir = W_out[n]  # [d_model]
        logits = out_dir @ W_U + b_U  # [vocab]
        top_idx = int(jnp.argmax(logits))
        bot_idx = int(jnp.argmin(logits))
        logit_range = float(jnp.max(logits) - jnp.min(logits))

        per_neuron.append({
            "neuron": n,
            "top_promoted_token": top_idx,
            "top_promoted_logit": float(logits[top_idx]),
            "top_suppressed_token": bot_idx,
            "top_suppressed_logit": float(logits[bot_idx]),
            "logit_range": logit_range,
        })

    per_neuron.sort(key=lambda x: x["logit_range"], reverse=True)

    return {
        "per_neuron": per_neuron,
        "layer": layer,
        "mean_logit_range": sum(p["logit_range"] for p in per_neuron) / max(len(per_neuron), 1),
    }


def mlp_cross_layer_division(model, tokens):
    """Analyze how MLP functionality is divided across layers.

    Compares activation patterns and output magnitudes to detect specialization.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        post_key = f"blocks.{layer}.mlp.hook_post"
        out_key = f"blocks.{layer}.hook_mlp_out"

        sparsity = 0.0
        output_norm = 0.0
        if post_key in cache:
            acts = cache[post_key]  # [seq, d_mlp]
            # Sparsity: fraction of near-zero activations
            sparsity = float(jnp.mean(jnp.abs(acts) < 0.01))
        if out_key in cache:
            output_norm = float(jnp.linalg.norm(cache[out_key]))

        per_layer.append({
            "layer": layer,
            "sparsity": sparsity,
            "output_norm": output_norm,
        })

    # Cross-layer similarity of activation patterns
    similarities = []
    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            key_i = f"blocks.{i}.mlp.hook_post"
            key_j = f"blocks.{j}.mlp.hook_post"
            if key_i in cache and key_j in cache:
                a = cache[key_i].reshape(-1)
                b = cache[key_j].reshape(-1)
                cos = float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-10))
                similarities.append({
                    "layer_i": i, "layer_j": j, "cosine": cos,
                })

    mean_sim = sum(s["cosine"] for s in similarities) / max(len(similarities), 1)

    return {
        "per_layer": per_layer,
        "cross_layer_similarities": similarities,
        "mean_cross_layer_similarity": mean_sim,
        "is_specialized": mean_sim < 0.5,
    }


def mlp_activation_sparsity_profile(model, tokens):
    """Profile activation sparsity patterns across layers and positions.

    Tracks which neurons are active and how activation density changes.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)

    per_layer = []
    for layer in range(n_layers):
        post_key = f"blocks.{layer}.mlp.hook_post"
        if post_key not in cache:
            per_layer.append({
                "layer": layer, "mean_sparsity": 0.0,
                "per_position_sparsity": [], "active_neuron_overlap": 0.0,
            })
            continue

        acts = cache[post_key]  # [seq, d_mlp]
        active = jnp.abs(acts) > 0.01

        # Per-position sparsity
        pos_sparsity = []
        for pos in range(seq_len):
            sp = float(1.0 - jnp.mean(active[pos].astype(jnp.float32)))
            pos_sparsity.append(sp)

        mean_sparsity = float(jnp.mean(1.0 - active.astype(jnp.float32)))

        # Active neuron overlap between positions (Jaccard)
        if seq_len > 1:
            overlaps = []
            for i in range(min(seq_len, 5)):
                for j in range(i + 1, min(seq_len, 5)):
                    intersection = float(jnp.sum(active[i] & active[j]))
                    union = float(jnp.sum(active[i] | active[j]))
                    if union > 0:
                        overlaps.append(intersection / union)
            mean_overlap = sum(overlaps) / max(len(overlaps), 1)
        else:
            mean_overlap = 1.0

        per_layer.append({
            "layer": layer,
            "mean_sparsity": mean_sparsity,
            "per_position_sparsity": pos_sparsity,
            "active_neuron_overlap": mean_overlap,
        })

    return {
        "per_layer": per_layer,
        "overall_sparsity": sum(p["mean_sparsity"] for p in per_layer) / max(len(per_layer), 1),
        "sparsity_trend": "increasing" if per_layer and per_layer[-1]["mean_sparsity"] > per_layer[0]["mean_sparsity"] else "decreasing",
    }


def mlp_specialization_summary(model, tokens):
    """Cross-layer summary of MLP specialization.

    Combines selectivity, targeting, sparsity, and division info.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    per_layer = []
    for layer in range(n_layers):
        post_key = f"blocks.{layer}.mlp.hook_post"
        out_key = f"blocks.{layer}.hook_mlp_out"

        output_norm = 0.0
        sparsity = 0.0
        logit_impact = 0.0

        if post_key in cache:
            acts = cache[post_key]
            sparsity = float(jnp.mean(jnp.abs(acts) < 0.01))

        if out_key in cache:
            mlp_out = cache[out_key]  # [seq, d_model]
            output_norm = float(jnp.linalg.norm(mlp_out))
            # Logit impact: project through unembed
            logits = mlp_out[-1] @ W_U + b_U
            logit_impact = float(jnp.max(logits) - jnp.min(logits))

        per_layer.append({
            "layer": layer,
            "output_norm": output_norm,
            "sparsity": sparsity,
            "logit_impact": logit_impact,
        })

    return {
        "per_layer": per_layer,
        "most_impactful_layer": max(per_layer, key=lambda p: p["logit_impact"])["layer"] if per_layer else 0,
        "n_layers": n_layers,
    }
