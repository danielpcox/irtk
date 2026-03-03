"""MLP information routing: how MLP layers route information between features.

Analyze input-output mapping, feature amplification, suppression,
and cross-feature routing patterns in MLP layers.
"""

import jax
import jax.numpy as jnp


def mlp_input_output_mapping(model, tokens, layer=0, position=-1, top_k=5):
    """Map how MLP transforms input features to output features.

    Identifies which input directions get amplified or suppressed.
    """
    _, cache = model.run_with_cache(tokens)
    if position < 0:
        position = len(tokens) + position

    pre_key = f"blocks.{layer}.mlp.hook_pre"
    post_key = f"blocks.{layer}.mlp.hook_post"
    out_key = f"blocks.{layer}.hook_mlp_out"

    if out_key not in cache:
        return {"input_norm": 0.0, "output_norm": 0.0, "amplification": 0.0, "layer": layer}

    # Get input to MLP (from residual)
    resid_key = f"blocks.{layer}.hook_resid_mid"
    if resid_key not in cache:
        resid_key = f"blocks.{layer}.hook_resid_pre"

    if resid_key in cache:
        mlp_input = cache[resid_key][position]
    else:
        mlp_input = jnp.zeros(model.cfg.d_model)

    mlp_output = cache[out_key][position]  # [d_model]

    input_norm = float(jnp.linalg.norm(mlp_input))
    output_norm = float(jnp.linalg.norm(mlp_output))

    # Cosine between input and output
    cos = float(jnp.dot(mlp_input, mlp_output) /
                (input_norm * output_norm + 1e-10))

    return {
        "input_norm": input_norm,
        "output_norm": output_norm,
        "amplification": output_norm / (input_norm + 1e-10),
        "input_output_cosine": cos,
        "layer": layer,
        "position": position,
    }


def mlp_feature_amplification(model, tokens, layer=0, top_k=10):
    """Identify which neurons amplify or suppress their input most strongly.

    Compares pre-activation to post-activation magnitudes.
    """
    _, cache = model.run_with_cache(tokens)

    pre_key = f"blocks.{layer}.mlp.hook_pre"
    post_key = f"blocks.{layer}.mlp.hook_post"

    if pre_key not in cache or post_key not in cache:
        return {"per_neuron": [], "layer": layer}

    pre = cache[pre_key]  # [seq, d_mlp]
    post = cache[post_key]  # [seq, d_mlp]
    d_mlp = pre.shape[-1]

    per_neuron = []
    for n in range(min(top_k, d_mlp)):
        pre_norm = float(jnp.linalg.norm(pre[:, n]))
        post_norm = float(jnp.linalg.norm(post[:, n]))
        ratio = post_norm / (pre_norm + 1e-10)

        per_neuron.append({
            "neuron": n,
            "pre_norm": pre_norm,
            "post_norm": post_norm,
            "amplification_ratio": ratio,
            "is_amplifying": ratio > 1.0,
        })

    per_neuron.sort(key=lambda x: x["amplification_ratio"], reverse=True)

    return {
        "per_neuron": per_neuron,
        "layer": layer,
        "mean_amplification": sum(p["amplification_ratio"] for p in per_neuron) / max(len(per_neuron), 1),
    }


def mlp_routing_direction_analysis(model, tokens, layer=0, top_k=5):
    """Analyze the directions MLP writes into the residual stream.

    Decomposes MLP output into principal directions via SVD.
    """
    _, cache = model.run_with_cache(tokens)

    out_key = f"blocks.{layer}.hook_mlp_out"
    if out_key not in cache:
        return {"directions": [], "layer": layer}

    mlp_out = cache[out_key]  # [seq, d_model]

    # Center and SVD
    centered = mlp_out - jnp.mean(mlp_out, axis=0, keepdims=True)
    _, svals, Vt = jnp.linalg.svd(centered, full_matrices=False)

    total_var = float(jnp.sum(svals ** 2)) + 1e-10

    directions = []
    for i in range(min(top_k, len(svals))):
        var_explained = float(svals[i] ** 2 / total_var)
        directions.append({
            "rank": i,
            "singular_value": float(svals[i]),
            "variance_explained": var_explained,
        })

    return {
        "directions": directions,
        "layer": layer,
        "effective_rank": float(jnp.sum(svals ** 2) ** 2 / (jnp.sum(svals ** 4) + 1e-10)),
    }


def mlp_cross_position_routing(model, tokens, layer=0):
    """Analyze whether MLP treats different positions similarly or differently.

    Measures how correlated MLP outputs are across positions.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)

    out_key = f"blocks.{layer}.hook_mlp_out"
    if out_key not in cache:
        return {"mean_similarity": 0.0, "layer": layer}

    mlp_out = cache[out_key]  # [seq, d_model]

    # Normalize
    norms = jnp.linalg.norm(mlp_out, axis=-1, keepdims=True) + 1e-10
    normed = mlp_out / norms
    sim = normed @ normed.T  # [seq, seq]

    off_diag_mask = 1.0 - jnp.eye(seq_len)
    mean_sim = float(jnp.sum(sim * off_diag_mask) / (jnp.sum(off_diag_mask) + 1e-10))

    per_position = []
    for pos in range(seq_len):
        pos_sim = float(jnp.sum(sim[pos] * (1.0 - jnp.eye(seq_len)[pos])) / max(seq_len - 1, 1))
        per_position.append({
            "position": pos,
            "mean_similarity_to_others": pos_sim,
            "output_norm": float(norms[pos, 0]),
        })

    return {
        "mean_similarity": mean_sim,
        "per_position": per_position,
        "is_position_specific": mean_sim < 0.5,
        "layer": layer,
    }


def mlp_routing_summary(model, tokens):
    """Cross-layer summary of MLP information routing.

    Combines amplification, direction, and position routing metrics.
    """
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        io = mlp_input_output_mapping(model, tokens, layer=layer)
        cross = mlp_cross_position_routing(model, tokens, layer=layer)
        dirs = mlp_routing_direction_analysis(model, tokens, layer=layer)

        per_layer.append({
            "layer": layer,
            "amplification": io["amplification"],
            "input_output_cosine": io.get("input_output_cosine", 0.0),
            "position_similarity": cross["mean_similarity"],
            "effective_rank": dirs.get("effective_rank", 0.0),
        })

    return {
        "per_layer": per_layer,
        "n_layers": n_layers,
    }
