"""Weight structure profiling: analyze weight matrix properties.

Profile the structure of weight matrices: condition number, spectral properties,
sparsity patterns, and weight alignment between components.
"""

import jax
import jax.numpy as jnp


def weight_spectral_profile(model, layer=0):
    """Spectral analysis of attention and MLP weights.

    Returns:
        dict with per-matrix spectral properties (condition number,
        top singular values, effective rank).
    """
    W_Q = model.blocks[layer].attn.W_Q  # [n_heads, d_model, d_head]
    W_K = model.blocks[layer].attn.W_K
    W_V = model.blocks[layer].attn.W_V
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]
    W_in = model.blocks[layer].mlp.W_in  # [d_model, d_mlp]
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]

    def _spectral(W):
        if W.ndim == 3:
            W = W.reshape(-1, W.shape[-1])
        s = jnp.linalg.svd(W, compute_uv=False)
        cond = float(s[0] / (s[-1] + 1e-10))
        # Effective rank
        s_norm = s / (jnp.sum(s) + 1e-10)
        s_safe = jnp.where(s_norm > 1e-10, s_norm, 1e-10)
        eff_rank = float(jnp.exp(-jnp.sum(s_safe * jnp.log(s_safe))))
        return {
            "condition_number": cond,
            "effective_rank": eff_rank,
            "top_5_singular": [float(s[i]) for i in range(min(5, len(s)))],
            "spectral_norm": float(s[0]),
        }

    return {
        "W_Q": _spectral(W_Q),
        "W_K": _spectral(W_K),
        "W_V": _spectral(W_V),
        "W_O": _spectral(W_O),
        "W_in": _spectral(W_in),
        "W_out": _spectral(W_out),
    }


def weight_sparsity_profile(model, layer=0, threshold=0.01):
    """How sparse are the weight matrices?

    Returns:
        dict with per-matrix sparsity metrics.
    """
    matrices = {
        "W_Q": model.blocks[layer].attn.W_Q,
        "W_K": model.blocks[layer].attn.W_K,
        "W_V": model.blocks[layer].attn.W_V,
        "W_O": model.blocks[layer].attn.W_O,
        "W_in": model.blocks[layer].mlp.W_in,
        "W_out": model.blocks[layer].mlp.W_out,
    }
    result = {}
    for name, W in matrices.items():
        abs_W = jnp.abs(W)
        mean_abs = float(jnp.mean(abs_W))
        near_zero = float(jnp.mean(abs_W < threshold * mean_abs))
        result[name] = {
            "sparsity": near_zero,
            "mean_abs_weight": mean_abs,
            "max_abs_weight": float(jnp.max(abs_W)),
            "std_weight": float(jnp.std(W)),
        }
    return result


def weight_alignment_profile(model, layer=0):
    """How aligned are different weight matrices within a layer?

    Measures cosine similarity between flattened weight vectors.

    Returns:
        dict with pairwise alignment scores.
    """
    matrices = {
        "W_Q": model.blocks[layer].attn.W_Q.reshape(-1),
        "W_K": model.blocks[layer].attn.W_K.reshape(-1),
        "W_V": model.blocks[layer].attn.W_V.reshape(-1),
    }
    pairs = []
    names = list(matrices.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = matrices[names[i]], matrices[names[j]]
            # Truncate to same length
            min_len = min(len(a), len(b))
            cos = float(jnp.dot(a[:min_len], b[:min_len]) / (
                jnp.linalg.norm(a[:min_len]) * jnp.linalg.norm(b[:min_len]) + 1e-10
            ))
            pairs.append({
                "matrix_a": names[i],
                "matrix_b": names[j],
                "cosine_similarity": cos,
            })
    return {"pairwise_alignments": pairs}


def weight_norm_distribution(model):
    """Distribution of weight norms across all layers.

    Returns:
        dict with 'per_layer' norm statistics.
    """
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        W_Q_norm = float(jnp.linalg.norm(model.blocks[layer].attn.W_Q))
        W_K_norm = float(jnp.linalg.norm(model.blocks[layer].attn.W_K))
        W_V_norm = float(jnp.linalg.norm(model.blocks[layer].attn.W_V))
        W_O_norm = float(jnp.linalg.norm(model.blocks[layer].attn.W_O))
        W_in_norm = float(jnp.linalg.norm(model.blocks[layer].mlp.W_in))
        W_out_norm = float(jnp.linalg.norm(model.blocks[layer].mlp.W_out))
        per_layer.append({
            "layer": layer,
            "W_Q_norm": W_Q_norm,
            "W_K_norm": W_K_norm,
            "W_V_norm": W_V_norm,
            "W_O_norm": W_O_norm,
            "W_in_norm": W_in_norm,
            "W_out_norm": W_out_norm,
            "total_attn_norm": W_Q_norm + W_K_norm + W_V_norm + W_O_norm,
            "total_mlp_norm": W_in_norm + W_out_norm,
        })
    return {"per_layer": per_layer}


def weight_structure_summary(model):
    """Summary of weight structure across all layers.

    Returns:
        dict with 'per_layer' summary metrics.
    """
    n_layers = len(model.blocks)
    per_layer = []
    for layer in range(n_layers):
        spec = weight_spectral_profile(model, layer=layer)
        sparse = weight_sparsity_profile(model, layer=layer)
        per_layer.append({
            "layer": layer,
            "W_Q_condition": spec["W_Q"]["condition_number"],
            "W_in_condition": spec["W_in"]["condition_number"],
            "W_Q_sparsity": sparse["W_Q"]["sparsity"],
            "W_in_sparsity": sparse["W_in"]["sparsity"],
        })
    return {"per_layer": per_layer}
