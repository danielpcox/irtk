"""Weight matrix probing for mechanistic interpretability.

Probe weight matrices for structure without running the model:
spectral signatures, role specialization, weight-activation alignment,
pruning sensitivity estimation, and weight geometry analysis.

References:
- Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"
- Sharma et al. (2023) "The Truth is in There: Improving Reasoning in Language
  Models with Layer-Selective Rank Reduction"
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional


def spectral_signatures(
    model,
    layer: int,
    component: str = "attn",
    top_k: int = 10,
) -> dict:
    """Analyze spectral properties of weight matrices.

    Computes SVD of key weight matrices and extracts signatures
    that indicate functional specialization.

    Args:
        model: HookedTransformer model.
        layer: Layer index.
        component: "attn" for attention weights, "mlp" for MLP weights.
        top_k: Number of top singular values to return.

    Returns:
        Dict with singular_values, effective_rank, spectral_entropy,
        condition_number per weight matrix.
    """
    results = {}

    if component == "attn":
        block = model.blocks[layer].attn
        matrices = {
            "W_Q": np.array(block.W_Q).reshape(-1, block.W_Q.shape[-1]),
            "W_K": np.array(block.W_K).reshape(-1, block.W_K.shape[-1]),
            "W_V": np.array(block.W_V).reshape(-1, block.W_V.shape[-1]),
            "W_O": np.array(block.W_O).reshape(-1, block.W_O.shape[-2]),
        }
    else:
        block = model.blocks[layer].mlp
        matrices = {
            "W_in": np.array(block.W_in),
            "W_out": np.array(block.W_out),
        }

    for name, W in matrices.items():
        svs = np.linalg.svd(W, compute_uv=False)
        svs_pos = svs[svs > 1e-10]

        # Effective rank (exponential of entropy of normalized SVs)
        svs_norm = svs_pos / np.sum(svs_pos)
        entropy = -np.sum(svs_norm * np.log(svs_norm + 1e-10))
        eff_rank = float(np.exp(entropy))

        # Condition number
        cond = float(svs[0] / max(svs[-1], 1e-10))

        results[name] = {
            "singular_values": jnp.array(svs[:top_k]),
            "effective_rank": eff_rank,
            "spectral_entropy": float(entropy),
            "condition_number": cond,
            "total_rank": len(svs_pos),
            "frobenius_norm": float(np.sqrt(np.sum(svs ** 2))),
        }

    return results


def role_specialization(
    model,
    layer: int,
    head: Optional[int] = None,
) -> dict:
    """Measure how specialized each head's weights are.

    Compares weight subspaces across heads to determine if they
    serve distinct roles.

    Args:
        model: HookedTransformer model.
        layer: Layer index.
        head: Specific head (if None, analyzes all heads).

    Returns:
        Dict with specialization_scores, weight_similarity_matrix,
        most_unique_head, most_similar_pair.
    """
    block = model.blocks[layer].attn
    n_heads = model.cfg.n_heads

    if head is not None:
        # Single head analysis
        W_Q = np.array(block.W_Q[head])  # [d_model, d_head]
        W_K = np.array(block.W_K[head])
        W_V = np.array(block.W_V[head])
        W_O = np.array(block.W_O[head])  # [d_head, d_model]

        # QK and OV circuit norms
        qk = W_Q @ W_K.T
        ov = W_V @ W_O
        return {
            "qk_norm": float(np.linalg.norm(qk)),
            "ov_norm": float(np.linalg.norm(ov)),
            "q_rank": float(np.linalg.matrix_rank(W_Q)),
            "v_rank": float(np.linalg.matrix_rank(W_V)),
        }

    # Cross-head comparison
    similarity = np.zeros((n_heads, n_heads))
    head_vecs = []
    for h in range(n_heads):
        W_Q = np.array(block.W_Q[h]).flatten()
        W_K = np.array(block.W_K[h]).flatten()
        W_V = np.array(block.W_V[h]).flatten()
        W_O = np.array(block.W_O[h]).flatten()
        vec = np.concatenate([W_Q, W_K, W_V, W_O])
        head_vecs.append(vec / (np.linalg.norm(vec) + 1e-10))

    for i in range(n_heads):
        for j in range(n_heads):
            similarity[i, j] = float(np.dot(head_vecs[i], head_vecs[j]))

    # Specialization = 1 - mean similarity to others
    specialization = np.zeros(n_heads)
    for h in range(n_heads):
        others = [similarity[h, j] for j in range(n_heads) if j != h]
        specialization[h] = 1.0 - np.mean(others) if others else 1.0

    # Most similar pair
    min_spec = (0, 1, float(similarity[0, 1]))
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            if similarity[i, j] > min_spec[2]:
                min_spec = (i, j, float(similarity[i, j]))

    return {
        "specialization_scores": jnp.array(specialization),
        "similarity_matrix": jnp.array(similarity),
        "most_unique_head": int(np.argmax(specialization)),
        "most_similar_pair": (min_spec[0], min_spec[1]),
    }


def weight_activation_alignment(
    model,
    tokens,
    layer: int,
    head: int = 0,
) -> dict:
    """Measure alignment between weight structure and activation patterns.

    Checks how well the weight matrix SVD directions match actual
    activation directions during inference.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        layer: Layer index.
        head: Head index.

    Returns:
        Dict with q_alignment, k_alignment, v_alignment, ov_alignment.
    """
    from irtk.hook_points import HookState

    cache = {}
    hook_state = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hook_state)

    block = model.blocks[layer].attn

    def compute_alignment(W, activations):
        """Compute alignment between weight SVD and actual activations."""
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        # Project activations onto top weight directions
        # U columns span the row space, Vt rows span the column space
        # activations live in the same space as the input to W
        # For W: [m, n], input is [*, m], so project onto U (left SVs)
        act_normed = activations / (np.linalg.norm(activations, axis=-1, keepdims=True) + 1e-10)
        d_act = activations.shape[-1]
        if U.shape[0] == d_act:
            projections = act_normed @ U  # [seq, k]
        elif Vt.shape[1] == d_act:
            projections = act_normed @ Vt.T  # [seq, k]
        else:
            return 0.0
        variance_explained = np.mean(projections ** 2, axis=0)
        return float(np.sum(variance_explained[:min(3, len(variance_explained))]))

    results = {}

    # Q alignment
    q_key = f"blocks.{layer}.attn.hook_q"
    if q_key in cache:
        q_act = np.array(cache[q_key])[:, head, :]  # [seq, d_head]
        W_Q = np.array(block.W_Q[head])  # [d_model, d_head]
        results["q_alignment"] = compute_alignment(W_Q, q_act)

    # K alignment
    k_key = f"blocks.{layer}.attn.hook_k"
    if k_key in cache:
        k_act = np.array(cache[k_key])[:, head, :]
        W_K = np.array(block.W_K[head])
        results["k_alignment"] = compute_alignment(W_K, k_act)

    # V alignment
    v_key = f"blocks.{layer}.attn.hook_v"
    if v_key in cache:
        v_act = np.array(cache[v_key])[:, head, :]
        W_V = np.array(block.W_V[head])
        results["v_alignment"] = compute_alignment(W_V, v_act)

    # OV alignment (using z → result)
    z_key = f"blocks.{layer}.attn.hook_z"
    if z_key in cache:
        z_act = np.array(cache[z_key])[:, head, :]  # [seq, d_head]
        W_O = np.array(block.W_O[head])  # [d_head, d_model]
        results["ov_alignment"] = compute_alignment(W_O, z_act)

    return results


def pruning_sensitivity(
    model,
    layer: int,
    component: str = "attn",
) -> dict:
    """Estimate pruning sensitivity from weight structure alone.

    Uses weight magnitude and spectral properties to predict which
    components are most important without running inference.

    Args:
        model: HookedTransformer model.
        layer: Layer index.
        component: "attn" or "mlp".

    Returns:
        Dict with importance_scores, pruning_order, cumulative_norm_loss.
    """
    if component == "attn":
        block = model.blocks[layer].attn
        n_heads = model.cfg.n_heads
        scores = []
        for h in range(n_heads):
            W_Q = np.array(block.W_Q[h])
            W_K = np.array(block.W_K[h])
            W_V = np.array(block.W_V[h])
            W_O = np.array(block.W_O[h])

            # Importance = Frobenius norm of OV circuit
            ov = W_V @ W_O
            ov_norm = float(np.linalg.norm(ov))

            # Also consider QK circuit
            qk = W_Q @ W_K.T
            qk_norm = float(np.linalg.norm(qk))

            scores.append(ov_norm + qk_norm)

        scores = np.array(scores)
        pruning_order = np.argsort(scores).tolist()  # Least important first

        # Cumulative norm loss
        total_norm = float(np.sum(scores))
        cumulative = []
        removed = 0
        for h in pruning_order:
            removed += scores[h]
            cumulative.append(float(removed / max(total_norm, 1e-10)))

        return {
            "importance_scores": jnp.array(scores),
            "pruning_order": pruning_order,
            "cumulative_norm_loss": jnp.array(cumulative),
            "n_components": n_heads,
        }
    else:
        block = model.blocks[layer].mlp
        W_in = np.array(block.W_in)  # [d_model, d_mlp]
        W_out = np.array(block.W_out)  # [d_mlp, d_model]

        # Per-neuron importance
        in_norms = np.linalg.norm(W_in, axis=0)  # [d_mlp]
        out_norms = np.linalg.norm(W_out, axis=1)  # [d_mlp]
        scores = in_norms * out_norms

        pruning_order = np.argsort(scores).tolist()
        total = float(np.sum(scores))
        cumulative = []
        removed = 0
        for n in pruning_order:
            removed += scores[n]
            cumulative.append(float(removed / max(total, 1e-10)))

        return {
            "importance_scores": jnp.array(scores),
            "pruning_order": pruning_order[:20],  # Top 20 for compactness
            "cumulative_norm_loss": jnp.array(cumulative),
            "n_components": len(scores),
        }


def weight_geometry(
    model,
    layers: Optional[list] = None,
) -> dict:
    """Analyze geometric properties of weight matrices across layers.

    Measures how weight geometry changes through the network: norms,
    angles, isotropy, and inter-layer similarity.

    Args:
        model: HookedTransformer model.
        layers: Layers to analyze (default: all).

    Returns:
        Dict with norm_profile, isotropy_profile, inter_layer_similarity.
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    norm_profile = []
    isotropy_profile = []
    weight_vecs = []

    for l in layers:
        block = model.blocks[l]

        # Collect all attention weight norms
        W_Q = np.array(block.attn.W_Q)  # [n_heads, d_model, d_head]
        W_K = np.array(block.attn.W_K)
        W_V = np.array(block.attn.W_V)
        W_O = np.array(block.attn.W_O)

        q_norm = float(np.linalg.norm(W_Q))
        k_norm = float(np.linalg.norm(W_K))
        v_norm = float(np.linalg.norm(W_V))
        o_norm = float(np.linalg.norm(W_O))

        W_in = np.array(block.mlp.W_in)
        W_out = np.array(block.mlp.W_out)
        in_norm = float(np.linalg.norm(W_in))
        out_norm = float(np.linalg.norm(W_out))

        norm_profile.append({
            "layer": l,
            "q_norm": q_norm,
            "k_norm": k_norm,
            "v_norm": v_norm,
            "o_norm": o_norm,
            "mlp_in_norm": in_norm,
            "mlp_out_norm": out_norm,
            "total": q_norm + k_norm + v_norm + o_norm + in_norm + out_norm,
        })

        # Isotropy (how uniformly spread singular values are)
        all_weights = np.concatenate([
            W_Q.reshape(-1), W_K.reshape(-1), W_V.reshape(-1), W_O.reshape(-1),
        ])
        weight_vecs.append(all_weights / (np.linalg.norm(all_weights) + 1e-10))

        # Compute isotropy from W_V as representative
        W_V_flat = W_V.reshape(-1, W_V.shape[-1])
        svs = np.linalg.svd(W_V_flat, compute_uv=False)
        svs_norm = svs / (np.sum(svs) + 1e-10)
        isotropy = float(np.exp(-np.sum(svs_norm * np.log(svs_norm + 1e-10))) / len(svs))
        isotropy_profile.append(isotropy)

    # Inter-layer similarity
    n = len(layers)
    similarity = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            min_len = min(len(weight_vecs[i]), len(weight_vecs[j]))
            similarity[i, j] = float(np.dot(weight_vecs[i][:min_len], weight_vecs[j][:min_len]))

    return {
        "norm_profile": norm_profile,
        "isotropy_profile": jnp.array(isotropy_profile),
        "inter_layer_similarity": jnp.array(similarity),
        "n_layers": len(layers),
    }
