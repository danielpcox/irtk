"""MLP decomposition: interpretable analysis of MLP computation.

Decomposes MLP layers into interpretable components — understanding what
individual neurons compute, how the MLP transforms features, and where
factual knowledge is stored.

Functions:
- neuron_contribution_decompose: Decompose MLP output into per-neuron contributions
- mlp_feature_directions: Extract meaningful directions from MLP weight matrices
- mlp_input_output_alignment: Alignment between MLP input/output weight directions
- mlp_knowledge_storage: Locate knowledge storage in MLP weights
- mlp_nonlinearity_analysis: Analyze the activation function's gating behavior

References:
    - Geva et al. (2021) "Transformer FFN Layers Are Key-Value Memories"
    - Dai et al. (2022) "Knowledge Neurons in Pretrained Transformers"
    - Geva et al. (2022) "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts"
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def neuron_contribution_decompose(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
    pos: int = -1,
    top_k: int = 10,
) -> dict:
    """Decompose MLP output into per-neuron contributions.

    Each neuron's contribution = activation_value * W_out_row. Returns
    the top contributing neurons and their effect on the residual stream.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        layer: Layer index.
        pos: Token position to analyze (-1 = last).
        top_k: Number of top neurons to return.

    Returns:
        Dict with:
            "top_neurons": list of (neuron_idx, contribution_norm)
            "neuron_contributions": [d_mlp] contribution norms per neuron
            "total_contribution": total MLP output norm
            "top_k_fraction": fraction of output explained by top_k neurons
    """
    _, cache = model.run_with_cache(tokens)

    post_key = f"blocks.{layer}.mlp.hook_post"
    if post_key not in cache.cache_dict:
        return {
            "top_neurons": [],
            "neuron_contributions": np.array([]),
            "total_contribution": 0.0,
            "top_k_fraction": 0.0,
        }

    activations = np.array(cache.cache_dict[post_key][pos])  # [d_mlp]
    W_out = np.array(model.blocks[layer].mlp.W_out)  # [d_mlp, d_model]

    # Per-neuron contribution
    contributions = activations[:, None] * W_out  # [d_mlp, d_model]
    contrib_norms = np.linalg.norm(contributions, axis=1)  # [d_mlp]

    # Top neurons
    top_idx = np.argsort(contrib_norms)[::-1][:top_k]
    top_neurons = [(int(idx), float(contrib_norms[idx])) for idx in top_idx]

    total = float(np.sum(contrib_norms))
    top_sum = float(np.sum(contrib_norms[top_idx]))
    fraction = top_sum / (total + 1e-10)

    return {
        "top_neurons": top_neurons,
        "neuron_contributions": contrib_norms,
        "total_contribution": total,
        "top_k_fraction": fraction,
    }


def mlp_feature_directions(
    model: HookedTransformer,
    layer: int,
    top_k: int = 10,
) -> dict:
    """Extract meaningful directions from MLP weight matrices.

    Analyzes W_in and W_out to find the principal input and output
    directions of the MLP, treating it as a key-value memory.

    Args:
        model: HookedTransformer.
        layer: Layer index.
        top_k: Number of top directions.

    Returns:
        Dict with:
            "input_directions": [top_k, d_model] top input (key) directions
            "output_directions": [top_k, d_model] top output (value) directions
            "singular_values": [top_k] importance of each direction pair
            "effective_rank": effective rank of the MLP weight matrix
    """
    W_in = np.array(model.blocks[layer].mlp.W_in)   # [d_model, d_mlp]
    W_out = np.array(model.blocks[layer].mlp.W_out)  # [d_mlp, d_model]

    # SVD of the combined MLP matrix W_out @ W_in^T would give key-value pairs,
    # but let's analyze W_in and W_out separately

    # Input directions (keys): what the MLP reads
    U_in, S_in, Vt_in = np.linalg.svd(W_in, full_matrices=False)
    k = min(top_k, len(S_in))

    # Output directions (values): what the MLP writes
    U_out, S_out, Vt_out = np.linalg.svd(W_out.T, full_matrices=False)

    # Effective rank
    s_normalized = S_in / (np.sum(S_in) + 1e-10)
    s_normalized = s_normalized[s_normalized > 1e-10]
    eff_rank = float(np.exp(-np.sum(s_normalized * np.log(s_normalized + 1e-10))))

    # Pad if needed
    input_dirs = U_in[:, :k].T  # [k, d_model]
    output_dirs = U_out[:, :k].T  # [k, d_model]

    if k < top_k:
        d = W_in.shape[0]
        input_dirs = np.concatenate([input_dirs, np.zeros((top_k - k, d))], axis=0)
        output_dirs = np.concatenate([output_dirs, np.zeros((top_k - k, d))], axis=0)

    return {
        "input_directions": input_dirs,
        "output_directions": output_dirs,
        "singular_values": S_in[:top_k] if len(S_in) >= top_k else np.concatenate([S_in, np.zeros(top_k - len(S_in))]),
        "effective_rank": eff_rank,
    }


def mlp_input_output_alignment(
    model: HookedTransformer,
    layer: int,
) -> dict:
    """Measure alignment between MLP input and output weight directions.

    High alignment means the MLP amplifies existing features; low alignment
    means it creates new features not present in the input.

    Args:
        model: HookedTransformer.
        layer: Layer index.

    Returns:
        Dict with:
            "per_neuron_alignment": [d_mlp] cosine similarity between W_in and W_out per neuron
            "mean_alignment": mean alignment across neurons
            "amplifying_neurons": count of neurons with alignment > 0.5
            "transforming_neurons": count of neurons with alignment < 0.1
    """
    W_in = np.array(model.blocks[layer].mlp.W_in)   # [d_model, d_mlp]
    W_out = np.array(model.blocks[layer].mlp.W_out)  # [d_mlp, d_model]

    d_mlp = W_in.shape[1]
    alignments = np.zeros(d_mlp)

    for n in range(d_mlp):
        in_dir = W_in[:, n]  # [d_model] - what this neuron reads
        out_dir = W_out[n, :]  # [d_model] - what this neuron writes

        norm_in = np.linalg.norm(in_dir)
        norm_out = np.linalg.norm(out_dir)

        if norm_in > 1e-10 and norm_out > 1e-10:
            alignments[n] = float(np.dot(in_dir, out_dir) / (norm_in * norm_out))

    return {
        "per_neuron_alignment": alignments,
        "mean_alignment": float(np.mean(np.abs(alignments))),
        "amplifying_neurons": int(np.sum(np.abs(alignments) > 0.5)),
        "transforming_neurons": int(np.sum(np.abs(alignments) < 0.1)),
    }


def mlp_knowledge_storage(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
    target_token: int,
    top_k: int = 10,
) -> dict:
    """Locate which MLP neurons store knowledge about a target token.

    Following Geva et al. (2022), identifies neurons whose output direction
    promotes the target token in the vocabulary distribution.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        layer: Layer index.
        target_token: Token to find knowledge about.
        top_k: Number of top neurons.

    Returns:
        Dict with:
            "knowledge_neurons": list of (neuron_idx, logit_contribution)
            "neuron_logit_effects": [d_mlp] each neuron's effect on target logit
            "total_promotion": total logit promotion from MLP
            "knowledge_concentration": fraction of promotion from top_k neurons
    """
    _, cache = model.run_with_cache(tokens)
    post_key = f"blocks.{layer}.mlp.hook_post"

    if post_key not in cache.cache_dict:
        return {
            "knowledge_neurons": [],
            "neuron_logit_effects": np.array([]),
            "total_promotion": 0.0,
            "knowledge_concentration": 0.0,
        }

    activations = np.array(cache.cache_dict[post_key][-1])  # [d_mlp] at last position
    W_out = np.array(model.blocks[layer].mlp.W_out)  # [d_mlp, d_model]
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]

    # Each neuron's contribution to the target logit
    # neuron_output = activation * W_out_row -> project through W_U
    logit_effects = np.zeros(W_out.shape[0])
    for n in range(W_out.shape[0]):
        neuron_out = activations[n] * W_out[n]  # [d_model]
        logit_effects[n] = float(neuron_out @ W_U[:, target_token])

    # Top neurons
    top_idx = np.argsort(np.abs(logit_effects))[::-1][:top_k]
    knowledge_neurons = [(int(idx), float(logit_effects[idx])) for idx in top_idx]

    total = float(np.sum(logit_effects))
    top_sum = float(np.sum(np.abs(logit_effects[top_idx])))
    total_abs = float(np.sum(np.abs(logit_effects)))
    concentration = top_sum / (total_abs + 1e-10)

    return {
        "knowledge_neurons": knowledge_neurons,
        "neuron_logit_effects": logit_effects,
        "total_promotion": total,
        "knowledge_concentration": concentration,
    }


def mlp_nonlinearity_analysis(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
    pos: int = -1,
) -> dict:
    """Analyze the activation function's gating behavior.

    Examines which neurons are active/inactive and the distribution
    of pre-activation values to understand the MLP's nonlinear behavior.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        layer: Layer index.
        pos: Token position (-1 = last).

    Returns:
        Dict with:
            "active_fraction": fraction of neurons with positive activation
            "pre_activation_stats": {mean, std, min, max} of pre-activations
            "gating_sharpness": std of post/pre ratio (sharp = binary gating)
            "dead_neurons": count of neurons with zero activation
    """
    _, cache = model.run_with_cache(tokens)

    pre_key = f"blocks.{layer}.mlp.hook_pre"
    post_key = f"blocks.{layer}.mlp.hook_post"

    if pre_key not in cache.cache_dict or post_key not in cache.cache_dict:
        return {
            "active_fraction": 0.0,
            "pre_activation_stats": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            "gating_sharpness": 0.0,
            "dead_neurons": 0,
        }

    pre = np.array(cache.cache_dict[pre_key][pos])  # [d_mlp]
    post = np.array(cache.cache_dict[post_key][pos])  # [d_mlp]

    active = post > 1e-8
    active_fraction = float(np.mean(active))

    # Gating sharpness: how binary is the activation?
    ratios = np.abs(post) / (np.abs(pre) + 1e-10)
    gating_sharpness = float(np.std(ratios))

    dead = int(np.sum(np.abs(post) < 1e-8))

    return {
        "active_fraction": active_fraction,
        "pre_activation_stats": {
            "mean": float(np.mean(pre)),
            "std": float(np.std(pre)),
            "min": float(np.min(pre)),
            "max": float(np.max(pre)),
        },
        "gating_sharpness": gating_sharpness,
        "dead_neurons": dead,
    }
