"""Prediction geometry analysis.

Analyzes the geometric structure of how predictions form and sharpen
across layers — the shape of the unembedding manifold, calibration of
intermediate predictions, and the trajectory through vocabulary space.

Functions:
- vocab_projection_trajectory: Top-k predictions at each layer
- prediction_sharpening_rate: Rate of prediction sharpening per layer
- unembedding_alignment_per_head: Which tokens each head promotes/demotes
- token_promotion_geometry: Geometry of unembedding directions for target tokens
- final_layer_residual_decomposition: Per-layer contribution parallel to prediction

References:
    - Dar et al. (2023) "Analyzing Transformers in Embedding Space"
    - Geva et al. (2022) "Transformer Feed-Forward Layers Build Predictions
      by Promoting Concepts in the Vocabulary Space"
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def vocab_projection_trajectory(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Top-k predictions at each layer via unembedding projection.

    For each layer's residual stream at a position, projects through
    the unembedding to record top-k predicted tokens.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        pos: Position to analyze (-1 = last).
        top_k: Number of top tokens to track.

    Returns:
        Dict with:
            "layer_top_tokens": [n_layers] list of top-k token indices per layer
            "layer_top_logits": [n_layers] list of top-k logit values per layer
            "final_enters_top_k_layer": first layer where final prediction enters top-k
            "final_becomes_top_1_layer": first layer where final prediction is top-1
            "final_prediction": the final output prediction
    """
    _, cache = model.run_with_cache(tokens)
    logits = np.array(model(tokens))
    seq_len = len(tokens)
    n_layers = model.cfg.n_layers

    if pos == -1:
        pos = seq_len - 1

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') else np.zeros(W_U.shape[1])

    final_pred = int(np.argmax(logits[pos]))

    layer_tops = []
    layer_logits_list = []
    first_in_topk = -1
    first_top1 = -1

    for l in range(n_layers):
        key = f"blocks.{l}.hook_resid_post"
        if key in cache.cache_dict:
            resid = np.array(cache.cache_dict[key][pos])
            layer_logits = resid @ W_U + b_U
            sorted_idx = np.argsort(layer_logits)[::-1][:top_k]
            layer_tops.append(sorted_idx.tolist())
            layer_logits_list.append([float(layer_logits[i]) for i in sorted_idx])

            if first_in_topk == -1 and final_pred in sorted_idx:
                first_in_topk = l
            if first_top1 == -1 and sorted_idx[0] == final_pred:
                first_top1 = l
        else:
            layer_tops.append([])
            layer_logits_list.append([])

    return {
        "layer_top_tokens": layer_tops,
        "layer_top_logits": layer_logits_list,
        "final_enters_top_k_layer": first_in_topk,
        "final_becomes_top_1_layer": first_top1,
        "final_prediction": final_pred,
    }


def prediction_sharpening_rate(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Rate of prediction sharpening per layer.

    Computes how the top-1 logit increases and entropy decreases at each
    layer. High sharpening at a layer means focused prediction work.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        pos: Position to analyze (-1 = last).

    Returns:
        Dict with:
            "layer_entropies": [n_layers] entropy of prediction at each layer
            "layer_top1_probs": [n_layers] top-1 probability at each layer
            "sharpening_rates": [n_layers-1] entropy decrease per layer transition
            "crystallization_layer": layer with maximum sharpening rate
            "total_sharpening": total entropy reduction from first to last layer
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)
    n_layers = model.cfg.n_layers

    if pos == -1:
        pos = seq_len - 1

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') else np.zeros(W_U.shape[1])

    entropies = np.zeros(n_layers)
    top1_probs = np.zeros(n_layers)

    for l in range(n_layers):
        key = f"blocks.{l}.hook_resid_post"
        if key in cache.cache_dict:
            resid = np.array(cache.cache_dict[key][pos])
            layer_logits = resid @ W_U + b_U
            # Stable softmax
            probs = np.exp(layer_logits - np.max(layer_logits))
            probs = probs / np.sum(probs)
            # Entropy
            log_probs = np.log(probs + 1e-10)
            entropies[l] = -float(np.sum(probs * log_probs))
            top1_probs[l] = float(np.max(probs))

    # Sharpening rates
    if n_layers >= 2:
        rates = -np.diff(entropies)  # positive = entropy decreased = sharpening
        crystal = int(np.argmax(rates))
    else:
        rates = np.array([])
        crystal = 0

    total_sharp = float(entropies[0] - entropies[-1]) if n_layers >= 2 else 0.0

    return {
        "layer_entropies": entropies,
        "layer_top1_probs": top1_probs,
        "sharpening_rates": rates,
        "crystallization_layer": crystal,
        "total_sharpening": total_sharp,
    }


def unembedding_alignment_per_head(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Which tokens each head promotes/demotes most strongly.

    For each attention head at a layer, projects its output through
    the unembedding to find which vocabulary items it votes for.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        layer: Layer to analyze.
        pos: Position (-1 = last).
        top_k: Number of top promoted/demoted tokens per head.

    Returns:
        Dict with:
            "promoted_tokens": [n_heads] list of top-k promoted token indices
            "promoted_logits": [n_heads] list of top-k promotion magnitudes
            "demoted_tokens": [n_heads] list of top-k demoted token indices
            "demoted_logits": [n_heads] list of top-k demotion magnitudes
            "head_logit_norms": [n_heads] L2 norm of each head's logit contribution
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)
    n_heads = model.cfg.n_heads

    if pos == -1:
        pos = seq_len - 1

    W_U = np.array(model.unembed.W_U)

    promoted = []
    promoted_vals = []
    demoted = []
    demoted_vals = []
    norms = np.zeros(n_heads)

    z_key = f"blocks.{layer}.attn.hook_z"
    if z_key not in cache.cache_dict:
        return {
            "promoted_tokens": [[] for _ in range(n_heads)],
            "promoted_logits": [[] for _ in range(n_heads)],
            "demoted_tokens": [[] for _ in range(n_heads)],
            "demoted_logits": [[] for _ in range(n_heads)],
            "head_logit_norms": norms,
        }

    z = np.array(cache.cache_dict[z_key])  # [seq, n_heads, d_head]

    for h in range(n_heads):
        W_O = np.array(model.blocks[layer].attn.W_O[h])  # [d_head, d_model]
        head_out = z[pos, h] @ W_O  # [d_model]
        head_logits = head_out @ W_U  # [d_vocab]

        norms[h] = float(np.linalg.norm(head_logits))

        # Top promoted
        top_idx = np.argsort(head_logits)[::-1][:top_k]
        promoted.append(top_idx.tolist())
        promoted_vals.append([float(head_logits[i]) for i in top_idx])

        # Top demoted
        bot_idx = np.argsort(head_logits)[:top_k]
        demoted.append(bot_idx.tolist())
        demoted_vals.append([float(head_logits[i]) for i in bot_idx])

    return {
        "promoted_tokens": promoted,
        "promoted_logits": promoted_vals,
        "demoted_tokens": demoted,
        "demoted_logits": demoted_vals,
        "head_logit_norms": norms,
    }


def token_promotion_geometry(
    model: HookedTransformer,
    target_tokens: list,
) -> dict:
    """Geometry of unembedding directions for target tokens.

    Analyzes pairwise cosine similarities of W_U columns for a set
    of tokens, revealing the geometric structure of the prediction space.

    Args:
        model: HookedTransformer.
        target_tokens: list of token indices to analyze.

    Returns:
        Dict with:
            "pairwise_similarity": [n, n] cosine similarity matrix
            "mean_pairwise_similarity": mean off-diagonal similarity
            "norms": [n] L2 norm of each unembedding direction
            "mean_cosine_to_centroid": mean cosine to the group centroid
    """
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]
    n = len(target_tokens)

    # Extract unembedding directions
    directions = np.array([W_U[:, t] for t in target_tokens])  # [n, d_model]
    norms = np.linalg.norm(directions, axis=1)  # [n]

    # Pairwise cosine similarity
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if norms[i] > 1e-10 and norms[j] > 1e-10:
                sim[i, j] = float(np.dot(directions[i], directions[j]) / (norms[i] * norms[j]))
            elif i == j:
                sim[i, j] = 1.0

    # Mean off-diagonal
    if n > 1:
        off_diag = sim[~np.eye(n, dtype=bool)]
        mean_sim = float(np.mean(off_diag))
    else:
        mean_sim = 0.0

    # Centroid cosine
    centroid = np.mean(directions, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    cos_to_centroid = []
    for i in range(n):
        if norms[i] > 1e-10 and centroid_norm > 1e-10:
            cos_to_centroid.append(float(np.dot(directions[i], centroid) / (norms[i] * centroid_norm)))
        else:
            cos_to_centroid.append(0.0)

    return {
        "pairwise_similarity": sim,
        "mean_pairwise_similarity": mean_sim,
        "norms": norms,
        "mean_cosine_to_centroid": float(np.mean(cos_to_centroid)),
    }


def final_layer_residual_decomposition(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Per-layer contribution parallel to the predicted token's direction.

    Decomposes the final residual stream into per-layer contributions
    that are parallel vs orthogonal to the top prediction's unembedding.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        pos: Position (-1 = last).
        top_k: Unused, kept for API consistency.

    Returns:
        Dict with:
            "parallel_contributions": [n_layers+1] projection onto prediction direction
                (index 0 = embedding, index l+1 = layer l)
            "orthogonal_norms": [n_layers+1] orthogonal component norms
            "prediction_token": the predicted token
            "total_parallel": sum of parallel contributions
            "dominant_layer": layer contributing most to the prediction
    """
    _, cache = model.run_with_cache(tokens)
    logits = np.array(model(tokens))
    seq_len = len(tokens)
    n_layers = model.cfg.n_layers

    if pos == -1:
        pos = seq_len - 1

    W_U = np.array(model.unembed.W_U)
    pred_token = int(np.argmax(logits[pos]))
    pred_dir = W_U[:, pred_token]
    pred_dir_norm = np.linalg.norm(pred_dir)
    if pred_dir_norm < 1e-10:
        pred_dir_unit = pred_dir
    else:
        pred_dir_unit = pred_dir / pred_dir_norm

    # Embedding contribution
    embed_key = "blocks.0.hook_resid_pre"
    parallel = np.zeros(n_layers + 1)
    orthogonal = np.zeros(n_layers + 1)

    if embed_key in cache.cache_dict:
        embed = np.array(cache.cache_dict[embed_key][pos])
        par = float(np.dot(embed, pred_dir_unit))
        parallel[0] = par
        orthogonal[0] = float(np.linalg.norm(embed - par * pred_dir_unit))

    # Per-layer contributions (layer output - layer input)
    for l in range(n_layers):
        pre_key = f"blocks.{l}.hook_resid_pre" if l == 0 else f"blocks.{l-1}.hook_resid_post"
        post_key = f"blocks.{l}.hook_resid_post"

        if pre_key in cache.cache_dict and post_key in cache.cache_dict:
            pre = np.array(cache.cache_dict[pre_key][pos])
            post = np.array(cache.cache_dict[post_key][pos])
            layer_contrib = post - pre
        elif post_key in cache.cache_dict:
            # Just use the post if we can't get the diff
            layer_contrib = np.array(cache.cache_dict[post_key][pos])
            if l > 0:
                prev_key = f"blocks.{l-1}.hook_resid_post"
                if prev_key in cache.cache_dict:
                    layer_contrib = layer_contrib - np.array(cache.cache_dict[prev_key][pos])
        else:
            layer_contrib = np.zeros(model.cfg.d_model)

        par = float(np.dot(layer_contrib, pred_dir_unit))
        parallel[l + 1] = par
        orthogonal[l + 1] = float(np.linalg.norm(layer_contrib - par * pred_dir_unit))

    dominant = int(np.argmax(np.abs(parallel)))

    return {
        "parallel_contributions": parallel,
        "orthogonal_norms": orthogonal,
        "prediction_token": pred_token,
        "total_parallel": float(np.sum(parallel)),
        "dominant_layer": dominant,
    }
