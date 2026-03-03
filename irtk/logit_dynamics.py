"""Logit dynamics: how the output prediction evolves through computation.

Tracks the logit vector's evolution across layers to understand prediction
flips, stability, commitment timing, and per-component contributions.
Goes beyond logit lens (which just projects) to analyze the dynamics
of prediction formation.

Functions:
- logit_flip_analysis: Where does the top-1 prediction change between layers?
- prediction_stability_across_layers: How stable is the top-k set across layers?
- commitment_timing: When does the model "commit" to its final prediction?
- logit_contribution_by_component: Contribution of each attention/MLP to final logits
- top_k_trajectory: Track how top-k tokens change through layers

References:
    - Nostalgebraist (2020) "Interpreting GPT: the Logit Lens"
    - Din et al. (2023) "Jump to Conclusions: Short-Cutting Transformers"
    - Geva et al. (2022) "Transformer Feed-Forward Layers Build Predictions"
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def logit_flip_analysis(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Identify where the top-1 prediction changes between layers.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        pos: Token position to analyze (-1 = last).

    Returns:
        Dict with:
            "layer_predictions": [n_layers] top-1 token at each layer
            "flip_layers": list of layers where the top prediction changes
            "n_flips": total number of prediction flips
            "final_prediction": the final output prediction
            "first_correct_layer": first layer predicting the final token (-1 if never)
    """
    _, cache = model.run_with_cache(tokens)
    logits = model(tokens)
    n_layers = model.cfg.n_layers

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') else np.zeros(W_U.shape[1])

    final_pred = int(np.argmax(np.array(logits[pos])))
    layer_preds = []

    for l in range(n_layers):
        key = f"blocks.{l}.hook_resid_post"
        if key in cache.cache_dict:
            resid = np.array(cache.cache_dict[key][pos])
            layer_logits = resid @ W_U + b_U
            layer_preds.append(int(np.argmax(layer_logits)))
        else:
            layer_preds.append(-1)

    # Find flips
    flips = []
    for i in range(1, len(layer_preds)):
        if layer_preds[i] != layer_preds[i - 1]:
            flips.append(i)

    # First layer predicting the correct token
    first_correct = -1
    for i, p in enumerate(layer_preds):
        if p == final_pred:
            first_correct = i
            break

    return {
        "layer_predictions": layer_preds,
        "flip_layers": flips,
        "n_flips": len(flips),
        "final_prediction": final_pred,
        "first_correct_layer": first_correct,
    }


def prediction_stability_across_layers(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Measure how stable the top-k prediction set is across layers.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        pos: Token position (-1 = last).
        top_k: Size of the prediction set to track.

    Returns:
        Dict with:
            "layer_top_k": [n_layers] list of top-k tokens at each layer
            "overlap_with_final": [n_layers] Jaccard overlap with final top-k
            "stability_scores": [n_layers-1] overlap between consecutive layers
            "mean_stability": mean consecutive overlap
            "final_top_k": the final output's top-k tokens
    """
    _, cache = model.run_with_cache(tokens)
    logits = model(tokens)
    n_layers = model.cfg.n_layers

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') else np.zeros(W_U.shape[1])

    final_logits = np.array(logits[pos])
    final_topk = set(np.argsort(final_logits)[::-1][:top_k].tolist())

    layer_topk = []
    overlaps_with_final = []

    for l in range(n_layers):
        key = f"blocks.{l}.hook_resid_post"
        if key in cache.cache_dict:
            resid = np.array(cache.cache_dict[key][pos])
            layer_logits = resid @ W_U + b_U
            topk_set = set(np.argsort(layer_logits)[::-1][:top_k].tolist())
            layer_topk.append(list(topk_set))

            # Jaccard overlap with final
            overlap = len(topk_set & final_topk) / len(topk_set | final_topk)
            overlaps_with_final.append(overlap)
        else:
            layer_topk.append([])
            overlaps_with_final.append(0.0)

    # Consecutive stability
    stability = []
    for i in range(1, len(layer_topk)):
        s1 = set(layer_topk[i - 1])
        s2 = set(layer_topk[i])
        if s1 and s2:
            stability.append(len(s1 & s2) / len(s1 | s2))
        else:
            stability.append(0.0)

    return {
        "layer_top_k": layer_topk,
        "overlap_with_final": np.array(overlaps_with_final),
        "stability_scores": np.array(stability),
        "mean_stability": float(np.mean(stability)) if stability else 0.0,
        "final_top_k": list(final_topk),
    }


def commitment_timing(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    confidence_threshold: float = 0.5,
) -> dict:
    """When does the model "commit" to its final prediction?

    Measures the probability of the final output token at each layer
    to find the commitment point.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        pos: Token position (-1 = last).
        confidence_threshold: Probability threshold for "committed".

    Returns:
        Dict with:
            "layer_confidence": [n_layers] probability of final token at each layer
            "commitment_layer": first layer where confidence exceeds threshold (-1 if never)
            "confidence_growth_rate": mean per-layer confidence increase
            "max_confidence_jump": layer with largest single-step confidence increase
            "final_confidence": confidence at the output
    """
    _, cache = model.run_with_cache(tokens)
    logits = model(tokens)
    n_layers = model.cfg.n_layers

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') else np.zeros(W_U.shape[1])

    final_logits = np.array(logits[pos])
    final_probs = np.exp(final_logits - np.max(final_logits))
    final_probs = final_probs / np.sum(final_probs)
    final_token = int(np.argmax(final_probs))
    final_conf = float(final_probs[final_token])

    layer_conf = np.zeros(n_layers)
    for l in range(n_layers):
        key = f"blocks.{l}.hook_resid_post"
        if key in cache.cache_dict:
            resid = np.array(cache.cache_dict[key][pos])
            layer_logits = resid @ W_U + b_U
            probs = np.exp(layer_logits - np.max(layer_logits))
            probs = probs / np.sum(probs)
            layer_conf[l] = float(probs[final_token])

    # Commitment layer
    commit = -1
    for i, c in enumerate(layer_conf):
        if c >= confidence_threshold:
            commit = i
            break

    # Growth rate
    if n_layers >= 2:
        growth = float((layer_conf[-1] - layer_conf[0]) / (n_layers - 1))
    else:
        growth = 0.0

    # Max jump
    if n_layers >= 2:
        jumps = np.diff(layer_conf)
        max_jump = int(np.argmax(jumps))
    else:
        max_jump = 0

    return {
        "layer_confidence": layer_conf,
        "commitment_layer": commit,
        "confidence_growth_rate": growth,
        "max_confidence_jump": max_jump,
        "final_confidence": final_conf,
    }


def logit_contribution_by_component(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_token: int,
    pos: int = -1,
) -> dict:
    """Measure each component's contribution to a target token's logit.

    Decomposes the final logit into contributions from each attention
    head and MLP layer.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        target_token: Token to analyze.
        pos: Position (-1 = last).

    Returns:
        Dict with:
            "attn_contributions": [n_layers] attention contribution to target logit
            "mlp_contributions": [n_layers] MLP contribution to target logit
            "embedding_contribution": embedding layer's contribution
            "total_logit": final logit value for target token
            "dominant_component": which component type contributes most
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') else np.zeros(W_U.shape[1])

    unembed_dir = W_U[:, target_token]  # [d_model]

    attn_contribs = np.zeros(n_layers)
    mlp_contribs = np.zeros(n_layers)

    for l in range(n_layers):
        # Attention contribution
        attn_key = f"blocks.{l}.attn.hook_result"
        if attn_key in cache.cache_dict:
            attn_out = np.array(cache.cache_dict[attn_key][pos])
            attn_contribs[l] = float(np.dot(attn_out, unembed_dir))

        # MLP contribution
        mlp_pre_key = f"blocks.{l}.hook_resid_mid"
        mlp_post_key = f"blocks.{l}.hook_resid_post"
        if mlp_pre_key in cache.cache_dict and mlp_post_key in cache.cache_dict:
            mlp_out = np.array(cache.cache_dict[mlp_post_key][pos]) - np.array(cache.cache_dict[mlp_pre_key][pos])
            mlp_contribs[l] = float(np.dot(mlp_out, unembed_dir))

    # Embedding contribution
    embed_key = "blocks.0.hook_resid_pre"
    embed_contrib = 0.0
    if embed_key in cache.cache_dict:
        embed_out = np.array(cache.cache_dict[embed_key][pos])
        embed_contrib = float(np.dot(embed_out, unembed_dir))

    # Total logit
    logits = model(tokens)
    total = float(np.array(logits[pos])[target_token])

    # Dominant component
    total_attn = float(np.sum(np.abs(attn_contribs)))
    total_mlp = float(np.sum(np.abs(mlp_contribs)))
    dominant = "attention" if total_attn > total_mlp else "mlp"

    return {
        "attn_contributions": attn_contribs,
        "mlp_contributions": mlp_contribs,
        "embedding_contribution": embed_contrib,
        "total_logit": total,
        "dominant_component": dominant,
    }


def top_k_trajectory(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Track how top-k token probabilities evolve through layers.

    For each layer, records the probability of the final top-k tokens.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        pos: Token position (-1 = last).
        top_k: Number of top tokens to track.

    Returns:
        Dict with:
            "tracked_tokens": [top_k] token indices being tracked
            "probability_trajectories": dict mapping token -> [n_layers] probability curve
            "convergence_layer": layer where top-1 prob first exceeds others
            "final_probabilities": [top_k] final probability for each tracked token
    """
    _, cache = model.run_with_cache(tokens)
    logits = model(tokens)
    n_layers = model.cfg.n_layers

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') else np.zeros(W_U.shape[1])

    final_logits = np.array(logits[pos])
    final_probs = np.exp(final_logits - np.max(final_logits))
    final_probs = final_probs / np.sum(final_probs)
    tracked = np.argsort(final_probs)[::-1][:top_k].tolist()

    trajectories = {t: np.zeros(n_layers) for t in tracked}
    final_p = {t: float(final_probs[t]) for t in tracked}

    for l in range(n_layers):
        key = f"blocks.{l}.hook_resid_post"
        if key in cache.cache_dict:
            resid = np.array(cache.cache_dict[key][pos])
            layer_logits = resid @ W_U + b_U
            probs = np.exp(layer_logits - np.max(layer_logits))
            probs = probs / np.sum(probs)
            for t in tracked:
                trajectories[t][l] = float(probs[t])

    # Convergence: when top-1 first exceeds all others
    convergence = -1
    top_1 = tracked[0] if tracked else 0
    for l in range(n_layers):
        if all(trajectories[top_1][l] > trajectories[t][l] for t in tracked[1:]):
            convergence = l
            break

    return {
        "tracked_tokens": tracked,
        "probability_trajectories": trajectories,
        "convergence_layer": convergence,
        "final_probabilities": [final_p[t] for t in tracked],
    }
