"""Distributed representation analysis.

Tools for studying how information is encoded across multiple dimensions
and how representations evolve through the model:
- linear_representation_probe: Validate linear encoding of concepts
- representation_rank: How many dimensions encode a concept
- cross_layer_concept_tracking: Track concept representation across layers
- writing_reading_decomposition: What directions does a head write to / read from
- token_geometry: Pairwise distances and angles in representation space
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def linear_representation_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    n_directions: int = 1,
) -> dict:
    """Fit a linear probe and return accuracy + encoding direction(s).

    Validates whether a concept is linearly represented in the activations.

    Args:
        activations: [n_samples, d_model] activation vectors.
        labels: [n_samples] integer class labels.
        n_directions: Number of directions to extract (1 for binary).

    Returns:
        Dict with:
        - "accuracy": classification accuracy
        - "directions": [n_directions, d_model] concept encoding directions
        - "explained_variance": per-direction variance explained
        - "class_means": [n_classes, d_model] mean per class
    """
    activations = np.array(activations)
    labels = np.array(labels)
    classes = np.unique(labels)
    n_classes = len(classes)

    # Compute class means
    class_means = np.zeros((n_classes, activations.shape[1]))
    for i, c in enumerate(classes):
        class_means[i] = activations[labels == c].mean(axis=0)

    # Between-class scatter matrix
    grand_mean = activations.mean(axis=0)
    S_b = np.zeros((activations.shape[1], activations.shape[1]))
    for i, c in enumerate(classes):
        n_c = np.sum(labels == c)
        diff = (class_means[i] - grand_mean).reshape(-1, 1)
        S_b += n_c * (diff @ diff.T)

    # Within-class scatter
    S_w = np.zeros_like(S_b)
    for i, c in enumerate(classes):
        centered = activations[labels == c] - class_means[i]
        S_w += centered.T @ centered

    # Regularize
    S_w += np.eye(S_w.shape[0]) * 1e-6

    # Solve generalized eigenvalue problem
    try:
        eigvals, eigvecs = np.linalg.eigh(np.linalg.inv(S_w) @ S_b)
    except np.linalg.LinAlgError:
        eigvals = np.zeros(activations.shape[1])
        eigvecs = np.eye(activations.shape[1])

    # Sort by eigenvalue descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    k = min(n_directions, len(eigvals))
    directions = eigvecs[:, :k].T  # [k, d_model]

    # Normalize
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-10)

    # Explained variance
    total_var = max(eigvals.sum(), 1e-10)
    explained = eigvals[:k] / total_var

    # Classification accuracy (nearest centroid in projected space)
    projected = activations @ directions.T  # [n_samples, k]
    proj_means = class_means @ directions.T  # [n_classes, k]

    dists = np.zeros((len(activations), n_classes))
    for i in range(n_classes):
        dists[:, i] = np.linalg.norm(projected - proj_means[i], axis=1)
    preds = classes[np.argmin(dists, axis=1)]
    accuracy = float(np.mean(preds == labels))

    return {
        "accuracy": accuracy,
        "directions": directions,
        "explained_variance": explained,
        "class_means": class_means,
    }


def representation_rank(
    model: HookedTransformer,
    token_sequences: list,
    hook_name: str,
    labels: np.ndarray,
    max_rank: int = 10,
) -> dict:
    """Estimate how many dimensions encode a concept.

    Trains probes with increasing dimensionality to find the
    rank at which accuracy saturates.

    Args:
        model: HookedTransformer.
        token_sequences: List of token arrays.
        hook_name: Hook point to analyze.
        labels: [n_sequences] per-sequence concept labels.
        max_rank: Maximum dimensionality to test.

    Returns:
        Dict with:
        - "accuracies": [max_rank] accuracy at each rank
        - "estimated_rank": rank at which accuracy saturates
        - "saturation_accuracy": accuracy at estimated rank
    """
    labels = np.array(labels)
    # Collect activations (last position per sequence)
    all_acts = []
    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)
        if hook_name in cache.cache_dict:
            act = np.array(cache.cache_dict[hook_name][-1])  # last position
            all_acts.append(act)

    if not all_acts or len(all_acts) != len(labels):
        return {"accuracies": np.zeros(max_rank), "estimated_rank": 1,
                "saturation_accuracy": 0.0}

    activations = np.stack(all_acts)
    accs = []

    for k in range(1, max_rank + 1):
        result = linear_representation_probe(activations, labels, n_directions=k)
        accs.append(result["accuracy"])

    accs = np.array(accs)

    # Estimate saturation: first rank where improvement < 1%
    est_rank = 1
    for i in range(1, len(accs)):
        if accs[i] - accs[i - 1] < 0.01:
            est_rank = i  # 1-indexed
            break
        est_rank = i + 1

    return {
        "accuracies": accs,
        "estimated_rank": est_rank,
        "saturation_accuracy": float(accs[est_rank - 1]),
    }


def cross_layer_concept_tracking(
    model: HookedTransformer,
    token_sequences: list,
    labels: np.ndarray,
    positions: Optional[list[int]] = None,
) -> dict:
    """Track how a concept's linear representation evolves across layers.

    Args:
        model: HookedTransformer.
        token_sequences: List of token arrays.
        labels: [n_sequences] concept labels.
        positions: Positions to average over (default: last).

    Returns:
        Dict with:
        - "layer_accuracies": [n_layers+1] probe accuracy at each layer
        - "layer_directions": [n_layers+1, d_model] concept direction at each
        - "direction_similarities": [n_layers] cosine sim between consecutive layers
        - "labels": ["embed", "L0", "L1", ...] layer names
    """
    labels = np.array(labels)
    n_layers = model.cfg.n_layers
    hook_names = ["hook_embed"] + [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]
    layer_labels = ["embed"] + [f"L{i}" for i in range(n_layers)]

    accs = []
    dirs = []

    for hook_name in hook_names:
        all_acts = []
        for tokens in token_sequences:
            tokens = jnp.array(tokens)
            _, cache = model.run_with_cache(tokens)
            if hook_name in cache.cache_dict:
                act = np.array(cache.cache_dict[hook_name])
                if positions is not None:
                    act = act[positions].mean(axis=0)
                else:
                    act = act[-1]  # last position
                all_acts.append(act)

        if len(all_acts) != len(labels):
            accs.append(0.0)
            dirs.append(np.zeros(model.cfg.d_model))
            continue

        activations = np.stack(all_acts)
        result = linear_representation_probe(activations, labels, n_directions=1)
        accs.append(result["accuracy"])
        dirs.append(result["directions"][0] if len(result["directions"]) > 0
                     else np.zeros(model.cfg.d_model))

    # Direction similarities
    dirs_arr = np.array(dirs)
    sims = []
    for i in range(len(dirs_arr) - 1):
        n1 = np.linalg.norm(dirs_arr[i])
        n2 = np.linalg.norm(dirs_arr[i + 1])
        if n1 > 1e-10 and n2 > 1e-10:
            sims.append(float(np.dot(dirs_arr[i], dirs_arr[i + 1]) / (n1 * n2)))
        else:
            sims.append(0.0)

    return {
        "layer_accuracies": np.array(accs),
        "layer_directions": dirs_arr,
        "direction_similarities": np.array(sims),
        "labels": layer_labels,
    }


def writing_reading_decomposition(
    model: HookedTransformer,
    layer: int,
    directions: np.ndarray,
) -> dict:
    """Decompose how much attention writes to / reads from given directions.

    For each direction, computes:
    - Writing: how much does W_O project into this direction?
    - Reading: how much does W_Q/W_K align with this direction?

    Args:
        model: HookedTransformer.
        layer: Layer to analyze.
        directions: [n_directions, d_model] directions to decompose against.

    Returns:
        Dict with:
        - "writing_scores": [n_heads, n_directions] how much each head writes to each direction
        - "reading_q_scores": [n_heads, n_directions] how much W_Q reads from each direction
        - "reading_k_scores": [n_heads, n_directions] how much W_K reads from each direction
    """
    directions = np.array(directions)
    if directions.ndim == 1:
        directions = directions.reshape(1, -1)

    # Normalize
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-10)

    n_heads = model.cfg.n_heads
    n_dirs = len(directions)

    W_O = np.array(model.blocks[layer].attn.W_O)  # [n_heads, d_head, d_model]
    W_Q = np.array(model.blocks[layer].attn.W_Q)  # [n_heads, d_model, d_head]
    W_K = np.array(model.blocks[layer].attn.W_K)  # [n_heads, d_model, d_head]

    writing = np.zeros((n_heads, n_dirs))
    reading_q = np.zeros((n_heads, n_dirs))
    reading_k = np.zeros((n_heads, n_dirs))

    for h in range(n_heads):
        for d in range(n_dirs):
            # Writing: ||direction @ W_O^T|| (how much output projects onto direction)
            writing[h, d] = float(np.linalg.norm(directions[d] @ W_O[h].T))
            # Reading Q: ||W_Q^T @ direction|| (how much Q responds to direction)
            reading_q[h, d] = float(np.linalg.norm(W_Q[h].T @ directions[d]))
            # Reading K: ||W_K^T @ direction|| (how much K responds to direction)
            reading_k[h, d] = float(np.linalg.norm(W_K[h].T @ directions[d]))

    return {
        "writing_scores": writing,
        "reading_q_scores": reading_q,
        "reading_k_scores": reading_k,
    }


def token_geometry(
    model: HookedTransformer,
    token_ids: list[int],
    hook_name: str,
    token_sequences: list,
) -> dict:
    """Compute pairwise distances and angles between token representations.

    Args:
        model: HookedTransformer.
        token_ids: Token IDs to analyze.
        hook_name: Hook point to get representations from.
        token_sequences: Token sequences containing the tokens.

    Returns:
        Dict with:
        - "distance_matrix": [n_tokens, n_tokens] pairwise L2 distances
        - "cosine_matrix": [n_tokens, n_tokens] pairwise cosine similarities
        - "mean_representations": [n_tokens, d_model] averaged representations
    """
    # Collect representations for each target token
    token_reps = {tid: [] for tid in token_ids}

    for tokens in token_sequences:
        tokens_arr = np.array(tokens)
        tokens_jax = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens_jax)
        if hook_name not in cache.cache_dict:
            continue
        acts = np.array(cache.cache_dict[hook_name])  # [seq, d_model]

        for pos, tid in enumerate(tokens_arr):
            tid_int = int(tid)
            if tid_int in token_reps:
                token_reps[tid_int].append(acts[pos])

    # Compute mean representations
    n = len(token_ids)
    d = model.cfg.d_model
    means = np.zeros((n, d))
    for i, tid in enumerate(token_ids):
        if token_reps[tid]:
            means[i] = np.mean(token_reps[tid], axis=0)

    # Pairwise distances and cosines
    dist_matrix = np.zeros((n, n))
    cos_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = float(np.linalg.norm(means[i] - means[j]))
            ni = np.linalg.norm(means[i])
            nj = np.linalg.norm(means[j])
            if ni > 1e-10 and nj > 1e-10:
                cos_matrix[i, j] = float(np.dot(means[i], means[j]) / (ni * nj))

    return {
        "distance_matrix": dist_matrix,
        "cosine_matrix": cos_matrix,
        "mean_representations": means,
    }
