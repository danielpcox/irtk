"""Representation similarity analysis across layers and positions.

Compare internal representations using CKA, cosine similarity matrices,
and other distance metrics to understand how representations evolve.
"""

import jax
import jax.numpy as jnp
from irtk.hook_points import HookState


def _run_and_cache(model, tokens):
    """Run model and return activation cache."""
    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    return hook_state.cache


def _linear_cka(X, Y):
    """Compute linear CKA between two representation matrices.

    X, Y: [n_samples, d] activation matrices.
    Returns scalar CKA similarity in [0, 1].
    """
    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Gram matrices
    XtX = X @ X.T  # [n, n]
    YtY = Y @ Y.T  # [n, n]
    XtY = X @ Y.T  # [n, n]

    # HSIC estimates
    hsic_xy = jnp.sum(XtY * XtY)
    hsic_xx = jnp.sum(XtX * XtX)
    hsic_yy = jnp.sum(YtY * YtY)

    denom = jnp.sqrt(hsic_xx * hsic_yy)
    return jnp.where(denom > 1e-10, hsic_xy / denom, 0.0)


def layer_representation_similarity(model, tokens):
    """Compute pairwise CKA similarity between all layer representations.

    Returns:
        dict with:
        - similarity_matrix: [n_stages, n_stages] CKA matrix
        - stage_names: list of stage names
        - mean_similarity: mean off-diagonal CKA
        - block_diagonal_score: how block-diagonal the matrix is
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers

    # Collect residual stream at each stage
    stages = []
    names = []

    # Embedding
    embed_key = 'blocks.0.hook_resid_pre'
    if embed_key in cache:
        stages.append(cache[embed_key])
        names.append('embed')

    # Each layer's output
    for l in range(n_layers):
        key = f'blocks.{l}.hook_resid_post'
        if key in cache:
            stages.append(cache[key])
            names.append(f'L{l}_post')

    n = len(stages)
    sim_matrix = jnp.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix = sim_matrix.at[i, j].set(1.0)
            elif j > i:
                cka = _linear_cka(stages[i], stages[j])
                sim_matrix = sim_matrix.at[i, j].set(cka)
                sim_matrix = sim_matrix.at[j, i].set(cka)

    # Mean off-diagonal
    mask = 1.0 - jnp.eye(n)
    mean_sim = float(jnp.sum(sim_matrix * mask) / jnp.sum(mask))

    # Block diagonal score: ratio of near-diagonal to far-diagonal similarity
    near_diag = 0.0
    far_diag = 0.0
    near_count = 0
    far_count = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if abs(i - j) <= 1:
                near_diag += float(sim_matrix[i, j])
                near_count += 1
            else:
                far_diag += float(sim_matrix[i, j])
                far_count += 1

    near_avg = near_diag / max(near_count, 1)
    far_avg = far_diag / max(far_count, 1)
    block_score = near_avg - far_avg  # positive = block-diagonal structure

    return {
        'similarity_matrix': sim_matrix,
        'stage_names': names,
        'mean_similarity': mean_sim,
        'block_diagonal_score': float(block_score),
    }


def position_representation_similarity(model, tokens):
    """Compare representations across positions at each layer.

    Returns:
        dict with per_layer list, each containing:
        - layer: layer index
        - position_similarity: [seq_len, seq_len] cosine similarity matrix
        - mean_similarity: mean off-diagonal similarity
        - most_similar_pair: (pos_i, pos_j) most similar positions
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)

    results = []
    for l in range(n_layers):
        key = f'blocks.{l}.hook_resid_post'
        if key not in cache:
            continue

        resid = cache[key]  # [seq, d_model]

        # Cosine similarity between positions
        norms = jnp.linalg.norm(resid, axis=-1, keepdims=True)  # [seq, 1]
        normalized = resid / jnp.maximum(norms, 1e-10)
        sim = normalized @ normalized.T  # [seq, seq]

        # Stats
        mask = 1.0 - jnp.eye(seq_len)
        mean_sim = float(jnp.sum(sim * mask) / jnp.maximum(jnp.sum(mask), 1.0))

        # Most similar pair (off-diagonal)
        masked_sim = sim * mask - jnp.eye(seq_len) * 1e10
        best_idx = int(jnp.argmax(masked_sim))
        best_i = best_idx // seq_len
        best_j = best_idx % seq_len

        results.append({
            'layer': l,
            'position_similarity': sim,
            'mean_similarity': mean_sim,
            'most_similar_pair': (best_i, best_j),
        })

    return {'per_layer': results}


def component_output_similarity(model, tokens):
    """Compare attention and MLP outputs within each layer.

    Returns:
        dict with per_layer list, each containing:
        - layer: layer index
        - attn_mlp_cosine: cosine similarity between attn and MLP outputs
        - attn_norm: norm of attention output
        - mlp_norm: norm of MLP output
        - alignment: how aligned the components are (-1 to 1)
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers

    results = []
    for l in range(n_layers):
        attn_key = f'blocks.{l}.hook_attn_out'
        mlp_key = f'blocks.{l}.hook_mlp_out'

        if attn_key not in cache or mlp_key not in cache:
            continue

        attn_out = cache[attn_key]  # [seq, d_model]
        mlp_out = cache[mlp_key]    # [seq, d_model]

        # Flatten for overall comparison
        attn_flat = attn_out.reshape(-1)
        mlp_flat = mlp_out.reshape(-1)

        attn_norm = float(jnp.linalg.norm(attn_flat))
        mlp_norm = float(jnp.linalg.norm(mlp_flat))

        cos = float(jnp.dot(attn_flat, mlp_flat) /
                     jnp.maximum(attn_norm * mlp_norm, 1e-10))

        results.append({
            'layer': l,
            'attn_mlp_cosine': cos,
            'attn_norm': attn_norm,
            'mlp_norm': mlp_norm,
            'alignment': cos,
        })

    return {'per_layer': results}


def representation_drift(model, tokens):
    """Measure how much representations change at each layer.

    Returns:
        dict with per_layer list, each containing:
        - layer: layer index
        - cosine_with_previous: cosine similarity with previous layer
        - l2_distance: L2 distance from previous layer
        - relative_change: ||delta|| / ||resid||
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers

    # Collect representations
    stages = []
    embed_key = 'blocks.0.hook_resid_pre'
    if embed_key in cache:
        stages.append(cache[embed_key])

    for l in range(n_layers):
        key = f'blocks.{l}.hook_resid_post'
        if key in cache:
            stages.append(cache[key])

    results = []
    for i in range(1, len(stages)):
        prev = stages[i - 1].reshape(-1)
        curr = stages[i].reshape(-1)
        delta = curr - prev

        prev_norm = float(jnp.linalg.norm(prev))
        curr_norm = float(jnp.linalg.norm(curr))
        delta_norm = float(jnp.linalg.norm(delta))

        cos = float(jnp.dot(prev, curr) / jnp.maximum(prev_norm * curr_norm, 1e-10))
        relative = delta_norm / max(curr_norm, 1e-10)

        results.append({
            'layer': i - 1,
            'cosine_with_previous': cos,
            'l2_distance': delta_norm,
            'relative_change': relative,
        })

    return {
        'per_layer': results,
        'total_drift': sum(r['l2_distance'] for r in results),
    }


def cross_input_similarity(model, tokens1, tokens2):
    """Compare representations of two different inputs at each layer.

    Returns:
        dict with per_layer list, each containing:
        - layer: layer index
        - mean_cosine: mean cosine similarity across positions
        - per_position: list of per-position cosine similarities
    """
    cache1 = _run_and_cache(model, tokens1)
    cache2 = _run_and_cache(model, tokens2)
    n_layers = model.cfg.n_layers
    min_len = min(len(tokens1), len(tokens2))

    results = []
    for l in range(n_layers):
        key = f'blocks.{l}.hook_resid_post'
        if key not in cache1 or key not in cache2:
            continue

        r1 = cache1[key][:min_len]  # [min_len, d_model]
        r2 = cache2[key][:min_len]

        # Per-position cosine
        n1 = jnp.linalg.norm(r1, axis=-1)  # [min_len]
        n2 = jnp.linalg.norm(r2, axis=-1)
        dots = jnp.sum(r1 * r2, axis=-1)
        cosines = dots / jnp.maximum(n1 * n2, 1e-10)

        results.append({
            'layer': l,
            'mean_cosine': float(jnp.mean(cosines)),
            'per_position': [float(c) for c in cosines],
        })

    return {'per_layer': results}
