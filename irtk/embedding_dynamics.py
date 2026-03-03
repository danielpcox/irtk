"""Embedding dynamics analysis.

Tracks how token embeddings evolve through the network: identity decay,
semantic drift, positional encoding persistence, and embedding subspace usage.

References:
    Ethayarajh (2019) "How Contextual are Contextualized Word Representations?"
    Cai et al. (2021) "Isotropy in the Contextual Embedding Space"
"""

import jax
import jax.numpy as jnp
import numpy as np


def token_identity_decay(model, tokens, pos=-1):
    """Track how similar the residual stream stays to the initial embedding.

    At each layer, measures cosine similarity between the residual stream
    and the initial token embedding, revealing identity decay.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position to track.

    Returns:
        dict with:
            identity_trajectory: array [n_layers+1] of cosine similarity to initial
            half_life_layer: int, layer where similarity drops below 0.5
            decay_rate: float, mean per-layer similarity drop
            final_identity: float
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    # Initial embedding
    W_E = np.array(model.embed.W_E)
    token_idx = int(tokens[pos])
    initial = W_E[token_idx]
    initial_norm = np.linalg.norm(initial) + 1e-10

    trajectory = np.zeros(n_layers + 1)

    for layer in range(n_layers + 1):
        if layer == 0:
            key = "blocks.0.hook_resid_pre"
        else:
            key = f"blocks.{layer - 1}.hook_resid_post"
        resid = cache.get(key)
        if resid is not None:
            vec = np.array(resid[pos])
            vec_norm = np.linalg.norm(vec) + 1e-10
            trajectory[layer] = float(np.dot(initial, vec) / (initial_norm * vec_norm))

    # Half-life
    half_life = n_layers
    for l in range(n_layers + 1):
        if trajectory[l] < 0.5:
            half_life = l
            break

    # Decay rate
    diffs = np.diff(trajectory)
    decay_rate = float(np.mean(diffs)) if len(diffs) > 0 else 0.0

    return {
        "identity_trajectory": trajectory,
        "half_life_layer": half_life,
        "decay_rate": decay_rate,
        "final_identity": float(trajectory[-1]),
    }


def semantic_drift_analysis(model, tokens, pos=-1):
    """Measure semantic drift: how much the representation changes direction per layer.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.

    Returns:
        dict with:
            drift_angles: array [n_layers] of angle change (radians) per layer
            cumulative_drift: array [n_layers] of accumulated angular change
            fastest_drift_layer: int
            total_drift: float, total angular change in radians
            drift_rate: float, mean angular change per layer
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    states = []
    for layer in range(n_layers + 1):
        if layer == 0:
            key = "blocks.0.hook_resid_pre"
        else:
            key = f"blocks.{layer - 1}.hook_resid_post"
        resid = cache.get(key)
        if resid is not None:
            states.append(np.array(resid[pos]))
        else:
            states.append(np.zeros(model.cfg.d_model))

    angles = np.zeros(n_layers)
    for l in range(n_layers):
        a = states[l]
        b = states[l + 1]
        na = np.linalg.norm(a) + 1e-10
        nb = np.linalg.norm(b) + 1e-10
        cos = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
        angles[l] = float(np.arccos(cos))

    cumulative = np.cumsum(angles)
    fastest = int(np.argmax(angles))

    return {
        "drift_angles": angles,
        "cumulative_drift": cumulative,
        "fastest_drift_layer": fastest,
        "total_drift": float(cumulative[-1]) if len(cumulative) > 0 else 0.0,
        "drift_rate": float(np.mean(angles)) if len(angles) > 0 else 0.0,
    }


def positional_encoding_persistence(model, tokens):
    """Measure how position information persists through layers.

    Compares representations at different positions to detect whether
    positional structure is maintained, enhanced, or lost.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            position_discriminability: array [n_layers+1] of how distinguishable positions are
            mean_inter_position_distance: array [n_layers+1]
            position_order_preserved: array [n_layers+1] of bool
            persistence_score: float, mean discriminability across layers
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    seq_len = len(tokens)

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    discriminability = np.zeros(n_layers + 1)
    mean_distance = np.zeros(n_layers + 1)
    order_preserved = np.zeros(n_layers + 1, dtype=bool)

    for layer in range(n_layers + 1):
        if layer == 0:
            key = "blocks.0.hook_resid_pre"
        else:
            key = f"blocks.{layer - 1}.hook_resid_post"
        resid = cache.get(key)
        if resid is None:
            continue

        matrix = np.array(resid)  # [seq_len, d_model]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        normed = matrix / norms

        # Pairwise distances
        dists = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                d = np.linalg.norm(matrix[i] - matrix[j])
                dists[i, j] = d
                dists[j, i] = d

        upper = dists[np.triu_indices(seq_len, k=1)]
        mean_distance[layer] = float(np.mean(upper)) if len(upper) > 0 else 0.0

        # Discriminability: variance of distances / mean
        if mean_distance[layer] > 1e-10:
            discriminability[layer] = float(np.std(upper) / mean_distance[layer])
        else:
            discriminability[layer] = 0.0

        # Order preserved: do adjacent positions have smaller distances than non-adjacent?
        if seq_len > 2:
            adj_dists = [dists[i, i + 1] for i in range(seq_len - 1)]
            nonadj_dists = [dists[i, j] for i in range(seq_len) for j in range(i + 2, seq_len)]
            order_preserved[layer] = np.mean(adj_dists) < np.mean(nonadj_dists) if nonadj_dists else True

    return {
        "position_discriminability": discriminability,
        "mean_inter_position_distance": mean_distance,
        "position_order_preserved": order_preserved,
        "persistence_score": float(np.mean(discriminability)),
    }


def embedding_subspace_tracking(model, tokens, pos=-1, n_components=5):
    """Track which subspace of the embedding the representation occupies.

    Projects residual stream onto top PCA components of the embedding matrix
    to see how "embedding-like" the representation remains.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.
        n_components: Number of PCA components.

    Returns:
        dict with:
            embedding_subspace_projection: array [n_layers+1] of fraction in embedding subspace
            orthogonal_growth: array [n_layers+1] of norm outside embedding subspace
            subspace_exit_layer: int, layer where projection drops below 50%
            final_subspace_fraction: float
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    # Get embedding PCA basis
    W_E = np.array(model.embed.W_E)  # [d_vocab, d_model]
    centered = W_E - np.mean(W_E, axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    actual_k = min(n_components, Vt.shape[0])
    basis = Vt[:actual_k]  # [k, d_model]

    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    cache = hook_state.cache

    projection = np.zeros(n_layers + 1)
    orthogonal = np.zeros(n_layers + 1)

    for layer in range(n_layers + 1):
        if layer == 0:
            key = "blocks.0.hook_resid_pre"
        else:
            key = f"blocks.{layer - 1}.hook_resid_post"
        resid = cache.get(key)
        if resid is None:
            continue

        vec = np.array(resid[pos])
        total_norm = np.linalg.norm(vec) + 1e-10

        # Project onto embedding subspace
        proj_coeffs = basis @ vec  # [k]
        proj_vec = proj_coeffs @ basis  # [d_model]
        proj_norm = np.linalg.norm(proj_vec)

        projection[layer] = float(proj_norm / total_norm)
        orthogonal[layer] = float(np.linalg.norm(vec - proj_vec))

    # Exit layer
    exit_layer = n_layers
    for l in range(n_layers + 1):
        if projection[l] < 0.5:
            exit_layer = l
            break

    return {
        "embedding_subspace_projection": projection,
        "orthogonal_growth": orthogonal,
        "subspace_exit_layer": exit_layer,
        "final_subspace_fraction": float(projection[-1]),
    }


def context_mixing_rate(model, tokens, target_pos=-1):
    """Measure how quickly information from other positions mixes into the target.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        target_pos: Position to track.

    Returns:
        dict with:
            self_similarity: array [n_layers+1] tracking similarity to self-only representation
            mixing_rate: array [n_layers] of per-layer mixing
            context_dependence: array [n_layers+1] of 1 - self_similarity
            fastest_mixing_layer: int
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    # Run full model
    full_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=full_state)

    # Run with only target token (to get context-free representation)
    single_token = jnp.array([tokens[target_pos]])
    single_state = HookState(hook_fns={}, cache={})
    model(single_token, hook_state=single_state)

    self_sim = np.zeros(n_layers + 1)

    for layer in range(n_layers + 1):
        if layer == 0:
            key = "blocks.0.hook_resid_pre"
        else:
            key = f"blocks.{layer - 1}.hook_resid_post"

        full_act = full_state.cache.get(key)
        single_act = single_state.cache.get(key)

        if full_act is not None and single_act is not None:
            a = np.array(full_act[target_pos])
            b = np.array(single_act[0])  # single token is at position 0
            na = np.linalg.norm(a) + 1e-10
            nb = np.linalg.norm(b) + 1e-10
            self_sim[layer] = float(np.dot(a, b) / (na * nb))

    mixing = np.zeros(n_layers)
    for l in range(n_layers):
        mixing[l] = max(0.0, self_sim[l] - self_sim[l + 1])

    context_dep = 1.0 - self_sim
    fastest = int(np.argmax(mixing)) if len(mixing) > 0 else 0

    return {
        "self_similarity": self_sim,
        "mixing_rate": mixing,
        "context_dependence": context_dep,
        "fastest_mixing_layer": fastest,
    }
