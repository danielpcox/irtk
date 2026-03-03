"""Cross-layer feature tracking for mechanistic interpretability.

Track how features evolve across layers: persistence, transformation,
creation/destruction events, lineage, and representation drift.

References:
- Geva et al. (2023) "Dissecting Recall of Factual Associations in Auto-Regressive
  Language Models"
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional


def _get_layer_activations(model, tokens, pos=-1):
    """Get activations at each layer."""
    from irtk.hook_points import HookState

    cache = {}
    hs = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hs)

    acts = []
    for l in range(model.cfg.n_layers):
        key = f"blocks.{l}.hook_resid_post"
        if key in cache:
            acts.append(np.array(cache[key][pos]))
    return acts


def feature_persistence(
    model,
    tokens,
    direction: Optional[np.ndarray] = None,
    pos: int = -1,
) -> dict:
    """Track how much a feature direction persists across layers.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        direction: Direction to track (if None, uses top PCA direction from layer 0).
        pos: Position to track.

    Returns:
        Dict with persistence_scores, projection_magnitudes, peak_layer,
        decay_rate.
    """
    acts = _get_layer_activations(model, tokens, pos)
    n_layers = len(acts)

    if direction is None:
        # Use top SVD direction from first layer
        U, S, Vt = np.linalg.svd(acts[0].reshape(1, -1), full_matrices=False)
        direction = Vt[0]

    direction = direction / (np.linalg.norm(direction) + 1e-10)

    projections = []
    for act in acts:
        proj = float(np.dot(act, direction))
        projections.append(proj)

    proj_arr = np.array(projections)
    magnitudes = np.abs(proj_arr)
    peak_layer = int(np.argmax(magnitudes))

    # Persistence: normalized by max
    max_mag = np.max(magnitudes) + 1e-10
    persistence = magnitudes / max_mag

    # Decay rate (from peak onwards)
    if peak_layer < n_layers - 1:
        post_peak = magnitudes[peak_layer:]
        if len(post_peak) > 1 and post_peak[0] > 1e-10:
            decay = float(1 - post_peak[-1] / post_peak[0])
        else:
            decay = 0.0
    else:
        decay = 0.0

    return {
        "persistence_scores": jnp.array(persistence),
        "projection_magnitudes": jnp.array(proj_arr),
        "peak_layer": peak_layer,
        "decay_rate": decay,
    }


def transformation_analysis(
    model,
    tokens,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Analyze how the residual stream transforms between layers.

    Computes the effective transformation matrix between adjacent layers.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        pos: Position.
        top_k: Number of top singular values.

    Returns:
        Dict with inter_layer_cosines, transformation_norms, rotation_angles,
        stretch_factors.
    """
    acts = _get_layer_activations(model, tokens, pos)
    n_layers = len(acts)

    cosines = []
    norms = []
    rotation_angles = []
    stretch_factors = []

    for l in range(n_layers - 1):
        a = acts[l]
        b = acts[l + 1]
        norm_a = np.linalg.norm(a) + 1e-10
        norm_b = np.linalg.norm(b) + 1e-10

        cos = float(np.dot(a, b) / (norm_a * norm_b))
        cosines.append(cos)
        norms.append(float(np.linalg.norm(b - a)))

        # Rotation angle
        angle = float(np.arccos(np.clip(cos, -1, 1)))
        rotation_angles.append(angle)

        # Stretch
        stretch = float(norm_b / norm_a)
        stretch_factors.append(stretch)

    return {
        "inter_layer_cosines": jnp.array(cosines),
        "transformation_norms": jnp.array(norms),
        "rotation_angles": jnp.array(rotation_angles),
        "stretch_factors": jnp.array(stretch_factors),
    }


def creation_destruction_events(
    model,
    tokens,
    pos: int = -1,
    threshold: float = 0.5,
) -> dict:
    """Detect feature creation and destruction events across layers.

    A feature is "created" when a new direction appears (low cosine to
    previous layer), and "destroyed" when a direction disappears.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        pos: Position.
        threshold: Cosine threshold for creation/destruction.

    Returns:
        Dict with creation_layers, destruction_layers, net_change,
        dimension_utilization.
    """
    acts = _get_layer_activations(model, tokens, pos)
    n_layers = len(acts)
    d = acts[0].shape[0]

    creations = []
    destructions = []
    utilization = []

    for l in range(n_layers):
        # Effective rank as proxy for dimension utilization
        act = acts[l]
        # Use absolute values as proxy for active dimensions
        active = np.sum(np.abs(act) > 0.01 * np.max(np.abs(act)))
        utilization.append(float(active / d))

    for l in range(1, n_layers):
        prev = acts[l - 1]
        curr = acts[l]
        delta = curr - prev

        # Creation: large component of delta orthogonal to prev
        prev_norm = prev / (np.linalg.norm(prev) + 1e-10)
        delta_parallel = np.dot(delta, prev_norm) * prev_norm
        delta_perp = delta - delta_parallel
        perp_ratio = np.linalg.norm(delta_perp) / (np.linalg.norm(delta) + 1e-10)

        if perp_ratio > threshold:
            creations.append(l)

        # Destruction: significant reduction in some directions
        cos = np.dot(prev, curr) / (np.linalg.norm(prev) * np.linalg.norm(curr) + 1e-10)
        if cos < threshold:
            destructions.append(l)

    return {
        "creation_layers": creations,
        "destruction_layers": destructions,
        "net_change": len(creations) - len(destructions),
        "dimension_utilization": jnp.array(utilization),
    }


def feature_lineage(
    model,
    tokens,
    source_layer: int = 0,
    pos: int = -1,
    n_directions: int = 3,
) -> dict:
    """Track the top directions from a source layer through the network.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        source_layer: Layer to track from.
        pos: Position.
        n_directions: Number of directions to track.

    Returns:
        Dict with direction_persistence (per direction, per layer),
        most_persistent_direction, least_persistent_direction.
    """
    acts = _get_layer_activations(model, tokens, pos)
    source_act = acts[source_layer]

    # Top directions from SVD of source activation
    U, S, Vt = np.linalg.svd(source_act.reshape(1, -1), full_matrices=False)
    n_dirs = min(n_directions, len(source_act))

    # For a single vector, SVD gives 1 direction. Use random orthogonal directions.
    directions = np.zeros((n_dirs, len(source_act)))
    directions[0] = source_act / (np.linalg.norm(source_act) + 1e-10)
    if n_dirs > 1:
        rng = np.random.RandomState(42)
        for i in range(1, n_dirs):
            d = rng.randn(len(source_act))
            # Orthogonalize against previous directions
            for j in range(i):
                d -= np.dot(d, directions[j]) * directions[j]
            directions[i] = d / (np.linalg.norm(d) + 1e-10)

    # Track each direction
    persistence = np.zeros((n_dirs, len(acts)))
    for d_idx in range(n_dirs):
        for l_idx, act in enumerate(acts):
            cos = np.dot(act, directions[d_idx]) / (np.linalg.norm(act) + 1e-10)
            persistence[d_idx, l_idx] = abs(cos)

    # Summary
    mean_persistence = np.mean(persistence, axis=1)
    most_persistent = int(np.argmax(mean_persistence))
    least_persistent = int(np.argmin(mean_persistence))

    return {
        "direction_persistence": jnp.array(persistence),
        "most_persistent_direction": most_persistent,
        "least_persistent_direction": least_persistent,
        "mean_persistence": jnp.array(mean_persistence),
    }


def representation_drift(
    model,
    tokens_list: list,
    pos: int = -1,
) -> dict:
    """Measure how representations drift across layers for multiple inputs.

    Args:
        model: HookedTransformer model.
        tokens_list: List of token arrays.
        pos: Position.

    Returns:
        Dict with mean_drift_per_layer, drift_variance, convergence_rate.
    """
    n_layers = model.cfg.n_layers

    all_drifts = np.zeros((len(tokens_list), n_layers - 1))

    for i, tok in enumerate(tokens_list):
        acts = _get_layer_activations(model, tok, pos)
        for l in range(n_layers - 1):
            drift = np.linalg.norm(acts[l + 1] - acts[l])
            all_drifts[i, l] = drift

    mean_drift = np.mean(all_drifts, axis=0)
    drift_var = np.var(all_drifts, axis=0)

    # Convergence: is drift decreasing?
    if len(mean_drift) > 1:
        convergence = float(mean_drift[0] - mean_drift[-1]) / (float(mean_drift[0]) + 1e-10)
    else:
        convergence = 0.0

    return {
        "mean_drift_per_layer": jnp.array(mean_drift),
        "drift_variance": jnp.array(drift_var),
        "convergence_rate": convergence,
        "total_drift": float(np.sum(mean_drift)),
    }
