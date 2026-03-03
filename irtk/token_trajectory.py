"""Token trajectory analysis.

Tracks how token representations evolve through layers: velocity,
acceleration, convergence, divergence between tokens, and trajectory
curvature.

References:
    Brunner et al. (2020) "On Identifiability in Transformers"
    Ethayarajh (2019) "How Contextual are Contextualized Word Representations?"
"""

import jax
import jax.numpy as jnp
import numpy as np


def token_representation_trajectory(model, tokens, positions=None):
    """Track token representations through each layer.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        positions: Positions to track (default: all).

    Returns:
        dict with:
            trajectories: [n_positions, n_layers+1, d_model] representations
            norms: [n_positions, n_layers+1] representation norms
            positions_tracked: list of positions
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    seq_len = len(tokens)
    d_model = model.cfg.d_model

    if positions is None:
        positions = list(range(seq_len))

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    n_pos = len(positions)
    trajs = np.zeros((n_pos, n_layers + 1, d_model))
    norms = np.zeros((n_pos, n_layers + 1))

    for pi, pos in enumerate(positions):
        # Layer 0 input
        r = cache.get("blocks.0.hook_resid_pre")
        if r is not None:
            trajs[pi, 0] = np.array(r[pos])
            norms[pi, 0] = float(np.linalg.norm(trajs[pi, 0]))

        for layer in range(n_layers):
            r = cache.get(f"blocks.{layer}.hook_resid_post")
            if r is not None:
                trajs[pi, layer + 1] = np.array(r[pos])
                norms[pi, layer + 1] = float(np.linalg.norm(trajs[pi, layer + 1]))

    return {
        "trajectories": trajs,
        "norms": norms,
        "positions_tracked": positions,
    }


def trajectory_velocity(model, tokens, positions=None):
    """Compute velocity (rate of change) of token representations.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        positions: Positions to track.

    Returns:
        dict with:
            velocities: [n_positions, n_layers] norm of change per layer
            directions: [n_positions, n_layers, d_model] direction of change
            mean_velocity: [n_layers] mean velocity across positions
            fastest_layer: int
    """
    traj = token_representation_trajectory(model, tokens, positions)
    t = traj["trajectories"]
    n_pos, n_steps, d = t.shape
    n_layers = n_steps - 1

    velocities = np.zeros((n_pos, n_layers))
    directions = np.zeros((n_pos, n_layers, d))

    for pi in range(n_pos):
        for l in range(n_layers):
            delta = t[pi, l + 1] - t[pi, l]
            velocities[pi, l] = float(np.linalg.norm(delta))
            if velocities[pi, l] > 1e-10:
                directions[pi, l] = delta / velocities[pi, l]

    mean_vel = np.mean(velocities, axis=0)

    return {
        "velocities": velocities,
        "directions": directions,
        "mean_velocity": mean_vel,
        "fastest_layer": int(np.argmax(mean_vel)),
    }


def trajectory_acceleration(model, tokens, positions=None):
    """Compute acceleration (change in velocity) of representations.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        positions: Positions to track.

    Returns:
        dict with:
            accelerations: [n_positions, n_layers-1] change in velocity
            mean_acceleration: [n_layers-1]
            is_decelerating: bool, whether velocity generally decreases
    """
    vel = trajectory_velocity(model, tokens, positions)
    v = vel["velocities"]
    n_pos, n_layers = v.shape

    if n_layers < 2:
        return {
            "accelerations": np.zeros((n_pos, 0)),
            "mean_acceleration": np.array([]),
            "is_decelerating": True,
        }

    acc = np.zeros((n_pos, n_layers - 1))
    for pi in range(n_pos):
        for l in range(n_layers - 1):
            acc[pi, l] = v[pi, l + 1] - v[pi, l]

    mean_acc = np.mean(acc, axis=0)
    is_decel = bool(np.mean(mean_acc) < 0)

    return {
        "accelerations": acc,
        "mean_acceleration": mean_acc,
        "is_decelerating": is_decel,
    }


def token_convergence(model, tokens, pos_a=0, pos_b=-1):
    """Measure convergence or divergence between two token representations.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos_a: First position.
        pos_b: Second position.

    Returns:
        dict with:
            distances: [n_layers+1] L2 distance per layer
            cosine_similarities: [n_layers+1] cosine similarity per layer
            is_converging: bool, distance decreasing
            convergence_rate: float, mean distance change
    """
    traj = token_representation_trajectory(model, tokens, [pos_a, pos_b])
    t = traj["trajectories"]
    n_steps = t.shape[1]

    distances = np.zeros(n_steps)
    cosines = np.zeros(n_steps)

    for l in range(n_steps):
        a = t[0, l]
        b = t[1, l]
        distances[l] = float(np.linalg.norm(a - b))
        na = np.linalg.norm(a) + 1e-10
        nb = np.linalg.norm(b) + 1e-10
        cosines[l] = float(np.dot(a, b) / (na * nb))

    # Convergence: is distance decreasing?
    if n_steps > 1:
        changes = distances[1:] - distances[:-1]
        is_conv = bool(np.mean(changes) < 0)
        conv_rate = float(np.mean(changes))
    else:
        is_conv = False
        conv_rate = 0.0

    return {
        "distances": distances,
        "cosine_similarities": cosines,
        "is_converging": is_conv,
        "convergence_rate": conv_rate,
    }


def trajectory_curvature(model, tokens, positions=None):
    """Compute curvature of representation trajectory.

    Curvature measures how much the direction of change changes between layers.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        positions: Positions to track.

    Returns:
        dict with:
            curvatures: [n_positions, n_layers-1] angle change between steps
            mean_curvature: [n_layers-1]
            straightest_position: int
            most_curved_position: int
    """
    vel = trajectory_velocity(model, tokens, positions)
    dirs = vel["directions"]
    n_pos, n_layers, d = dirs.shape

    if n_layers < 2:
        return {
            "curvatures": np.zeros((n_pos, 0)),
            "mean_curvature": np.array([]),
            "straightest_position": 0,
            "most_curved_position": 0,
        }

    curvatures = np.zeros((n_pos, n_layers - 1))

    for pi in range(n_pos):
        for l in range(n_layers - 1):
            d1 = dirs[pi, l]
            d2 = dirs[pi, l + 1]
            n1 = np.linalg.norm(d1) + 1e-10
            n2 = np.linalg.norm(d2) + 1e-10
            cos = np.dot(d1, d2) / (n1 * n2)
            curvatures[pi, l] = float(np.arccos(np.clip(cos, -1, 1)))

    mean_curv = np.mean(curvatures, axis=1)
    straightest = int(np.argmin(mean_curv))
    most_curved = int(np.argmax(mean_curv))

    return {
        "curvatures": curvatures,
        "mean_curvature": np.mean(curvatures, axis=0),
        "straightest_position": straightest,
        "most_curved_position": most_curved,
    }
