"""Latent space navigation for mechanistic interpretability.

Navigate the model's latent space: interpolation between representations,
geodesic-like paths, latent arithmetic, boundary detection, and
manifold exploration.

References:
- Park et al. (2023) "The Linear Representation Hypothesis and the Geometry
  of Large Language Models"
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional


def representation_interpolation(
    model,
    tokens_a,
    tokens_b,
    layer: int = -1,
    pos: int = -1,
    n_steps: int = 10,
) -> dict:
    """Interpolate between two representations in activation space.

    Creates a linear path between two activation vectors and projects
    each point through the remaining network to see predictions.

    Args:
        model: HookedTransformer model.
        tokens_a: First input token array.
        tokens_b: Second input token array.
        layer: Layer at which to interpolate.
        pos: Position to interpolate.
        n_steps: Number of interpolation steps.

    Returns:
        Dict with interpolation_path, top_predictions, prediction_entropy,
        path_norm.
    """
    from irtk.hook_points import HookState

    if layer < 0:
        layer = model.cfg.n_layers + layer
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Get activations for both inputs
    cache_a = {}
    hs_a = HookState(hook_fns={}, cache=cache_a)
    model(tokens_a, hook_state=hs_a)
    act_a = np.array(cache_a[hook_name][pos])

    cache_b = {}
    hs_b = HookState(hook_fns={}, cache=cache_b)
    model(tokens_b, hook_state=hs_b)
    act_b = np.array(cache_b[hook_name][pos])

    # Interpolate
    alphas = np.linspace(0, 1, n_steps)
    path = []
    top_preds = []
    entropies = []

    for alpha in alphas:
        interp = (1 - alpha) * act_a + alpha * act_b
        path.append(interp)

        # Project through remaining layers by injecting at hook point
        interp_jnp = jnp.array(interp)

        def inject_hook(x, name, _interp=interp_jnp):
            return x.at[pos].set(_interp)

        hs = HookState(hook_fns={hook_name: inject_hook}, cache=None)
        logits = model(tokens_a, hook_state=hs)

        # Top prediction at interpolated position
        last_logits = np.array(logits[pos])
        probs = np.exp(last_logits - np.max(last_logits))
        probs = probs / np.sum(probs)
        top_preds.append(int(np.argmax(probs)))
        entropies.append(float(-np.sum(probs * np.log(probs + 1e-10))))

    path = np.stack(path)
    norms = np.linalg.norm(np.diff(path, axis=0), axis=1)

    return {
        "interpolation_path": jnp.array(path),
        "alphas": jnp.array(alphas),
        "top_predictions": top_preds,
        "prediction_entropy": jnp.array(entropies),
        "path_norms": jnp.array(norms),
        "total_path_length": float(np.sum(norms)),
    }


def latent_arithmetic(
    model,
    tokens_a,
    tokens_b,
    tokens_c,
    layer: int = -1,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Perform arithmetic in latent space: a - b + c.

    Tests whether linear operations in activation space produce
    semantically meaningful results (analogous to word2vec arithmetic).

    Args:
        model: HookedTransformer model.
        tokens_a: First input (the "result" template).
        tokens_b: Second input (to subtract).
        tokens_c: Third input (to add).
        layer: Layer for arithmetic.
        pos: Position for arithmetic.
        top_k: Number of top predictions.

    Returns:
        Dict with result_vector, top_predictions, prediction_probs,
        analogy_coherence.
    """
    from irtk.hook_points import HookState

    if layer < 0:
        layer = model.cfg.n_layers + layer
    hook_name = f"blocks.{layer}.hook_resid_post"

    # Collect activations
    def get_act(tokens):
        cache = {}
        hs = HookState(hook_fns={}, cache=cache)
        model(tokens, hook_state=hs)
        return np.array(cache[hook_name][pos])

    act_a = get_act(tokens_a)
    act_b = get_act(tokens_b)
    act_c = get_act(tokens_c)

    # Arithmetic: a - b + c
    result = act_a - act_b + act_c
    result_jnp = jnp.array(result)

    # Inject result and get predictions
    def inject_hook(x, name, _result=result_jnp):
        return x.at[pos].set(_result)

    hs = HookState(hook_fns={hook_name: inject_hook}, cache=None)
    logits = model(tokens_a, hook_state=hs)

    last_logits = np.array(logits[pos])
    probs = np.exp(last_logits - np.max(last_logits))
    probs = probs / np.sum(probs)
    top_idx = np.argsort(probs)[::-1][:top_k]

    # Coherence: how similar is the result to actual activations?
    # Compare to each input's activation
    cos_a = float(np.dot(result, act_a) / (np.linalg.norm(result) * np.linalg.norm(act_a) + 1e-10))
    cos_b = float(np.dot(result, act_b) / (np.linalg.norm(result) * np.linalg.norm(act_b) + 1e-10))
    cos_c = float(np.dot(result, act_c) / (np.linalg.norm(result) * np.linalg.norm(act_c) + 1e-10))

    return {
        "result_vector": jnp.array(result),
        "top_predictions": [(int(idx), float(probs[idx])) for idx in top_idx],
        "cosine_to_a": cos_a,
        "cosine_to_b": cos_b,
        "cosine_to_c": cos_c,
        "result_norm": float(np.linalg.norm(result)),
    }


def boundary_detection(
    model,
    tokens_a,
    tokens_b,
    layer: int = -1,
    pos: int = -1,
    n_probes: int = 20,
) -> dict:
    """Detect decision boundaries between two representations.

    Searches for the interpolation point where the model's top
    prediction changes, indicating a decision boundary.

    Args:
        model: HookedTransformer model.
        tokens_a: First input.
        tokens_b: Second input.
        layer: Layer to interpolate at.
        pos: Position to analyze.
        n_probes: Number of probe points along the interpolation.

    Returns:
        Dict with boundary_alpha, prediction_sequence, boundary_sharpness,
        n_boundaries.
    """
    from irtk.hook_points import HookState

    if layer < 0:
        layer = model.cfg.n_layers + layer
    hook_name = f"blocks.{layer}.hook_resid_post"

    cache_a = {}
    hs_a = HookState(hook_fns={}, cache=cache_a)
    model(tokens_a, hook_state=hs_a)
    act_a = np.array(cache_a[hook_name][pos])

    cache_b = {}
    hs_b = HookState(hook_fns={}, cache=cache_b)
    model(tokens_b, hook_state=hs_b)
    act_b = np.array(cache_b[hook_name][pos])

    alphas = np.linspace(0, 1, n_probes)
    predictions = []
    confidences = []

    for alpha in alphas:
        interp = (1 - alpha) * act_a + alpha * act_b
        interp_jnp = jnp.array(interp)

        def inject_hook(x, name, _interp=interp_jnp):
            return x.at[pos].set(_interp)

        hs = HookState(hook_fns={hook_name: inject_hook}, cache=None)
        logits = model(tokens_a, hook_state=hs)

        last_logits = np.array(logits[pos])
        probs = np.exp(last_logits - np.max(last_logits))
        probs = probs / np.sum(probs)
        predictions.append(int(np.argmax(probs)))
        confidences.append(float(np.max(probs)))

    # Find boundaries (where prediction changes)
    boundaries = []
    for i in range(1, len(predictions)):
        if predictions[i] != predictions[i - 1]:
            boundaries.append(float(alphas[i]))

    # Boundary sharpness: min confidence near boundary
    sharpness = 1.0
    if boundaries:
        for b in boundaries:
            idx = int(b * (n_probes - 1))
            nearby_conf = confidences[max(0, idx - 1):min(len(confidences), idx + 2)]
            sharpness = min(sharpness, min(nearby_conf))

    return {
        "boundary_alphas": boundaries,
        "n_boundaries": len(boundaries),
        "prediction_sequence": predictions,
        "confidences": jnp.array(confidences),
        "boundary_sharpness": sharpness,
    }


def manifold_exploration(
    model,
    tokens,
    layer: int = -1,
    pos: int = -1,
    n_directions: int = 5,
    step_size: float = 0.5,
    n_steps: int = 5,
) -> dict:
    """Explore the activation manifold around a point.

    Takes random steps in principal directions to map out the
    local structure of the representation space.

    Args:
        model: HookedTransformer model.
        tokens: Input token array.
        layer: Layer to explore.
        pos: Position to explore.
        n_directions: Number of random directions.
        step_size: Size of each step (relative to activation norm).
        n_steps: Number of steps per direction.

    Returns:
        Dict with center_prediction, direction_effects, prediction_stability,
        effective_dimensionality.
    """
    from irtk.hook_points import HookState

    if layer < 0:
        layer = model.cfg.n_layers + layer
    hook_name = f"blocks.{layer}.hook_resid_post"

    cache = {}
    hs = HookState(hook_fns={}, cache=cache)
    logits = model(tokens, hook_state=hs)
    center_act = np.array(cache[hook_name][pos])
    act_norm = float(np.linalg.norm(center_act))

    center_logits = np.array(logits[pos])
    center_probs = np.exp(center_logits - np.max(center_logits))
    center_probs = center_probs / np.sum(center_probs)
    center_pred = int(np.argmax(center_probs))

    # Generate random directions
    rng = np.random.RandomState(42)
    directions = rng.randn(n_directions, len(center_act))
    for i in range(n_directions):
        directions[i] /= np.linalg.norm(directions[i]) + 1e-10

    direction_effects = []
    pred_changes = 0
    total_probes = 0

    for d_idx in range(n_directions):
        direction = directions[d_idx]
        effects = []

        for step in range(1, n_steps + 1):
            for sign in [-1, 1]:
                displaced = center_act + sign * step * step_size * act_norm * direction
                displaced_jnp = jnp.array(displaced)

                def inject_hook(x, name, _d=displaced_jnp):
                    return x.at[pos].set(_d)

                hs = HookState(hook_fns={hook_name: inject_hook}, cache=None)
                step_logits = model(tokens, hook_state=hs)
                step_logits_np = np.array(step_logits[pos])
                step_probs = np.exp(step_logits_np - np.max(step_logits_np))
                step_probs = step_probs / np.sum(step_probs)
                step_pred = int(np.argmax(step_probs))

                effects.append({
                    "step": sign * step,
                    "prediction": step_pred,
                    "confidence": float(np.max(step_probs)),
                    "changed": step_pred != center_pred,
                })

                total_probes += 1
                if step_pred != center_pred:
                    pred_changes += 1

        direction_effects.append(effects)

    # Prediction stability
    stability = 1.0 - pred_changes / max(total_probes, 1)

    return {
        "center_prediction": center_pred,
        "direction_effects": direction_effects,
        "prediction_stability": stability,
        "n_directions": n_directions,
        "n_prediction_changes": pred_changes,
        "total_probes": total_probes,
    }


def latent_distance_map(
    model,
    tokens_list: list,
    layer: int = -1,
    pos: int = -1,
) -> dict:
    """Compute pairwise distances between representations.

    Args:
        model: HookedTransformer model.
        tokens_list: List of token arrays.
        layer: Layer to analyze.
        pos: Position to analyze.

    Returns:
        Dict with distance_matrix, cosine_similarity_matrix, nearest_neighbors.
    """
    from irtk.hook_points import HookState

    if layer < 0:
        layer = model.cfg.n_layers + layer
    hook_name = f"blocks.{layer}.hook_resid_post"

    activations = []
    for tok in tokens_list:
        cache = {}
        hs = HookState(hook_fns={}, cache=cache)
        model(tok, hook_state=hs)
        activations.append(np.array(cache[hook_name][pos]))
    activations = np.stack(activations)

    n = len(activations)
    distances = np.zeros((n, n))
    cosines = np.zeros((n, n))

    norms = np.linalg.norm(activations, axis=1, keepdims=True) + 1e-10
    normed = activations / norms

    for i in range(n):
        for j in range(n):
            distances[i, j] = float(np.linalg.norm(activations[i] - activations[j]))
            cosines[i, j] = float(np.dot(normed[i], normed[j]))

    # Nearest neighbors
    nearest = []
    for i in range(n):
        dists = distances[i].copy()
        dists[i] = float('inf')
        nn = int(np.argmin(dists))
        nearest.append((i, nn, float(distances[i, nn])))

    return {
        "distance_matrix": jnp.array(distances),
        "cosine_similarity_matrix": jnp.array(cosines),
        "nearest_neighbors": nearest,
        "mean_distance": float(np.mean(distances[np.triu_indices(n, k=1)])),
        "mean_cosine": float(np.mean(cosines[np.triu_indices(n, k=1)])),
    }
