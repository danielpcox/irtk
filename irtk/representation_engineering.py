"""Representation Engineering (RepE) tools.

Extract reading vectors via PCA of contrastive pair activations,
apply control vector interventions, and score concept presence
across layers. Based on Zou et al. 2023.
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def extract_reading_vectors(
    model: HookedTransformer,
    positive_prompts: list,
    negative_prompts: list,
    hook_name: str,
    n_components: int = 1,
    pos: int = -1,
) -> dict:
    """Extract reading vectors for a concept via contrastive PCA.

    Runs positive and negative prompts through the model, collects
    activations at the specified hook, and finds the principal direction(s)
    separating them.

    Args:
        model: HookedTransformer.
        positive_prompts: Token sequences exhibiting the concept.
        negative_prompts: Token sequences not exhibiting the concept.
        hook_name: Hook point to extract activations from.
        n_components: Number of PCA components to extract.
        pos: Token position to use (-1 for last).

    Returns:
        Dict with:
        - "reading_vectors": [n_components, d_model] principal directions
        - "explained_variance": [n_components] variance explained per component
        - "mean_positive": [d_model] mean positive activation
        - "mean_negative": [d_model] mean negative activation
        - "separation_score": cosine similarity between mean diff and top PC
    """
    pos_acts = []
    neg_acts = []

    for tokens in positive_prompts:
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)
        if hook_name in cache.cache_dict:
            act = np.array(cache.cache_dict[hook_name])
            resolved = pos if pos >= 0 else act.shape[0] + pos
            if 0 <= resolved < act.shape[0]:
                pos_acts.append(act[resolved])

    for tokens in negative_prompts:
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)
        if hook_name in cache.cache_dict:
            act = np.array(cache.cache_dict[hook_name])
            resolved = pos if pos >= 0 else act.shape[0] + pos
            if 0 <= resolved < act.shape[0]:
                neg_acts.append(act[resolved])

    if not pos_acts or not neg_acts:
        d = model.cfg.d_model
        return {
            "reading_vectors": np.zeros((n_components, d)),
            "explained_variance": np.zeros(n_components),
            "mean_positive": np.zeros(d),
            "mean_negative": np.zeros(d),
            "separation_score": 0.0,
        }

    pos_arr = np.array(pos_acts)
    neg_arr = np.array(neg_acts)

    mean_pos = np.mean(pos_arr, axis=0)
    mean_neg = np.mean(neg_arr, axis=0)

    # PCA on the difference vectors
    all_acts = np.concatenate([pos_arr, neg_arr], axis=0)
    mean_all = np.mean(all_acts, axis=0)
    centered = all_acts - mean_all

    # Covariance and SVD
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    try:
        U, S, Vt = np.linalg.svd(cov, full_matrices=False)
        reading_vecs = Vt[:n_components]
        total_var = np.sum(S)
        explained = S[:n_components] / max(total_var, 1e-10)
    except np.linalg.LinAlgError:
        d = model.cfg.d_model
        reading_vecs = np.zeros((n_components, d))
        explained = np.zeros(n_components)

    # Ensure reading vector points from negative to positive
    diff = mean_pos - mean_neg
    for i in range(min(n_components, len(reading_vecs))):
        if np.dot(reading_vecs[i], diff) < 0:
            reading_vecs[i] = -reading_vecs[i]

    # Separation score
    diff_norm = np.linalg.norm(diff)
    rv_norm = np.linalg.norm(reading_vecs[0]) if len(reading_vecs) > 0 else 0
    if diff_norm > 1e-10 and rv_norm > 1e-10:
        sep = float(np.dot(reading_vecs[0], diff) / (rv_norm * diff_norm))
    else:
        sep = 0.0

    return {
        "reading_vectors": reading_vecs,
        "explained_variance": explained,
        "mean_positive": mean_pos,
        "mean_negative": mean_neg,
        "separation_score": float(sep),
    }


def reading_vector_scan(
    model: HookedTransformer,
    positive_prompts: list,
    negative_prompts: list,
    pos: int = -1,
) -> dict:
    """Extract reading vectors across all layers.

    Produces a layer-by-layer profile of where a concept is encoded.

    Args:
        model: HookedTransformer.
        positive_prompts: Token sequences exhibiting the concept.
        negative_prompts: Token sequences not exhibiting it.
        pos: Token position to use.

    Returns:
        Dict with:
        - "reading_vectors": [n_layers, d_model] direction per layer
        - "separation_scores": [n_layers] cosine alignment per layer
        - "explained_variances": [n_layers] top-PC variance per layer
        - "best_layer": layer with highest separation
    """
    n_layers = model.cfg.n_layers
    d = model.cfg.d_model

    vectors = np.zeros((n_layers, d))
    scores = np.zeros(n_layers)
    variances = np.zeros(n_layers)

    for layer in range(n_layers):
        hook = f"blocks.{layer}.hook_resid_post"
        result = extract_reading_vectors(
            model, positive_prompts, negative_prompts, hook, n_components=1, pos=pos
        )
        vectors[layer] = result["reading_vectors"][0]
        scores[layer] = result["separation_score"]
        variances[layer] = result["explained_variance"][0]

    return {
        "reading_vectors": vectors,
        "separation_scores": scores,
        "explained_variances": variances,
        "best_layer": int(np.argmax(np.abs(scores))),
    }


def control_vector_intervention(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    reading_vectors: dict,
    coefficient: float = 1.0,
) -> dict:
    """Apply reading vectors as control vectors to steer model behavior.

    Args:
        model: HookedTransformer.
        tokens: Token sequence.
        reading_vectors: Dict mapping hook_name -> [d_model] direction vector.
        coefficient: Scaling factor for the intervention.

    Returns:
        Dict with:
        - "original_logits": logits without intervention
        - "steered_logits": logits after steering
        - "logit_diff": steered - original at last position
        - "top_changed_tokens": list of (token_idx, logit_change) for most affected
    """
    tokens = jnp.array(tokens)
    original_logits = np.array(model(tokens))

    # Build hooks
    def make_steer_hook(direction, coeff):
        direction = jnp.array(direction)
        def hook(x, name):
            return x + coeff * direction
        return hook

    fwd_hooks = []
    for hook_name, vec in reading_vectors.items():
        fwd_hooks.append((hook_name, make_steer_hook(vec, coefficient)))

    steered_logits = np.array(model.run_with_hooks(tokens, fwd_hooks=fwd_hooks))

    diff = steered_logits[-1] - original_logits[-1]
    top_changed = np.argsort(np.abs(diff))[::-1][:20]
    top_list = [(int(i), float(diff[i])) for i in top_changed]

    return {
        "original_logits": original_logits,
        "steered_logits": steered_logits,
        "logit_diff": diff,
        "top_changed_tokens": top_list,
    }


def representation_score(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    reading_vector: np.ndarray,
    hook_name: str,
) -> dict:
    """Project activations onto a reading vector for concept presence scoring.

    Args:
        model: HookedTransformer.
        tokens: Token sequence.
        reading_vector: [d_model] direction to project onto.
        hook_name: Hook point to analyze.

    Returns:
        Dict with:
        - "scores": [seq_len] concept presence score per position
        - "mean_score": average score across positions
        - "max_score": maximum score
        - "max_pos": position with highest score
    """
    tokens = jnp.array(tokens)
    _, cache = model.run_with_cache(tokens)

    if hook_name not in cache.cache_dict:
        return {"scores": np.array([]), "mean_score": 0.0, "max_score": 0.0, "max_pos": 0}

    acts = np.array(cache.cache_dict[hook_name])  # [seq_len, d_model]
    rv = reading_vector / max(np.linalg.norm(reading_vector), 1e-10)

    if acts.ndim == 1:
        scores = np.array([float(np.dot(acts, rv))])
    else:
        scores = acts @ rv  # [seq_len]

    return {
        "scores": scores,
        "mean_score": float(np.mean(scores)),
        "max_score": float(np.max(scores)),
        "max_pos": int(np.argmax(scores)),
    }


def concept_suppression_curve(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    reading_vectors: dict,
    metric_fn: Callable,
    coefficients: Optional[list[float]] = None,
) -> dict:
    """Sweep control vector coefficient and measure metric response.

    Produces a curve showing how the metric changes as the concept
    is amplified or suppressed.

    Args:
        model: HookedTransformer.
        tokens: Token sequence.
        reading_vectors: Dict mapping hook_name -> [d_model] direction.
        metric_fn: Function from logits -> float.
        coefficients: List of coefficient values to try.

    Returns:
        Dict with:
        - "coefficients": [n] coefficient values
        - "metrics": [n] metric values at each coefficient
        - "baseline_metric": metric at coefficient=0
        - "sensitivity": absolute change per unit coefficient (linear approx)
    """
    if coefficients is None:
        coefficients = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]

    tokens = jnp.array(tokens)
    metrics = []

    for coeff in coefficients:
        if abs(coeff) < 1e-10:
            logits = model(tokens)
        else:
            def make_hook(direction, c):
                direction = jnp.array(direction)
                def hook(x, name):
                    return x + c * direction
                return hook

            fwd_hooks = [(hook_name, make_hook(vec, coeff))
                         for hook_name, vec in reading_vectors.items()]
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        metrics.append(float(metric_fn(logits)))

    baseline_idx = None
    for i, c in enumerate(coefficients):
        if abs(c) < 1e-10:
            baseline_idx = i
            break
    baseline = metrics[baseline_idx] if baseline_idx is not None else metrics[len(metrics) // 2]

    # Sensitivity: linear approx of |dmetric/dcoeff|
    if len(coefficients) >= 2:
        diffs = np.diff(metrics) / np.diff(coefficients)
        sensitivity = float(np.mean(np.abs(diffs)))
    else:
        sensitivity = 0.0

    return {
        "coefficients": np.array(coefficients),
        "metrics": np.array(metrics),
        "baseline_metric": float(baseline),
        "sensitivity": sensitivity,
    }
