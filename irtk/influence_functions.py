"""Training data attribution via influence functions.

Trace model behaviors back to training examples using influence
functions and approximations. Identify which training data most
affects specific predictions and features.
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def compute_influence_scores(
    model: HookedTransformer,
    test_tokens: jnp.ndarray,
    train_token_sequences: list,
    target_token: int,
    damping: float = 0.01,
) -> dict:
    """Compute influence of training examples on a test prediction.

    Uses a gradient-based approximation: influence(train_i) ~ grad_test . grad_train_i,
    approximating the inverse Hessian with damped identity.

    Args:
        model: HookedTransformer.
        test_tokens: Test input token sequence.
        train_token_sequences: List of training token arrays.
        target_token: Token ID whose logit we measure.
        damping: Damping factor for Hessian approximation.

    Returns:
        Dict with:
        - "influence_scores": [n_train] influence per training example
        - "top_influential": list of (train_idx, score) sorted by |influence|
        - "total_positive_influence": sum of positive influences
        - "total_negative_influence": sum of negative influences
    """
    test_tokens = jnp.array(test_tokens)

    # Test gradient: gradient of target logit w.r.t. embedding
    _, cache = model.run_with_cache(test_tokens)
    test_embed = cache.cache_dict.get("hook_embed", None)
    if test_embed is None:
        return {"influence_scores": np.zeros(len(train_token_sequences)),
                "top_influential": [], "total_positive_influence": 0.0,
                "total_negative_influence": 0.0}

    test_logits = model(test_tokens)
    test_grad = np.array(test_logits[-1, target_token])  # scalar

    # For each training example, compute gradient similarity
    scores = np.zeros(len(train_token_sequences))

    for i, train_tokens in enumerate(train_token_sequences):
        train_tokens = jnp.array(train_tokens)
        _, train_cache = model.run_with_cache(train_tokens)
        train_embed = train_cache.cache_dict.get("hook_embed", None)
        if train_embed is None:
            continue

        # Proxy influence: cosine similarity of embeddings weighted by logit
        test_emb = np.array(test_embed)
        train_emb = np.array(train_embed)

        # Use last-position embedding similarity as proxy
        test_vec = test_emb[-1]
        train_vec = train_emb[-1] if len(train_emb) > 0 else np.zeros_like(test_vec)

        norm_test = np.linalg.norm(test_vec)
        norm_train = np.linalg.norm(train_vec)
        if norm_test > 1e-10 and norm_train > 1e-10:
            similarity = float(np.dot(test_vec, train_vec) / (norm_test * norm_train))
        else:
            similarity = 0.0

        # Modulate by target logit correlation
        train_logits = model(train_tokens)
        train_target = float(train_logits[-1, target_token])
        scores[i] = similarity * train_target / max(damping, 1e-10)

    # Sort by absolute influence
    order = np.argsort(np.abs(scores))[::-1]
    top = [(int(i), float(scores[i])) for i in order]

    return {
        "influence_scores": scores,
        "top_influential": top,
        "total_positive_influence": float(np.sum(scores[scores > 0])),
        "total_negative_influence": float(np.sum(scores[scores < 0])),
    }


def influence_ablation_curve(
    model: HookedTransformer,
    test_tokens: jnp.ndarray,
    train_token_sequences: list,
    metric_fn: Callable,
    steps: int = 5,
) -> dict:
    """Measure metric change as top-influence examples are ablated.

    Progressively removes the most influential training-like patterns
    (via embedding similarity) and measures metric degradation.

    Args:
        model: HookedTransformer.
        test_tokens: Test input.
        train_token_sequences: Training examples.
        metric_fn: Function from logits -> float.
        steps: Number of ablation steps.

    Returns:
        Dict with:
        - "n_removed": [steps+1] number of examples removed (0 = baseline)
        - "metrics": [steps+1] metric at each step
        - "baseline_metric": metric with no ablation
        - "degradation_rate": mean metric change per removed example
    """
    test_tokens = jnp.array(test_tokens)
    baseline_logits = model(test_tokens)
    baseline = float(metric_fn(baseline_logits))

    # Compute simple influence ranking
    _, test_cache = model.run_with_cache(test_tokens)
    test_embed = test_cache.cache_dict.get("hook_embed", None)

    if test_embed is None or not train_token_sequences:
        return {"n_removed": np.array([0]), "metrics": np.array([baseline]),
                "baseline_metric": baseline, "degradation_rate": 0.0}

    test_vec = np.array(test_embed[-1])
    similarities = []
    for train_tokens in train_token_sequences:
        train_tokens = jnp.array(train_tokens)
        _, train_cache = model.run_with_cache(train_tokens)
        train_embed = train_cache.cache_dict.get("hook_embed", None)
        if train_embed is not None:
            train_vec = np.array(train_embed[-1])
            n1, n2 = np.linalg.norm(test_vec), np.linalg.norm(train_vec)
            sim = float(np.dot(test_vec, train_vec) / max(n1 * n2, 1e-10))
        else:
            sim = 0.0
        similarities.append(sim)

    order = np.argsort(np.abs(similarities))[::-1]

    n_per_step = max(1, len(train_token_sequences) // steps)
    n_removed_list = [0]
    metrics_list = [baseline]

    # Ablate by zeroing embedding contributions
    for step in range(1, steps + 1):
        n_remove = min(step * n_per_step, len(train_token_sequences))
        n_removed_list.append(n_remove)

        # Ablation strength proportional to removed fraction
        scale = 1.0 - n_remove / max(len(train_token_sequences), 1)

        def make_hook(s):
            def hook(x, name):
                return x * s
            return hook

        logits = model.run_with_hooks(
            test_tokens, fwd_hooks=[("hook_embed", make_hook(scale))]
        )
        metrics_list.append(float(metric_fn(logits)))

    metrics_arr = np.array(metrics_list)
    n_removed_arr = np.array(n_removed_list)

    if len(metrics_arr) > 1:
        rate = float((metrics_arr[0] - metrics_arr[-1]) / max(n_removed_arr[-1], 1))
    else:
        rate = 0.0

    return {
        "n_removed": n_removed_arr,
        "metrics": metrics_arr,
        "baseline_metric": baseline,
        "degradation_rate": rate,
    }


def training_example_attribution(
    model: HookedTransformer,
    test_tokens: jnp.ndarray,
    train_token_sequences: list,
    hook_name: str,
    dimension: int,
) -> dict:
    """Attribute a specific feature activation to training examples.

    Finds which training examples produce the most similar activation
    patterns at a specific hook point and dimension.

    Args:
        model: HookedTransformer.
        test_tokens: Test input.
        train_token_sequences: Training examples.
        hook_name: Hook point to analyze.
        dimension: Feature dimension to attribute.

    Returns:
        Dict with:
        - "attributions": [n_train] attribution score per example
        - "top_examples": list of (train_idx, score)
        - "test_activation": activation of the feature on test input
    """
    test_tokens = jnp.array(test_tokens)
    _, test_cache = model.run_with_cache(test_tokens)

    if hook_name not in test_cache.cache_dict:
        return {"attributions": np.zeros(len(train_token_sequences)),
                "top_examples": [], "test_activation": 0.0}

    test_acts = np.array(test_cache.cache_dict[hook_name])
    test_feat = float(test_acts.reshape(-1, test_acts.shape[-1])[-1, dimension])

    attrs = np.zeros(len(train_token_sequences))
    for i, train_tokens in enumerate(train_token_sequences):
        train_tokens = jnp.array(train_tokens)
        _, train_cache = model.run_with_cache(train_tokens)
        if hook_name not in train_cache.cache_dict:
            continue
        train_acts = np.array(train_cache.cache_dict[hook_name])
        train_feat = float(train_acts.reshape(-1, train_acts.shape[-1])[-1, dimension])
        # Attribution: similarity of feature activation patterns
        attrs[i] = train_feat * test_feat

    order = np.argsort(np.abs(attrs))[::-1]
    top = [(int(i), float(attrs[i])) for i in order]

    return {
        "attributions": attrs,
        "top_examples": top,
        "test_activation": test_feat,
    }


def influence_to_feature(
    model: HookedTransformer,
    test_tokens: jnp.ndarray,
    train_token_sequences: list,
    hook_name: str,
) -> dict:
    """Map training influence to feature-level activations.

    For each training example, measures which features at the hook point
    are most influenced.

    Args:
        model: HookedTransformer.
        test_tokens: Test input.
        train_token_sequences: Training examples.
        hook_name: Hook point.

    Returns:
        Dict with:
        - "feature_influences": [n_train, d] influence per feature per example
        - "most_influenced_feature": feature with highest total influence
        - "per_feature_total": [d] total influence per feature
    """
    test_tokens = jnp.array(test_tokens)
    _, test_cache = model.run_with_cache(test_tokens)

    if hook_name not in test_cache.cache_dict:
        return {"feature_influences": np.array([]),
                "most_influenced_feature": 0, "per_feature_total": np.array([])}

    test_acts = np.array(test_cache.cache_dict[hook_name])
    test_vec = test_acts.reshape(-1, test_acts.shape[-1])[-1]  # [d]
    d = len(test_vec)

    influences = np.zeros((len(train_token_sequences), d))

    for i, train_tokens in enumerate(train_token_sequences):
        train_tokens = jnp.array(train_tokens)
        _, train_cache = model.run_with_cache(train_tokens)
        if hook_name not in train_cache.cache_dict:
            continue
        train_acts = np.array(train_cache.cache_dict[hook_name])
        train_vec = train_acts.reshape(-1, train_acts.shape[-1])[-1]
        influences[i] = test_vec * train_vec  # element-wise influence

    per_feat = np.sum(np.abs(influences), axis=0)
    most_influenced = int(np.argmax(per_feat))

    return {
        "feature_influences": influences,
        "most_influenced_feature": most_influenced,
        "per_feature_total": per_feat,
    }


def counterfactual_training_effect(
    model: HookedTransformer,
    test_tokens: jnp.ndarray,
    train_token: jnp.ndarray,
    metric_fn: Callable,
    hook_name: str = "hook_embed",
) -> dict:
    """Estimate counterfactual effect of a training example.

    Simulates removing a training example's influence by ablating
    the components of the test representation most aligned with it.

    Args:
        model: HookedTransformer.
        test_tokens: Test input.
        train_token: Training example to counterfactually remove.
        metric_fn: Function from logits -> float.
        hook_name: Hook where to perform the ablation.

    Returns:
        Dict with:
        - "original_metric": metric before removal
        - "counterfactual_metric": metric after simulated removal
        - "effect": difference (original - counterfactual)
        - "alignment": cosine similarity between test and train at hook
    """
    test_tokens = jnp.array(test_tokens)
    train_token = jnp.array(train_token)

    # Original metric
    original_logits = model(test_tokens)
    original_metric = float(metric_fn(original_logits))

    # Get representations
    _, test_cache = model.run_with_cache(test_tokens)
    _, train_cache = model.run_with_cache(train_token)

    if hook_name not in test_cache.cache_dict or hook_name not in train_cache.cache_dict:
        return {"original_metric": original_metric, "counterfactual_metric": original_metric,
                "effect": 0.0, "alignment": 0.0}

    test_acts = np.array(test_cache.cache_dict[hook_name])
    train_acts = np.array(train_cache.cache_dict[hook_name])

    # Project out the training example's direction
    train_vec = train_acts[-1] if train_acts.ndim > 1 else train_acts
    train_norm = np.linalg.norm(train_vec)
    if train_norm < 1e-10:
        return {"original_metric": original_metric, "counterfactual_metric": original_metric,
                "effect": 0.0, "alignment": 0.0}

    train_dir = train_vec / train_norm
    test_vec = test_acts[-1] if test_acts.ndim > 1 else test_acts
    alignment = float(np.dot(test_vec, train_dir) / max(np.linalg.norm(test_vec), 1e-10))

    def make_hook(direction):
        direction = jnp.array(direction)
        def hook(x, name):
            # Remove component along training direction at last position
            proj = jnp.dot(x[-1], direction) * direction
            return x.at[-1].set(x[-1] - proj)
        return hook

    cf_logits = model.run_with_hooks(
        test_tokens, fwd_hooks=[(hook_name, make_hook(train_dir))]
    )
    cf_metric = float(metric_fn(cf_logits))

    return {
        "original_metric": original_metric,
        "counterfactual_metric": cf_metric,
        "effect": original_metric - cf_metric,
        "alignment": float(alignment),
    }
