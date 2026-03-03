"""Developmental interpretability: how circuits form during training.

Analyzes model checkpoints across training to understand when and how
interpretable structures crystallize. Tracks phase transitions,
circuit formation order, and representation crystallization.

Functions:
- detect_phase_transitions: Find discontinuous structural changes across checkpoints
- track_circuit_formation: Measure when a specific circuit becomes functional
- measure_representation_crystallization: Track representation structure formation
- compare_learning_order: Determine temporal ordering of circuit acquisition
- grokking_detector: Detect delayed generalization (grokking) phenomena

References:
    - Hoogland et al. (2024) "Developmental Interpretability"
    - Olsson et al. (2022) "In-context Learning and Induction Heads" (phase transitions)
    - Power et al. (2022) "Grokking: Generalization Beyond Overfitting"
"""

from typing import Optional, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def detect_phase_transitions(
    checkpoints: Sequence[HookedTransformer],
    tokens: jnp.ndarray,
    hook_names: Optional[list] = None,
) -> dict:
    """Find discontinuous structural changes across training checkpoints.

    Computes representational similarity (CKA-like) between consecutive
    checkpoints and identifies jumps indicating qualitative reorganization.

    Args:
        checkpoints: Sequence of model checkpoints in training order.
        tokens: [seq_len] input tokens for evaluation.
        hook_names: Hook names to compare. Defaults to all resid_post.

    Returns:
        Dict with:
            "similarity_curve": [n_checkpoints-1] similarity between consecutive checkpoints
            "transition_points": indices where similarity drops sharply
            "transition_magnitudes": magnitude of each transition
            "smoothness": overall smoothness of training trajectory
    """
    n = len(checkpoints)
    if n < 2:
        return {
            "similarity_curve": np.array([]),
            "transition_points": [],
            "transition_magnitudes": [],
            "smoothness": 1.0,
        }

    if hook_names is None:
        hook_names = [f"blocks.{l}.hook_resid_post" for l in range(checkpoints[0].cfg.n_layers)]

    # Collect activations for each checkpoint
    def get_activations(model):
        _, cache = model.run_with_cache(tokens)
        acts = []
        for name in hook_names:
            if name in cache.cache_dict:
                acts.append(np.array(cache.cache_dict[name]).flatten())
        return np.concatenate(acts) if acts else np.zeros(1)

    act_list = [get_activations(m) for m in checkpoints]

    # Similarity between consecutive checkpoints (cosine)
    similarities = []
    for i in range(n - 1):
        a, b = act_list[i], act_list[i + 1]
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        similarities.append(float(sim))

    similarities = np.array(similarities)

    # Detect transitions: significant drops in similarity
    if len(similarities) >= 2:
        diffs = np.diff(similarities)
        mean_diff = np.mean(np.abs(diffs))
        threshold = mean_diff * 2
        transitions = [int(i) for i in range(len(diffs)) if abs(diffs[i]) > threshold]
        magnitudes = [float(abs(diffs[i])) for i in transitions]
    else:
        transitions = []
        magnitudes = []

    smoothness = float(1.0 - np.std(similarities)) if len(similarities) > 0 else 1.0

    return {
        "similarity_curve": similarities,
        "transition_points": transitions,
        "transition_magnitudes": magnitudes,
        "smoothness": max(0.0, smoothness),
    }


def track_circuit_formation(
    checkpoints: Sequence[HookedTransformer],
    tokens: jnp.ndarray,
    circuit_hooks: list,
    metric_fn: Callable,
) -> dict:
    """Measure when a specific circuit becomes functional across training.

    Ablates the specified circuit components at each checkpoint and measures
    the metric effect, tracking when the circuit "turns on."

    Args:
        checkpoints: Sequence of model checkpoints.
        tokens: [seq_len] input tokens.
        circuit_hooks: List of (hook_name, hook_fn) pairs defining the circuit ablation.
        metric_fn: Function(logits) -> float.

    Returns:
        Dict with:
            "clean_metrics": [n_checkpoints] metric values without ablation
            "ablated_metrics": [n_checkpoints] metric values with circuit ablated
            "circuit_effects": [n_checkpoints] absolute metric change from ablation
            "formation_checkpoint": first checkpoint where circuit effect > 50% of final
            "formation_curve": [n_checkpoints] normalized circuit importance (0 to 1)
    """
    n = len(checkpoints)
    clean = np.zeros(n)
    ablated = np.zeros(n)

    for i, model in enumerate(checkpoints):
        clean_logits = model(tokens)
        clean[i] = float(metric_fn(clean_logits))

        try:
            abl_logits = model.run_with_hooks(tokens, fwd_hooks=circuit_hooks)
            ablated[i] = float(metric_fn(abl_logits))
        except Exception:
            ablated[i] = clean[i]

    effects = np.abs(clean - ablated)
    final_effect = effects[-1] if n > 0 else 0.0

    # Normalize to [0, 1] for formation curve
    if final_effect > 1e-10:
        formation = effects / final_effect
    else:
        formation = np.zeros(n)

    # Formation checkpoint: first where effect > 50% of final
    threshold = 0.5
    formation_idx = -1
    for i, f in enumerate(formation):
        if f >= threshold:
            formation_idx = i
            break

    return {
        "clean_metrics": clean,
        "ablated_metrics": ablated,
        "circuit_effects": effects,
        "formation_checkpoint": formation_idx,
        "formation_curve": formation,
    }


def measure_representation_crystallization(
    checkpoints: Sequence[HookedTransformer],
    tokens: jnp.ndarray,
    hook_name: Optional[str] = None,
) -> dict:
    """Track representation structure formation across training.

    Measures intrinsic dimensionality and clustering structure of
    intermediate representations at each checkpoint.

    Args:
        checkpoints: Sequence of model checkpoints.
        tokens: [seq_len] input tokens.
        hook_name: Hook name to analyze. Defaults to last layer resid_post.

    Returns:
        Dict with:
            "effective_ranks": [n_checkpoints] effective rank of representation matrix
            "condition_numbers": [n_checkpoints] condition number of representation
            "crystallization_index": rate of rank decrease (more structured = lower rank)
            "most_structured_checkpoint": checkpoint with lowest effective rank
    """
    n = len(checkpoints)

    if hook_name is None:
        hook_name = f"blocks.{checkpoints[0].cfg.n_layers - 1}.hook_resid_post"

    ranks = np.zeros(n)
    cond_nums = np.zeros(n)

    for i, model in enumerate(checkpoints):
        _, cache = model.run_with_cache(tokens)
        if hook_name in cache.cache_dict:
            act = np.array(cache.cache_dict[hook_name])  # [seq, d_model]
            s = np.linalg.svd(act, compute_uv=False)
            s_norm = s / (np.sum(s) + 1e-10)
            s_norm = s_norm[s_norm > 1e-10]
            ranks[i] = float(np.exp(-np.sum(s_norm * np.log(s_norm + 1e-10))))
            cond_nums[i] = float(s[0] / (s[-1] + 1e-10)) if len(s) > 0 else 1.0

    # Crystallization: how much rank decreased over training
    if n >= 2 and ranks[0] > 1e-10:
        crystallization = float((ranks[0] - ranks[-1]) / ranks[0])
    else:
        crystallization = 0.0

    most_structured = int(np.argmin(ranks)) if n > 0 else 0

    return {
        "effective_ranks": ranks,
        "condition_numbers": cond_nums,
        "crystallization_index": crystallization,
        "most_structured_checkpoint": most_structured,
    }


def compare_learning_order(
    checkpoints: Sequence[HookedTransformer],
    tokens: jnp.ndarray,
    circuit_specs: dict,
    metric_fn: Callable,
) -> dict:
    """Determine temporal ordering of circuit acquisition during training.

    Given multiple circuit specifications, measures when each becomes
    functional and produces a learning order.

    Args:
        checkpoints: Sequence of model checkpoints.
        tokens: [seq_len] input tokens.
        circuit_specs: Dict mapping circuit_name -> list of (hook_name, hook_fn).
        metric_fn: Function(logits) -> float.

    Returns:
        Dict with:
            "formation_order": list of (circuit_name, formation_checkpoint) sorted by timing
            "per_circuit_curves": dict mapping circuit_name -> [n_checkpoints] effect curve
            "earliest_circuit": name of first circuit to form
            "latest_circuit": name of last circuit to form
    """
    results = {}
    for name, hooks in circuit_specs.items():
        r = track_circuit_formation(checkpoints, tokens, hooks, metric_fn)
        results[name] = r

    # Sort by formation checkpoint
    order = []
    for name, r in results.items():
        order.append((name, r["formation_checkpoint"]))

    # Sort by formation time (-1 means never formed, put last)
    order.sort(key=lambda x: x[1] if x[1] >= 0 else float('inf'))

    curves = {name: r["circuit_effects"] for name, r in results.items()}

    earliest = order[0][0] if order else ""
    latest = order[-1][0] if order else ""

    return {
        "formation_order": order,
        "per_circuit_curves": curves,
        "earliest_circuit": earliest,
        "latest_circuit": latest,
    }


def grokking_detector(
    checkpoints: Sequence[HookedTransformer],
    train_tokens: jnp.ndarray,
    test_tokens: jnp.ndarray,
    metric_fn: Callable,
    gap_threshold: float = 0.3,
) -> dict:
    """Detect delayed generalization (grokking) across checkpoints.

    Monitors the gap between train and test performance, combined with
    weight norm tracking, to detect grokking phenomena.

    Args:
        checkpoints: Sequence of model checkpoints.
        train_tokens: [seq_len] training input tokens.
        test_tokens: [seq_len] test input tokens.
        metric_fn: Function(logits) -> float.
        gap_threshold: Min train-test gap to count as memorization phase.

    Returns:
        Dict with:
            "train_metrics": [n_checkpoints] metric on training data
            "test_metrics": [n_checkpoints] metric on test data
            "generalization_gap": [n_checkpoints] train - test metric
            "grokking_detected": whether grokking pattern was found
            "grokking_checkpoint": checkpoint where generalization catches up (-1 if not)
            "weight_norms": [n_checkpoints] total model weight norm
    """
    n = len(checkpoints)
    train_m = np.zeros(n)
    test_m = np.zeros(n)
    w_norms = np.zeros(n)

    for i, model in enumerate(checkpoints):
        train_logits = model(train_tokens)
        test_logits = model(test_tokens)
        train_m[i] = float(metric_fn(train_logits))
        test_m[i] = float(metric_fn(test_logits))

        # Weight norms
        leaves = jax.tree.leaves(model)
        total_norm = 0.0
        for leaf in leaves:
            if isinstance(leaf, jnp.ndarray) and leaf.dtype in (jnp.float32, jnp.float16, jnp.bfloat16):
                total_norm += float(jnp.sum(leaf ** 2))
        w_norms[i] = float(np.sqrt(total_norm))

    gap = train_m - test_m

    # Detect grokking: memorization phase (gap > threshold) followed by generalization (gap < threshold)
    in_memorization = False
    grokking_detected = False
    grok_checkpoint = -1

    for i in range(n):
        if gap[i] > gap_threshold:
            in_memorization = True
        elif in_memorization and gap[i] < gap_threshold:
            grokking_detected = True
            grok_checkpoint = i
            break

    return {
        "train_metrics": train_m,
        "test_metrics": test_m,
        "generalization_gap": gap,
        "grokking_detected": grokking_detected,
        "grokking_checkpoint": grok_checkpoint,
        "weight_norms": w_norms,
    }
