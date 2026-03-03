"""Mechanistic analysis of in-context learning.

Tools for understanding how transformers perform in-context learning:
task vector extraction, ICL head identification, implicit gradient
descent testing, label sensitivity, and demonstration order effects.

References:
    - Olsson et al. (2022) "In-context Learning and Induction Heads"
    - Akyurek et al. (2023) "What Learning Algorithm Is In-Context Learning?"
    - Hendel et al. (2023) "In-Context Learning Creates Task Vectors"
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def extract_task_vector(
    model: HookedTransformer,
    demonstration_tokens: jnp.ndarray,
    query_tokens: jnp.ndarray,
    hook_name: str,
    pos: int = -1,
) -> dict:
    """Extract the task vector encoded by ICL demonstrations.

    The task vector is the difference in the query position's residual
    stream between running with demonstrations vs. without.

    Args:
        model: HookedTransformer.
        demonstration_tokens: Full prompt including demonstrations and query.
        query_tokens: Query-only prompt (no demonstrations).
        hook_name: Hook point to extract from.
        pos: Position to extract (-1 = last).

    Returns:
        Dict with:
        - "task_vector": [d] vector encoding the ICL task
        - "task_vector_norm": L2 norm of the task vector
        - "baseline_norm": norm of the query-only representation
        - "relative_magnitude": task_vector_norm / baseline_norm
    """
    demo_tokens = jnp.array(demonstration_tokens)
    q_tokens = jnp.array(query_tokens)

    _, cache_demo = model.run_with_cache(demo_tokens)
    _, cache_query = model.run_with_cache(q_tokens)

    if hook_name not in cache_demo.cache_dict or hook_name not in cache_query.cache_dict:
        d = model.cfg.d_model
        return {"task_vector": np.zeros(d), "task_vector_norm": 0.0,
                "baseline_norm": 0.0, "relative_magnitude": 0.0}

    demo_acts = np.array(cache_demo.cache_dict[hook_name])
    query_acts = np.array(cache_query.cache_dict[hook_name])

    # Get the representation at the query position
    if demo_acts.ndim > 1:
        demo_vec = demo_acts[pos]
        query_vec = query_acts[pos]
    else:
        demo_vec = demo_acts
        query_vec = query_acts

    # Ensure same dimension
    min_d = min(len(demo_vec), len(query_vec))
    demo_vec = demo_vec[:min_d]
    query_vec = query_vec[:min_d]

    task_vec = demo_vec - query_vec
    tv_norm = float(np.linalg.norm(task_vec))
    q_norm = float(np.linalg.norm(query_vec))

    return {
        "task_vector": task_vec,
        "task_vector_norm": tv_norm,
        "baseline_norm": q_norm,
        "relative_magnitude": tv_norm / max(q_norm, 1e-10),
    }


def icl_head_identification(
    model: HookedTransformer,
    demonstration_tokens: jnp.ndarray,
    query_tokens: jnp.ndarray,
    metric_fn: Callable,
) -> dict:
    """Identify which attention heads drive in-context learning.

    Uses activation patching: for each head, replace its output in the
    full-prompt run with its output from the query-only run, measuring
    the metric change.

    Args:
        model: HookedTransformer.
        demonstration_tokens: Full prompt with demonstrations.
        query_tokens: Query-only prompt.
        metric_fn: Function from logits -> float.

    Returns:
        Dict with:
        - "head_icl_scores": [n_layers, n_heads] ICL importance per head
        - "top_icl_heads": list of (layer, head, score) sorted by importance
        - "total_icl_effect": sum of all head scores
        - "layer_icl_profile": [n_layers] summed ICL score per layer
    """
    demo_tokens = jnp.array(demonstration_tokens)
    q_tokens = jnp.array(query_tokens)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Baseline: metric with full demonstrations
    demo_logits = model(demo_tokens)
    baseline = float(metric_fn(demo_logits))

    # Query-only metric
    query_logits = model(q_tokens)
    query_metric = float(metric_fn(query_logits))

    # Get query-only attention outputs for patching
    _, query_cache = model.run_with_cache(q_tokens)

    scores = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        hook = f"blocks.{layer}.attn.hook_result"
        if hook not in query_cache.cache_dict:
            continue

        query_result = query_cache.cache_dict[hook]

        for head in range(n_heads):
            # Patch this head's output with query-only version
            def make_patch(h, q_res):
                q_res_np = np.array(q_res)
                def patch(x, name):
                    # x is [n_heads, seq, d_head] or similar
                    if x.ndim >= 2 and h < x.shape[0]:
                        min_s = min(x.shape[1], q_res_np.shape[1]) if x.ndim > 2 else x.shape[-1]
                        if x.ndim == 3 and q_res_np.ndim == 3:
                            min_seq = min(x.shape[1], q_res_np.shape[1])
                            min_d = min(x.shape[2], q_res_np.shape[2])
                            return x.at[h, :min_seq, :min_d].set(
                                jnp.array(q_res_np[h, :min_seq, :min_d])
                            )
                    return x
                return patch

            patched_logits = model.run_with_hooks(
                demo_tokens, fwd_hooks=[(hook, make_patch(head, query_result))]
            )
            patched_metric = float(metric_fn(patched_logits))
            scores[layer, head] = abs(baseline - patched_metric)

    # Build sorted list
    top_heads = []
    for l in range(n_layers):
        for h in range(n_heads):
            top_heads.append((l, h, float(scores[l, h])))
    top_heads.sort(key=lambda x: x[2], reverse=True)

    layer_profile = np.sum(scores, axis=1)

    return {
        "head_icl_scores": scores,
        "top_icl_heads": top_heads,
        "total_icl_effect": float(np.sum(scores)),
        "layer_icl_profile": layer_profile,
    }


def implicit_gradient_descent_test(
    model: HookedTransformer,
    demonstration_tokens: jnp.ndarray,
    query_tokens: jnp.ndarray,
    layer: int,
) -> dict:
    """Test whether a transformer layer implements something like gradient descent.

    Compares the attention-mediated update at a layer with the direction
    from query-only to demonstration-augmented representations.

    Args:
        model: HookedTransformer.
        demonstration_tokens: Full prompt with demonstrations.
        query_tokens: Query-only prompt.
        layer: Layer to test.

    Returns:
        Dict with:
        - "alignment_score": cosine between attention update and GD direction
        - "attention_update_norm": norm of the attention contribution
        - "gd_direction_norm": norm of the representation shift
        - "layer_tested": layer index
    """
    demo_tokens = jnp.array(demonstration_tokens)
    q_tokens = jnp.array(query_tokens)

    _, demo_cache = model.run_with_cache(demo_tokens)
    _, query_cache = model.run_with_cache(q_tokens)

    # Attention output at this layer (the "update")
    attn_hook = f"blocks.{layer}.hook_attn_out"
    resid_pre = f"blocks.{layer}.hook_resid_pre"
    resid_post = f"blocks.{layer}.hook_resid_post"

    d = model.cfg.d_model
    zero = {"alignment_score": 0.0, "attention_update_norm": 0.0,
            "gd_direction_norm": 0.0, "layer_tested": layer}

    if attn_hook not in demo_cache.cache_dict:
        return zero

    # Attention update in the demo run
    attn_out = np.array(demo_cache.cache_dict[attn_hook])
    attn_vec = attn_out[-1] if attn_out.ndim > 1 else attn_out

    # "GD direction": how the residual stream shifts from query-only to demo
    if resid_post in demo_cache.cache_dict and resid_post in query_cache.cache_dict:
        demo_resid = np.array(demo_cache.cache_dict[resid_post])
        query_resid = np.array(query_cache.cache_dict[resid_post])
        demo_vec = demo_resid[-1] if demo_resid.ndim > 1 else demo_resid
        query_vec = query_resid[-1] if query_resid.ndim > 1 else query_resid
        min_d = min(len(demo_vec), len(query_vec))
        gd_dir = demo_vec[:min_d] - query_vec[:min_d]
    else:
        return zero

    attn_vec = attn_vec[:min_d]
    n_attn = float(np.linalg.norm(attn_vec))
    n_gd = float(np.linalg.norm(gd_dir))

    if n_attn > 1e-10 and n_gd > 1e-10:
        alignment = float(np.dot(attn_vec, gd_dir) / (n_attn * n_gd))
    else:
        alignment = 0.0

    return {
        "alignment_score": alignment,
        "attention_update_norm": n_attn,
        "gd_direction_norm": n_gd,
        "layer_tested": layer,
    }


def icl_label_sensitivity(
    model: HookedTransformer,
    clean_tokens: jnp.ndarray,
    metric_fn: Callable,
    corruption_positions: list,
    vocab_size: Optional[int] = None,
    n_corruptions: int = 5,
    seed: int = 42,
) -> dict:
    """Measure how sensitive ICL is to corrupting specific token positions.

    Replaces tokens at corruption_positions with random tokens and
    measures metric change. High sensitivity suggests genuine ICL.

    Args:
        model: HookedTransformer.
        clean_tokens: Full prompt with correct demonstrations.
        metric_fn: Function from logits -> float.
        corruption_positions: Token positions to corrupt (e.g., label positions).
        vocab_size: Vocab size for random replacements (inferred if None).
        n_corruptions: Number of random corruptions to average over.
        seed: Random seed.

    Returns:
        Dict with:
        - "clean_metric": metric with original tokens
        - "mean_corrupted_metric": average metric under corruption
        - "sensitivity_score": |clean - corrupted| / max(|clean|, 1e-10)
        - "corrupted_metrics": [n_corruptions] individual corrupted metrics
    """
    clean = jnp.array(clean_tokens)
    v = vocab_size or model.cfg.d_vocab

    clean_logits = model(clean)
    clean_metric = float(metric_fn(clean_logits))

    rng = np.random.RandomState(seed)
    corrupted_metrics = []

    for _ in range(n_corruptions):
        corrupted = np.array(clean)
        for pos in corruption_positions:
            if 0 <= pos < len(corrupted):
                corrupted[pos] = rng.randint(0, v)
        c_logits = model(jnp.array(corrupted))
        corrupted_metrics.append(float(metric_fn(c_logits)))

    mean_corr = float(np.mean(corrupted_metrics))
    sensitivity = abs(clean_metric - mean_corr) / max(abs(clean_metric), 1e-10)

    return {
        "clean_metric": clean_metric,
        "mean_corrupted_metric": mean_corr,
        "sensitivity_score": float(sensitivity),
        "corrupted_metrics": np.array(corrupted_metrics),
    }


def demonstration_order_effect(
    model: HookedTransformer,
    demonstration_chunks: list,
    query_tokens: jnp.ndarray,
    metric_fn: Callable,
    n_shuffles: int = 10,
    seed: int = 42,
) -> dict:
    """Measure variance in ICL performance across demonstration orderings.

    Args:
        model: HookedTransformer.
        demonstration_chunks: List of token arrays, one per demonstration.
        query_tokens: Query tokens appended after demonstrations.
        metric_fn: Function from logits -> float.
        n_shuffles: Number of random orderings.
        seed: Random seed.

    Returns:
        Dict with:
        - "mean_metric": mean metric across orderings
        - "std_metric": standard deviation
        - "min_metric": worst ordering metric
        - "max_metric": best ordering metric
        - "order_sensitivity": std / max(|mean|, 1e-10)
    """
    q = jnp.array(query_tokens)
    rng = np.random.RandomState(seed)
    n_demos = len(demonstration_chunks)

    metrics = []

    for _ in range(n_shuffles):
        order = rng.permutation(n_demos)
        # Concatenate demos in this order + query
        parts = [np.array(demonstration_chunks[i]) for i in order]
        parts.append(np.array(q))
        full = jnp.array(np.concatenate(parts))

        # Truncate if too long
        if len(full) > model.cfg.n_ctx:
            full = full[:model.cfg.n_ctx]

        logits = model(full)
        metrics.append(float(metric_fn(logits)))

    metrics_arr = np.array(metrics)
    mean_m = float(np.mean(metrics_arr))
    std_m = float(np.std(metrics_arr))

    return {
        "mean_metric": mean_m,
        "std_metric": std_m,
        "min_metric": float(np.min(metrics_arr)),
        "max_metric": float(np.max(metrics_arr)),
        "order_sensitivity": std_m / max(abs(mean_m), 1e-10),
    }
