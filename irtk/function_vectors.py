"""Function vector analysis.

Implements the Todd et al. (2023) function vector framework. A function vector
is a single d_model vector extractable from a few-shot prompt by reading a
specific attention head's output. It causally encodes an abstract task and
can transfer that task to new contexts.

Functions:
- extract_function_vector: Extract a function vector from a few-shot prompt
- scan_for_function_heads: Identify heads that encode function vectors
- function_vector_transfer: Inject a function vector into a new context
- function_vector_arithmetic: Test composition of function vectors
- function_vector_similarity_matrix: Compare function vectors across tasks

References:
    - Todd et al. (2023) "Function Vectors in Large Language Models"
    - Hendel et al. (2023) "In-Context Learning Creates Task Vectors"
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def extract_function_vector(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
    head: int,
    pos: int = -1,
) -> dict:
    """Extract a function vector from a few-shot prompt.

    Reads the output of a specific attention head at the final query
    position in a few-shot prompt to extract the function vector.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] few-shot prompt tokens.
        layer: Layer of the function head.
        head: Head index of the function head.
        pos: Position to read from (-1 = last).

    Returns:
        Dict with:
            "function_vector": [d_model] the extracted function vector
            "fv_norm": L2 norm of the function vector
            "head_output_norm": norm of the full head output at that position
            "extraction_position": actual position used
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)
    if pos == -1:
        pos = seq_len - 1

    z_key = f"blocks.{layer}.attn.hook_z"
    if z_key not in cache.cache_dict:
        d_model = model.cfg.d_model
        return {
            "function_vector": np.zeros(d_model),
            "fv_norm": 0.0,
            "head_output_norm": 0.0,
            "extraction_position": pos,
        }

    z = np.array(cache.cache_dict[z_key])  # [seq, n_heads, d_head]
    W_O = np.array(model.blocks[layer].attn.W_O[head])  # [d_head, d_model]

    # Project through output matrix to get d_model vector
    fv = z[pos, head] @ W_O  # [d_model]
    fv_norm = float(np.linalg.norm(fv))

    # Full head output norm
    result_key = f"blocks.{layer}.attn.hook_result"
    if result_key in cache.cache_dict:
        result = np.array(cache.cache_dict[result_key])
        head_norm = float(np.linalg.norm(result[pos]))
    else:
        head_norm = fv_norm

    return {
        "function_vector": fv,
        "fv_norm": fv_norm,
        "head_output_norm": head_norm,
        "extraction_position": pos,
    }


def scan_for_function_heads(
    model: HookedTransformer,
    few_shot_tokens: jnp.ndarray,
    zero_shot_tokens: jnp.ndarray,
    metric_fn: Callable,
    top_k: int = 5,
) -> dict:
    """Identify heads that encode function vectors.

    Patches each head's output from the few-shot run into a zero-shot run
    and measures metric improvement.

    Args:
        model: HookedTransformer.
        few_shot_tokens: [seq_len] few-shot prompt tokens.
        zero_shot_tokens: [seq_len] zero-shot prompt tokens.
        metric_fn: Function(logits) -> float metric to measure.
        top_k: Number of top function heads to return.

    Returns:
        Dict with:
            "transfer_scores": [n_layers, n_heads] metric improvement from transfer
            "top_function_heads": list of (layer, head, score) ranked by transfer effect
            "baseline_metric": metric on zero-shot without transfer
    """
    _, few_cache = model.run_with_cache(few_shot_tokens)

    # Baseline: zero-shot metric
    baseline_logits = np.array(model(zero_shot_tokens))
    baseline = float(metric_fn(baseline_logits))

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    scores = np.zeros((n_layers, n_heads))

    zero_len = len(zero_shot_tokens)
    few_len = len(few_shot_tokens)

    for l in range(n_layers):
        z_key = f"blocks.{l}.attn.hook_z"
        if z_key not in few_cache.cache_dict:
            continue

        few_z = np.array(few_cache.cache_dict[z_key])  # [few_seq, n_heads, d_head]

        for h in range(n_heads):
            W_O = np.array(model.blocks[l].attn.W_O[h])  # [d_head, d_model]
            # Extract function vector from few-shot (last position)
            fv = few_z[few_len - 1, h] @ W_O  # [d_model]

            # Inject into zero-shot via hook
            def make_hook(fv_arr, head_idx, layer_idx):
                fv_jax = jnp.array(fv_arr)
                def hook_fn(x, name):
                    # x is hook_result: [seq, d_model]
                    # Add function vector at last position
                    return x.at[-1].add(fv_jax)
                return hook_fn

            hook_name = f"blocks.{l}.attn.hook_result"
            hook_fn = make_hook(fv, h, l)

            try:
                patched_logits = np.array(
                    model.run_with_hooks(zero_shot_tokens, fwd_hooks=[(hook_name, hook_fn)])
                )
                patched_metric = float(metric_fn(patched_logits))
                scores[l, h] = patched_metric - baseline
            except Exception:
                scores[l, h] = 0.0

    # Rank
    flat = scores.flatten()
    top_idx = np.argsort(np.abs(flat))[::-1][:top_k]
    top_heads = []
    for idx in top_idx:
        l_idx = int(idx // n_heads)
        h_idx = int(idx % n_heads)
        top_heads.append((l_idx, h_idx, float(scores[l_idx, h_idx])))

    return {
        "transfer_scores": scores,
        "top_function_heads": top_heads,
        "baseline_metric": baseline,
    }


def function_vector_transfer(
    model: HookedTransformer,
    function_vector: np.ndarray,
    target_tokens: jnp.ndarray,
    inject_layer: int,
    inject_pos: int = -1,
) -> dict:
    """Inject a function vector into a new context.

    Adds the function vector to the residual stream at the specified
    layer and position, then returns the resulting logits.

    Args:
        model: HookedTransformer.
        function_vector: [d_model] function vector to inject.
        target_tokens: [seq_len] tokens to run with injection.
        inject_layer: Layer at which to inject.
        inject_pos: Position at which to inject (-1 = last).

    Returns:
        Dict with:
            "patched_logits": [seq_len, d_vocab] logits after injection
            "clean_logits": [seq_len, d_vocab] logits without injection
            "logit_diff": [d_vocab] difference in logits at inject_pos
            "top_promoted": list of (token_idx, logit_increase) for top-5
    """
    seq_len = len(target_tokens)
    if inject_pos == -1:
        inject_pos = seq_len - 1

    clean_logits = np.array(model(target_tokens))

    fv_jax = jnp.array(function_vector)
    actual_pos = inject_pos

    def hook_fn(x, name):
        return x.at[actual_pos].add(fv_jax)

    hook_name = f"blocks.{inject_layer}.hook_resid_pre"
    patched_logits = np.array(
        model.run_with_hooks(target_tokens, fwd_hooks=[(hook_name, hook_fn)])
    )

    diff = patched_logits[inject_pos] - clean_logits[inject_pos]
    top_idx = np.argsort(diff)[::-1][:5]
    top_promoted = [(int(idx), float(diff[idx])) for idx in top_idx]

    return {
        "patched_logits": patched_logits,
        "clean_logits": clean_logits,
        "logit_diff": diff,
        "top_promoted": top_promoted,
    }


def function_vector_arithmetic(
    model: HookedTransformer,
    fv_a: np.ndarray,
    fv_b: np.ndarray,
    target_tokens: jnp.ndarray,
    inject_layer: int,
    inject_pos: int = -1,
) -> dict:
    """Test whether function vectors compose.

    Injects fv_a + fv_b and measures whether the composed vector
    produces effects consistent with both tasks.

    Args:
        model: HookedTransformer.
        fv_a: [d_model] first function vector.
        fv_b: [d_model] second function vector.
        target_tokens: [seq_len] tokens.
        inject_layer: Layer at which to inject.
        inject_pos: Position (-1 = last).

    Returns:
        Dict with:
            "combined_logits": logits from injecting fv_a + fv_b
            "fv_a_logits": logits from injecting fv_a alone
            "fv_b_logits": logits from injecting fv_b alone
            "linearity_score": how well combined ≈ sum of individual effects
            "cosine_similarity": cosine between fv_a and fv_b
    """
    seq_len = len(target_tokens)
    if inject_pos == -1:
        inject_pos = seq_len - 1

    clean_logits = np.array(model(target_tokens))

    # Individual effects
    result_a = function_vector_transfer(model, fv_a, target_tokens, inject_layer, inject_pos)
    result_b = function_vector_transfer(model, fv_b, target_tokens, inject_layer, inject_pos)

    # Combined
    fv_combined = fv_a + fv_b
    result_combined = function_vector_transfer(
        model, fv_combined, target_tokens, inject_layer, inject_pos
    )

    # Linearity: does combined effect ≈ sum of individual effects?
    diff_a = result_a["patched_logits"][inject_pos] - clean_logits[inject_pos]
    diff_b = result_b["patched_logits"][inject_pos] - clean_logits[inject_pos]
    diff_combined = result_combined["patched_logits"][inject_pos] - clean_logits[inject_pos]

    expected = diff_a + diff_b
    actual = diff_combined

    norm_expected = np.linalg.norm(expected)
    norm_diff = np.linalg.norm(actual - expected)
    linearity = 1.0 - float(norm_diff / (norm_expected + 1e-10))

    # Cosine similarity
    cos = float(np.dot(fv_a, fv_b) / (np.linalg.norm(fv_a) * np.linalg.norm(fv_b) + 1e-10))

    return {
        "combined_logits": result_combined["patched_logits"],
        "fv_a_logits": result_a["patched_logits"],
        "fv_b_logits": result_b["patched_logits"],
        "linearity_score": max(0.0, linearity),
        "cosine_similarity": cos,
    }


def function_vector_similarity_matrix(
    model: HookedTransformer,
    task_token_list: list,
    layer: int,
    head: int,
    pos: int = -1,
) -> dict:
    """Compare function vectors across tasks.

    Extracts function vectors for a list of task prompts and computes
    pairwise cosine similarities.

    Args:
        model: HookedTransformer.
        task_token_list: List of [seq_len] token arrays (one per task).
        layer: Layer of the function head.
        head: Head index.
        pos: Position to read from (-1 = last).

    Returns:
        Dict with:
            "similarity_matrix": [n_tasks, n_tasks] pairwise cosine similarity
            "function_vectors": list of [d_model] vectors
            "norms": [n_tasks] L2 norms
            "mean_similarity": mean off-diagonal similarity
    """
    n_tasks = len(task_token_list)
    fvs = []

    for tokens in task_token_list:
        result = extract_function_vector(model, tokens, layer, head, pos)
        fvs.append(result["function_vector"])

    fvs_arr = np.array(fvs)  # [n_tasks, d_model]
    norms = np.linalg.norm(fvs_arr, axis=1)  # [n_tasks]

    sim_matrix = np.zeros((n_tasks, n_tasks))
    for i in range(n_tasks):
        for j in range(n_tasks):
            if norms[i] > 1e-10 and norms[j] > 1e-10:
                sim_matrix[i, j] = float(np.dot(fvs_arr[i], fvs_arr[j]) / (norms[i] * norms[j]))
            elif i == j:
                sim_matrix[i, j] = 1.0

    # Mean off-diagonal
    if n_tasks > 1:
        off_diag = sim_matrix[~np.eye(n_tasks, dtype=bool)]
        mean_sim = float(np.mean(off_diag))
    else:
        mean_sim = 0.0

    return {
        "similarity_matrix": sim_matrix,
        "function_vectors": fvs,
        "norms": norms,
        "mean_similarity": mean_sim,
    }
