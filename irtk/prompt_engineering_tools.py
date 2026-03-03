"""Interpretability-guided prompt engineering tools.

Use mechanistic insights to understand and optimize prompts: token importance
for prompt construction, attention steering analysis, critical context
positions, prompt sensitivity mapping, and prompt comparison analysis.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Callable


def _get_all_caches(model, tokens):
    """Run model and return full cache."""
    from irtk.hook_points import HookState
    cache = {}
    hs = HookState(hook_fns={}, cache=cache)
    model(tokens, hook_state=hs)
    return cache


def token_importance_map(
    model,
    tokens,
    target_pos: int = -1,
    method: str = "attention",
) -> dict:
    """Map importance of each token for the prediction at target_pos.

    Uses attention aggregation or gradient-based methods to rank
    which input tokens most influence the output.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        target_pos: Position whose output to analyze.
        method: "attention" for attention-based, "ablation" for
            leave-one-out ablation.

    Returns:
        Dict with importance_scores, ranked_positions,
        most_important, least_important.
    """
    seq_len = len(tokens)

    if method == "attention":
        cache = _get_all_caches(model, tokens)
        n_layers = model.cfg.n_layers

        scores = np.zeros(seq_len)
        count = 0
        for l in range(n_layers):
            pattern_key = f"blocks.{l}.attn.hook_pattern"
            if pattern_key not in cache:
                continue
            patterns = np.array(cache[pattern_key])  # [n_heads, seq, seq]
            # Sum attention from target to each source across all heads
            scores += np.sum(patterns[:, target_pos, :], axis=0)
            count += patterns.shape[0]

        if count > 0:
            scores /= count
    else:
        # Ablation-based
        base_logits = np.array(model(tokens))
        base_output = base_logits[target_pos]

        scores = np.zeros(seq_len)
        for i in range(seq_len):
            # Replace token with 0 (simple ablation)
            ablated = np.array(tokens)
            ablated = jnp.array(ablated).at[i].set(0)
            abl_logits = np.array(model(ablated))
            diff = np.linalg.norm(abl_logits[target_pos] - base_output)
            scores[i] = diff

    ranked = np.argsort(scores)[::-1]
    ranked_positions = [(int(i), float(scores[i])) for i in ranked]

    return {
        "importance_scores": jnp.array(scores),
        "ranked_positions": ranked_positions,
        "most_important": int(ranked[0]),
        "least_important": int(ranked[-1]),
    }


def attention_steering_analysis(
    model,
    tokens,
    target_pos: int = -1,
    source_pos: Optional[int] = None,
) -> dict:
    """Analyze which heads steer attention to/from specific positions.

    For prompt engineering: understanding which heads control where
    the model "looks" helps design prompts that leverage these patterns.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        target_pos: Query position.
        source_pos: Source position (default: most attended).

    Returns:
        Dict with steering_heads (sorted by influence), attention_distribution,
        concentration_score.
    """
    cache = _get_all_caches(model, tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    steering_heads = []
    for l in range(n_layers):
        pattern_key = f"blocks.{l}.attn.hook_pattern"
        if pattern_key not in cache:
            continue

        patterns = np.array(cache[pattern_key])  # [n_heads, seq, seq]
        for h in range(n_heads):
            p = patterns[h, target_pos]  # [seq]

            # Entropy as measure of how focused this head is
            entropy = -float(np.sum(p * np.log(p + 1e-10)))
            peak_pos = int(np.argmax(p))
            peak_weight = float(p[peak_pos])

            sp = source_pos if source_pos is not None else peak_pos
            attn_to_source = float(p[sp])

            steering_heads.append({
                "layer": l,
                "head": h,
                "attention_to_source": attn_to_source,
                "peak_position": peak_pos,
                "peak_weight": peak_weight,
                "entropy": entropy,
            })

    steering_heads.sort(key=lambda x: -x["attention_to_source"])

    # Overall attention distribution from target
    all_patterns = []
    for l in range(n_layers):
        pattern_key = f"blocks.{l}.attn.hook_pattern"
        if pattern_key in cache:
            all_patterns.append(np.array(cache[pattern_key])[:, target_pos, :])  # [n_heads, seq]
    if all_patterns:
        avg_distribution = np.mean(np.concatenate(all_patterns, axis=0), axis=0)
    else:
        avg_distribution = np.ones(len(tokens)) / len(tokens)

    concentration = float(np.max(avg_distribution) / (np.mean(avg_distribution) + 1e-10))

    return {
        "steering_heads": steering_heads,
        "attention_distribution": jnp.array(avg_distribution),
        "concentration_score": concentration,
    }


def critical_context_positions(
    model,
    tokens,
    top_k: int = 5,
) -> dict:
    """Identify the most critical positions in the context.

    Combines attention and residual stream analysis to find which
    positions carry the most influence on the final output.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        top_k: Number of top positions.

    Returns:
        Dict with critical_positions, position_scores,
        attention_score, residual_score.
    """
    cache = _get_all_caches(model, tokens)
    seq_len = len(tokens)
    n_layers = model.cfg.n_layers

    # Attention-based: average incoming attention at last position across all layers/heads
    attn_scores = np.zeros(seq_len)
    count = 0
    for l in range(n_layers):
        pattern_key = f"blocks.{l}.attn.hook_pattern"
        if pattern_key in cache:
            patterns = np.array(cache[pattern_key])
            attn_scores += np.sum(patterns[:, -1, :], axis=0)
            count += patterns.shape[0]
    if count > 0:
        attn_scores /= count

    # Residual-based: norm of residual at each position in last layer
    last_resid_key = f"blocks.{n_layers - 1}.hook_resid_post"
    if last_resid_key in cache:
        resid = np.array(cache[last_resid_key])
        resid_scores = np.linalg.norm(resid, axis=-1)
        resid_scores /= (np.max(resid_scores) + 1e-10)
    else:
        resid_scores = np.zeros(seq_len)

    # Combined score
    combined = 0.5 * attn_scores + 0.5 * resid_scores

    ranked = np.argsort(combined)[::-1][:top_k]
    critical = [(int(i), float(combined[i])) for i in ranked]

    return {
        "critical_positions": critical,
        "position_scores": jnp.array(combined),
        "attention_scores": jnp.array(attn_scores),
        "residual_scores": jnp.array(resid_scores),
    }


def prompt_sensitivity_map(
    model,
    tokens,
    target_pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Map how sensitive the output is to each input token.

    Uses leave-one-out analysis: for each position, replaces the token
    with token 0 and measures the output change.

    Args:
        model: HookedTransformer model.
        tokens: Input token ids.
        target_pos: Output position to monitor.
        top_k: Number of top sensitive positions.

    Returns:
        Dict with sensitivity_scores, most_sensitive_positions,
        least_sensitive_positions, mean_sensitivity.
    """
    base_logits = np.array(model(tokens))
    base_output = base_logits[target_pos]
    seq_len = len(tokens)

    sensitivities = np.zeros(seq_len)
    for i in range(seq_len):
        ablated = jnp.array(tokens).at[i].set(0)
        abl_logits = np.array(model(ablated))
        sensitivities[i] = float(np.linalg.norm(abl_logits[target_pos] - base_output))

    ranked = np.argsort(sensitivities)[::-1]
    most_sensitive = [(int(i), float(sensitivities[i])) for i in ranked[:top_k]]
    least_sensitive = [(int(i), float(sensitivities[i])) for i in ranked[-top_k:]]

    return {
        "sensitivity_scores": jnp.array(sensitivities),
        "most_sensitive_positions": most_sensitive,
        "least_sensitive_positions": least_sensitive,
        "mean_sensitivity": float(np.mean(sensitivities)),
    }


def prompt_comparison(
    model,
    tokens_a,
    tokens_b,
    pos: int = -1,
) -> dict:
    """Compare two prompts to understand why they produce different outputs.

    Analyzes both prompts and identifies the key differences in
    internal representations and attention patterns.

    Args:
        model: HookedTransformer model.
        tokens_a: First prompt tokens.
        tokens_b: Second prompt tokens.
        pos: Position to compare.

    Returns:
        Dict with logit_diff, attention_diff_per_layer,
        residual_diff_per_layer, most_different_layer.
    """
    cache_a = _get_all_caches(model, tokens_a)
    cache_b = _get_all_caches(model, tokens_b)

    logits_a = np.array(model(tokens_a))
    logits_b = np.array(model(tokens_b))
    logit_diff = float(np.linalg.norm(logits_a[pos] - logits_b[pos]))

    n_layers = model.cfg.n_layers
    attn_diffs = []
    resid_diffs = []

    for l in range(n_layers):
        # Attention pattern difference
        pattern_key = f"blocks.{l}.attn.hook_pattern"
        if pattern_key in cache_a and pattern_key in cache_b:
            pa = np.array(cache_a[pattern_key])
            pb = np.array(cache_b[pattern_key])
            min_seq = min(pa.shape[1], pb.shape[1])
            diff = float(np.mean(np.abs(pa[:, :min_seq, :min_seq] - pb[:, :min_seq, :min_seq])))
            attn_diffs.append({"layer": l, "diff": diff})
        else:
            attn_diffs.append({"layer": l, "diff": 0.0})

        # Residual stream difference
        resid_key = f"blocks.{l}.hook_resid_post"
        if resid_key in cache_a and resid_key in cache_b:
            ra = np.array(cache_a[resid_key][pos])
            rb = np.array(cache_b[resid_key][pos])
            diff = float(np.linalg.norm(ra - rb))
            resid_diffs.append({"layer": l, "diff": diff})
        else:
            resid_diffs.append({"layer": l, "diff": 0.0})

    most_diff_layer = max(resid_diffs, key=lambda x: x["diff"])["layer"] if resid_diffs else 0

    return {
        "logit_diff": logit_diff,
        "attention_diff_per_layer": attn_diffs,
        "residual_diff_per_layer": resid_diffs,
        "most_different_layer": most_diff_layer,
    }
