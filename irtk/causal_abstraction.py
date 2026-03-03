"""Causal abstraction testing.

Tests alignment between the model's computation and hypothesized causal
models via interchange interventions and abstraction quality metrics.

References:
    Geiger et al. (2021) "Causal Abstractions of Neural Networks"
    Geiger et al. (2023) "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations"
"""

import jax
import jax.numpy as jnp
import numpy as np


def interchange_intervention(model, tokens_base, tokens_source, hook_name, pos=-1):
    """Perform an interchange intervention: patch activation from source into base.

    Runs both inputs, then re-runs base with the specified activation
    replaced by the source's activation at that hook point.

    Args:
        model: HookedTransformer model.
        tokens_base: Base input token IDs [seq_len].
        tokens_source: Source input token IDs [seq_len].
        hook_name: Hook point name to patch at.
        pos: Position to patch (-1 = all positions).

    Returns:
        dict with:
            base_logits: array of base model logits
            source_logits: array of source model logits
            patched_logits: array of logits after intervention
            kl_divergence: float, KL between patched and base
            logit_diff: float, mean absolute logit difference
    """
    from irtk.hook_points import HookState

    # Run source to get activation
    source_state = HookState(hook_fns={}, cache={})
    source_logits = np.array(model(tokens_source, hook_state=source_state))
    source_act = source_state.cache.get(hook_name)

    # Run base
    base_logits = np.array(model(tokens_base))

    # Run base with patched activation
    if source_act is not None:
        if pos == -1:
            def patch_fn(x, name):
                return jnp.array(source_act)
        else:
            def patch_fn(x, name):
                return x.at[pos].set(jnp.array(source_act[pos]))
    else:
        def patch_fn(x, name):
            return x

    patched_state = HookState(hook_fns={hook_name: patch_fn}, cache={})
    patched_logits = np.array(model(tokens_base, hook_state=patched_state))

    # KL divergence
    base_probs = np.exp(base_logits - np.max(base_logits, axis=-1, keepdims=True))
    base_probs = base_probs / np.sum(base_probs, axis=-1, keepdims=True)
    patched_probs = np.exp(patched_logits - np.max(patched_logits, axis=-1, keepdims=True))
    patched_probs = patched_probs / np.sum(patched_probs, axis=-1, keepdims=True)

    kl = float(np.mean(np.sum(base_probs * np.log((base_probs + 1e-10) / (patched_probs + 1e-10)), axis=-1)))
    logit_diff = float(np.mean(np.abs(patched_logits - base_logits)))

    return {
        "base_logits": base_logits,
        "source_logits": source_logits,
        "patched_logits": patched_logits,
        "kl_divergence": kl,
        "logit_diff": logit_diff,
    }


def causal_abstraction_test(model, input_pairs, hook_name, metric_fn, pos=-1):
    """Test if a hook point implements a causal variable.

    For pairs of inputs where the hypothesized variable differs,
    checks whether interchange intervention at the hook point
    produces the expected output change.

    Args:
        model: HookedTransformer model.
        input_pairs: List of (base_tokens, source_tokens, expected_change) tuples.
        hook_name: Hook to test as implementing the variable.
        metric_fn: Function from logits -> scalar.
        pos: Position.

    Returns:
        dict with:
            alignment_scores: array of per-pair alignment scores
            mean_alignment: float
            n_aligned: int, number of pairs where intervention had expected effect direction
            alignment_rate: float
    """
    scores = []
    n_aligned = 0

    for base_tokens, source_tokens, expected_change in input_pairs:
        result = interchange_intervention(model, base_tokens, source_tokens, hook_name, pos)
        base_metric = metric_fn(jnp.array(result["base_logits"]))
        patched_metric = metric_fn(jnp.array(result["patched_logits"]))
        actual_change = patched_metric - base_metric

        if abs(expected_change) > 1e-10:
            alignment = actual_change / expected_change
        else:
            alignment = 1.0 if abs(actual_change) < 1e-5 else 0.0

        scores.append(float(alignment))
        if (expected_change > 0 and actual_change > 0) or \
           (expected_change < 0 and actual_change < 0) or \
           (abs(expected_change) < 1e-10 and abs(actual_change) < 1e-5):
            n_aligned += 1

    scores = np.array(scores)
    return {
        "alignment_scores": scores,
        "mean_alignment": float(np.mean(scores)),
        "n_aligned": n_aligned,
        "alignment_rate": n_aligned / len(input_pairs) if input_pairs else 0.0,
    }


def distributed_alignment_search(model, tokens_base, tokens_source, metric_fn, layer=0, pos=-1):
    """Search for distributed representations at a layer that align with a causal variable.

    Tests different projection directions to find the best alignment
    between activation subspaces and the target metric.

    Args:
        model: HookedTransformer model.
        tokens_base: Base input.
        tokens_source: Source input.
        metric_fn: Function from logits -> scalar.
        layer: Layer to search in.
        pos: Position.

    Returns:
        dict with:
            full_patch_effect: float, effect of patching all dims
            top_direction: array [d_model], best single direction
            direction_effects: array [d_model], effect of patching each dim
            cumulative_effects: array [d_model], cumulative from top dims
            dims_for_half_effect: int
    """
    from irtk.hook_points import HookState

    hook_name = f"blocks.{layer}.hook_resid_post" if layer > 0 else "blocks.0.hook_resid_pre"
    if layer > 0:
        hook_name = f"blocks.{layer - 1}.hook_resid_post"

    # Get activations
    base_state = HookState(hook_fns={}, cache={})
    base_logits = model(tokens_base, hook_state=base_state)
    base_metric = metric_fn(base_logits)

    source_state = HookState(hook_fns={}, cache={})
    model(tokens_source, hook_state=source_state)

    base_act = np.array(base_state.cache.get(hook_name, np.zeros(1)))
    source_act = np.array(source_state.cache.get(hook_name, np.zeros(1)))

    if base_act.ndim == 0:
        d_model = model.cfg.d_model
        return {
            "full_patch_effect": 0.0,
            "top_direction": np.zeros(d_model),
            "direction_effects": np.zeros(d_model),
            "cumulative_effects": np.zeros(d_model),
            "dims_for_half_effect": d_model,
        }

    diff = source_act[pos] - base_act[pos]
    d_model = len(diff)

    # Full patch
    def full_patch(x, name):
        return x.at[pos].set(jnp.array(source_act[pos]))

    full_state = HookState(hook_fns={hook_name: full_patch}, cache={})
    full_logits = model(tokens_base, hook_state=full_state)
    full_effect = float(metric_fn(full_logits) - base_metric)

    # Per-dimension effects
    dim_effects = np.zeros(d_model)
    for d in range(d_model):
        def make_dim_patch(dim_idx, diff_vec=diff):
            def fn(x, name):
                delta = jnp.zeros_like(x[pos])
                delta = delta.at[dim_idx].set(diff_vec[dim_idx])
                return x.at[pos].set(x[pos] + delta)
            return fn

        state = HookState(hook_fns={hook_name: make_dim_patch(d)}, cache={})
        logits = model(tokens_base, hook_state=state)
        dim_effects[d] = float(metric_fn(logits) - base_metric)

    # Sort by effect magnitude
    order = np.argsort(np.abs(dim_effects))[::-1]
    cumulative = np.cumsum(dim_effects[order])

    # Dims for half effect
    half_target = abs(full_effect) * 0.5
    dims_half = d_model
    for i, c in enumerate(np.abs(cumulative)):
        if c >= half_target:
            dims_half = i + 1
            break

    top_dir = np.zeros(d_model)
    top_dir[order[0]] = 1.0

    return {
        "full_patch_effect": float(full_effect),
        "top_direction": top_dir,
        "direction_effects": dim_effects,
        "cumulative_effects": cumulative,
        "dims_for_half_effect": dims_half,
    }


def multi_variable_alignment(model, tokens_list, hook_names, metric_fn, pos=-1):
    """Test alignment of multiple hook points with a metric simultaneously.

    Args:
        model: HookedTransformer model.
        tokens_list: List of token arrays.
        hook_names: List of hook point names to test.
        metric_fn: Function from logits -> scalar.
        pos: Position.

    Returns:
        dict with:
            hook_effects: dict mapping hook_name -> mean effect
            hook_rankings: list of hook_names sorted by effect
            total_effect: float, sum of all hook effects
            complementarity: float, how much hooks explain non-redundantly
    """
    from irtk.hook_points import HookState

    effects = {name: [] for name in hook_names}

    for tokens in tokens_list:
        baseline = metric_fn(model(tokens))

        for name in hook_names:
            def zero_fn(x, nm):
                return jnp.zeros_like(x)

            state = HookState(hook_fns={name: zero_fn}, cache={})
            logits = model(tokens, hook_state=state)
            effect = abs(baseline - metric_fn(logits))
            effects[name].append(float(effect))

    mean_effects = {name: float(np.mean(effects[name])) for name in hook_names}
    rankings = sorted(hook_names, key=lambda n: mean_effects[n], reverse=True)
    total = sum(mean_effects.values())

    # Test all-zero vs individual sum for complementarity
    combined_effect = 0.0
    if tokens_list:
        tokens = tokens_list[0]
        baseline = metric_fn(model(tokens))
        all_hooks = {name: (lambda x, nm: jnp.zeros_like(x)) for name in hook_names}
        state = HookState(hook_fns=all_hooks, cache={})
        logits = model(tokens, hook_state=state)
        combined_effect = abs(baseline - metric_fn(logits))

    complementarity = combined_effect / (total + 1e-10) if total > 0 else 0.0

    return {
        "hook_effects": mean_effects,
        "hook_rankings": rankings,
        "total_effect": total,
        "complementarity": float(complementarity),
    }


def abstraction_quality_score(model, tokens_base, tokens_source, hook_name, metric_fn, pos=-1):
    """Compute a quality score for treating a hook point as a causal abstraction.

    Combines faithfulness (does intervention produce expected change?)
    and specificity (does it affect only the target behavior?).

    Args:
        model: HookedTransformer model.
        tokens_base: Base input.
        tokens_source: Source input.
        metric_fn: Target metric function.
        hook_name: Hook point to evaluate.
        pos: Position.

    Returns:
        dict with:
            faithfulness: float, how well intervention reproduces source behavior
            specificity: float, how much intervention changes only the target metric
            quality_score: float, combined score
            intervention_effect: float
    """
    from irtk.hook_points import HookState

    base_logits = model(tokens_base)
    base_metric = metric_fn(base_logits)

    source_logits = model(tokens_source)
    source_metric = metric_fn(source_logits)

    result = interchange_intervention(model, tokens_base, tokens_source, hook_name, pos)
    patched_metric = metric_fn(jnp.array(result["patched_logits"]))

    # Faithfulness: does patched match source?
    target_change = source_metric - base_metric
    actual_change = patched_metric - base_metric

    if abs(target_change) > 1e-10:
        faithfulness = 1.0 - abs(actual_change - target_change) / (abs(target_change) + 1e-10)
        faithfulness = max(0.0, faithfulness)
    else:
        faithfulness = 1.0 if abs(actual_change) < 1e-5 else 0.0

    # Specificity: KL divergence should be small apart from target metric
    specificity = max(0.0, 1.0 - result["kl_divergence"])

    quality = (faithfulness + specificity) / 2

    return {
        "faithfulness": float(faithfulness),
        "specificity": float(specificity),
        "quality_score": float(quality),
        "intervention_effect": float(actual_change),
    }
