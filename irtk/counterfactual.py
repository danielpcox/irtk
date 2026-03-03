"""Counterfactual and contrastive analysis tools.

Compare model behavior between a clean and corrupted/alternative input to
understand what drives predictions. Find where computations diverge,
measure necessity/sufficiency of tokens, and attribute prediction differences.
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def contrastive_activation_diff(
    model: HookedTransformer,
    tokens_a: jnp.ndarray,
    tokens_b: jnp.ndarray,
) -> dict:
    """Compare activations between two inputs at every layer.

    Finds where in the network two inputs start to produce different
    internal representations.

    Args:
        model: HookedTransformer.
        tokens_a: First token sequence.
        tokens_b: Second token sequence.

    Returns:
        Dict with:
        - "layer_diffs": [n_layers+1] L2 norm of activation difference per layer
        - "divergence_layer": first layer where diff exceeds 10% of max diff
        - "max_diff_layer": layer with largest activation difference
        - "relative_diffs": [n_layers+1] diff normalized by mean activation norm
    """
    tokens_a = jnp.array(tokens_a)
    tokens_b = jnp.array(tokens_b)

    _, cache_a = model.run_with_cache(tokens_a)
    _, cache_b = model.run_with_cache(tokens_b)

    resid_a = cache_a.accumulated_resid()  # [n_components, seq_len, d_model]
    resid_b = cache_b.accumulated_resid()

    n_components = resid_a.shape[0]
    # Use last position for comparison
    diffs = []
    norms_a = []
    for i in range(n_components):
        act_a = np.array(resid_a[i, -1])
        act_b = np.array(resid_b[i, -1])
        diffs.append(float(np.linalg.norm(act_a - act_b)))
        norms_a.append(float(np.linalg.norm(act_a)))

    diffs = np.array(diffs)
    norms = np.array(norms_a)
    relative = diffs / np.maximum(norms, 1e-10)

    max_diff = np.max(diffs)
    divergence = None
    for i in range(len(diffs)):
        if diffs[i] > 0.1 * max_diff:
            divergence = i
            break

    return {
        "layer_diffs": diffs,
        "divergence_layer": divergence,
        "max_diff_layer": int(np.argmax(diffs)),
        "relative_diffs": relative,
    }


def minimal_change_tokens(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
    target_change: float = 0.0,
    candidates: Optional[list[int]] = None,
) -> dict:
    """Find the minimal token change that most affects the metric.

    Tests replacing each token with alternatives to find which single-token
    change has the largest (or most targeted) effect.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] original tokens.
        metric_fn: Function from logits -> float.
        target_change: Target metric value after change (0 = maximize change).
        candidates: Alternative token IDs to try. None = try [0, 1, ..., 49].

    Returns:
        Dict with:
        - "best_pos": position to change
        - "best_replacement": replacement token ID
        - "original_metric": metric on original input
        - "changed_metric": metric after the best change
        - "per_position_effects": [seq_len] max effect at each position
    """
    tokens = jnp.array(tokens)
    seq_len = len(tokens)

    if candidates is None:
        candidates = list(range(min(50, model.cfg.d_vocab)))

    original_logits = model(tokens)
    original_metric = float(metric_fn(original_logits))

    best_pos = 0
    best_replacement = int(tokens[0])
    best_effect = 0.0
    best_metric = original_metric
    per_pos_effects = np.zeros(seq_len)

    for pos in range(seq_len):
        for cand in candidates:
            if cand == int(tokens[pos]):
                continue
            modified = tokens.at[pos].set(cand)
            logits = model(modified)
            new_metric = float(metric_fn(logits))
            effect = abs(new_metric - target_change) if target_change != 0 else abs(new_metric - original_metric)

            per_pos_effects[pos] = max(per_pos_effects[pos], abs(new_metric - original_metric))

            if target_change != 0:
                # Minimize distance to target
                if abs(new_metric - target_change) < abs(best_metric - target_change) or best_effect == 0:
                    best_effect = effect
                    best_pos = pos
                    best_replacement = cand
                    best_metric = new_metric
            else:
                # Maximize change
                if effect > best_effect:
                    best_effect = effect
                    best_pos = pos
                    best_replacement = cand
                    best_metric = new_metric

    return {
        "best_pos": best_pos,
        "best_replacement": best_replacement,
        "original_metric": original_metric,
        "changed_metric": best_metric,
        "per_position_effects": per_pos_effects,
    }


def counterfactual_effect_by_layer(
    model: HookedTransformer,
    clean_tokens: jnp.ndarray,
    corrupted_tokens: jnp.ndarray,
    metric_fn: Callable,
) -> dict:
    """Measure how much each layer contributes to the prediction difference.

    Patches each layer's output from clean into corrupted run to see
    how much each layer restores the clean metric.

    Args:
        model: HookedTransformer.
        clean_tokens: Tokens with correct prediction.
        corrupted_tokens: Tokens with wrong prediction.
        metric_fn: Function from logits -> float.

    Returns:
        Dict with:
        - "clean_metric": metric on clean input
        - "corrupted_metric": metric on corrupted input
        - "restoration_by_layer": [n_layers] metric restored by patching each layer
        - "most_important_layer": layer with highest restoration
    """
    clean_tokens = jnp.array(clean_tokens)
    corrupted_tokens = jnp.array(corrupted_tokens)

    clean_logits = model(clean_tokens)
    clean_metric = float(metric_fn(clean_logits))

    corrupted_logits = model(corrupted_tokens)
    corrupted_metric = float(metric_fn(corrupted_logits))

    # Get clean activations
    _, clean_cache = model.run_with_cache(clean_tokens)

    n_layers = model.cfg.n_layers
    restoration = np.zeros(n_layers)

    for layer in range(n_layers):
        hook = f"blocks.{layer}.hook_resid_post"
        if hook not in clean_cache.cache_dict:
            continue
        clean_act = clean_cache.cache_dict[hook]

        def make_patch(ca):
            def patch(x, name):
                # Replace at last position (or all positions if same length)
                if x.shape[0] == ca.shape[0]:
                    return ca
                return x.at[-1].set(ca[-1])
            return patch

        logits = model.run_with_hooks(
            corrupted_tokens, fwd_hooks=[(hook, make_patch(clean_act))]
        )
        patched_metric = float(metric_fn(logits))
        restoration[layer] = patched_metric - corrupted_metric

    return {
        "clean_metric": clean_metric,
        "corrupted_metric": corrupted_metric,
        "restoration_by_layer": restoration,
        "most_important_layer": int(np.argmax(np.abs(restoration))),
    }


def token_necessity_sufficiency(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable,
    threshold: float = 0.0,
) -> dict:
    """Measure necessity and sufficiency of each token for the prediction.

    Necessity: how much does removing the token hurt?
    Sufficiency: how much does keeping only this token help?

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        metric_fn: Function from logits -> float.
        threshold: Metric threshold for "success".

    Returns:
        Dict with:
        - "necessity": [seq_len] metric drop when each token is ablated
        - "sufficiency": [seq_len] metric when only each token is present
        - "most_necessary": position most necessary for prediction
        - "most_sufficient": position most sufficient for prediction
    """
    tokens = jnp.array(tokens)
    seq_len = len(tokens)

    # Baseline
    _, cache = model.run_with_cache(tokens)
    clean_embed = cache.cache_dict.get("hook_embed", None)
    if clean_embed is None:
        return {"necessity": np.zeros(seq_len), "sufficiency": np.zeros(seq_len),
                "most_necessary": 0, "most_sufficient": 0}

    clean_embed_np = np.array(clean_embed)
    full_logits = model(tokens)
    full_metric = float(metric_fn(full_logits))

    necessity = np.zeros(seq_len)
    sufficiency = np.zeros(seq_len)

    for pos in range(seq_len):
        # Necessity: zero out this token's embedding
        ablated = jnp.array(clean_embed_np.copy())
        ablated = ablated.at[pos].set(0.0)

        def make_hook(emb):
            def hook(x, name):
                return emb
            return hook

        logits = model.run_with_hooks(tokens, fwd_hooks=[("hook_embed", make_hook(ablated))])
        necessity[pos] = full_metric - float(metric_fn(logits))

        # Sufficiency: zero out all other tokens
        only_one = jnp.zeros_like(ablated)
        only_one = only_one.at[pos].set(clean_embed_np[pos])
        logits = model.run_with_hooks(tokens, fwd_hooks=[("hook_embed", make_hook(only_one))])
        sufficiency[pos] = float(metric_fn(logits))

    return {
        "necessity": necessity,
        "sufficiency": sufficiency,
        "most_necessary": int(np.argmax(necessity)),
        "most_sufficient": int(np.argmax(sufficiency)),
    }


def contrastive_feature_attribution(
    model: HookedTransformer,
    tokens_a: jnp.ndarray,
    tokens_b: jnp.ndarray,
    metric_fn: Callable,
) -> dict:
    """Attribute metric difference between two inputs to model components.

    For each attention and MLP output, measures how much replacing it from
    input A into input B's run changes the metric.

    Args:
        model: HookedTransformer.
        tokens_a: First token sequence.
        tokens_b: Second token sequence.
        metric_fn: Function from logits -> float.

    Returns:
        Dict with:
        - "attn_attribution": [n_layers] attribution from attention
        - "mlp_attribution": [n_layers] attribution from MLP
        - "total_diff": metric_a - metric_b
        - "most_important_component": (type, layer) of largest attribution
    """
    tokens_a = jnp.array(tokens_a)
    tokens_b = jnp.array(tokens_b)

    logits_a = model(tokens_a)
    logits_b = model(tokens_b)
    metric_a = float(metric_fn(logits_a))
    metric_b = float(metric_fn(logits_b))

    _, cache_a = model.run_with_cache(tokens_a)

    n_layers = model.cfg.n_layers
    attn_attr = np.zeros(n_layers)
    mlp_attr = np.zeros(n_layers)

    for layer in range(n_layers):
        # Patch attention from A into B
        attn_hook = f"blocks.{layer}.hook_attn_out"
        if attn_hook in cache_a.cache_dict:
            clean_act = cache_a.cache_dict[attn_hook]

            def make_patch(ca):
                def patch(x, name):
                    if x.shape == ca.shape:
                        return ca
                    return x
                return patch

            logits = model.run_with_hooks(
                tokens_b, fwd_hooks=[(attn_hook, make_patch(clean_act))]
            )
            attn_attr[layer] = float(metric_fn(logits)) - metric_b

        # Patch MLP from A into B
        mlp_hook = f"blocks.{layer}.hook_mlp_out"
        if mlp_hook in cache_a.cache_dict:
            clean_act = cache_a.cache_dict[mlp_hook]

            def make_patch2(ca):
                def patch(x, name):
                    if x.shape == ca.shape:
                        return ca
                    return x
                return patch

            logits = model.run_with_hooks(
                tokens_b, fwd_hooks=[(mlp_hook, make_patch2(clean_act))]
            )
            mlp_attr[layer] = float(metric_fn(logits)) - metric_b

    # Find most important
    all_effects = list(zip(["attn"] * n_layers, range(n_layers), np.abs(attn_attr))) + \
                  list(zip(["mlp"] * n_layers, range(n_layers), np.abs(mlp_attr)))
    all_effects.sort(key=lambda x: x[2], reverse=True)
    most_important = (all_effects[0][0], all_effects[0][1]) if all_effects else ("attn", 0)

    return {
        "attn_attribution": attn_attr,
        "mlp_attribution": mlp_attr,
        "total_diff": metric_a - metric_b,
        "most_important_component": most_important,
    }
