"""Patchscopes: representation inspection via model self-decoding.

Patch an intermediate activation from one forward pass into a different
"inspection prompt" in a second forward pass, letting the model's own
generation capabilities decode what a representation contains.

References:
    - Ghandeharioun et al. (2024) "Patchscopes: A Unifying Framework for
      Inspecting Hidden Representations of Language Models" (ICML 2024)
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def patchscope(
    model: HookedTransformer,
    source_tokens: jnp.ndarray,
    source_hook: str,
    target_tokens: jnp.ndarray,
    target_hook: str,
    target_pos: int = -1,
    source_pos: Optional[int] = None,
) -> np.ndarray:
    """Core patchscope primitive: patch a source activation into a target context.

    Extracts activation at source_hook from a run on source_tokens,
    then patches it into target_pos of target_hook during a second run
    on target_tokens.

    Args:
        model: HookedTransformer.
        source_tokens: Tokens for the source forward pass.
        source_hook: Hook name to extract activation from (e.g., "blocks.5.hook_resid_post").
        target_tokens: Tokens for the target (inspection) forward pass.
        target_hook: Hook name to inject activation into.
        target_pos: Position in target sequence to patch into. -1 for last position.
        source_pos: Position in source to extract from. None for last position.

    Returns:
        [seq_len, d_vocab] logits from the patched target run.
    """
    source_tokens = jnp.array(source_tokens)
    target_tokens = jnp.array(target_tokens)

    # Extract source activation
    _, source_cache = model.run_with_cache(source_tokens)
    source_act = source_cache.cache_dict[source_hook]  # [seq_len, d_model]

    sp = source_pos if source_pos is not None else -1
    patch_vec = source_act[sp]  # [d_model]

    # Resolve target_pos
    tp = target_pos if target_pos >= 0 else len(target_tokens) + target_pos

    def patch_hook(x, name):
        return x.at[tp].set(patch_vec)

    logits = model.run_with_hooks(target_tokens, fwd_hooks=[(target_hook, patch_hook)])
    return np.array(logits)


def logit_lens_patchscope(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
    pos: int = -1,
) -> np.ndarray:
    """Recover logit-lens as a Patchscope: patch layer residual into final layer.

    Patches the residual stream at a given layer into the last layer's
    residual for a single-token identity context, recovering the standard
    logit lens result via the Patchscope framework.

    Args:
        model: HookedTransformer.
        tokens: Input token sequence.
        layer: Which layer to read from (0 to n_layers-1).
        pos: Position to inspect (-1 for last).

    Returns:
        [d_vocab] logits (or probabilities) from the patchscope.
    """
    tokens = jnp.array(tokens)
    resolved_pos = pos if pos >= 0 else len(tokens) + pos

    # Get activation at this layer
    _, cache = model.run_with_cache(tokens)
    source_hook = f"blocks.{layer}.hook_resid_post"
    source_act = cache.cache_dict[source_hook][resolved_pos]  # [d_model]

    # For logit-lens patchscope, we need a minimal target context.
    # Use a single token and patch at the final layer's residual.
    # We use the same token to create a minimal 1-token context.
    target_tokens = jnp.array([int(tokens[resolved_pos])])
    final_hook = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"

    def patch_hook(x, name):
        return x.at[0].set(source_act)

    logits = model.run_with_hooks(target_tokens, fwd_hooks=[(final_hook, patch_hook)])
    return np.array(logits[0])  # [d_vocab]


def token_identity_inspection(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
    pos: int = -1,
    k: int = 10,
) -> dict:
    """Probe what token identity the model has resolved at a position and layer.

    Patches the activation into a simple identity context and reads the
    top-k predicted tokens.

    Args:
        model: HookedTransformer.
        tokens: Input token sequence.
        layer: Layer to inspect.
        pos: Position to inspect (-1 for last).
        k: Number of top tokens to return.

    Returns:
        Dict with:
        - "top_tokens": [(token_id, probability), ...] top-k predicted tokens
        - "top_logits": [(token_id, logit), ...] top-k by logit
        - "entropy": entropy of the predicted distribution
    """
    logits = logit_lens_patchscope(model, tokens, layer, pos)

    probs = np.array(jax.nn.softmax(jnp.array(logits)))
    log_probs = np.array(jax.nn.log_softmax(jnp.array(logits)))

    # Top-k by probability
    top_prob_idx = np.argsort(probs)[::-1][:k]
    top_tokens = [(int(i), float(probs[i])) for i in top_prob_idx]

    # Top-k by logit
    top_logit_idx = np.argsort(logits)[::-1][:k]
    top_logits = [(int(i), float(logits[i])) for i in top_logit_idx]

    # Entropy
    entropy = float(-np.sum(probs * log_probs))

    return {
        "top_tokens": top_tokens,
        "top_logits": top_logits,
        "entropy": entropy,
    }


def attribute_to_source(
    model: HookedTransformer,
    source_tokens: jnp.ndarray,
    target_tokens: jnp.ndarray,
    target_hook: str,
    target_pos: int,
    metric_fn: Callable,
    layers: Optional[list[int]] = None,
) -> dict:
    """Sweep over all (source_layer, source_pos) pairs to find what drives a target.

    For each pair, patches the source activation into target_pos of
    target_hook and records metric_fn(logits).

    Args:
        model: HookedTransformer.
        source_tokens: Source token sequence.
        target_tokens: Target (inspection) token sequence.
        target_hook: Hook in target run to patch into.
        target_pos: Position in target to patch.
        metric_fn: Function from logits -> float to evaluate.
        layers: Which layers to sweep. None for all layers.

    Returns:
        Dict with:
        - "attribution_matrix": [n_layers, seq_len] metric values
        - "best_layer": layer index with highest absolute attribution
        - "best_pos": position with highest absolute attribution
        - "baseline_metric": metric from unpatched target run
    """
    source_tokens = jnp.array(source_tokens)
    target_tokens = jnp.array(target_tokens)
    seq_len = len(source_tokens)

    if layers is None:
        layers = list(range(model.cfg.n_layers))

    # Baseline: unpatched target run
    baseline_logits = model(target_tokens)
    baseline_metric = float(metric_fn(baseline_logits))

    # Get all source activations
    _, source_cache = model.run_with_cache(source_tokens)

    matrix = np.zeros((len(layers), seq_len))

    for li, layer in enumerate(layers):
        source_hook = f"blocks.{layer}.hook_resid_post"
        if source_hook not in source_cache.cache_dict:
            continue
        layer_acts = source_cache.cache_dict[source_hook]  # [seq_len, d_model]

        for sp in range(seq_len):
            patch_vec = layer_acts[sp]  # [d_model]
            tp = target_pos if target_pos >= 0 else len(target_tokens) + target_pos

            def make_hook(pv, tpos):
                def patch_hook(x, name):
                    return x.at[tpos].set(pv)
                return patch_hook

            logits = model.run_with_hooks(
                target_tokens, fwd_hooks=[(target_hook, make_hook(patch_vec, tp))]
            )
            matrix[li, sp] = float(metric_fn(logits))

    # Find best
    abs_diff = np.abs(matrix - baseline_metric)
    best_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)

    return {
        "attribution_matrix": matrix,
        "best_layer": layers[best_idx[0]],
        "best_pos": int(best_idx[1]),
        "baseline_metric": baseline_metric,
    }


def cross_model_inspection(
    source_model: HookedTransformer,
    target_model: HookedTransformer,
    source_tokens: jnp.ndarray,
    source_hook: str,
    target_tokens: jnp.ndarray,
    target_hook: str,
    target_pos: int = -1,
    source_pos: Optional[int] = None,
    projection: Optional[jnp.ndarray] = None,
) -> np.ndarray:
    """Patch a representation from source_model into target_model's computation.

    Useful for "use a larger model to explain a smaller model's representations."
    If models have different d_model, provide a projection matrix.

    Args:
        source_model: Model to extract activation from.
        target_model: Model to inject activation into.
        source_tokens: Tokens for source forward pass.
        source_hook: Hook to extract from in source_model.
        target_tokens: Tokens for target forward pass.
        target_hook: Hook to inject into in target_model.
        target_pos: Position in target to patch. -1 for last.
        source_pos: Position in source to extract. None for last.
        projection: Optional [source_d_model, target_d_model] projection matrix
            for when models have different hidden dimensions.

    Returns:
        [seq_len, d_vocab] logits from the patched target run.
    """
    source_tokens = jnp.array(source_tokens)
    target_tokens = jnp.array(target_tokens)

    # Extract source activation
    _, source_cache = source_model.run_with_cache(source_tokens)
    source_act = source_cache.cache_dict[source_hook]
    sp = source_pos if source_pos is not None else -1
    patch_vec = source_act[sp]  # [source_d_model]

    # Project if needed
    if projection is not None:
        patch_vec = patch_vec @ jnp.array(projection)

    tp = target_pos if target_pos >= 0 else len(target_tokens) + target_pos

    def patch_hook(x, name):
        return x.at[tp].set(patch_vec)

    logits = target_model.run_with_hooks(
        target_tokens, fwd_hooks=[(target_hook, patch_hook)]
    )
    return np.array(logits)
