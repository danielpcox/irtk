"""Activation patching and ablation utilities for causal analysis.

Provides high-level APIs for:
- Activation patching (swap activations between clean/corrupted runs)
- Mean ablation (replace with mean activation)
- Zero ablation
- Noising/denoising experiments
- Per-head and per-layer patching
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer
from irtk.hook_points import HookState


def _logit_diff(
    logits: jnp.ndarray,
    correct_token: int,
    wrong_token: int,
    pos: int = -1,
) -> float:
    """Compute logit difference: logit(correct) - logit(wrong)."""
    return float(logits[pos, correct_token] - logits[pos, wrong_token])


def _loss_at_pos(logits: jnp.ndarray, target_token: int, pos: int = -1) -> float:
    """Compute cross-entropy loss at a specific position."""
    log_probs = jax.nn.log_softmax(logits[pos])
    return -float(log_probs[target_token])


# ─── Activation Patching ─────────────────────────────────────────────────────


def activation_patch(
    model: HookedTransformer,
    clean_tokens: jnp.ndarray,
    corrupted_tokens: jnp.ndarray,
    hook_names: list[str],
    metric_fn: Callable[[jnp.ndarray], float],
    patch_from: str = "corrupted",
) -> dict[str, float]:
    """Patch activations from one run into another and measure the effect.

    For each hook point, run the model on clean input but swap in the
    activation from the corrupted run (or vice versa) at that hook.

    Args:
        model: HookedTransformer.
        clean_tokens: Clean input tokens.
        corrupted_tokens: Corrupted input tokens.
        hook_names: List of hook point names to patch.
        metric_fn: Function(logits) -> float that measures the behavior of interest.
        patch_from: "corrupted" = patch corrupted activations into clean run,
                    "clean" = patch clean activations into corrupted run.

    Returns:
        Dict mapping hook_name -> metric value after patching.
    """
    # Get source activations
    if patch_from == "corrupted":
        base_tokens = clean_tokens
        _, source_cache = model.run_with_cache(corrupted_tokens)
    else:
        base_tokens = corrupted_tokens
        _, source_cache = model.run_with_cache(clean_tokens)

    results = {}
    for hook_name in hook_names:
        source_act = source_cache[hook_name]

        def patch_hook(x, name, _source=source_act):
            return _source

        logits = model.run_with_hooks(
            base_tokens,
            fwd_hooks=[(hook_name, patch_hook)],
        )
        results[hook_name] = metric_fn(logits)

    return results


def patch_by_layer(
    model: HookedTransformer,
    clean_tokens: jnp.ndarray,
    corrupted_tokens: jnp.ndarray,
    metric_fn: Callable[[jnp.ndarray], float],
    hook_template: str = "blocks.{layer}.hook_resid_post",
) -> np.ndarray:
    """Patch activations at each layer and measure the effect.

    Args:
        model: HookedTransformer.
        clean_tokens: Clean input tokens.
        corrupted_tokens: Corrupted input tokens.
        metric_fn: Function(logits) -> float.
        hook_template: Template with {layer} placeholder.

    Returns:
        [n_layers] array of metric values after patching each layer.
    """
    n_layers = model.cfg.n_layers
    hook_names = [hook_template.format(layer=l) for l in range(n_layers)]
    results = activation_patch(model, clean_tokens, corrupted_tokens, hook_names, metric_fn)
    return np.array([results[name] for name in hook_names])


def patch_by_head(
    model: HookedTransformer,
    clean_tokens: jnp.ndarray,
    corrupted_tokens: jnp.ndarray,
    metric_fn: Callable[[jnp.ndarray], float],
    activation: str = "z",
) -> np.ndarray:
    """Patch individual attention head outputs and measure the effect.

    For each head, patches its z (pre-output-projection) activation from
    the corrupted run into the clean run.

    Args:
        model: HookedTransformer.
        clean_tokens: Clean tokens.
        corrupted_tokens: Corrupted tokens.
        metric_fn: Function(logits) -> float.
        activation: Which activation to patch ("z", "q", "k", "v", "result").

    Returns:
        [n_layers, n_heads] array of metric values.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    _, corrupted_cache = model.run_with_cache(corrupted_tokens)
    results = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.attn.hook_{activation}"
        corrupted_act = corrupted_cache[hook_name]

        for head in range(n_heads):
            def patch_head_hook(x, name, _ca=corrupted_act, _h=head):
                return x.at[:, _h, :].set(_ca[:, _h, :])

            logits = model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[(hook_name, patch_head_hook)],
            )
            results[layer, head] = metric_fn(logits)

    return results


def patch_by_position(
    model: HookedTransformer,
    clean_tokens: jnp.ndarray,
    corrupted_tokens: jnp.ndarray,
    metric_fn: Callable[[jnp.ndarray], float],
    hook_template: str = "blocks.{layer}.hook_resid_post",
) -> np.ndarray:
    """Patch activations at each (layer, position) and measure the effect.

    Args:
        model: HookedTransformer.
        clean_tokens: Clean tokens.
        corrupted_tokens: Corrupted tokens.
        metric_fn: Function(logits) -> float.
        hook_template: Template with {layer} placeholder.

    Returns:
        [n_layers, seq_len] array of metric values.
    """
    n_layers = model.cfg.n_layers
    seq_len = clean_tokens.shape[0]

    _, corrupted_cache = model.run_with_cache(corrupted_tokens)
    results = np.zeros((n_layers, seq_len))

    for layer in range(n_layers):
        hook_name = hook_template.format(layer=layer)
        corrupted_act = corrupted_cache[hook_name]

        for pos in range(seq_len):
            def patch_pos_hook(x, name, _ca=corrupted_act, _p=pos):
                return x.at[_p].set(_ca[_p])

            logits = model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[(hook_name, patch_pos_hook)],
            )
            results[layer, pos] = metric_fn(logits)

    return results


# ─── Ablation ────────────────────────────────────────────────────────────────


def zero_ablate(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    hook_names: list[str],
    metric_fn: Callable[[jnp.ndarray], float],
) -> dict[str, float]:
    """Zero-ablate each hook point and measure the effect.

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        hook_names: Hook points to ablate.
        metric_fn: Function(logits) -> float.

    Returns:
        Dict mapping hook_name -> metric value after ablation.
    """
    results = {}
    for hook_name in hook_names:
        def zero_hook(x, name):
            return jnp.zeros_like(x)

        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, zero_hook)])
        results[hook_name] = metric_fn(logits)
    return results


def mean_ablate(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    hook_names: list[str],
    metric_fn: Callable[[jnp.ndarray], float],
    mean_cache: dict[str, jnp.ndarray] | None = None,
) -> dict[str, float]:
    """Mean-ablate each hook point (replace with mean activation) and measure.

    If mean_cache is not provided, uses the activations from the input as
    the mean (per-position mean across the sequence).

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        hook_names: Hook points to ablate.
        metric_fn: Function(logits) -> float.
        mean_cache: Optional pre-computed mean activations.

    Returns:
        Dict mapping hook_name -> metric value after ablation.
    """
    if mean_cache is None:
        _, cache = model.run_with_cache(tokens)
        mean_cache = {}
        for name in hook_names:
            act = cache[name]
            # Mean across sequence dimension (keep shape for broadcasting)
            mean_cache[name] = jnp.mean(act, axis=0, keepdims=True) * jnp.ones_like(act)

    results = {}
    for hook_name in hook_names:
        mean_act = mean_cache[hook_name]

        def mean_hook(x, name, _mean=mean_act):
            return _mean

        logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, mean_hook)])
        results[hook_name] = metric_fn(logits)
    return results


def ablate_heads(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable[[jnp.ndarray], float],
    method: str = "zero",
) -> np.ndarray:
    """Ablate individual attention heads and measure the effect.

    Args:
        model: HookedTransformer.
        tokens: Input tokens.
        metric_fn: Function(logits) -> float.
        method: "zero" or "mean".

    Returns:
        [n_layers, n_heads] array of metric values.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    results = np.zeros((n_layers, n_heads))

    _, cache = model.run_with_cache(tokens)

    for layer in range(n_layers):
        hook_name = f"blocks.{layer}.attn.hook_z"
        z = cache[hook_name]

        for head in range(n_heads):
            if method == "zero":
                def ablate_hook(x, name, _h=head):
                    return x.at[:, _h, :].set(0.0)
            elif method == "mean":
                head_mean = jnp.mean(z[:, head, :], axis=0)
                def ablate_hook(x, name, _h=head, _m=head_mean):
                    return x.at[:, _h, :].set(_m)
            else:
                raise ValueError(f"Unknown ablation method: {method}")

            logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, ablate_hook)])
            results[layer, head] = metric_fn(logits)

    return results


# ─── Convenience Functions ───────────────────────────────────────────────────


def make_logit_diff_metric(
    correct_token: int, wrong_token: int, pos: int = -1
) -> Callable[[jnp.ndarray], float]:
    """Create a metric function that computes logit difference.

    Args:
        correct_token: Token ID of the correct answer.
        wrong_token: Token ID of the wrong answer.
        pos: Position to measure at (-1 for last).

    Returns:
        Function(logits) -> float.
    """
    def metric(logits):
        return _logit_diff(logits, correct_token, wrong_token, pos)
    return metric


def path_patch(
    model: HookedTransformer,
    clean_tokens: jnp.ndarray,
    corrupted_tokens: jnp.ndarray,
    sender_hook: str,
    receiver_hooks: list[str],
    metric_fn: Callable[[jnp.ndarray], float],
) -> float:
    """Path patching: patch a specific edge in the computational graph.

    Corrupts the sender's output, then measures the effect only through
    specific receiver hooks. All other paths use clean activations.

    This is more fine-grained than activation patching: instead of replacing
    an entire activation, it only corrupts the component that flows through
    a specific downstream path.

    Algorithm:
    1. Run corrupted input, cache sender's output.
    2. Run clean input with the sender's output replaced by corrupted.
    3. For each receiver, only use the corrupted-sender version at that hook.

    Args:
        model: HookedTransformer.
        clean_tokens: Clean input tokens.
        corrupted_tokens: Corrupted input tokens.
        sender_hook: Hook name of the sending component (e.g., "blocks.0.hook_attn_out").
        receiver_hooks: Hook names where the corrupted signal is received
            (e.g., ["blocks.1.attn.hook_q", "blocks.1.attn.hook_k"]).
        metric_fn: Function(logits) -> float.

    Returns:
        Metric value with the sender->receiver path corrupted.
    """
    # Step 1: Get corrupted sender activation
    _, corrupted_cache = model.run_with_cache(corrupted_tokens)
    corrupted_sender = corrupted_cache[sender_hook]

    # Step 2: Run clean with sender patched in
    def patch_sender(x, name, _cs=corrupted_sender):
        return _cs

    _, patched_cache = model.run_with_cache(clean_tokens)
    # Actually need to get the effect of the corrupted sender on the receiver
    # Run the model with sender patched
    def make_hook_fn(patched_val):
        def hook_fn(x, name, _pv=patched_val):
            return _pv
        return hook_fn

    # The clean run with sender corrupted gives us the "corrupted path" activations
    # We need to capture what the receivers look like when the sender is corrupted
    capture = {}
    def capture_hook(name):
        def fn(x, hook_name, _n=name):
            capture[_n] = x
            return x
        return fn

    fwd_hooks = [(sender_hook, patch_sender)]
    for rh in receiver_hooks:
        fwd_hooks.append((rh, capture_hook(rh)))

    model.run_with_hooks(clean_tokens, fwd_hooks=fwd_hooks)

    # Step 3: Run clean with only the receiver hooks getting the corrupted-path values
    receiver_fwd_hooks = []
    for rh in receiver_hooks:
        if rh in capture:
            patched_val = capture[rh]
            receiver_fwd_hooks.append((rh, make_hook_fn(patched_val)))

    logits = model.run_with_hooks(clean_tokens, fwd_hooks=receiver_fwd_hooks)
    return metric_fn(logits)


def path_patch_by_receiver(
    model: HookedTransformer,
    clean_tokens: jnp.ndarray,
    corrupted_tokens: jnp.ndarray,
    sender_hook: str,
    metric_fn: Callable[[jnp.ndarray], float],
    receiver_templates: list[str] | None = None,
) -> dict[str, float]:
    """Path patch from a sender to each possible receiver independently.

    For each downstream hook point, measures the effect of the corrupted
    sender flowing only through that specific receiver.

    Args:
        model: HookedTransformer.
        clean_tokens: Clean tokens.
        corrupted_tokens: Corrupted tokens.
        sender_hook: Sender hook name.
        metric_fn: Metric function.
        receiver_templates: List of receiver hook name templates.
            Defaults to q, k, v hooks in all downstream layers.

    Returns:
        Dict mapping receiver hook name -> metric value.
    """
    if receiver_templates is None:
        # Default: Q, K, V of all downstream attention heads
        # Parse sender layer
        sender_layer = None
        for part in sender_hook.split("."):
            try:
                sender_layer = int(part)
                break
            except ValueError:
                continue

        receiver_templates = []
        if sender_layer is not None:
            for l in range(sender_layer + 1, model.cfg.n_layers):
                for act in ["q", "k", "v"]:
                    receiver_templates.append(f"blocks.{l}.attn.hook_{act}")

    results = {}
    for receiver in receiver_templates:
        metric_val = path_patch(
            model, clean_tokens, corrupted_tokens,
            sender_hook, [receiver], metric_fn,
        )
        results[receiver] = metric_val

    return results


def make_loss_metric(
    target_token: int, pos: int = -1
) -> Callable[[jnp.ndarray], float]:
    """Create a metric function that computes loss for a target token.

    Args:
        target_token: Token ID to measure loss for.
        pos: Position to measure at (-1 for last).

    Returns:
        Function(logits) -> float.
    """
    def metric(logits):
        return _loss_at_pos(logits, target_token, pos)
    return metric
