"""Model comparison and diffing utilities.

Tools for comparing two models (e.g., before/after finetuning):
- weight_diff: Compare weights layer by layer
- activation_diff: Compare cached activations at each hook
- logit_diff_on_dataset: Compare logit outputs across prompts
- finetuning_attribution: Identify which weights changed most
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from irtk.hooked_transformer import HookedTransformer


def weight_diff(
    model_a: HookedTransformer,
    model_b: HookedTransformer,
) -> dict[str, dict[str, float]]:
    """Compare weights between two models, per named parameter.

    For each weight matrix, computes the L2 norm of the difference
    and the relative change (norm of diff / norm of original).

    Args:
        model_a: First model (e.g., original).
        model_b: Second model (e.g., finetuned).

    Returns:
        Dict mapping parameter path -> {"abs_diff": float, "rel_diff": float, "norm_a": float}.
    """
    results = {}

    # Compare embeddings
    for name, get_fn in [
        ("embed.W_E", lambda m: m.embed.W_E),
        ("unembed.W_U", lambda m: m.unembed.W_U),
        ("unembed.b_U", lambda m: m.unembed.b_U),
    ]:
        a = get_fn(model_a)
        b = get_fn(model_b)
        norm_a = float(jnp.linalg.norm(a))
        abs_diff = float(jnp.linalg.norm(a - b))
        rel_diff = abs_diff / max(norm_a, 1e-10)
        results[name] = {"abs_diff": abs_diff, "rel_diff": rel_diff, "norm_a": norm_a}

    if model_a.pos_embed is not None and model_b.pos_embed is not None:
        a = model_a.pos_embed.W_pos
        b = model_b.pos_embed.W_pos
        norm_a = float(jnp.linalg.norm(a))
        abs_diff = float(jnp.linalg.norm(a - b))
        results["pos_embed.W_pos"] = {
            "abs_diff": abs_diff,
            "rel_diff": abs_diff / max(norm_a, 1e-10),
            "norm_a": norm_a,
        }

    # Compare per-layer weights
    n_layers = min(model_a.cfg.n_layers, model_b.cfg.n_layers)
    for l in range(n_layers):
        block_a = model_a.blocks[l]
        block_b = model_b.blocks[l]

        # Attention weights
        for attr in ["W_Q", "W_K", "W_V", "W_O", "b_Q", "b_K", "b_V", "b_O"]:
            a = getattr(block_a.attn, attr)
            b = getattr(block_b.attn, attr)
            norm_a = float(jnp.linalg.norm(a))
            abs_diff = float(jnp.linalg.norm(a - b))
            results[f"blocks.{l}.attn.{attr}"] = {
                "abs_diff": abs_diff,
                "rel_diff": abs_diff / max(norm_a, 1e-10),
                "norm_a": norm_a,
            }

        # MLP weights
        for attr in ["W_in", "W_out", "b_in", "b_out"]:
            a = getattr(block_a.mlp, attr)
            b = getattr(block_b.mlp, attr)
            norm_a = float(jnp.linalg.norm(a))
            abs_diff = float(jnp.linalg.norm(a - b))
            results[f"blocks.{l}.mlp.{attr}"] = {
                "abs_diff": abs_diff,
                "rel_diff": abs_diff / max(norm_a, 1e-10),
                "norm_a": norm_a,
            }

        # LayerNorm weights
        for ln_name in ["ln1", "ln2"]:
            ln_a = getattr(block_a, ln_name)
            ln_b = getattr(block_b, ln_name)
            if ln_a is not None and ln_b is not None:
                if hasattr(ln_a, 'w'):
                    a = ln_a.w
                    b = ln_b.w
                    norm_a = float(jnp.linalg.norm(a))
                    abs_diff = float(jnp.linalg.norm(a - b))
                    results[f"blocks.{l}.{ln_name}.w"] = {
                        "abs_diff": abs_diff,
                        "rel_diff": abs_diff / max(norm_a, 1e-10),
                        "norm_a": norm_a,
                    }

    return results


def weight_diff_summary(
    model_a: HookedTransformer,
    model_b: HookedTransformer,
    top_k: int = 10,
) -> list[tuple[str, float, float]]:
    """Summarize the largest weight differences between two models.

    Args:
        model_a: First model.
        model_b: Second model.
        top_k: Number of largest differences to return.

    Returns:
        List of (param_name, abs_diff, rel_diff) tuples sorted by rel_diff.
    """
    diffs = weight_diff(model_a, model_b)
    entries = [(name, d["abs_diff"], d["rel_diff"]) for name, d in diffs.items()]
    entries.sort(key=lambda x: x[2], reverse=True)
    return entries[:top_k]


def activation_diff(
    model_a: HookedTransformer,
    model_b: HookedTransformer,
    tokens: jnp.ndarray,
) -> dict[str, float]:
    """Compare cached activations between two models on the same input.

    For each hook point, computes the L2 norm of the activation difference.

    Args:
        model_a: First model.
        model_b: Second model.
        tokens: Input tokens to run through both models.

    Returns:
        Dict mapping hook_name -> L2 norm of activation difference.
    """
    _, cache_a = model_a.run_with_cache(tokens)
    _, cache_b = model_b.run_with_cache(tokens)

    results = {}
    for name in cache_a.cache_dict:
        if name in cache_b.cache_dict:
            a = cache_a.cache_dict[name]
            b = cache_b.cache_dict[name]
            if a.shape == b.shape:
                results[name] = float(jnp.linalg.norm(a - b))

    return results


def logit_diff_on_dataset(
    model_a: HookedTransformer,
    model_b: HookedTransformer,
    token_sequences: list[jnp.ndarray],
) -> dict[str, np.ndarray]:
    """Compare logit outputs of two models across multiple prompts.

    For each prompt, computes various divergence measures between
    the two models' output distributions.

    Args:
        model_a: First model.
        model_b: Second model.
        token_sequences: List of token arrays to evaluate on.

    Returns:
        Dict with:
        - "logit_l2": [n_prompts] L2 norm of logit difference at last position
        - "kl_divergence": [n_prompts] KL(model_a || model_b) at last position
        - "top1_agree": [n_prompts] bool whether top-1 prediction matches
        - "mean_logit_l2": scalar mean of logit_l2
    """
    logit_l2s = []
    kl_divs = []
    top1_agrees = []

    for tokens in token_sequences:
        logits_a = model_a(tokens)
        logits_b = model_b(tokens)

        # L2 norm at last position
        diff = logits_a[-1] - logits_b[-1]
        logit_l2s.append(float(jnp.linalg.norm(diff)))

        # KL divergence: KL(a || b)
        log_probs_a = jax.nn.log_softmax(logits_a[-1])
        log_probs_b = jax.nn.log_softmax(logits_b[-1])
        probs_a = jax.nn.softmax(logits_a[-1])
        kl = float(jnp.sum(probs_a * (log_probs_a - log_probs_b)))
        kl_divs.append(max(kl, 0.0))  # Clamp small negative values from float precision

        # Top-1 agreement
        top1_agrees.append(bool(jnp.argmax(logits_a[-1]) == jnp.argmax(logits_b[-1])))

    return {
        "logit_l2": np.array(logit_l2s),
        "kl_divergence": np.array(kl_divs),
        "top1_agree": np.array(top1_agrees),
        "mean_logit_l2": float(np.mean(logit_l2s)),
    }


def finetuning_attribution(
    model_a: HookedTransformer,
    model_b: HookedTransformer,
) -> dict[str, np.ndarray]:
    """Attribute finetuning changes to attention heads and MLP layers.

    For each layer, computes:
    - Per-head weight change (sum of W_Q, W_K, W_V, W_O diffs for each head)
    - MLP weight change

    This helps identify which components were most affected by finetuning.

    Args:
        model_a: Original model.
        model_b: Finetuned model.

    Returns:
        Dict with:
        - "head_diff": [n_layers, n_heads] per-head total weight change
        - "mlp_diff": [n_layers] per-layer MLP total weight change
        - "ln_diff": [n_layers] per-layer LayerNorm weight change
    """
    n_layers = min(model_a.cfg.n_layers, model_b.cfg.n_layers)
    n_heads = model_a.cfg.n_heads

    head_diff = np.zeros((n_layers, n_heads))
    mlp_diff = np.zeros(n_layers)
    ln_diff = np.zeros(n_layers)

    for l in range(n_layers):
        block_a = model_a.blocks[l]
        block_b = model_b.blocks[l]

        # Per-head attribution
        for h in range(n_heads):
            total = 0.0
            for attr in ["W_Q", "W_K", "W_V"]:
                a = getattr(block_a.attn, attr)[h]
                b = getattr(block_b.attn, attr)[h]
                total += float(jnp.linalg.norm(a - b))
            # W_O
            a = block_a.attn.W_O[h]
            b = block_b.attn.W_O[h]
            total += float(jnp.linalg.norm(a - b))
            head_diff[l, h] = total

        # MLP
        mlp_total = 0.0
        for attr in ["W_in", "W_out", "b_in", "b_out"]:
            a = getattr(block_a.mlp, attr)
            b = getattr(block_b.mlp, attr)
            mlp_total += float(jnp.linalg.norm(a - b))
        mlp_diff[l] = mlp_total

        # LayerNorm
        ln_total = 0.0
        for ln_name in ["ln1", "ln2"]:
            ln_a = getattr(block_a, ln_name)
            ln_b = getattr(block_b, ln_name)
            if ln_a is not None and ln_b is not None and hasattr(ln_a, 'w'):
                ln_total += float(jnp.linalg.norm(ln_a.w - ln_b.w))
        ln_diff[l] = ln_total

    return {
        "head_diff": head_diff,
        "mlp_diff": mlp_diff,
        "ln_diff": ln_diff,
    }
