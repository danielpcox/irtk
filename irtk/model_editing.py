"""Localized model editing (ROME-style).

Tools for surgically editing factual associations in model weights:
- locate_decisive_layer: Find which MLP layer is most responsible for a fact
- compute_key_vector: Extract the key representation for a subject at a layer
- compute_value_target: Compute the target value vector for a new fact
- apply_rank_one_edit: Apply rank-one weight update to an MLP layer
- edit_fact: High-level API to change a factual association
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from irtk.hooked_transformer import HookedTransformer
from irtk.hook_points import HookState


def locate_decisive_layer(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    metric_fn: Callable[[jnp.ndarray], float],
    corrupt_pos: int,
    noise_std: float = 0.1,
) -> dict:
    """Find which MLP layer is most decisive for recovering a fact.

    Uses causal tracing: corrupts the embedding at a position, then
    restores the MLP output at each layer to find which layer best
    recovers the metric.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        metric_fn: Function(logits) -> float measuring the fact.
        corrupt_pos: Position to corrupt (usually the subject token).
        noise_std: Noise std for corruption.

    Returns:
        Dict with:
        - "layer_effects": [n_layers] recovery when restoring each MLP layer
        - "decisive_layer": index of the most decisive layer
        - "clean_metric": clean model metric
        - "corrupted_metric": corrupted model metric
    """
    tokens = jnp.array(tokens)
    n_layers = model.cfg.n_layers

    # Clean metric
    clean_logits = model(tokens)
    clean_metric = metric_fn(clean_logits)

    # Get clean cache
    _, clean_cache = model.run_with_cache(tokens)

    # Corruption hook
    key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key, (model.cfg.d_model,)) * noise_std

    def corrupt_hook(x, name):
        return x.at[corrupt_pos].add(noise)

    corrupted_logits = model.run_with_hooks(
        tokens, fwd_hooks=[("hook_embed", corrupt_hook)]
    )
    corrupted_metric = metric_fn(corrupted_logits)

    # Restore each MLP layer and measure recovery
    layer_effects = np.zeros(n_layers)
    for layer in range(n_layers):
        mlp_key = f"blocks.{layer}.hook_mlp_out"
        if mlp_key not in clean_cache.cache_dict:
            continue

        clean_mlp = clean_cache.cache_dict[mlp_key]

        def restore_hook(x, name, _cm=clean_mlp, _p=corrupt_pos):
            return x.at[_p].set(_cm[_p])

        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[("hook_embed", corrupt_hook), (mlp_key, restore_hook)],
        )
        layer_effects[layer] = metric_fn(logits) - corrupted_metric

    decisive = int(np.argmax(layer_effects))

    return {
        "layer_effects": layer_effects,
        "decisive_layer": decisive,
        "clean_metric": float(clean_metric),
        "corrupted_metric": float(corrupted_metric),
    }


def compute_key_vector(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
    pos: int,
) -> np.ndarray:
    """Extract the MLP input (key) vector at a specific layer and position.

    In ROME, the key vector is the input to the MLP at the decisive layer,
    at the subject token position. This is what the MLP "recognizes".

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        layer: MLP layer index.
        pos: Token position.

    Returns:
        [d_model] key vector.
    """
    tokens = jnp.array(tokens)
    _, cache = model.run_with_cache(tokens)

    # MLP input is the residual stream after attention + layernorm
    # We use hook_mlp_in if available, otherwise hook_resid_mid
    mlp_in_key = f"blocks.{layer}.hook_mlp_in"
    resid_mid_key = f"blocks.{layer}.hook_resid_mid"

    if mlp_in_key in cache.cache_dict:
        return np.array(cache.cache_dict[mlp_in_key][pos])
    elif resid_mid_key in cache.cache_dict:
        return np.array(cache.cache_dict[resid_mid_key][pos])
    else:
        # Fallback: use resid_pre + attn_out
        resid_pre_key = f"blocks.{layer}.hook_resid_pre"
        attn_out_key = f"blocks.{layer}.hook_attn_out"
        if resid_pre_key in cache.cache_dict and attn_out_key in cache.cache_dict:
            resid = cache.cache_dict[resid_pre_key][pos]
            attn = cache.cache_dict[attn_out_key][pos]
            return np.array(resid + attn)
        elif resid_pre_key in cache.cache_dict:
            return np.array(cache.cache_dict[resid_pre_key][pos])
        else:
            return np.zeros(model.cfg.d_model)


def compute_value_target(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
    pos: int,
    target_token: int,
    coeff: float = 1.0,
) -> np.ndarray:
    """Compute the target value vector for a fact edit.

    The value target is a vector that, when added to the MLP output,
    increases the probability of the target token at the output position.

    Uses a simple approach: project the unembedding direction for the
    target token, scaled by the coefficient.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        layer: MLP layer index.
        pos: Token position where we want the new prediction.
        target_token: Token ID of the desired output.
        coeff: Scaling coefficient for the edit strength.

    Returns:
        [d_model] target value vector (desired MLP output delta).
    """
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    target_dir = np.array(W_U[:, target_token])  # [d_model]

    # Normalize and scale
    norm = np.linalg.norm(target_dir)
    if norm > 1e-10:
        target_dir = target_dir / norm

    # Get current MLP output to calibrate scale
    _, cache = model.run_with_cache(jnp.array(tokens))
    mlp_out_key = f"blocks.{layer}.hook_mlp_out"
    if mlp_out_key in cache.cache_dict:
        current_scale = float(np.linalg.norm(cache.cache_dict[mlp_out_key][pos]))
    else:
        current_scale = 1.0

    return target_dir * coeff * max(current_scale, 1.0)


def apply_rank_one_edit(
    model: HookedTransformer,
    layer: int,
    key_vector: np.ndarray,
    value_vector: np.ndarray,
) -> HookedTransformer:
    """Apply a rank-one edit to an MLP layer's output weights.

    Modifies W_out such that the MLP maps the key vector to the
    value vector in addition to its existing behavior:

        W_out_new = W_out + value_vector @ key_vector^T / (key_vector^T @ key_vector)

    This is a simplified version of the ROME update rule.

    Args:
        model: HookedTransformer.
        layer: MLP layer index.
        key_vector: [d_model] key vector (what to match).
        value_vector: [d_model] value vector (what to produce).

    Returns:
        New model with the edited weights.
    """
    k = jnp.array(key_vector, dtype=jnp.float32)
    v = jnp.array(value_vector, dtype=jnp.float32)

    # Project key through W_in to get the hidden representation
    W_in = model.blocks[layer].mlp.W_in  # [d_model, d_mlp]
    k_hidden = k @ W_in  # [d_mlp]

    # Rank-one update to W_out: W_out += v @ k_hidden^T / (k_hidden^T @ k_hidden)
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    k_norm_sq = jnp.dot(k_hidden, k_hidden)

    if k_norm_sq > 1e-10:
        delta = jnp.outer(k_hidden, v) / k_norm_sq  # [d_mlp, d_model]
        new_W_out = W_out + delta
    else:
        new_W_out = W_out

    return eqx.tree_at(
        lambda m: m.blocks[layer].mlp.W_out, model, new_W_out
    )


def edit_fact(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    subject_pos: int,
    target_token: int,
    layer: Optional[int] = None,
    metric_fn: Optional[Callable] = None,
    coeff: float = 1.0,
) -> dict:
    """High-level API to edit a factual association in the model.

    Combines locate_decisive_layer, compute_key_vector, compute_value_target,
    and apply_rank_one_edit into a single call.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens (prompt containing the fact).
        subject_pos: Position of the subject token.
        target_token: Token ID of the desired new answer.
        layer: MLP layer to edit (auto-detected if None).
        metric_fn: Metric for locating decisive layer (required if layer is None).
        coeff: Edit strength coefficient.

    Returns:
        Dict with:
        - "edited_model": the new model with the edit applied
        - "layer": which layer was edited
        - "key_vector": the key vector used
        - "value_vector": the value target used
    """
    tokens = jnp.array(tokens)

    # Auto-detect layer
    if layer is None:
        if metric_fn is None:
            def metric_fn(logits, _t=target_token):
                return float(logits[-1, _t])

        result = locate_decisive_layer(
            model, tokens, metric_fn, corrupt_pos=subject_pos
        )
        layer = result["decisive_layer"]

    # Compute edit vectors
    key_vec = compute_key_vector(model, tokens, layer, subject_pos)
    val_vec = compute_value_target(model, tokens, layer, -1, target_token, coeff=coeff)

    # Apply edit
    edited = apply_rank_one_edit(model, layer, key_vec, val_vec)

    return {
        "edited_model": edited,
        "layer": layer,
        "key_vector": key_vec,
        "value_vector": val_vec,
    }
