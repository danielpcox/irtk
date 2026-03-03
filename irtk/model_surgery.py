"""Model surgery tools for component transplant and replacement.

Functional tools for surgically modifying model weights:
- transplant_heads: Copy attention head weights between models
- transplant_mlp: Copy MLP weights between models
- knockout_head_weights: Zero out a head's weights permanently
- compare_heads_across_models: Compare corresponding heads in two models
- zero_out_layer: Make a layer act as identity (skip connection only)
"""

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def transplant_heads(
    donor: HookedTransformer,
    recipient: HookedTransformer,
    heads: list[tuple[int, int]],
) -> HookedTransformer:
    """Transplant attention head weights from donor to recipient.

    Copies W_Q, W_K, W_V, W_O and associated biases for the specified
    heads. Returns a new model (functional, no mutation).

    Args:
        donor: Model to copy head weights from.
        recipient: Model to receive the head weights.
        heads: List of (layer, head) tuples to transplant.

    Returns:
        New HookedTransformer with transplanted heads.
    """
    model = recipient
    for layer, head in heads:
        d_attn = donor.blocks[layer].attn
        r_attn = model.blocks[layer].attn

        # W_Q
        new_W_Q = r_attn.W_Q.at[head].set(d_attn.W_Q[head])
        model = eqx.tree_at(
            lambda m, _l=layer: m.blocks[_l].attn.W_Q, model, new_W_Q
        )

        # b_Q
        new_b_Q = r_attn.b_Q.at[head].set(d_attn.b_Q[head])
        model = eqx.tree_at(
            lambda m, _l=layer: m.blocks[_l].attn.b_Q, model, new_b_Q
        )

        # For K/V, handle GQA (n_kv_heads may differ from n_heads)
        n_kv = d_attn.W_K.shape[0]
        kv_head = head if head < n_kv else head % n_kv

        # W_K
        new_W_K = r_attn.W_K.at[kv_head].set(d_attn.W_K[kv_head])
        model = eqx.tree_at(
            lambda m, _l=layer: m.blocks[_l].attn.W_K, model, new_W_K
        )

        # b_K
        new_b_K = r_attn.b_K.at[kv_head].set(d_attn.b_K[kv_head])
        model = eqx.tree_at(
            lambda m, _l=layer: m.blocks[_l].attn.b_K, model, new_b_K
        )

        # W_V
        new_W_V = r_attn.W_V.at[kv_head].set(d_attn.W_V[kv_head])
        model = eqx.tree_at(
            lambda m, _l=layer: m.blocks[_l].attn.W_V, model, new_W_V
        )

        # b_V
        new_b_V = r_attn.b_V.at[kv_head].set(d_attn.b_V[kv_head])
        model = eqx.tree_at(
            lambda m, _l=layer: m.blocks[_l].attn.b_V, model, new_b_V
        )

        # W_O
        new_W_O = r_attn.W_O.at[head].set(d_attn.W_O[head])
        model = eqx.tree_at(
            lambda m, _l=layer: m.blocks[_l].attn.W_O, model, new_W_O
        )

        # b_O (shared across heads, so only update if this is the only head)
        # b_O is [d_model], not per-head, so skip per-head transplant

    return model


def transplant_mlp(
    donor: HookedTransformer,
    recipient: HookedTransformer,
    layer: int,
) -> HookedTransformer:
    """Transplant MLP weights from donor to recipient at a specific layer.

    Args:
        donor: Model to copy MLP weights from.
        recipient: Model to receive the MLP weights.
        layer: Layer index to transplant.

    Returns:
        New HookedTransformer with transplanted MLP.
    """
    model = recipient
    d_mlp = donor.blocks[layer].mlp

    # W_in
    model = eqx.tree_at(
        lambda m, _l=layer: m.blocks[_l].mlp.W_in, model, d_mlp.W_in
    )
    # b_in
    model = eqx.tree_at(
        lambda m, _l=layer: m.blocks[_l].mlp.b_in, model, d_mlp.b_in
    )
    # W_out
    model = eqx.tree_at(
        lambda m, _l=layer: m.blocks[_l].mlp.W_out, model, d_mlp.W_out
    )
    # b_out
    model = eqx.tree_at(
        lambda m, _l=layer: m.blocks[_l].mlp.b_out, model, d_mlp.b_out
    )

    # Handle gated MLP if present
    if hasattr(d_mlp, 'W_gate') and d_mlp.W_gate is not None:
        r_mlp = recipient.blocks[layer].mlp
        if hasattr(r_mlp, 'W_gate') and r_mlp.W_gate is not None:
            model = eqx.tree_at(
                lambda m, _l=layer: m.blocks[_l].mlp.W_gate, model, d_mlp.W_gate
            )

    return model


def knockout_head_weights(
    model: HookedTransformer,
    layer: int,
    head: int,
) -> HookedTransformer:
    """Zero out all weights for a specific attention head permanently.

    Unlike runtime ablation (which hooks the output), this modifies the
    weights so the head produces zero output. More efficient for batch
    evaluation across many prompts.

    Args:
        model: HookedTransformer.
        layer: Attention layer index.
        head: Attention head index.

    Returns:
        New HookedTransformer with the head's weights zeroed.
    """
    attn = model.blocks[layer].attn

    # Zero W_O for this head (this is sufficient to zero the head's contribution)
    new_W_O = attn.W_O.at[head].set(jnp.zeros_like(attn.W_O[head]))
    model = eqx.tree_at(
        lambda m, _l=layer: m.blocks[_l].attn.W_O, model, new_W_O
    )

    return model


def compare_heads_across_models(
    model_a: HookedTransformer,
    model_b: HookedTransformer,
    layer: int,
    head: int,
) -> dict:
    """Compare corresponding attention heads in two models.

    Computes cosine similarity and L2 distance between the weight
    matrices of the same head in two models.

    Args:
        model_a: First model.
        model_b: Second model.
        layer: Attention layer index.
        head: Attention head index.

    Returns:
        Dict with per-matrix comparison:
        - "W_Q_cosine", "W_K_cosine", "W_V_cosine", "W_O_cosine": cosine similarities
        - "W_Q_l2", "W_K_l2", "W_V_l2", "W_O_l2": L2 distances
        - "overall_cosine": Average cosine similarity across all matrices
    """
    attn_a = model_a.blocks[layer].attn
    attn_b = model_b.blocks[layer].attn

    results = {}
    cosines = []

    for name, wa, wb in [
        ("W_Q", attn_a.W_Q[head], attn_b.W_Q[head]),
        ("W_K", attn_a.W_K[head], attn_b.W_K[head]),
        ("W_V", attn_a.W_V[head], attn_b.W_V[head]),
        ("W_O", attn_a.W_O[head], attn_b.W_O[head]),
    ]:
        wa_flat = np.array(wa).flatten()
        wb_flat = np.array(wb).flatten()

        norm_a = np.linalg.norm(wa_flat)
        norm_b = np.linalg.norm(wb_flat)

        if norm_a > 1e-10 and norm_b > 1e-10:
            cos = float(np.dot(wa_flat, wb_flat) / (norm_a * norm_b))
        else:
            cos = 0.0

        l2 = float(np.linalg.norm(wa_flat - wb_flat))
        results[f"{name}_cosine"] = cos
        results[f"{name}_l2"] = l2
        cosines.append(cos)

    results["overall_cosine"] = float(np.mean(cosines))
    return results


def zero_out_layer(
    model: HookedTransformer,
    layer: int,
    component: str = "both",
) -> HookedTransformer:
    """Make a layer produce zero output (identity residual connection only).

    Zeros the output projections so the layer's attention and/or MLP
    contribute nothing to the residual stream.

    Args:
        model: HookedTransformer.
        layer: Layer index.
        component: "attn", "mlp", or "both".

    Returns:
        New HookedTransformer with the layer's output zeroed.
    """
    if component in ("attn", "both"):
        attn = model.blocks[layer].attn
        new_W_O = jnp.zeros_like(attn.W_O)
        model = eqx.tree_at(
            lambda m, _l=layer: m.blocks[_l].attn.W_O, model, new_W_O
        )
        new_b_O = jnp.zeros_like(attn.b_O)
        model = eqx.tree_at(
            lambda m, _l=layer: m.blocks[_l].attn.b_O, model, new_b_O
        )

    if component in ("mlp", "both"):
        mlp = model.blocks[layer].mlp
        new_W_out = jnp.zeros_like(mlp.W_out)
        model = eqx.tree_at(
            lambda m, _l=layer: m.blocks[_l].mlp.W_out, model, new_W_out
        )
        new_b_out = jnp.zeros_like(mlp.b_out)
        model = eqx.tree_at(
            lambda m, _l=layer: m.blocks[_l].mlp.b_out, model, new_b_out
        )

    return model
