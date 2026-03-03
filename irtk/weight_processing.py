"""Weight processing utilities for cleaner mechanistic analysis.

These operations return new models (via eqx.tree_at) with processed weights:
- fold_layer_norm: Fold LayerNorm into adjacent weights
- center_writing_weights: Center the writing weights of each component
- center_unembed: Center the unembedding matrix

After these transformations, the residual stream decomposition and logit lens
give cleaner results because they remove the confounding effect of LayerNorm.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx

from irtk.hooked_transformer import HookedTransformer
from irtk.components.layer_norm import LayerNorm, RMSNorm


def fold_layer_norm(model: HookedTransformer) -> HookedTransformer:
    """Fold LayerNorm parameters into adjacent linear weights.

    For each LayerNorm with learned w and b, fold them into the downstream
    linear weights so the LayerNorm becomes a pure normalization (no affine).

    This makes the model equivalent but with the affine part absorbed:
    - ln.w, ln.b folded into W_Q, W_K, W_V (for attention)
    - ln.w, ln.b folded into W_in (for MLP)
    - ln_final.w, ln_final.b folded into W_U

    After folding, all LayerNorm w are set to ones and b to zeros.

    Args:
        model: HookedTransformer.

    Returns:
        New model with folded weights.
    """
    new_model = model

    for l in range(model.cfg.n_layers):
        block = new_model.blocks[l]

        # Fold ln1 (pre-attention LayerNorm) into W_Q, W_K, W_V
        ln1 = block.ln1
        if ln1 is not None and isinstance(ln1, (LayerNorm, RMSNorm)):
            w_ln = ln1.w  # [d_model]

            # W_Q, W_K, W_V: [n_heads, d_model, d_head]
            # Each column of W_Q is multiplied by the corresponding w_ln element
            for attr in ["W_Q", "W_K", "W_V"]:
                W = getattr(block.attn, attr)
                # w_ln broadcast: [d_model] -> [1, d_model, 1]
                new_W = W * w_ln[None, :, None]
                new_model = eqx.tree_at(
                    lambda m, _l=l, _a=attr: getattr(m.blocks[_l].attn, _a),
                    new_model, new_W,
                )

            # Also fold bias if LayerNorm has one
            if isinstance(ln1, LayerNorm):
                b_ln = ln1.b  # [d_model]
                for attr, b_attr in [("W_Q", "b_Q"), ("W_K", "b_K"), ("W_V", "b_V")]:
                    W = getattr(block.attn, attr)
                    b = getattr(block.attn, b_attr)
                    # b_ln @ W_attr for each head: [d_model] @ [n_heads, d_model, d_head] -> [n_heads, d_head]
                    new_b = b + jnp.einsum("m,nmh->nh", b_ln, W)
                    new_model = eqx.tree_at(
                        lambda m, _l=l, _b=b_attr: getattr(m.blocks[_l].attn, _b),
                        new_model, new_b,
                    )

            # Reset ln1 to identity
            new_model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].ln1.w, new_model, jnp.ones_like(ln1.w))
            if isinstance(ln1, LayerNorm):
                new_model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].ln1.b, new_model, jnp.zeros_like(ln1.b))

        # Fold ln2 (pre-MLP LayerNorm) into W_in
        ln2 = block.ln2
        if ln2 is not None and isinstance(ln2, (LayerNorm, RMSNorm)):
            w_ln = ln2.w
            W_in = block.mlp.W_in  # [d_model, d_mlp]
            new_W_in = W_in * w_ln[:, None]
            new_model = eqx.tree_at(
                lambda m, _l=l: m.blocks[_l].mlp.W_in, new_model, new_W_in,
            )

            if isinstance(ln2, LayerNorm):
                b_ln = ln2.b
                b_in = block.mlp.b_in
                new_b_in = b_in + b_ln @ W_in
                new_model = eqx.tree_at(
                    lambda m, _l=l: m.blocks[_l].mlp.b_in, new_model, new_b_in,
                )

            new_model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].ln2.w, new_model, jnp.ones_like(ln2.w))
            if isinstance(ln2, LayerNorm):
                new_model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].ln2.b, new_model, jnp.zeros_like(ln2.b))

    # Fold ln_final into W_U
    ln_final = new_model.ln_final
    if ln_final is not None and isinstance(ln_final, (LayerNorm, RMSNorm)):
        w_ln = ln_final.w
        W_U = new_model.unembed.W_U  # [d_model, d_vocab]
        new_W_U = W_U * w_ln[:, None]
        new_model = eqx.tree_at(lambda m: m.unembed.W_U, new_model, new_W_U)

        if isinstance(ln_final, LayerNorm):
            b_ln = ln_final.b
            b_U = new_model.unembed.b_U
            new_b_U = b_U + b_ln @ W_U
            new_model = eqx.tree_at(lambda m: m.unembed.b_U, new_model, new_b_U)

        new_model = eqx.tree_at(lambda m: m.ln_final.w, new_model, jnp.ones_like(ln_final.w))
        if isinstance(ln_final, LayerNorm):
            new_model = eqx.tree_at(lambda m: m.ln_final.b, new_model, jnp.zeros_like(ln_final.b))

    return new_model


def center_writing_weights(model: HookedTransformer) -> HookedTransformer:
    """Center the writing weights of each component.

    Subtracts the mean of each writing weight matrix so that the
    residual stream decomposition sums correctly. This removes a
    constant bias from each component's contribution.

    Writing weights: W_O (attention output), W_out (MLP output).
    These write to the residual stream, so centering them removes
    the component's mean contribution to all positions equally.

    Args:
        model: HookedTransformer.

    Returns:
        New model with centered writing weights.
    """
    new_model = model

    for l in range(model.cfg.n_layers):
        # Center W_O: [n_heads, d_head, d_model] -> center over d_model
        W_O = new_model.blocks[l].attn.W_O
        W_O_mean = W_O.mean(axis=-1, keepdims=True)
        new_W_O = W_O - W_O_mean
        new_model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].attn.W_O, new_model, new_W_O,
        )

        # Center b_O: [d_model] -> subtract mean
        b_O = new_model.blocks[l].attn.b_O
        new_b_O = b_O - b_O.mean()
        new_model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].attn.b_O, new_model, new_b_O,
        )

        # Center W_out: [d_mlp, d_model] -> center over d_model
        W_out = new_model.blocks[l].mlp.W_out
        W_out_mean = W_out.mean(axis=-1, keepdims=True)
        new_W_out = W_out - W_out_mean
        new_model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].mlp.W_out, new_model, new_W_out,
        )

        # Center b_out: [d_model]
        b_out = new_model.blocks[l].mlp.b_out
        new_b_out = b_out - b_out.mean()
        new_model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].mlp.b_out, new_model, new_b_out,
        )

    return new_model


def center_unembed(model: HookedTransformer) -> HookedTransformer:
    """Center the unembedding matrix.

    Subtracts the mean logit across tokens, so the model's output is
    zero-centered. This doesn't change predictions (softmax is shift-invariant)
    but makes logit attribution more interpretable.

    Args:
        model: HookedTransformer.

    Returns:
        New model with centered unembed.
    """
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    b_U = model.unembed.b_U  # [d_vocab]

    # Center over vocab dimension
    new_W_U = W_U - W_U.mean(axis=-1, keepdims=True)
    new_b_U = b_U - b_U.mean()

    new_model = eqx.tree_at(lambda m: m.unembed.W_U, model, new_W_U)
    new_model = eqx.tree_at(lambda m: m.unembed.b_U, new_model, new_b_U)

    return new_model
