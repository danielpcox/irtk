"""Weight loading from HuggingFace pretrained models.

Loads weights and assigns them to HookedTransformer via eqx.tree_at.
Supports both Flax-native and PyTorch-only models.
"""

from typing import Optional

import numpy as np
import jax.numpy as jnp
import equinox as eqx

from irtk.hooked_transformer_config import HookedTransformerConfig


def load_pretrained_weights(model, cfg: HookedTransformerConfig, model_name: str, **hf_kwargs):
    """Load pretrained weights into a HookedTransformer.

    Dispatches to architecture-specific loaders.

    Args:
        model: HookedTransformer instance to populate.
        cfg: Model configuration.
        model_name: HuggingFace model name.
        **hf_kwargs: Additional keyword arguments passed to
            AutoModelForCausalLM.from_pretrained (e.g., revision).
    """
    arch = cfg.original_architecture

    if arch == "gpt2":
        return _load_gpt2(model, cfg, model_name, **hf_kwargs)
    elif arch == "gpt_neo":
        return _load_gpt_neo(model, cfg, model_name, **hf_kwargs)
    elif arch == "gpt_neox":
        return _load_gpt_neox(model, cfg, model_name, **hf_kwargs)
    elif arch in ("llama", "mistral"):
        return _load_llama(model, cfg, model_name, **hf_kwargs)
    else:
        raise ValueError(f"No weight loader for architecture: {arch}")


def _load_gpt2(model, cfg, model_name: str, **hf_kwargs):
    """Load GPT-2 weights from HuggingFace.

    GPT-2 has Flax support, but we use PyTorch for consistency across all models.
    """
    import torch
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, **hf_kwargs)
    sd = hf_model.state_dict()

    def to_jax(key):
        return jnp.array(sd[key].detach().cpu().numpy())

    # Token embeddings
    model = eqx.tree_at(lambda m: m.embed.W_E, model, to_jax("transformer.wte.weight"))
    # Position embeddings
    model = eqx.tree_at(lambda m: m.pos_embed.W_pos, model, to_jax("transformer.wpe.weight"))

    for l in range(cfg.n_layers):
        p = f"transformer.h.{l}"

        # Layer norms
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].ln1.w, model, to_jax(f"{p}.ln_1.weight")
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].ln1.b, model, to_jax(f"{p}.ln_1.bias")
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].ln2.w, model, to_jax(f"{p}.ln_2.weight")
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].ln2.b, model, to_jax(f"{p}.ln_2.bias")
        )

        # Attention: GPT-2 stores QKV as a combined [d_model, 3*d_model] matrix
        # Split into Q, K, V and reshape to [n_heads, d_model, d_head]
        qkv_w = to_jax(f"{p}.attn.c_attn.weight")  # [d_model, 3*d_model]
        qkv_b = to_jax(f"{p}.attn.c_attn.bias")     # [3*d_model]

        d_model = cfg.d_model
        n_heads = cfg.n_heads
        d_head = cfg.d_head

        # Split along last dim into Q, K, V
        W_Q, W_K, W_V = jnp.split(qkv_w, 3, axis=-1)  # each [d_model, d_model]
        b_Q, b_K, b_V = jnp.split(qkv_b, 3, axis=-1)    # each [d_model]

        # Reshape to per-head: [d_model, d_model] -> [n_heads, d_model, d_head]
        W_Q = W_Q.reshape(d_model, n_heads, d_head).transpose(1, 0, 2)
        W_K = W_K.reshape(d_model, n_heads, d_head).transpose(1, 0, 2)
        W_V = W_V.reshape(d_model, n_heads, d_head).transpose(1, 0, 2)
        b_Q = b_Q.reshape(n_heads, d_head)
        b_K = b_K.reshape(n_heads, d_head)
        b_V = b_V.reshape(n_heads, d_head)

        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.W_Q, model, W_Q)
        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.W_K, model, W_K)
        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.W_V, model, W_V)
        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.b_Q, model, b_Q)
        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.b_K, model, b_K)
        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.b_V, model, b_V)

        # Output projection: [d_model, d_model] -> [n_heads, d_head, d_model]
        W_O = to_jax(f"{p}.attn.c_proj.weight")  # [d_model, d_model]
        b_O = to_jax(f"{p}.attn.c_proj.bias")     # [d_model]
        W_O = W_O.reshape(n_heads, d_head, d_model)

        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.W_O, model, W_O)
        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.b_O, model, b_O)

        # MLP
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].mlp.W_in, model, to_jax(f"{p}.mlp.c_fc.weight")
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].mlp.b_in, model, to_jax(f"{p}.mlp.c_fc.bias")
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].mlp.W_out, model, to_jax(f"{p}.mlp.c_proj.weight")
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].mlp.b_out, model, to_jax(f"{p}.mlp.c_proj.bias")
        )

    # Final layer norm
    model = eqx.tree_at(lambda m: m.ln_final.w, model, to_jax("transformer.ln_f.weight"))
    model = eqx.tree_at(lambda m: m.ln_final.b, model, to_jax("transformer.ln_f.bias"))

    # Unembed: GPT-2 ties embed and unembed weights
    model = eqx.tree_at(lambda m: m.unembed.W_U, model, to_jax("transformer.wte.weight").T)
    # b_U stays zero (GPT-2 has no unembed bias)

    del hf_model, sd
    return model


def _load_gpt_neo(model, cfg, model_name: str, **hf_kwargs):
    """Load GPT-Neo weights from HuggingFace."""
    import torch
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, **hf_kwargs)
    sd = hf_model.state_dict()

    def to_jax(key):
        return jnp.array(sd[key].detach().cpu().numpy())

    d_model = cfg.d_model
    n_heads = cfg.n_heads
    d_head = cfg.d_head

    model = eqx.tree_at(lambda m: m.embed.W_E, model, to_jax("transformer.wte.weight"))
    model = eqx.tree_at(lambda m: m.pos_embed.W_pos, model, to_jax("transformer.wpe.weight"))

    for l in range(cfg.n_layers):
        p = f"transformer.h.{l}"

        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].ln1.w, model, to_jax(f"{p}.ln_1.weight")
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].ln1.b, model, to_jax(f"{p}.ln_1.bias")
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].ln2.w, model, to_jax(f"{p}.ln_2.weight")
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].ln2.b, model, to_jax(f"{p}.ln_2.bias")
        )

        # GPT-Neo has separate Q, K, V projections
        W_Q = to_jax(f"{p}.attn.attention.q_proj.weight").T  # [d_model, d_model]
        W_K = to_jax(f"{p}.attn.attention.k_proj.weight").T
        W_V = to_jax(f"{p}.attn.attention.v_proj.weight").T

        W_Q = W_Q.reshape(d_model, n_heads, d_head).transpose(1, 0, 2)
        W_K = W_K.reshape(d_model, n_heads, d_head).transpose(1, 0, 2)
        W_V = W_V.reshape(d_model, n_heads, d_head).transpose(1, 0, 2)

        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.W_Q, model, W_Q)
        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.W_K, model, W_K)
        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.W_V, model, W_V)

        # Zero biases (GPT-Neo attention has no bias)
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].attn.b_Q, model, jnp.zeros((n_heads, d_head))
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].attn.b_K, model, jnp.zeros((n_heads, d_head))
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].attn.b_V, model, jnp.zeros((n_heads, d_head))
        )

        W_O = to_jax(f"{p}.attn.attention.out_proj.weight").T  # [d_model, d_model]
        b_O = to_jax(f"{p}.attn.attention.out_proj.bias")
        W_O = W_O.reshape(n_heads, d_head, d_model)

        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.W_O, model, W_O)
        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.b_O, model, b_O)

        # MLP
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].mlp.W_in, model, to_jax(f"{p}.mlp.c_fc.weight").T
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].mlp.b_in, model, to_jax(f"{p}.mlp.c_fc.bias")
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].mlp.W_out, model, to_jax(f"{p}.mlp.c_proj.weight").T
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].mlp.b_out, model, to_jax(f"{p}.mlp.c_proj.bias")
        )

    model = eqx.tree_at(lambda m: m.ln_final.w, model, to_jax("transformer.ln_f.weight"))
    model = eqx.tree_at(lambda m: m.ln_final.b, model, to_jax("transformer.ln_f.bias"))

    # Tied embeddings
    model = eqx.tree_at(lambda m: m.unembed.W_U, model, to_jax("transformer.wte.weight").T)

    del hf_model, sd
    return model


def _load_gpt_neox(model, cfg, model_name: str, **hf_kwargs):
    """Load GPT-NeoX (Pythia) weights from HuggingFace."""
    import torch
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, **hf_kwargs)
    sd = hf_model.state_dict()

    def to_jax(key):
        return jnp.array(sd[key].detach().cpu().numpy())

    d_model = cfg.d_model
    n_heads = cfg.n_heads
    d_head = cfg.d_head

    model = eqx.tree_at(lambda m: m.embed.W_E, model, to_jax("gpt_neox.embed_in.weight"))

    for l in range(cfg.n_layers):
        p = f"gpt_neox.layers.{l}"

        # Layer norms (parallel attn+mlp: only ln1 is used)
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].ln1.w, model, to_jax(f"{p}.input_layernorm.weight")
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].ln1.b, model, to_jax(f"{p}.input_layernorm.bias")
        )

        # GPT-NeoX has combined QKV
        qkv_w = to_jax(f"{p}.attention.query_key_value.weight")  # [3*d_model, d_model]
        qkv_b = to_jax(f"{p}.attention.query_key_value.bias")     # [3*d_model]

        # GPT-NeoX interleaves Q, K, V per head: [n_heads, 3, d_head] in the output dim
        qkv_w = qkv_w.T  # [d_model, 3*d_model]
        qkv_w = qkv_w.reshape(d_model, n_heads, 3, d_head)
        qkv_b = qkv_b.reshape(n_heads, 3, d_head)

        W_Q = qkv_w[:, :, 0, :].transpose(1, 0, 2)  # [n_heads, d_model, d_head]
        W_K = qkv_w[:, :, 1, :].transpose(1, 0, 2)
        W_V = qkv_w[:, :, 2, :].transpose(1, 0, 2)
        b_Q = qkv_b[:, 0, :]  # [n_heads, d_head]
        b_K = qkv_b[:, 1, :]
        b_V = qkv_b[:, 2, :]

        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.W_Q, model, W_Q)
        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.W_K, model, W_K)
        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.W_V, model, W_V)
        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.b_Q, model, b_Q)
        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.b_K, model, b_K)
        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.b_V, model, b_V)

        W_O = to_jax(f"{p}.attention.dense.weight")  # [d_model, d_model]
        b_O = to_jax(f"{p}.attention.dense.bias")
        W_O = W_O.T.reshape(n_heads, d_head, d_model)

        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.W_O, model, W_O)
        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.b_O, model, b_O)

        # MLP
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].mlp.W_in, model, to_jax(f"{p}.mlp.dense_h_to_4h.weight").T
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].mlp.b_in, model, to_jax(f"{p}.mlp.dense_h_to_4h.bias")
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].mlp.W_out, model, to_jax(f"{p}.mlp.dense_4h_to_h.weight").T
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].mlp.b_out, model, to_jax(f"{p}.mlp.dense_4h_to_h.bias")
        )

    model = eqx.tree_at(lambda m: m.ln_final.w, model, to_jax("gpt_neox.final_layer_norm.weight"))
    model = eqx.tree_at(lambda m: m.ln_final.b, model, to_jax("gpt_neox.final_layer_norm.bias"))

    # Unembed (not tied in Pythia)
    model = eqx.tree_at(lambda m: m.unembed.W_U, model, to_jax("embed_out.weight").T)

    del hf_model, sd
    return model


def _load_llama(model, cfg, model_name: str, **hf_kwargs):
    """Load LLaMA/Mistral weights from HuggingFace."""
    import torch
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, **hf_kwargs)
    sd = hf_model.state_dict()

    def to_jax(key):
        return jnp.array(sd[key].detach().cpu().numpy())

    d_model = cfg.d_model
    n_heads = cfg.n_heads
    d_head = cfg.d_head
    n_kv_heads = cfg.n_key_value_heads

    model = eqx.tree_at(lambda m: m.embed.W_E, model, to_jax("model.embed_tokens.weight"))

    for l in range(cfg.n_layers):
        p = f"model.layers.{l}"

        # RMS norms
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].ln1.w, model, to_jax(f"{p}.input_layernorm.weight")
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].ln2.w, model, to_jax(f"{p}.post_attention_layernorm.weight")
        )

        # Attention: separate Q, K, V projections
        W_Q = to_jax(f"{p}.self_attn.q_proj.weight")  # [n_heads*d_head, d_model]
        W_K = to_jax(f"{p}.self_attn.k_proj.weight")  # [n_kv_heads*d_head, d_model]
        W_V = to_jax(f"{p}.self_attn.v_proj.weight")

        W_Q = W_Q.T.reshape(d_model, n_heads, d_head).transpose(1, 0, 2)
        W_K = W_K.T.reshape(d_model, n_kv_heads, d_head).transpose(1, 0, 2)
        W_V = W_V.T.reshape(d_model, n_kv_heads, d_head).transpose(1, 0, 2)

        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.W_Q, model, W_Q)
        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.W_K, model, W_K)
        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.W_V, model, W_V)

        # LLaMA has no attention biases
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].attn.b_Q, model, jnp.zeros((n_heads, d_head))
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].attn.b_K, model, jnp.zeros((n_kv_heads, d_head))
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].attn.b_V, model, jnp.zeros((n_kv_heads, d_head))
        )

        W_O = to_jax(f"{p}.self_attn.o_proj.weight")  # [d_model, n_heads*d_head]
        W_O = W_O.T.reshape(n_heads, d_head, d_model)
        model = eqx.tree_at(lambda m, _l=l: m.blocks[_l].attn.W_O, model, W_O)
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].attn.b_O, model, jnp.zeros(d_model)
        )

        # Gated MLP
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].mlp.W_gate, model, to_jax(f"{p}.mlp.gate_proj.weight").T
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].mlp.W_in, model, to_jax(f"{p}.mlp.up_proj.weight").T
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].mlp.W_out, model, to_jax(f"{p}.mlp.down_proj.weight").T
        )
        model = eqx.tree_at(
            lambda m, _l=l: m.blocks[_l].mlp.b_out, model, jnp.zeros(d_model)
        )

    model = eqx.tree_at(lambda m: m.ln_final.w, model, to_jax("model.norm.weight"))

    # Unembed
    model = eqx.tree_at(lambda m: m.unembed.W_U, model, to_jax("lm_head.weight").T)

    del hf_model, sd
    return model
