"""Convert HuggingFace model configs to HookedTransformerConfig."""

from transformers import AutoConfig

from irtk.hooked_transformer_config import HookedTransformerConfig


def convert_hf_config(model_name: str) -> HookedTransformerConfig:
    """Load a HuggingFace config and convert to HookedTransformerConfig.

    Supports: GPT2, GPTNeo, GPTNeoX (Pythia), LLaMA, Mistral.
    """
    hf_config = AutoConfig.from_pretrained(model_name)
    arch = hf_config.model_type

    if arch == "gpt2":
        return _convert_gpt2(hf_config, model_name)
    elif arch == "gpt_neo":
        return _convert_gpt_neo(hf_config, model_name)
    elif arch == "gpt_neox":
        return _convert_gpt_neox(hf_config, model_name)
    elif arch in ("llama", "mistral"):
        return _convert_llama(hf_config, model_name)
    else:
        raise ValueError(
            f"Unsupported architecture: {arch}. "
            f"Supported: gpt2, gpt_neo, gpt_neox, llama, mistral"
        )


def _convert_gpt2(hf_config, model_name: str) -> HookedTransformerConfig:
    return HookedTransformerConfig(
        n_layers=hf_config.n_layer,
        d_model=hf_config.n_embd,
        n_ctx=hf_config.n_positions,
        d_head=hf_config.n_embd // hf_config.n_head,
        n_heads=hf_config.n_head,
        d_vocab=hf_config.vocab_size,
        d_mlp=4 * hf_config.n_embd,
        act_fn="gelu_new",
        normalization_type="LN",
        positional_embedding_type="standard",
        attention_dir="causal",
        tokenizer_name=model_name,
        original_architecture="gpt2",
    )


def _convert_gpt_neo(hf_config, model_name: str) -> HookedTransformerConfig:
    return HookedTransformerConfig(
        n_layers=hf_config.num_layers,
        d_model=hf_config.hidden_size,
        n_ctx=hf_config.max_position_embeddings,
        d_head=hf_config.hidden_size // hf_config.num_heads,
        n_heads=hf_config.num_heads,
        d_vocab=hf_config.vocab_size,
        d_mlp=4 * hf_config.hidden_size,
        act_fn="gelu_new",
        normalization_type="LN",
        positional_embedding_type="standard",
        attention_dir="causal",
        tokenizer_name=model_name,
        original_architecture="gpt_neo",
    )


def _convert_gpt_neox(hf_config, model_name: str) -> HookedTransformerConfig:
    rotary_pct = getattr(hf_config, "rotary_pct", 0.25)
    d_head = hf_config.hidden_size // hf_config.num_attention_heads
    rotary_dim = int(d_head * rotary_pct)

    return HookedTransformerConfig(
        n_layers=hf_config.num_hidden_layers,
        d_model=hf_config.hidden_size,
        n_ctx=hf_config.max_position_embeddings,
        d_head=d_head,
        n_heads=hf_config.num_attention_heads,
        d_vocab=hf_config.vocab_size,
        d_mlp=hf_config.intermediate_size,
        act_fn="gelu",
        normalization_type="LN",
        positional_embedding_type="rotary",
        rotary_dim=rotary_dim,
        rotary_base=getattr(hf_config, "rotary_emb_base", 10000.0),
        attention_dir="causal",
        parallel_attn_mlp=True,
        tokenizer_name=model_name,
        original_architecture="gpt_neox",
    )


def _convert_llama(hf_config, model_name: str) -> HookedTransformerConfig:
    d_head = hf_config.hidden_size // hf_config.num_attention_heads
    n_kv_heads = getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)

    return HookedTransformerConfig(
        n_layers=hf_config.num_hidden_layers,
        d_model=hf_config.hidden_size,
        n_ctx=hf_config.max_position_embeddings,
        d_head=d_head,
        n_heads=hf_config.num_attention_heads,
        d_vocab=hf_config.vocab_size,
        d_mlp=hf_config.intermediate_size,
        act_fn="silu",
        gated_mlp=True,
        normalization_type="RMS",
        eps=getattr(hf_config, "rms_norm_eps", 1e-5),
        positional_embedding_type="rotary",
        rotary_dim=d_head,
        rotary_base=getattr(hf_config, "rope_theta", 10000.0),
        attention_dir="causal",
        n_key_value_heads=n_kv_heads,
        tokenizer_name=model_name,
        original_architecture=hf_config.model_type,
    )
