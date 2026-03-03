"""HookedTransformerConfig: all configuration fields for a hooked transformer model."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HookedTransformerConfig:
    """Configuration for a HookedTransformer model.

    Uses TransformerLens naming conventions.
    """

    # Core architecture
    n_layers: int
    d_model: int
    n_ctx: int
    d_head: int
    n_heads: int
    d_vocab: int

    # MLP
    d_mlp: Optional[int] = None  # defaults to 4 * d_model
    act_fn: str = "gelu"
    gated_mlp: bool = False

    # Normalization
    normalization_type: Optional[str] = "LN"  # "LN", "LNPre", "RMS", "RMSPre", None
    eps: float = 1e-5

    # Positional embeddings
    positional_embedding_type: str = "standard"  # "standard", "rotary", "none"
    rotary_dim: Optional[int] = None
    rotary_base: float = 10000.0

    # Attention
    attention_dir: str = "causal"  # "causal" or "bidirectional"
    n_key_value_heads: Optional[int] = None  # for GQA; None means same as n_heads
    attn_scale: Optional[float] = None  # defaults to sqrt(d_head)

    # Architecture variants
    parallel_attn_mlp: bool = False  # GPT-J style parallel attn+mlp

    # Unembedding
    d_vocab_out: Optional[int] = None  # defaults to d_vocab

    # Token types
    tokenizer_name: Optional[str] = None
    original_architecture: Optional[str] = None

    # Dtype
    dtype: str = "float32"

    # Misc
    init_weights: bool = False  # whether to initialize weights (vs loading pretrained)

    def __post_init__(self):
        if self.d_mlp is None:
            self.d_mlp = 4 * self.d_model
        if self.d_vocab_out is None:
            self.d_vocab_out = self.d_vocab
        if self.n_key_value_heads is None:
            self.n_key_value_heads = self.n_heads
        if self.attn_scale is None:
            self.attn_scale = self.d_head**0.5
        if self.rotary_dim is None and self.positional_embedding_type == "rotary":
            self.rotary_dim = self.d_head
