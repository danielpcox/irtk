"""HookedTransformer: the main model class with hook-based activation inspection."""

from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx

from irtk.hook_points import HookPoint, HookState, HookFn
from irtk.hooked_transformer_config import HookedTransformerConfig
from irtk.activation_cache import ActivationCache
from irtk.components.embed import Embed, PosEmbed
from irtk.components.unembed import Unembed
from irtk.components.layer_norm import get_layer_norm
from irtk.components.transformer_block import TransformerBlock


class HookedTransformer(eqx.Module):
    """A transformer model with hooks at every intermediate activation.

    Designed for mechanistic interpretability. Every component threads
    an optional HookState through its forward pass for activation
    caching and intervention.

    Components operate on unbatched data [seq_len, ...].
    Use jax.vmap for batched inference.
    """

    cfg: HookedTransformerConfig = eqx.field(static=True)
    embed: Embed
    pos_embed: Optional[PosEmbed]
    blocks: list[TransformerBlock]
    ln_final: Optional[eqx.Module]
    unembed: Unembed

    def __init__(self, cfg: HookedTransformerConfig):
        self.cfg = cfg
        self.embed = Embed(cfg.d_vocab, cfg.d_model)

        if cfg.positional_embedding_type == "standard":
            self.pos_embed = PosEmbed(cfg.n_ctx, cfg.d_model)
        else:
            self.pos_embed = None

        self.blocks = [
            TransformerBlock(cfg, layer_idx=i) for i in range(cfg.n_layers)
        ]
        self.ln_final = get_layer_norm(
            cfg.normalization_type, cfg.d_model, cfg.eps, name_prefix="ln_final."
        )
        self.unembed = Unembed(cfg.d_model, cfg.d_vocab_out)

    def __call__(
        self, tokens: jnp.ndarray, hook_state: Optional[HookState] = None
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            tokens: [seq_len] integer token IDs (unbatched).
            hook_state: Optional HookState for caching/intervention.

        Returns:
            logits: [seq_len, d_vocab_out]
        """
        # Embeddings
        residual = self.embed(tokens, hook_state)
        if self.pos_embed is not None:
            residual = residual + self.pos_embed(tokens, hook_state)

        # Transformer blocks
        for block in self.blocks:
            residual = block(residual, hook_state)

        # Final layer norm
        if self.ln_final is not None:
            residual = self.ln_final(residual, hook_state)

        # Unembed
        logits = self.unembed(residual, hook_state)
        return logits

    def run_with_cache(
        self, tokens: jnp.ndarray
    ) -> tuple[jnp.ndarray, ActivationCache]:
        """Run forward pass and cache all activations.

        Args:
            tokens: [seq_len] integer token IDs.

        Returns:
            (logits, cache): logits and ActivationCache with all hook activations.
        """
        cache_dict: dict[str, jnp.ndarray] = {}
        hook_state = HookState(hook_fns={}, cache=cache_dict)
        logits = self(tokens, hook_state)
        return logits, ActivationCache(cache_dict, self)

    def run_with_hooks(
        self,
        tokens: jnp.ndarray,
        fwd_hooks: list[tuple[str, HookFn]] | None = None,
    ) -> jnp.ndarray:
        """Run forward pass with hook functions for activation intervention.

        Args:
            tokens: [seq_len] integer token IDs.
            fwd_hooks: List of (hook_name, hook_fn) pairs.

        Returns:
            logits with interventions applied.
        """
        hook_fns = {}
        if fwd_hooks is not None:
            for name, fn in fwd_hooks:
                hook_fns[name] = fn
        hook_state = HookState(hook_fns=hook_fns, cache=None)
        return self(tokens, hook_state)

    def to_tokens(
        self, text: str, prepend_bos: bool = False
    ) -> jnp.ndarray:
        """Tokenize text using this model's tokenizer.

        Args:
            text: Input string.
            prepend_bos: Whether to prepend BOS token.

        Returns:
            [seq_len] array of token IDs.
        """
        from irtk.datasets import to_tokens, get_tokenizer
        tokenizer = get_tokenizer(self.cfg.tokenizer_name or "openai-community/gpt2")
        return to_tokens(text, tokenizer=tokenizer, prepend_bos=prepend_bos)

    def to_string(self, tokens: jnp.ndarray) -> str:
        """Decode token IDs back to text.

        Args:
            tokens: [seq_len] token IDs.

        Returns:
            Decoded string.
        """
        from irtk.datasets import to_string, get_tokenizer
        tokenizer = get_tokenizer(self.cfg.tokenizer_name or "openai-community/gpt2")
        return to_string(tokens, tokenizer=tokenizer)

    @property
    def tokenizer(self):
        """Get this model's tokenizer."""
        from irtk.datasets import get_tokenizer
        return get_tokenizer(self.cfg.tokenizer_name or "openai-community/gpt2")

    @property
    def hook_dict(self) -> dict[str, HookPoint]:
        """Return a dict of all hook point names -> HookPoint modules."""
        hooks = {}

        def _collect(module, prefix=""):
            if isinstance(module, HookPoint):
                hooks[module.name] = module
            elif isinstance(module, eqx.Module):
                for field_name in vars(module):
                    child = getattr(module, field_name)
                    if isinstance(child, list):
                        for i, item in enumerate(child):
                            _collect(item, f"{prefix}{field_name}[{i}].")
                    elif isinstance(child, eqx.Module):
                        _collect(child, f"{prefix}{field_name}.")

        _collect(self)
        return hooks

    @staticmethod
    def from_pretrained(model_name: str, **kwargs) -> "HookedTransformer":
        """Load a pretrained model from HuggingFace.

        Args:
            model_name: Model name or alias (e.g., "gpt2", "gpt2-small", "pythia-70m").

        Returns:
            HookedTransformer with pretrained weights.
        """
        from irtk.loading.model_names import get_official_model_name
        from irtk.loading.pretrained_config import convert_hf_config
        from irtk.loading.weight_conversions import load_pretrained_weights

        official_name = get_official_model_name(model_name)
        cfg = convert_hf_config(official_name)
        model = HookedTransformer(cfg)
        model = load_pretrained_weights(model, cfg, official_name)
        return model
