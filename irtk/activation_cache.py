"""ActivationCache: wraps cached activations with analysis methods."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from irtk.hooked_transformer import HookedTransformer


class ActivationCache:
    """Cache of activations from a HookedTransformer forward pass.

    Supports shorthand access like cache["q", 0] for cache["blocks.0.attn.hook_q"],
    plus analysis methods for residual stream decomposition and logit attribution.
    """

    def __init__(
        self, cache_dict: dict[str, jnp.ndarray], model: Optional["HookedTransformer"] = None
    ):
        self.cache_dict = cache_dict
        self.model = model

    def __getitem__(self, key) -> jnp.ndarray:
        """Access cached activations.

        Supports:
            cache["blocks.0.attn.hook_q"] - full name
            cache["q", 0] - shorthand (activation_name, layer)
            cache["embed"] - non-layer activation
        """
        if isinstance(key, tuple):
            name, layer = key
            full_name = f"blocks.{layer}.attn.hook_{name}"
            if full_name in self.cache_dict:
                return self.cache_dict[full_name]
            # Try MLP hooks
            full_name = f"blocks.{layer}.mlp.hook_{name}"
            if full_name in self.cache_dict:
                return self.cache_dict[full_name]
            # Try block-level hooks
            full_name = f"blocks.{layer}.hook_{name}"
            if full_name in self.cache_dict:
                return self.cache_dict[full_name]
            raise KeyError(f"Activation '{name}' not found at layer {layer}")
        return self.cache_dict[key]

    def __contains__(self, key) -> bool:
        if isinstance(key, tuple):
            try:
                self[key]
                return True
            except KeyError:
                return False
        return key in self.cache_dict

    def keys(self):
        return self.cache_dict.keys()

    def values(self):
        return self.cache_dict.values()

    def items(self):
        return self.cache_dict.items()

    def __len__(self):
        return len(self.cache_dict)

    def accumulated_resid(
        self, layer: Optional[int] = None, incl_mid: bool = False, return_labels: bool = False
    ):
        """Get accumulated residual stream at each layer.

        Returns a stack of residual streams from embedding through the specified layer.

        Args:
            layer: Up to which layer (exclusive). None = all layers.
            incl_mid: If True, include hook_resid_mid (between attn and MLP).
            return_labels: If True, also return list of labels.

        Returns:
            [n_components, seq_len, d_model] stacked residual streams.
            Optionally also returns labels list.
        """
        if self.model is None:
            raise ValueError("Model reference needed for accumulated_resid")
        n_layers = self.model.cfg.n_layers if layer is None else layer
        components = []
        labels = []

        # Embedding
        embed = self.cache_dict.get("hook_embed", None)
        pos_embed = self.cache_dict.get("hook_pos_embed", None)
        if embed is not None:
            base = embed
            if pos_embed is not None:
                base = base + pos_embed
            components.append(base)
            labels.append("embed")

        for l in range(n_layers):
            if incl_mid and f"blocks.{l}.hook_resid_mid" in self.cache_dict:
                components.append(self.cache_dict[f"blocks.{l}.hook_resid_mid"])
                labels.append(f"blocks.{l}.hook_resid_mid")
            if f"blocks.{l}.hook_resid_post" in self.cache_dict:
                components.append(self.cache_dict[f"blocks.{l}.hook_resid_post"])
                labels.append(f"blocks.{l}.hook_resid_post")

        result = jnp.stack(components, axis=0) if components else jnp.array([])
        if return_labels:
            return result, labels
        return result

    def decompose_resid(
        self, layer: Optional[int] = None, return_labels: bool = False
    ):
        """Decompose the residual stream into individual component contributions.

        Returns the contribution of each attention and MLP output at each layer.

        Args:
            layer: Up to which layer (exclusive). None = all layers.
            return_labels: If True, also return labels.

        Returns:
            [n_components, seq_len, d_model] individual contributions.
        """
        if self.model is None:
            raise ValueError("Model reference needed for decompose_resid")
        n_layers = self.model.cfg.n_layers if layer is None else layer
        components = []
        labels = []

        embed = self.cache_dict.get("hook_embed", None)
        pos_embed = self.cache_dict.get("hook_pos_embed", None)
        if embed is not None:
            base = embed
            if pos_embed is not None:
                base = base + pos_embed
            components.append(base)
            labels.append("embed")

        for l in range(n_layers):
            attn_key = f"blocks.{l}.hook_attn_out"
            mlp_key = f"blocks.{l}.hook_mlp_out"
            if attn_key in self.cache_dict:
                components.append(self.cache_dict[attn_key])
                labels.append(attn_key)
            if mlp_key in self.cache_dict:
                components.append(self.cache_dict[mlp_key])
                labels.append(mlp_key)

        result = jnp.stack(components, axis=0) if components else jnp.array([])
        if return_labels:
            return result, labels
        return result

    def stack_head_results(
        self, layer: int = -1
    ) -> jnp.ndarray:
        """Stack per-head attention results for a given layer.

        Args:
            layer: Layer index. -1 means all layers.

        Returns:
            If layer >= 0: [n_heads, seq_len, d_model] - per-head results for that layer.
            If layer == -1: [n_layers, n_heads, seq_len, d_model] - all layers.
        """
        if self.model is None:
            raise ValueError("Model reference needed for stack_head_results")

        if layer >= 0:
            z = self[("z", layer)]  # [seq, n_heads, d_head]
            W_O = self.model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]
            # Per-head: result_h = z_h @ W_O_h for each head
            # z:[s,n,h] W_O:[n,h,m] -> [n,s,m]
            return jnp.einsum("snh,nhm->nsm", z, W_O)
        else:
            all_layers = []
            for l in range(self.model.cfg.n_layers):
                all_layers.append(self.stack_head_results(layer=l))
            return jnp.stack(all_layers, axis=0)

    def stack_activation(self, name: str) -> jnp.ndarray:
        """Stack a named activation across all layers.

        Args:
            name: Activation name (e.g., "q", "k", "resid_pre").

        Returns:
            Stacked array with layer dimension first.
        """
        if self.model is None:
            raise ValueError("Model reference needed for stack_activation")
        activations = []
        for l in range(self.model.cfg.n_layers):
            if (name, l) in self:
                activations.append(self[(name, l)])
        return jnp.stack(activations, axis=0) if activations else jnp.array([])

    def logit_attrs(
        self,
        residual_stack: jnp.ndarray,
        tokens: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Compute logit attributions via the logit lens.

        Projects each component's contribution through the unembedding.

        Args:
            residual_stack: [n_components, seq_len, d_model]
            tokens: Optional target tokens for per-token attribution.

        Returns:
            If tokens is None: [n_components, seq_len, d_vocab] logit contributions.
            If tokens provided: [n_components, seq_len] logit attr for those tokens.
        """
        if self.model is None:
            raise ValueError("Model reference needed for logit_attrs")
        W_U = self.model.unembed.W_U  # [d_model, d_vocab]
        logits = jnp.einsum("csd,dv->csv", residual_stack, W_U)
        if tokens is not None:
            # Gather the logit for target token at each position
            logits = jnp.take_along_axis(
                logits, tokens[None, :, None].astype(jnp.int32), axis=-1
            ).squeeze(-1)
        return logits
