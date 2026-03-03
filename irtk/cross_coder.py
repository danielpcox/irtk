"""Cross-coders: joint sparse autoencoders across multiple activation streams.

Train a single sparse autoencoder across multiple activation sources
(different models, layers, or training stages) to identify shared vs.
stream-specific features.

References:
    - Lindsey et al. (2024) "Crosscoders"
"""

from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from irtk.hooked_transformer import HookedTransformer


class CrossCoder(eqx.Module):
    """Joint sparse autoencoder over multiple activation streams.

    Learns a shared encoder and per-stream decoders. Each stream
    (e.g., base model layer L, finetuned model layer L) shares the
    same feature dictionary but has its own read-out weights.

    Architecture:
        shared encoder: concat(x_1, ..., x_k) -> features (ReLU + L1)
        per-stream decoders: features -> x_hat_i for each stream i
    """

    W_enc: jnp.ndarray      # [total_d, n_features]
    b_enc: jnp.ndarray       # [n_features]
    W_decs: list             # list of [n_features, d_i] arrays
    b_decs: list             # list of [d_i] arrays
    n_streams: int = eqx.field(static=True)
    n_features: int = eqx.field(static=True)
    stream_dims: tuple = eqx.field(static=True)

    def __init__(
        self,
        stream_dims: tuple,
        n_features: int,
        *,
        key: jax.random.PRNGKey,
    ):
        """Initialize CrossCoder.

        Args:
            stream_dims: Tuple of d_model for each stream.
            n_features: Number of shared features.
            key: Random key.
        """
        self.n_streams = len(stream_dims)
        self.n_features = n_features
        self.stream_dims = stream_dims
        total_d = sum(stream_dims)

        keys = jax.random.split(key, self.n_streams + 1)

        # Shared encoder
        self.W_enc = jax.random.normal(keys[0], (total_d, n_features)) * (1.0 / jnp.sqrt(total_d))
        self.b_enc = jnp.zeros(n_features)

        # Per-stream decoders
        W_decs = []
        b_decs = []
        for i, d in enumerate(stream_dims):
            W_d = jax.random.normal(keys[i + 1], (n_features, d)) * (1.0 / jnp.sqrt(n_features))
            W_d = W_d / jnp.linalg.norm(W_d, axis=-1, keepdims=True)
            W_decs.append(W_d)
            b_decs.append(jnp.zeros(d))

        self.W_decs = W_decs
        self.b_decs = b_decs

    def encode(self, streams: list) -> jnp.ndarray:
        """Encode concatenated streams to shared features.

        Args:
            streams: List of [d_i] or [..., d_i] arrays, one per stream.

        Returns:
            [..., n_features] sparse feature activations.
        """
        centered = []
        for s, b in zip(streams, self.b_decs):
            centered.append(s - b)
        x = jnp.concatenate(centered, axis=-1)
        pre_acts = x @ self.W_enc + self.b_enc
        return jax.nn.relu(pre_acts)

    def decode(self, features: jnp.ndarray, stream_idx: int) -> jnp.ndarray:
        """Decode features for a specific stream.

        Args:
            features: [..., n_features] feature activations.
            stream_idx: Which stream to decode for.

        Returns:
            [..., d_i] reconstructed activations.
        """
        return features @ self.W_decs[stream_idx] + self.b_decs[stream_idx]

    def __call__(self, streams: list) -> list:
        """Full forward pass: encode then decode all streams.

        Args:
            streams: List of activation arrays.

        Returns:
            List of reconstructed arrays, one per stream.
        """
        features = self.encode(streams)
        return [self.decode(features, i) for i in range(self.n_streams)]


def train_crosscoder(
    models: list,
    hook_names: list,
    token_sequences: list,
    n_features: int,
    l1_coeff: float = 1e-3,
    n_steps: int = 100,
    lr: float = 1e-3,
    *,
    key: jax.random.PRNGKey,
) -> CrossCoder:
    """Train a cross-coder on activations from multiple models/hooks.

    Each (model, hook) pair is one stream. The cross-coder learns
    shared features across all streams.

    Args:
        models: List of HookedTransformer models.
        hook_names: List of hook names (one per model).
        token_sequences: Shared test inputs.
        n_features: Number of features in the cross-coder.
        l1_coeff: L1 sparsity coefficient.
        n_steps: Training steps.
        lr: Learning rate.
        key: Random key.

    Returns:
        Trained CrossCoder.
    """
    n_streams = len(models)

    # Collect activations
    stream_data = [[] for _ in range(n_streams)]

    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        for si, (model, hook) in enumerate(zip(models, hook_names)):
            _, cache = model.run_with_cache(tokens)
            if hook in cache.cache_dict:
                acts = np.array(cache.cache_dict[hook])
                flat = acts.reshape(-1, acts.shape[-1])
                stream_data[si].append(flat)

    # Align lengths
    min_samples = min(sum(d.shape[0] for d in sd) for sd in stream_data if sd)
    if min_samples == 0:
        dims = tuple(models[i].cfg.d_model for i in range(n_streams))
        return CrossCoder(dims, n_features, key=key)

    aligned = []
    for sd in stream_data:
        if sd:
            cat = np.concatenate(sd, axis=0)[:min_samples]
            aligned.append(jnp.array(cat))
        else:
            aligned.append(jnp.zeros((min_samples, models[0].cfg.d_model)))

    dims = tuple(a.shape[-1] for a in aligned)
    cc = CrossCoder(dims, n_features, key=key)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(cc, eqx.is_array))

    def loss_fn(cc, streams):
        features = cc.encode(streams)
        recon_loss = 0.0
        for i in range(len(streams)):
            recon = cc.decode(features, i)
            recon_loss = recon_loss + jnp.mean((streams[i] - recon) ** 2)
        l1_loss = l1_coeff * jnp.mean(jnp.abs(features))
        return recon_loss + l1_loss

    @eqx.filter_jit
    def step(cc, opt_state, streams):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(cc, streams)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(cc, eqx.is_array)
        )
        new_cc = eqx.apply_updates(cc, updates)
        return new_cc, new_opt_state, loss

    for _ in range(n_steps):
        cc, opt_state, _ = step(cc, opt_state, aligned)

    return cc


def shared_vs_specific_features(
    cc: CrossCoder,
    models: list,
    hook_names: list,
    token_sequences: list,
    threshold: float = 0.1,
) -> dict:
    """Classify features as shared or stream-specific.

    A feature is shared if it activates similarly across all streams.

    Args:
        cc: Trained CrossCoder.
        models: List of models.
        hook_names: List of hook names.
        token_sequences: Test inputs.
        threshold: Minimum mean activation to consider a feature active.

    Returns:
        Dict with:
        - "shared_features": feature indices active across all streams
        - "specific_features": dict of stream_idx -> feature indices
        - "sharing_scores": [n_features] mean cross-stream activation ratio
        - "n_shared": number of shared features
    """
    n_streams = len(models)

    # Collect per-stream feature activations
    stream_acts = [[] for _ in range(n_streams)]

    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        stream_inputs = []
        for si, (model, hook) in enumerate(zip(models, hook_names)):
            _, cache = model.run_with_cache(tokens)
            if hook in cache.cache_dict:
                acts = cache.cache_dict[hook]
                flat = acts.reshape(-1, acts.shape[-1])
                stream_inputs.append(flat)
            else:
                stream_inputs.append(jnp.zeros((1, cc.stream_dims[si])))

        if len(stream_inputs) == n_streams:
            features = np.array(cc.encode(stream_inputs))
            # Track mean activation per stream
            for si in range(n_streams):
                stream_acts[si].append(np.mean(features, axis=0))

    if not stream_acts[0]:
        return {"shared_features": [], "specific_features": {},
                "sharing_scores": np.zeros(cc.n_features), "n_shared": 0}

    # Average activations per stream
    mean_acts = np.zeros((n_streams, cc.n_features))
    for si in range(n_streams):
        if stream_acts[si]:
            mean_acts[si] = np.mean(stream_acts[si], axis=0)

    # Active in stream = mean activation > threshold
    active = mean_acts > threshold

    # Shared: active in all streams
    shared = []
    specific = {si: [] for si in range(n_streams)}
    sharing_scores = np.zeros(cc.n_features)

    for fi in range(cc.n_features):
        n_active = int(np.sum(active[:, fi]))
        sharing_scores[fi] = n_active / max(n_streams, 1)
        if n_active == n_streams:
            shared.append(fi)
        elif n_active == 1:
            for si in range(n_streams):
                if active[si, fi]:
                    specific[si].append(fi)

    return {
        "shared_features": shared,
        "specific_features": specific,
        "sharing_scores": sharing_scores,
        "n_shared": len(shared),
    }


def finetuning_feature_diff(
    base_model: HookedTransformer,
    finetuned_model: HookedTransformer,
    hook_name: str,
    token_sequences: list,
    n_features: int = 64,
    n_steps: int = 100,
    *,
    key: jax.random.PRNGKey,
) -> dict:
    """Find what fine-tuning changed using a cross-coder.

    Trains a cross-coder on (base, finetuned) activations and identifies
    features that are amplified, suppressed, or stream-specific.

    Args:
        base_model: Base model.
        finetuned_model: Fine-tuned model.
        hook_name: Hook to compare.
        token_sequences: Test inputs.
        n_features: Number of cross-coder features.
        n_steps: Training steps.
        key: Random key.

    Returns:
        Dict with:
        - "crosscoder": trained CrossCoder
        - "amplified_features": features stronger in finetuned
        - "suppressed_features": features weaker in finetuned
        - "base_specific": features only in base
        - "finetuned_specific": features only in finetuned
        - "mean_activation_diff": [n_features] finetuned - base mean activation
    """
    cc = train_crosscoder(
        [base_model, finetuned_model],
        [hook_name, hook_name],
        token_sequences,
        n_features,
        n_steps=n_steps,
        key=key,
    )

    # Collect mean feature activations per stream
    base_acts = []
    ft_acts = []

    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        _, base_cache = base_model.run_with_cache(tokens)
        _, ft_cache = finetuned_model.run_with_cache(tokens)

        if hook_name in base_cache.cache_dict and hook_name in ft_cache.cache_dict:
            b = base_cache.cache_dict[hook_name].reshape(-1, base_model.cfg.d_model)
            f = ft_cache.cache_dict[hook_name].reshape(-1, finetuned_model.cfg.d_model)
            min_n = min(b.shape[0], f.shape[0])
            features = np.array(cc.encode([b[:min_n], f[:min_n]]))
            base_acts.append(np.mean(features, axis=0))
            ft_acts.append(np.mean(features, axis=0))

    if not base_acts:
        return {"crosscoder": cc, "amplified_features": [], "suppressed_features": [],
                "base_specific": [], "finetuned_specific": [],
                "mean_activation_diff": np.zeros(n_features)}

    mean_base = np.mean(base_acts, axis=0)
    mean_ft = np.mean(ft_acts, axis=0)
    diff = mean_ft - mean_base

    # Classify
    amplified = [int(i) for i in np.where(diff > 0.01)[0]]
    suppressed = [int(i) for i in np.where(diff < -0.01)[0]]
    base_only = [int(i) for i in np.where((mean_base > 0.01) & (mean_ft < 0.001))[0]]
    ft_only = [int(i) for i in np.where((mean_ft > 0.01) & (mean_base < 0.001))[0]]

    return {
        "crosscoder": cc,
        "amplified_features": amplified,
        "suppressed_features": suppressed,
        "base_specific": base_only,
        "finetuned_specific": ft_only,
        "mean_activation_diff": diff,
    }


def cross_layer_crosscoder(
    model: HookedTransformer,
    hook_names: list,
    token_sequences: list,
    n_features: int = 64,
    n_steps: int = 100,
    *,
    key: jax.random.PRNGKey,
) -> dict:
    """Train a cross-coder across multiple layers of a single model.

    Identifies features reused across layers (universal) vs.
    layer-specific features.

    Args:
        model: HookedTransformer.
        hook_names: List of hook names (one per layer).
        token_sequences: Test inputs.
        n_features: Number of features.
        n_steps: Training steps.
        key: Random key.

    Returns:
        Dict with:
        - "crosscoder": trained CrossCoder
        - "universal_features": feature indices active in most layers
        - "layer_specific_features": dict of layer_idx -> feature indices
        - "per_layer_activity": [n_features, n_layers] mean activation
    """
    n_layers = len(hook_names)
    models = [model] * n_layers

    cc = train_crosscoder(
        models, hook_names, token_sequences, n_features,
        n_steps=n_steps, key=key,
    )

    # Measure per-layer activity
    per_layer = np.zeros((n_features, n_layers))

    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)

        streams = []
        valid = True
        for hook in hook_names:
            if hook in cache.cache_dict:
                acts = cache.cache_dict[hook].reshape(-1, model.cfg.d_model)
                streams.append(acts)
            else:
                valid = False
                break

        if valid and streams:
            min_n = min(s.shape[0] for s in streams)
            streams = [s[:min_n] for s in streams]
            features = np.array(cc.encode(streams))  # [n_pos, n_features]
            mean_f = np.mean(features, axis=0)
            for li in range(n_layers):
                per_layer[:, li] += mean_f

    n_inputs = max(len(token_sequences), 1)
    per_layer /= n_inputs

    # Classify
    active_thresh = 0.01
    universal = []
    layer_specific = {li: [] for li in range(n_layers)}

    for fi in range(n_features):
        n_active = int(np.sum(per_layer[fi] > active_thresh))
        if n_active >= n_layers * 0.8:
            universal.append(fi)
        elif n_active == 1:
            best_layer = int(np.argmax(per_layer[fi]))
            layer_specific[best_layer].append(fi)

    return {
        "crosscoder": cc,
        "universal_features": universal,
        "layer_specific_features": layer_specific,
        "per_layer_activity": per_layer,
    }
