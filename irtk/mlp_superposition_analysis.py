"""MLP superposition analysis: detect and characterize superposition in MLP weights."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def mlp_input_interference(model: HookedTransformer, layer: int) -> dict:
    """How much do MLP input directions interfere with each other?

    High interference = features stored in superposition.
    """
    W_in = model.blocks[layer].mlp.W_in  # [d_model, d_mlp]
    d_mlp = W_in.shape[1]

    # Normalize columns
    norms = jnp.linalg.norm(W_in, axis=0, keepdims=True) + 1e-10
    normed = W_in / norms

    # Gram matrix of input directions
    gram = normed.T @ normed  # [d_mlp, d_mlp]
    off_diag = gram - jnp.eye(d_mlp)

    mean_interference = float(jnp.mean(jnp.abs(off_diag)))
    max_interference = float(jnp.max(jnp.abs(off_diag)))

    # Count high-interference pairs
    high_pairs = int(jnp.sum(jnp.abs(off_diag) > 0.5) / 2)  # symmetric

    return {
        'layer': layer,
        'd_mlp': d_mlp,
        'mean_interference': mean_interference,
        'max_interference': max_interference,
        'n_high_interference_pairs': high_pairs,
        'has_superposition': mean_interference > 0.1,
    }


def mlp_output_interference(model: HookedTransformer, layer: int) -> dict:
    """How much do MLP output directions interfere?

    High interference = neurons write in overlapping directions.
    """
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    d_mlp = W_out.shape[0]

    norms = jnp.linalg.norm(W_out, axis=1, keepdims=True) + 1e-10
    normed = W_out / norms

    gram = normed @ normed.T  # [d_mlp, d_mlp]
    off_diag = gram - jnp.eye(d_mlp)

    mean_interference = float(jnp.mean(jnp.abs(off_diag)))
    max_interference = float(jnp.max(jnp.abs(off_diag)))
    high_pairs = int(jnp.sum(jnp.abs(off_diag) > 0.5) / 2)

    return {
        'layer': layer,
        'd_mlp': d_mlp,
        'mean_interference': mean_interference,
        'max_interference': max_interference,
        'n_high_interference_pairs': high_pairs,
        'has_superposition': mean_interference > 0.1,
    }


def mlp_feature_capacity(model: HookedTransformer, layer: int) -> dict:
    """Feature capacity ratio: how many features vs dimensions?

    Superposition allows d_mlp > d_model effective features.
    """
    W_in = model.blocks[layer].mlp.W_in  # [d_model, d_mlp]
    d_model = W_in.shape[0]
    d_mlp = W_in.shape[1]

    # Effective rank of W_in
    sv = jnp.linalg.svd(W_in, compute_uv=False)
    sv_norm = sv / (jnp.sum(sv) + 1e-10)
    entropy = -jnp.sum(sv_norm * jnp.log(sv_norm + 1e-10))
    eff_rank = float(jnp.exp(entropy))

    # Capacity ratio
    capacity_ratio = d_mlp / d_model
    utilization = eff_rank / d_model

    return {
        'layer': layer,
        'd_model': d_model,
        'd_mlp': d_mlp,
        'capacity_ratio': capacity_ratio,
        'effective_rank': eff_rank,
        'rank_utilization': utilization,
        'theoretical_max_features': d_mlp,
        'superposition_ratio': capacity_ratio / (utilization + 1e-10),
    }


def mlp_neuron_orthogonality(model: HookedTransformer, layer: int, sample_size: int = 50) -> dict:
    """Are neuron input/output directions approximately orthogonal?

    Perfect orthogonality = no superposition; random directions = moderate overlap.
    """
    W_in = model.blocks[layer].mlp.W_in  # [d_model, d_mlp]
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    d_mlp = W_in.shape[1]
    n = min(sample_size, d_mlp)

    # Input directions
    in_norms = jnp.linalg.norm(W_in[:, :n], axis=0, keepdims=True) + 1e-10
    in_normed = W_in[:, :n] / in_norms
    in_gram = in_normed.T @ in_normed
    in_off = jnp.abs(in_gram - jnp.eye(n))

    # Output directions
    out_norms = jnp.linalg.norm(W_out[:n], axis=1, keepdims=True) + 1e-10
    out_normed = W_out[:n] / out_norms
    out_gram = out_normed @ out_normed.T
    out_off = jnp.abs(out_gram - jnp.eye(n))

    return {
        'layer': layer,
        'n_sampled': n,
        'input_mean_overlap': float(jnp.mean(in_off)),
        'input_max_overlap': float(jnp.max(in_off)),
        'output_mean_overlap': float(jnp.mean(out_off)),
        'output_max_overlap': float(jnp.max(out_off)),
        'is_approximately_orthogonal': float(jnp.mean(in_off)) < 0.15,
    }


def mlp_superposition_summary(model: HookedTransformer) -> dict:
    """Cross-layer superposition summary."""
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        W_in = model.blocks[layer].mlp.W_in
        d_model = W_in.shape[0]
        d_mlp = W_in.shape[1]

        # Input interference
        in_norms = jnp.linalg.norm(W_in, axis=0, keepdims=True) + 1e-10
        in_normed = W_in / in_norms
        in_gram = in_normed.T @ in_normed
        in_off = jnp.abs(in_gram - jnp.eye(d_mlp))
        mean_in_interference = float(jnp.mean(in_off))

        # Effective rank
        sv = jnp.linalg.svd(W_in, compute_uv=False)
        sv_norm = sv / (jnp.sum(sv) + 1e-10)
        entropy = -jnp.sum(sv_norm * jnp.log(sv_norm + 1e-10))
        eff_rank = float(jnp.exp(entropy))

        per_layer.append({
            'layer': layer,
            'capacity_ratio': d_mlp / d_model,
            'mean_interference': mean_in_interference,
            'effective_rank': eff_rank,
            'superposition_level': 'high' if mean_in_interference > 0.15 else
                                   'moderate' if mean_in_interference > 0.08 else 'low',
        })

    return {
        'per_layer': per_layer,
    }
