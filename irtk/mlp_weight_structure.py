"""MLP weight structure: analyze the structure of MLP weight matrices."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def mlp_weight_spectrum(model: HookedTransformer, layer: int) -> dict:
    """Singular value spectrum of MLP weight matrices.

    Reveals the effective rank of W_in and W_out.
    """
    W_in = model.blocks[layer].mlp.W_in  # [d_model, d_mlp]
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]

    in_sv = jnp.linalg.svd(W_in, compute_uv=False)
    out_sv = jnp.linalg.svd(W_out, compute_uv=False)

    def eff_rank(sv):
        sv_norm = sv / (jnp.sum(sv) + 1e-10)
        entropy = -jnp.sum(sv_norm * jnp.log(sv_norm + 1e-10))
        return float(jnp.exp(entropy))

    return {
        'layer': layer,
        'W_in_effective_rank': eff_rank(in_sv),
        'W_out_effective_rank': eff_rank(out_sv),
        'W_in_spectral_norm': float(in_sv[0]),
        'W_out_spectral_norm': float(out_sv[0]),
        'W_in_top_sv_fraction': float(in_sv[0] / (jnp.sum(in_sv) + 1e-10)),
        'W_out_top_sv_fraction': float(out_sv[0] / (jnp.sum(out_sv) + 1e-10)),
    }


def mlp_neuron_norms(model: HookedTransformer, layer: int, top_k: int = 10) -> dict:
    """Norm of each neuron's input and output weight vectors.

    Large norms = high-impact neurons.
    """
    W_in = model.blocks[layer].mlp.W_in  # [d_model, d_mlp]
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    d_mlp = W_in.shape[1]

    in_norms = jnp.linalg.norm(W_in, axis=0)  # [d_mlp]
    out_norms = jnp.linalg.norm(W_out, axis=1)  # [d_mlp]
    combined = in_norms * out_norms

    top_indices = jnp.argsort(combined)[-top_k:][::-1]
    top_neurons = []
    for idx in top_indices:
        idx = int(idx)
        top_neurons.append({
            'neuron': idx,
            'in_norm': float(in_norms[idx]),
            'out_norm': float(out_norms[idx]),
            'combined_norm': float(combined[idx]),
        })

    return {
        'layer': layer,
        'd_mlp': d_mlp,
        'mean_in_norm': float(jnp.mean(in_norms)),
        'mean_out_norm': float(jnp.mean(out_norms)),
        'top_neurons': top_neurons,
    }


def mlp_in_out_alignment(model: HookedTransformer, layer: int) -> dict:
    """How aligned are W_in and W_out for each neuron?

    High alignment = neuron reads and writes in similar directions.
    """
    W_in = model.blocks[layer].mlp.W_in  # [d_model, d_mlp]
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    d_mlp = W_in.shape[1]

    # Per-neuron: cos(W_in[:, n], W_out[n, :])
    cosines = []
    for n in range(d_mlp):
        in_dir = W_in[:, n]
        out_dir = W_out[n, :]
        cos = float(jnp.dot(in_dir, out_dir) / (jnp.linalg.norm(in_dir) * jnp.linalg.norm(out_dir) + 1e-10))
        cosines.append(cos)

    mean_cos = sum(cosines) / len(cosines)
    aligned = sum(1 for c in cosines if abs(c) > 0.5)

    return {
        'layer': layer,
        'mean_alignment': mean_cos,
        'n_aligned': aligned,
        'n_total': d_mlp,
        'fraction_aligned': aligned / d_mlp,
    }


def mlp_cross_layer_similarity(model: HookedTransformer) -> dict:
    """How similar are MLP weights across layers?

    Similar weights → similar computations.
    """
    n_layers = model.cfg.n_layers

    in_dirs = []
    out_dirs = []
    for layer in range(n_layers):
        W_in = model.blocks[layer].mlp.W_in
        W_out = model.blocks[layer].mlp.W_out
        in_flat = W_in.reshape(-1)
        out_flat = W_out.reshape(-1)
        in_dirs.append(in_flat / (jnp.linalg.norm(in_flat) + 1e-10))
        out_dirs.append(out_flat / (jnp.linalg.norm(out_flat) + 1e-10))

    in_dirs = jnp.stack(in_dirs)
    out_dirs = jnp.stack(out_dirs)
    in_sims = in_dirs @ in_dirs.T
    out_sims = out_dirs @ out_dirs.T

    pairs = []
    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            pairs.append({
                'layer_a': i,
                'layer_b': j,
                'W_in_similarity': float(in_sims[i, j]),
                'W_out_similarity': float(out_sims[i, j]),
            })

    return {
        'pairs': pairs,
    }


def mlp_unembed_alignment(model: HookedTransformer, layer: int, top_k: int = 10) -> dict:
    """Which vocabulary tokens does each MLP neuron most promote?

    Projects W_out rows through unembedding.
    """
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    d_mlp = W_out.shape[0]

    # For efficiency, analyze top-norm neurons only
    out_norms = jnp.linalg.norm(W_out, axis=1)
    top_neurons = jnp.argsort(out_norms)[-top_k:][::-1]

    per_neuron = []
    for idx in top_neurons:
        idx = int(idx)
        neuron_dir = W_out[idx]  # [d_model]
        logits = neuron_dir @ W_U  # [d_vocab]
        top_token = int(jnp.argmax(logits))
        bottom_token = int(jnp.argmin(logits))

        per_neuron.append({
            'neuron': idx,
            'output_norm': float(out_norms[idx]),
            'top_promoted_token': top_token,
            'top_promoted_logit': float(logits[top_token]),
            'top_suppressed_token': bottom_token,
            'top_suppressed_logit': float(logits[bottom_token]),
        })

    return {
        'layer': layer,
        'per_neuron': per_neuron,
    }
