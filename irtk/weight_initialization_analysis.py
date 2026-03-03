"""Weight initialization analysis: analyze weight scales and distributions."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def weight_scale_profile(model: HookedTransformer) -> dict:
    """Profile the scale (norm) of each weight matrix.

    Compares actual norms to expected initialization scales.
    """
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    per_weight = []

    # Embedding weights
    w_e = model.embed.W_E
    per_weight.append({
        'name': 'W_E',
        'shape': list(w_e.shape),
        'mean_norm': float(jnp.mean(jnp.linalg.norm(w_e, axis=-1))),
        'std': float(jnp.std(w_e)),
        'max_abs': float(jnp.max(jnp.abs(w_e))),
    })

    # Unembed
    w_u = model.unembed.W_U
    per_weight.append({
        'name': 'W_U',
        'shape': list(w_u.shape),
        'mean_norm': float(jnp.mean(jnp.linalg.norm(w_u, axis=0))),
        'std': float(jnp.std(w_u)),
        'max_abs': float(jnp.max(jnp.abs(w_u))),
    })

    for layer in range(n_layers):
        attn = model.blocks[layer].attn
        mlp = model.blocks[layer].mlp

        for name, w in [('W_Q', attn.W_Q), ('W_K', attn.W_K),
                        ('W_V', attn.W_V), ('W_O', attn.W_O)]:
            per_weight.append({
                'name': f'L{layer}_{name}',
                'shape': list(w.shape),
                'mean_norm': float(jnp.mean(jnp.linalg.norm(w.reshape(-1, w.shape[-1]), axis=-1))),
                'std': float(jnp.std(w)),
                'max_abs': float(jnp.max(jnp.abs(w))),
            })

        per_weight.append({
            'name': f'L{layer}_W_in',
            'shape': list(mlp.W_in.shape),
            'mean_norm': float(jnp.mean(jnp.linalg.norm(mlp.W_in, axis=-1))),
            'std': float(jnp.std(mlp.W_in)),
            'max_abs': float(jnp.max(jnp.abs(mlp.W_in))),
        })
        per_weight.append({
            'name': f'L{layer}_W_out',
            'shape': list(mlp.W_out.shape),
            'mean_norm': float(jnp.mean(jnp.linalg.norm(mlp.W_out, axis=-1))),
            'std': float(jnp.std(mlp.W_out)),
            'max_abs': float(jnp.max(jnp.abs(mlp.W_out))),
        })

    return {
        'per_weight': per_weight,
        'n_weight_matrices': len(per_weight),
        'mean_std': sum(w['std'] for w in per_weight) / len(per_weight),
    }


def weight_distribution_stats(model: HookedTransformer, layer: int) -> dict:
    """Detailed distribution statistics for a specific layer's weights.

    Checks for normality, symmetry, and outliers.
    """
    attn = model.blocks[layer].attn
    mlp = model.blocks[layer].mlp

    def stats(w, name):
        flat = w.reshape(-1)
        return {
            'name': name,
            'n_params': int(flat.shape[0]),
            'mean': float(jnp.mean(flat)),
            'std': float(jnp.std(flat)),
            'skewness': float(jnp.mean(((flat - jnp.mean(flat)) / (jnp.std(flat) + 1e-10)) ** 3)),
            'kurtosis': float(jnp.mean(((flat - jnp.mean(flat)) / (jnp.std(flat) + 1e-10)) ** 4)),
            'min': float(jnp.min(flat)),
            'max': float(jnp.max(flat)),
            'near_zero_fraction': float(jnp.mean(jnp.abs(flat) < 0.01)),
        }

    per_weight = [
        stats(attn.W_Q, 'W_Q'),
        stats(attn.W_K, 'W_K'),
        stats(attn.W_V, 'W_V'),
        stats(attn.W_O, 'W_O'),
        stats(mlp.W_in, 'W_in'),
        stats(mlp.W_out, 'W_out'),
    ]

    total_params = sum(w['n_params'] for w in per_weight)

    return {
        'layer': layer,
        'per_weight': per_weight,
        'total_params': total_params,
    }


def weight_norm_comparison(model: HookedTransformer) -> dict:
    """Compare weight norms across layers to detect imbalances.

    Large norm differences between layers can indicate training issues.
    """
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        attn = model.blocks[layer].attn
        mlp = model.blocks[layer].mlp

        attn_norm = sum(float(jnp.linalg.norm(w)) for w in [attn.W_Q, attn.W_K, attn.W_V, attn.W_O])
        mlp_norm = float(jnp.linalg.norm(mlp.W_in)) + float(jnp.linalg.norm(mlp.W_out))

        per_layer.append({
            'layer': layer,
            'attn_norm': attn_norm,
            'mlp_norm': mlp_norm,
            'total_norm': attn_norm + mlp_norm,
            'attn_fraction': attn_norm / (attn_norm + mlp_norm) if (attn_norm + mlp_norm) > 0 else 0.5,
        })

    norms = [p['total_norm'] for p in per_layer]
    mean_norm = sum(norms) / len(norms)
    max_ratio = max(norms) / (min(norms) + 1e-10)

    return {
        'per_layer': per_layer,
        'mean_layer_norm': mean_norm,
        'max_layer_ratio': max_ratio,
        'is_balanced': max_ratio < 3.0,
    }


def weight_sparsity_profile(model: HookedTransformer, threshold: float = 0.01) -> dict:
    """How sparse are the model's weights?

    Measures fraction of near-zero weights at different thresholds.
    """
    n_layers = model.cfg.n_layers

    per_layer = []
    total_params = 0
    total_sparse = 0

    for layer in range(n_layers):
        attn = model.blocks[layer].attn
        mlp = model.blocks[layer].mlp

        weights = [attn.W_Q, attn.W_K, attn.W_V, attn.W_O, mlp.W_in, mlp.W_out]
        layer_params = sum(w.size for w in weights)
        layer_sparse = sum(int(jnp.sum(jnp.abs(w) < threshold)) for w in weights)

        per_layer.append({
            'layer': layer,
            'n_params': layer_params,
            'n_sparse': layer_sparse,
            'sparsity': layer_sparse / layer_params,
        })
        total_params += layer_params
        total_sparse += layer_sparse

    return {
        'per_layer': per_layer,
        'threshold': threshold,
        'total_params': total_params,
        'total_sparsity': total_sparse / total_params if total_params > 0 else 0.0,
    }


def embedding_weight_analysis(model: HookedTransformer) -> dict:
    """Analyze embedding and unembedding weight properties.

    Checks alignment, isotropy, and norm distribution.
    """
    W_E = model.embed.W_E  # [d_vocab, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    # Embedding norms
    embed_norms = jnp.linalg.norm(W_E, axis=-1)
    unembed_norms = jnp.linalg.norm(W_U, axis=0)

    # Isotropy: how uniform are the norms?
    embed_cv = float(jnp.std(embed_norms) / (jnp.mean(embed_norms) + 1e-10))
    unembed_cv = float(jnp.std(unembed_norms) / (jnp.mean(unembed_norms) + 1e-10))

    # Embed-unembed alignment (do they share directions?)
    # Sample a few tokens for efficiency
    n_sample = min(50, W_E.shape[0])
    sample_E = W_E[:n_sample]
    sample_U = W_U[:, :n_sample].T

    e_normed = sample_E / (jnp.linalg.norm(sample_E, axis=-1, keepdims=True) + 1e-10)
    u_normed = sample_U / (jnp.linalg.norm(sample_U, axis=-1, keepdims=True) + 1e-10)
    alignment = jnp.sum(e_normed * u_normed, axis=-1)
    mean_alignment = float(jnp.mean(alignment))

    return {
        'embed_mean_norm': float(jnp.mean(embed_norms)),
        'embed_norm_cv': embed_cv,
        'unembed_mean_norm': float(jnp.mean(unembed_norms)),
        'unembed_norm_cv': unembed_cv,
        'embed_unembed_alignment': mean_alignment,
        'is_isotropic': embed_cv < 0.5,
    }
