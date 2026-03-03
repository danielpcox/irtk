"""Weight space geometry.

Analyze the geometric structure of weight matrices: manifold properties,
weight space distances, parameter symmetries, and interpolation effects.
"""

import jax
import jax.numpy as jnp


def weight_manifold_dimension(model, layer):
    """Estimate the effective dimensionality of each weight matrix.

    Args:
        model: HookedTransformer
        layer: layer index

    Returns:
        dict with effective dimensionality for each weight matrix.
    """
    block = model.blocks[layer]

    matrices = {
        'W_Q': block.attn.W_Q.reshape(-1, block.attn.W_Q.shape[-1]),
        'W_K': block.attn.W_K.reshape(-1, block.attn.W_K.shape[-1]),
        'W_V': block.attn.W_V.reshape(-1, block.attn.W_V.shape[-1]),
        'W_O': block.attn.W_O.reshape(-1, block.attn.W_O.shape[-1]),
        'W_in': block.mlp.W_in,
        'W_out': block.mlp.W_out,
    }

    results = {}
    for name, W in matrices.items():
        S = jnp.linalg.svd(W, compute_uv=False)
        S_norm = S / jnp.maximum(jnp.sum(S), 1e-10)
        S_safe = jnp.maximum(S_norm, 1e-10)
        entropy = -float(jnp.sum(S_safe * jnp.log(S_safe)))
        eff_dim = float(jnp.exp(jnp.array(entropy)))
        results[name] = {
            'effective_dimension': eff_dim,
            'full_rank': min(W.shape),
            'rank_utilization': eff_dim / min(W.shape),
            'top_singular_value': float(S[0]),
            'condition_number': float(S[0] / jnp.maximum(S[-1], 1e-10)),
        }

    return {'layer': layer, 'matrices': results}


def weight_distance_profile(model):
    """Compute distances between corresponding weight matrices across layers.

    Returns:
        dict with pairwise weight distances.
    """
    n_layers = model.cfg.n_layers
    matrix_names = ['W_Q', 'W_K', 'W_V', 'W_O']

    distances = {}
    for name in matrix_names:
        dist_matrix = jnp.zeros((n_layers, n_layers))
        for i in range(n_layers):
            W_i = getattr(model.blocks[i].attn, name).reshape(-1)
            for j in range(i + 1, n_layers):
                W_j = getattr(model.blocks[j].attn, name).reshape(-1)
                d = float(jnp.linalg.norm(W_i - W_j))
                dist_matrix = dist_matrix.at[i, j].set(d)
                dist_matrix = dist_matrix.at[j, i].set(d)

        distances[name] = {
            'distance_matrix': [[float(dist_matrix[i, j]) for j in range(n_layers)]
                                for i in range(n_layers)],
            'mean_distance': float(jnp.mean(dist_matrix[jnp.triu_indices(n_layers, k=1)])) if n_layers > 1 else 0.0,
        }

    return {'n_layers': n_layers, 'distances': distances}


def weight_symmetry_analysis(model, layer):
    """Analyze symmetries in weight matrices (e.g., W_Q ≈ W_K).

    Args:
        model: HookedTransformer
        layer: layer index

    Returns:
        dict with symmetry scores between weight pairs.
    """
    block = model.blocks[layer]
    W_Q = block.attn.W_Q  # [n_heads, d_model, d_head]
    W_K = block.attn.W_K
    W_V = block.attn.W_V
    n_heads = W_Q.shape[0]

    def _cosine_flat(a, b):
        a_flat = a.reshape(-1)
        b_flat = b.reshape(-1)
        return float(jnp.sum(a_flat * b_flat) /
                     jnp.maximum(jnp.linalg.norm(a_flat) * jnp.linalg.norm(b_flat), 1e-10))

    # Per-head QK symmetry
    per_head = []
    for h in range(n_heads):
        qk_cos = _cosine_flat(W_Q[h], W_K[h])
        qv_cos = _cosine_flat(W_Q[h], W_V[h])
        kv_cos = _cosine_flat(W_K[h], W_V[h])
        per_head.append({
            'head': h,
            'qk_cosine': qk_cos,
            'qv_cosine': qv_cos,
            'kv_cosine': kv_cos,
        })

    return {
        'layer': layer,
        'per_head': per_head,
        'mean_qk_symmetry': float(jnp.mean(jnp.array([h['qk_cosine'] for h in per_head]))),
        'mean_kv_symmetry': float(jnp.mean(jnp.array([h['kv_cosine'] for h in per_head]))),
        'global_qk_cosine': _cosine_flat(W_Q, W_K),
    }


def weight_interpolation_effect(model, tokens, layer, matrix_name='W_in', n_steps=5):
    """Measure the effect of interpolating a weight matrix toward zero.

    Args:
        model: HookedTransformer
        tokens: input token IDs
        layer: layer index
        matrix_name: which weight to interpolate ('W_in', 'W_out', 'W_Q', etc.)
        n_steps: number of interpolation steps

    Returns:
        dict with logit changes at each interpolation step.
    """
    import equinox as eqx

    clean_logits = model(tokens)
    alphas = jnp.linspace(0.0, 1.0, n_steps)

    results = []
    for alpha in alphas:
        alpha_f = float(alpha)

        # Scale the weight matrix
        if matrix_name in ('W_Q', 'W_K', 'W_V', 'W_O'):
            original = getattr(model.blocks[layer].attn, matrix_name)
            scaled = original * (1.0 - alpha_f)
            new_attn = eqx.tree_at(lambda a: getattr(a, matrix_name),
                                    model.blocks[layer].attn, scaled)
            new_block = eqx.tree_at(lambda b: b.attn, model.blocks[layer], new_attn)
        elif matrix_name in ('W_in', 'W_out'):
            original = getattr(model.blocks[layer].mlp, matrix_name)
            scaled = original * (1.0 - alpha_f)
            new_mlp = eqx.tree_at(lambda m: getattr(m, matrix_name),
                                   model.blocks[layer].mlp, scaled)
            new_block = eqx.tree_at(lambda b: b.mlp, model.blocks[layer], new_mlp)
        else:
            raise ValueError(f"Unknown matrix: {matrix_name}")

        new_model = eqx.tree_at(lambda m: m.blocks[layer], model, new_block)
        mod_logits = new_model(tokens)
        diff = mod_logits - clean_logits

        results.append({
            'alpha': alpha_f,
            'max_logit_change': float(jnp.max(jnp.abs(diff))),
            'mean_logit_change': float(jnp.mean(jnp.abs(diff))),
        })

    return {
        'layer': layer,
        'matrix': matrix_name,
        'interpolation': results,
    }


def parameter_norm_geometry(model):
    """Analyze the norm geometry of all parameters.

    Returns:
        dict with parameter norms and their distribution.
    """
    n_layers = model.cfg.n_layers

    layer_stats = []
    all_norms = []
    for l in range(n_layers):
        block = model.blocks[l]
        stats = {}

        for name in ['W_Q', 'W_K', 'W_V', 'W_O']:
            W = getattr(block.attn, name)
            n = float(jnp.linalg.norm(W))
            stats[name] = n
            all_norms.append(n)

        for name in ['W_in', 'W_out']:
            W = getattr(block.mlp, name)
            n = float(jnp.linalg.norm(W))
            stats[name] = n
            all_norms.append(n)

        layer_stats.append({'layer': l, 'norms': stats})

    all_norms = jnp.array(all_norms)
    return {
        'per_layer': layer_stats,
        'global_mean_norm': float(jnp.mean(all_norms)),
        'global_std_norm': float(jnp.std(all_norms)),
        'global_max_norm': float(jnp.max(all_norms)),
        'global_min_norm': float(jnp.min(all_norms)),
        'embed_norm': float(jnp.linalg.norm(model.embed.W_E)),
        'unembed_norm': float(jnp.linalg.norm(model.unembed.W_U)),
    }
