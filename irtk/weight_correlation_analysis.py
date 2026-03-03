"""Weight correlation analysis.

Analyze correlations and patterns in model weights: cross-layer weight
similarity, parameter covariance, weight symmetry, and initialization
deviation.
"""

import jax
import jax.numpy as jnp


def cross_layer_weight_correlation(model, matrix_type='W_Q'):
    """Correlation between the same weight matrix across layers.

    Returns:
        dict with correlation_matrix and per_pair correlations.
    """
    n_layers = model.cfg.n_layers

    # Collect weight matrices
    matrices = []
    for l in range(n_layers):
        block = model.blocks[l]
        if matrix_type in ('W_Q', 'W_K', 'W_V', 'W_O'):
            mat = getattr(block.attn, matrix_type)  # [n_heads, d_model, d_head] or similar
        elif matrix_type in ('W_in', 'W_out'):
            mat = getattr(block.mlp, matrix_type)
        else:
            continue
        matrices.append(mat.reshape(-1))  # flatten

    n = len(matrices)
    corr_matrix = jnp.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix = corr_matrix.at[i, j].set(1.0)
            elif j > i:
                # Cosine similarity as correlation proxy
                ni = jnp.linalg.norm(matrices[i])
                nj = jnp.linalg.norm(matrices[j])
                cos = jnp.dot(matrices[i], matrices[j]) / jnp.maximum(ni * nj, 1e-10)
                corr_matrix = corr_matrix.at[i, j].set(cos)
                corr_matrix = corr_matrix.at[j, i].set(cos)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                'layer_i': i,
                'layer_j': j,
                'correlation': float(corr_matrix[i, j]),
            })

    mean_corr = float(jnp.mean(jnp.array([p['correlation'] for p in pairs]))) if pairs else 0.0

    return {
        'matrix_type': matrix_type,
        'correlation_matrix': corr_matrix,
        'per_pair': pairs,
        'mean_correlation': mean_corr,
    }


def weight_norm_pattern(model):
    """Analyze how weight norms vary across layers.

    Returns:
        dict with per_layer norm info and trends.
    """
    n_layers = model.cfg.n_layers

    results = []
    for l in range(n_layers):
        block = model.blocks[l]
        norms = {}
        for name in ['W_Q', 'W_K', 'W_V', 'W_O']:
            mat = getattr(block.attn, name)
            norms[name] = float(jnp.linalg.norm(mat))
        for name in ['W_in', 'W_out']:
            mat = getattr(block.mlp, name)
            norms[name] = float(jnp.linalg.norm(mat))

        total = sum(norms.values())
        results.append({
            'layer': l,
            'norms': norms,
            'total_norm': total,
        })

    # Trend: is total norm growing/shrinking?
    total_norms = [r['total_norm'] for r in results]
    if len(total_norms) >= 2:
        trend = (total_norms[-1] - total_norms[0]) / max(total_norms[0], 1e-10)
    else:
        trend = 0.0

    return {
        'per_layer': results,
        'norm_trend': trend,
    }


def head_weight_similarity(model, layer=0):
    """How similar are different heads' weight matrices within a layer?

    Returns:
        dict with per_pair head similarity.
    """
    block = model.blocks[layer]
    n_heads = model.cfg.n_heads

    # Compare QK weights across heads
    W_Q = block.attn.W_Q  # [n_heads, d_model, d_head]
    W_K = block.attn.W_K
    W_V = block.attn.W_V

    pairs = []
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            q_sim = float(jnp.dot(W_Q[i].reshape(-1), W_Q[j].reshape(-1)) /
                         jnp.maximum(jnp.linalg.norm(W_Q[i]) * jnp.linalg.norm(W_Q[j]), 1e-10))
            k_sim = float(jnp.dot(W_K[i].reshape(-1), W_K[j].reshape(-1)) /
                         jnp.maximum(jnp.linalg.norm(W_K[i]) * jnp.linalg.norm(W_K[j]), 1e-10))
            v_sim = float(jnp.dot(W_V[i].reshape(-1), W_V[j].reshape(-1)) /
                         jnp.maximum(jnp.linalg.norm(W_V[i]) * jnp.linalg.norm(W_V[j]), 1e-10))

            pairs.append({
                'head_i': i,
                'head_j': j,
                'q_similarity': q_sim,
                'k_similarity': k_sim,
                'v_similarity': v_sim,
                'mean_similarity': (q_sim + k_sim + v_sim) / 3,
            })

    return {'layer': layer, 'per_pair': pairs}


def qk_ov_weight_balance(model):
    """Compare QK circuit vs OV circuit weight magnitudes.

    Returns:
        dict with per_layer balance between QK and OV circuits.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    results = []
    for l in range(n_layers):
        block = model.blocks[l]
        W_Q = block.attn.W_Q  # [n_heads, d_model, d_head]
        W_K = block.attn.W_K
        W_V = block.attn.W_V
        W_O = block.attn.W_O

        per_head = []
        for h in range(n_heads):
            qk_norm = float(jnp.linalg.norm(W_Q[h]) * jnp.linalg.norm(W_K[h]))
            ov_norm = float(jnp.linalg.norm(W_V[h]) * jnp.linalg.norm(W_O[h]))
            total = qk_norm + ov_norm
            per_head.append({
                'head': h,
                'qk_norm': qk_norm,
                'ov_norm': ov_norm,
                'qk_fraction': qk_norm / max(total, 1e-10),
            })

        results.append({
            'layer': l,
            'per_head': per_head,
        })

    return {'per_layer': results}


def weight_initialization_deviation(model):
    """How far have weights deviated from a typical initialization?

    Compares weight statistics to expected values for random init.

    Returns:
        dict with per_layer deviation metrics.
    """
    n_layers = model.cfg.n_layers

    results = []
    for l in range(n_layers):
        block = model.blocks[l]

        deviations = {}
        for name in ['W_Q', 'W_K', 'W_V', 'W_O']:
            mat = getattr(block.attn, name)
            flat = mat.reshape(-1)
            # For Gaussian init, expected std ~ 1/sqrt(fan_in)
            actual_std = float(jnp.std(flat))
            actual_mean = float(jnp.mean(flat))
            # Kurtosis (excess): 0 for Gaussian
            centered = flat - actual_mean
            kurtosis = float(jnp.mean(centered ** 4) / jnp.maximum(actual_std ** 4, 1e-10) - 3.0)
            deviations[name] = {
                'mean': actual_mean,
                'std': actual_std,
                'kurtosis': kurtosis,
            }

        results.append({
            'layer': l,
            'deviations': deviations,
        })

    return {'per_layer': results}
