"""Unembedding analysis: detailed analysis of the unembedding matrix W_U."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def unembed_spectrum(model: HookedTransformer) -> dict:
    """Singular value spectrum of the unembedding matrix.

    Reveals the effective dimensionality of the output space.
    """
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    sv = jnp.linalg.svd(W_U, compute_uv=False)
    total = jnp.sum(sv)
    sv_norm = sv / (total + 1e-10)
    entropy = -float(jnp.sum(sv_norm * jnp.log(sv_norm + 1e-10)))
    eff_rank = float(jnp.exp(entropy))

    cumulative = jnp.cumsum(sv) / total
    dim_90 = int(jnp.searchsorted(cumulative, 0.9)) + 1

    return {
        'd_model': int(W_U.shape[0]),
        'd_vocab': int(W_U.shape[1]),
        'effective_rank': eff_rank,
        'dim_for_90_pct': dim_90,
        'spectral_norm': float(sv[0]),
        'top_5_sv': [float(sv[i]) for i in range(min(5, len(sv)))],
    }


def unembed_token_norms(model: HookedTransformer, token_ids: list = None) -> dict:
    """Norm of each token's unembedding column.

    Tokens with large norms are easier to promote.
    """
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    d_vocab = W_U.shape[1]

    all_norms = jnp.linalg.norm(W_U, axis=0)  # [d_vocab]
    mean_norm = float(jnp.mean(all_norms))
    std_norm = float(jnp.std(all_norms))

    if token_ids is None:
        token_ids = list(range(min(20, d_vocab)))

    per_token = []
    for tid in token_ids:
        norm = float(all_norms[tid])
        z_score = (norm - mean_norm) / (std_norm + 1e-10)
        per_token.append({
            'token': tid,
            'norm': norm,
            'z_score': z_score,
            'is_outlier': bool(abs(z_score) > 2),
        })

    return {
        'global_mean_norm': mean_norm,
        'global_std_norm': std_norm,
        'per_token': per_token,
    }


def unembed_direction_clustering(model: HookedTransformer, token_ids: list = None) -> dict:
    """How clustered are token unembedding directions?

    Similar directions = tokens that are hard to distinguish.
    """
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    if token_ids is None:
        token_ids = list(range(min(20, W_U.shape[1])))

    import numpy as np
    token_ids_np = np.array(token_ids)
    U_subset = W_U[:, token_ids_np].T  # [n_tokens, d_model]
    norms = jnp.linalg.norm(U_subset, axis=-1, keepdims=True)
    U_normed = U_subset / (norms + 1e-10)

    sim_matrix = U_normed @ U_normed.T

    pairs = []
    for i in range(len(token_ids)):
        for j in range(i + 1, len(token_ids)):
            pairs.append({
                'token_a': token_ids[i],
                'token_b': token_ids[j],
                'cosine': float(sim_matrix[i, j]),
            })

    mean_sim = sum(abs(p['cosine']) for p in pairs) / len(pairs) if pairs else 0.0

    return {
        'pairs': pairs,
        'mean_abs_similarity': mean_sim,
        'is_well_separated': bool(mean_sim < 0.3),
    }


def unembed_component_projection(model: HookedTransformer, tokens: jnp.ndarray, layer: int, position: int = -1) -> dict:
    """Project each component's output through the unembedding matrix.

    Shows what each component is trying to say.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position

    W_U = model.unembed.W_U
    n_layers = model.cfg.n_layers

    components = [
        ('embed', cache['hook_embed'][pos] + cache['hook_pos_embed'][pos]),
    ]
    for l in range(n_layers):
        components.append((f'attn_{l}', cache[f'blocks.{l}.hook_attn_out'][pos]))
        components.append((f'mlp_{l}', cache[f'blocks.{l}.hook_mlp_out'][pos]))

    per_component = []
    for name, vec in components:
        logits = vec @ W_U  # [d_vocab]
        top_token = int(jnp.argmax(logits))
        top_logit = float(logits[top_token])
        bottom_token = int(jnp.argmin(logits))
        bottom_logit = float(logits[bottom_token])

        per_component.append({
            'component': name,
            'top_token': top_token,
            'top_logit': top_logit,
            'bottom_token': bottom_token,
            'bottom_logit': bottom_logit,
            'logit_range': top_logit - bottom_logit,
        })

    return {
        'position': pos,
        'per_component': per_component,
    }


def unembed_bias_analysis(model: HookedTransformer) -> dict:
    """Analyze the unembedding bias b_U.

    Large biases make certain tokens more likely regardless of context.
    """
    b_U = model.unembed.b_U  # [d_vocab]
    d_vocab = b_U.shape[0]

    mean_bias = float(jnp.mean(b_U))
    std_bias = float(jnp.std(b_U))

    top_5 = jnp.argsort(b_U)[-5:][::-1]
    bottom_5 = jnp.argsort(b_U)[:5]

    return {
        'mean_bias': mean_bias,
        'std_bias': std_bias,
        'max_bias': float(jnp.max(b_U)),
        'min_bias': float(jnp.min(b_U)),
        'top_biased_tokens': [{'token': int(t), 'bias': float(b_U[t])} for t in top_5],
        'bottom_biased_tokens': [{'token': int(t), 'bias': float(b_U[t])} for t in bottom_5],
        'has_significant_bias': bool(std_bias > 0.1),
    }
