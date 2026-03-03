"""Embedding layer analysis: analyze the embedding and its relationship to computation."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def embedding_norm_structure(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Analyze the norm structure of token embeddings.

    Are all tokens embedded with similar norms, or do some stand out?
    """
    W_E = model.embed.W_E  # [d_vocab, d_model]
    d_vocab = W_E.shape[0]

    # Global statistics
    all_norms = jnp.linalg.norm(W_E, axis=-1)
    mean_norm = float(jnp.mean(all_norms))
    std_norm = float(jnp.std(all_norms))
    max_norm = float(jnp.max(all_norms))
    min_norm = float(jnp.min(all_norms))

    # Input token norms
    input_norms = jnp.linalg.norm(W_E[tokens], axis=-1)

    per_token = []
    for i, t in enumerate(tokens):
        t_int = int(t)
        norm = float(input_norms[i])
        percentile_rank = float(jnp.mean(all_norms < norm))
        per_token.append({
            'position': i,
            'token': t_int,
            'norm': norm,
            'percentile': percentile_rank,
            'is_outlier': norm > mean_norm + 2 * std_norm,
        })

    return {
        'global_mean_norm': mean_norm,
        'global_std_norm': std_norm,
        'global_max_norm': max_norm,
        'global_min_norm': min_norm,
        'per_token': per_token,
    }


def embedding_similarity_to_unembed(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How aligned are input embeddings with the corresponding unembed directions?

    High alignment = token has a direct path to output prediction.
    """
    W_E = model.embed.W_E  # [d_vocab, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    per_token = []
    for i, t in enumerate(tokens):
        t_int = int(t)
        embed = W_E[t_int]
        unembed = W_U[:, t_int]

        # Cosine similarity
        cos = float(
            jnp.dot(embed, unembed) /
            (jnp.linalg.norm(embed) * jnp.linalg.norm(unembed) + 1e-10)
        )

        # Self-logit: how much does this token's embedding promote itself?
        self_logit = float(jnp.dot(embed, unembed))

        per_token.append({
            'position': i,
            'token': t_int,
            'embed_unembed_cos': cos,
            'self_logit': self_logit,
            'is_self_promoting': self_logit > 0,
        })

    mean_cos = sum(p['embed_unembed_cos'] for p in per_token) / len(per_token)

    return {
        'per_token': per_token,
        'mean_embed_unembed_cos': mean_cos,
    }


def embedding_neighborhood(model: HookedTransformer, tokens: jnp.ndarray, top_k: int = 5) -> dict:
    """Find the nearest neighbors of each input token in embedding space.

    Shows which tokens are similar before any computation.
    """
    W_E = model.embed.W_E  # [d_vocab, d_model]
    d_vocab = W_E.shape[0]

    # Normalize embeddings
    W_E_normed = W_E / (jnp.linalg.norm(W_E, axis=-1, keepdims=True) + 1e-10)

    per_token = []
    for i, t in enumerate(tokens):
        t_int = int(t)
        query = W_E_normed[t_int]
        sims = query @ W_E_normed.T  # [d_vocab]
        # Exclude self
        sims = sims.at[t_int].set(-2.0)
        top_indices = jnp.argsort(sims)[-top_k:][::-1]

        neighbors = [
            {'token': int(idx), 'similarity': float(sims[idx])}
            for idx in top_indices
        ]

        per_token.append({
            'position': i,
            'token': t_int,
            'neighbors': neighbors,
            'mean_neighbor_sim': sum(n['similarity'] for n in neighbors) / len(neighbors),
        })

    return {
        'per_token': per_token,
    }


def positional_embedding_structure(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Analyze the positional embedding for input positions.

    Norm, pairwise similarity, and content-position interaction.
    """
    W_pos = model.pos_embed.W_pos  # [n_ctx, d_model]
    seq_len = tokens.shape[0]

    pos_embeds = W_pos[:seq_len]
    norms = jnp.linalg.norm(pos_embeds, axis=-1)

    per_position = []
    for i in range(seq_len):
        per_position.append({
            'position': i,
            'norm': float(norms[i]),
        })

    # Pairwise similarity
    normed = pos_embeds / (norms[:, None] + 1e-10)
    sim_matrix = normed @ normed.T

    # Content-position interaction
    W_E = model.embed.W_E
    content_embeds = W_E[tokens]  # [seq, d_model]
    content_normed = content_embeds / (jnp.linalg.norm(content_embeds, axis=-1, keepdims=True) + 1e-10)
    pos_normed = pos_embeds / (norms[:, None] + 1e-10)
    content_pos_cos = jnp.sum(content_normed * pos_normed, axis=-1)  # [seq]

    for i in range(seq_len):
        per_position[i]['content_position_cos'] = float(content_pos_cos[i])

    mean_pos_sim = float(jnp.mean(sim_matrix) - 1.0 / seq_len)  # Subtract self-sim

    return {
        'per_position': per_position,
        'mean_position_similarity': mean_pos_sim,
    }


def embedding_effective_dimension(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Measure the effective dimensionality of the input embeddings.

    How much of the d_model space do the input embeddings use?
    """
    W_E = model.embed.W_E
    content = W_E[tokens]  # [seq, d_model]

    # Center the embeddings
    centered = content - jnp.mean(content, axis=0, keepdims=True)

    # SVD
    U, S, Vt = jnp.linalg.svd(centered, full_matrices=False)
    S_normalized = S / (jnp.sum(S) + 1e-10)

    # Effective dimension via entropy
    entropy = -float(jnp.sum(S_normalized * jnp.log(S_normalized + 1e-10)))
    effective_dim = float(jnp.exp(entropy))

    # Variance explained by top components
    cumvar = jnp.cumsum(S ** 2) / (jnp.sum(S ** 2) + 1e-10)
    dim_90 = int(jnp.searchsorted(cumvar, 0.9) + 1)

    per_component = []
    for i in range(min(5, len(S))):
        per_component.append({
            'component': i,
            'singular_value': float(S[i]),
            'variance_explained': float(S[i] ** 2 / (jnp.sum(S ** 2) + 1e-10)),
            'cumulative': float(cumvar[i]),
        })

    return {
        'effective_dimension': effective_dim,
        'dim_for_90_pct': dim_90,
        'per_component': per_component,
        'd_model': model.cfg.d_model,
    }
