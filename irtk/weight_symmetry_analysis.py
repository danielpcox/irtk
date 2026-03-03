"""Weight symmetry analysis: detect symmetry and structure in model weights."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def qk_symmetry(model: HookedTransformer, layer: int, head: int) -> dict:
    """How symmetric is the QK circuit (W_Q @ W_K^T)?

    Symmetric QK = content-based matching; antisymmetric = positional.
    """
    W_Q = model.blocks[layer].attn.W_Q[head]  # [d_model, d_head]
    W_K = model.blocks[layer].attn.W_K[head]  # [d_model, d_head]

    QK = W_Q @ W_K.T  # [d_model, d_model]
    sym = (QK + QK.T) / 2
    anti = (QK - QK.T) / 2

    sym_norm = float(jnp.linalg.norm(sym))
    anti_norm = float(jnp.linalg.norm(anti))
    total = sym_norm + anti_norm + 1e-10

    return {
        'layer': layer,
        'head': head,
        'symmetric_fraction': sym_norm / total,
        'antisymmetric_fraction': anti_norm / total,
        'is_symmetric': sym_norm > 2 * anti_norm,
    }


def ov_symmetry(model: HookedTransformer, layer: int, head: int) -> dict:
    """How symmetric is the OV circuit (W_V @ W_O)?"""
    W_V = model.blocks[layer].attn.W_V[head]  # [d_model, d_head]
    W_O = model.blocks[layer].attn.W_O[head]  # [d_head, d_model]

    OV = W_V @ W_O  # [d_model, d_model]
    sym = (OV + OV.T) / 2
    anti = (OV - OV.T) / 2

    sym_norm = float(jnp.linalg.norm(sym))
    anti_norm = float(jnp.linalg.norm(anti))
    total = sym_norm + anti_norm + 1e-10

    return {
        'layer': layer,
        'head': head,
        'symmetric_fraction': sym_norm / total,
        'antisymmetric_fraction': anti_norm / total,
        'is_symmetric': sym_norm > 2 * anti_norm,
    }


def mlp_weight_symmetry(model: HookedTransformer, layer: int) -> dict:
    """Symmetry between W_in and W_out: is W_out ≈ W_in^T?"""
    W_in = model.blocks[layer].mlp.W_in  # [d_model, d_mlp]
    W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]

    # Compare W_out with W_in^T
    in_flat = W_in.T.reshape(-1)  # [d_mlp * d_model]
    out_flat = W_out.reshape(-1)

    in_norm = jnp.linalg.norm(in_flat) + 1e-10
    out_norm = jnp.linalg.norm(out_flat) + 1e-10

    cosine = float(jnp.dot(in_flat / in_norm, out_flat / out_norm))

    return {
        'layer': layer,
        'transpose_cosine': cosine,
        'is_approximately_transpose': abs(cosine) > 0.5,
        'W_in_norm': float(jnp.linalg.norm(W_in)),
        'W_out_norm': float(jnp.linalg.norm(W_out)),
    }


def cross_head_symmetry(model: HookedTransformer, layer: int) -> dict:
    """Are attention heads symmetric or specialized?"""
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    # Compare QK circuits
    qk_dirs = []
    ov_dirs = []
    for head in range(n_heads):
        W_Q = model.blocks[layer].attn.W_Q[head]
        W_K = model.blocks[layer].attn.W_K[head]
        W_V = model.blocks[layer].attn.W_V[head]
        W_O = model.blocks[layer].attn.W_O[head]

        qk = (W_Q @ W_K.T).reshape(-1)
        ov = (W_V @ W_O).reshape(-1)
        qk_dirs.append(qk / (jnp.linalg.norm(qk) + 1e-10))
        ov_dirs.append(ov / (jnp.linalg.norm(ov) + 1e-10))

    qk_dirs = jnp.stack(qk_dirs)
    ov_dirs = jnp.stack(ov_dirs)
    qk_sims = qk_dirs @ qk_dirs.T
    ov_sims = ov_dirs @ ov_dirs.T

    pairs = []
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            pairs.append({
                'head_a': i,
                'head_b': j,
                'qk_similarity': float(qk_sims[i, j]),
                'ov_similarity': float(ov_sims[i, j]),
            })

    mask = 1 - jnp.eye(n_heads)
    mean_qk = float(jnp.sum(qk_sims * mask) / (n_heads * (n_heads - 1) + 1e-10))
    mean_ov = float(jnp.sum(ov_sims * mask) / (n_heads * (n_heads - 1) + 1e-10))

    return {
        'layer': layer,
        'mean_qk_similarity': mean_qk,
        'mean_ov_similarity': mean_ov,
        'heads_are_diverse': mean_qk < 0.3 and mean_ov < 0.3,
        'pairs': pairs,
    }


def embed_unembed_symmetry(model: HookedTransformer) -> dict:
    """How symmetric is the relationship between embedding and unembedding?"""
    W_E = model.embed.W_E  # [d_vocab, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    # Compare W_E^T with W_U
    e_flat = W_E.T.reshape(-1)
    u_flat = W_U.reshape(-1)

    e_norm = jnp.linalg.norm(e_flat) + 1e-10
    u_norm = jnp.linalg.norm(u_flat) + 1e-10

    cosine = float(jnp.dot(e_flat / e_norm, u_flat / u_norm))

    # Per-token alignment
    d_model = W_E.shape[1]
    d_vocab = W_E.shape[0]
    n_sample = min(20, d_vocab)

    token_cosines = []
    for i in range(n_sample):
        e_dir = W_E[i] / (jnp.linalg.norm(W_E[i]) + 1e-10)
        u_dir = W_U[:, i] / (jnp.linalg.norm(W_U[:, i]) + 1e-10)
        token_cosines.append(float(jnp.dot(e_dir, u_dir)))

    mean_token_cos = sum(token_cosines) / len(token_cosines)

    return {
        'global_cosine': cosine,
        'mean_per_token_cosine': mean_token_cos,
        'is_weight_tied': cosine > 0.9,
        'W_E_norm': float(jnp.linalg.norm(W_E)),
        'W_U_norm': float(jnp.linalg.norm(W_U)),
    }
