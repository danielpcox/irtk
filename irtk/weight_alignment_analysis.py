"""Weight alignment analysis: how well do different weight matrices align."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def qk_ov_alignment(model: HookedTransformer, layer: int, head: int) -> dict:
    """How aligned are the QK and OV circuits for a given head?

    High alignment suggests the head looks at what it copies.
    """
    W_Q = model.blocks[layer].attn.W_Q[head]  # [d_model, d_head]
    W_K = model.blocks[layer].attn.W_K[head]  # [d_model, d_head]
    W_V = model.blocks[layer].attn.W_V[head]  # [d_model, d_head]
    W_O = model.blocks[layer].attn.W_O[head]  # [d_head, d_model]

    QK = W_Q @ W_K.T  # [d_model, d_model]
    OV = W_V @ W_O    # [d_model, d_model]

    # Flatten and compute cosine
    qk_flat = QK.reshape(-1)
    ov_flat = OV.reshape(-1)
    cos = float(jnp.dot(qk_flat, ov_flat) / (jnp.linalg.norm(qk_flat) * jnp.linalg.norm(ov_flat) + 1e-10))

    # Spectral analysis
    qk_sv = jnp.linalg.svd(QK, compute_uv=False)
    ov_sv = jnp.linalg.svd(OV, compute_uv=False)

    return {
        'layer': layer,
        'head': head,
        'qk_ov_cosine': cos,
        'qk_spectral_norm': float(qk_sv[0]),
        'ov_spectral_norm': float(ov_sv[0]),
        'is_aligned': abs(cos) > 0.3,
    }


def cross_head_weight_alignment(model: HookedTransformer, layer: int) -> dict:
    """How aligned are the weight matrices across heads within a layer?

    Measures pairwise OV circuit similarity.
    """
    n_heads = model.cfg.n_heads
    W_V = model.blocks[layer].attn.W_V  # [n_heads, d_model, d_head]
    W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

    ov_circuits = []
    for h in range(n_heads):
        OV = (W_V[h] @ W_O[h]).reshape(-1)
        OV = OV / (jnp.linalg.norm(OV) + 1e-10)
        ov_circuits.append(OV)

    ov_circuits = jnp.stack(ov_circuits)
    sim_matrix = ov_circuits @ ov_circuits.T

    pairs = []
    total_sim = 0.0
    n_pairs = 0
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            s = float(sim_matrix[i, j])
            pairs.append({'head_a': i, 'head_b': j, 'ov_similarity': s})
            total_sim += abs(s)
            n_pairs += 1

    mean_abs_sim = total_sim / n_pairs if n_pairs > 0 else 0.0

    return {
        'layer': layer,
        'pairs': pairs,
        'mean_abs_similarity': mean_abs_sim,
        'is_diverse': mean_abs_sim < 0.3,
    }


def cross_layer_weight_alignment(model: HookedTransformer, component: str = 'ov') -> dict:
    """How aligned are weight circuits across layers?

    Measures whether layers do similar or different things.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    layer_circuits = []
    for layer in range(n_layers):
        if component == 'ov':
            W_V = model.blocks[layer].attn.W_V
            W_O = model.blocks[layer].attn.W_O
            combined = jnp.mean(jnp.stack([
                (W_V[h] @ W_O[h]).reshape(-1) for h in range(n_heads)
            ]), axis=0)
        else:  # qk
            W_Q = model.blocks[layer].attn.W_Q
            W_K = model.blocks[layer].attn.W_K
            combined = jnp.mean(jnp.stack([
                (W_Q[h] @ W_K[h].T).reshape(-1) for h in range(n_heads)
            ]), axis=0)

        combined = combined / (jnp.linalg.norm(combined) + 1e-10)
        layer_circuits.append(combined)

    layer_circuits = jnp.stack(layer_circuits)
    sim_matrix = layer_circuits @ layer_circuits.T

    pairs = []
    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            pairs.append({
                'layer_a': i,
                'layer_b': j,
                'similarity': float(sim_matrix[i, j]),
            })

    return {
        'component': component,
        'pairs': pairs,
        'mean_similarity': sum(abs(p['similarity']) for p in pairs) / len(pairs) if pairs else 0,
    }


def embed_weight_alignment(model: HookedTransformer) -> dict:
    """How aligned are the embedding and unembedding weight matrices?

    In tied-weight models this would be perfect; measures actual alignment.
    """
    W_E = model.embed.W_E  # [d_vocab, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    # Normalize rows/columns
    W_E_normed = W_E / (jnp.linalg.norm(W_E, axis=-1, keepdims=True) + 1e-10)
    W_U_normed = W_U / (jnp.linalg.norm(W_U, axis=0, keepdims=True) + 1e-10)

    # Per-token alignment
    per_token_cos = jnp.sum(W_E_normed * W_U_normed.T, axis=-1)  # [d_vocab]
    mean_cos = float(jnp.mean(per_token_cos))
    std_cos = float(jnp.std(per_token_cos))

    # SVD alignment
    E_sv = jnp.linalg.svd(W_E, compute_uv=False)
    U_sv = jnp.linalg.svd(W_U, compute_uv=False)

    return {
        'mean_token_alignment': mean_cos,
        'std_token_alignment': std_cos,
        'embed_spectral_norm': float(E_sv[0]),
        'unembed_spectral_norm': float(U_sv[0]),
        'is_tied': mean_cos > 0.9,
    }


def mlp_weight_alignment(model: HookedTransformer) -> dict:
    """How aligned are MLP weight matrices across layers?

    Measures whether MLPs do similar transformations.
    """
    n_layers = model.cfg.n_layers

    w_in_dirs = []
    w_out_dirs = []
    for layer in range(n_layers):
        W_in = model.blocks[layer].mlp.W_in  # [d_model, d_mlp]
        W_out = model.blocks[layer].mlp.W_out  # [d_mlp, d_model]
        w_in_dirs.append(W_in.reshape(-1) / (jnp.linalg.norm(W_in) + 1e-10))
        w_out_dirs.append(W_out.reshape(-1) / (jnp.linalg.norm(W_out) + 1e-10))

    w_in_dirs = jnp.stack(w_in_dirs)
    w_out_dirs = jnp.stack(w_out_dirs)
    in_sims = w_in_dirs @ w_in_dirs.T
    out_sims = w_out_dirs @ w_out_dirs.T

    per_layer = []
    for i in range(n_layers):
        in_mean = float(jnp.mean(jnp.abs(in_sims[i])) - 1.0 / n_layers)
        out_mean = float(jnp.mean(jnp.abs(out_sims[i])) - 1.0 / n_layers)
        per_layer.append({
            'layer': i,
            'mean_in_similarity': in_mean,
            'mean_out_similarity': out_mean,
        })

    return {
        'per_layer': per_layer,
    }
