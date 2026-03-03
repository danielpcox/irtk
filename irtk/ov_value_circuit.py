"""OV value circuit analysis: what does each head copy/write through its OV matrix?"""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def ov_eigenspectrum(model: HookedTransformer, layer: int, head: int) -> dict:
    """Eigenspectrum of the OV matrix W_V @ W_O.

    Top eigenvalues reveal what the head copies most strongly.
    """
    W_V = model.blocks[layer].attn.W_V[head]  # [d_model, d_head]
    W_O = model.blocks[layer].attn.W_O[head]  # [d_head, d_model]
    OV = W_V @ W_O  # [d_model, d_model]

    sv = jnp.linalg.svd(OV, compute_uv=False)
    total = jnp.sum(sv)
    sv_norm = sv / (total + 1e-10)
    entropy = -float(jnp.sum(sv_norm * jnp.log(sv_norm + 1e-10)))
    eff_rank = float(jnp.exp(entropy))

    top_5 = [{'index': i, 'value': float(sv[i]), 'fraction': float(sv[i] / (total + 1e-10))} for i in range(min(5, len(sv)))]

    return {
        'layer': layer,
        'head': head,
        'effective_rank': eff_rank,
        'spectral_norm': float(sv[0]),
        'frobenius_norm': float(jnp.linalg.norm(OV)),
        'top_singular_values': top_5,
    }


def ov_token_copying_score(model: HookedTransformer, layer: int, head: int, token_ids: list = None) -> dict:
    """How well does the OV circuit copy token identity?

    Measures whether W_E @ OV @ W_U has high diagonal entries.
    """
    W_V = model.blocks[layer].attn.W_V[head]
    W_O = model.blocks[layer].attn.W_O[head]
    W_E = model.embed.W_E  # [d_vocab, d_model]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    OV = W_V @ W_O  # [d_model, d_model]

    if token_ids is None:
        token_ids = list(range(min(20, W_E.shape[0])))

    import numpy as np
    token_ids_np = np.array(token_ids)
    E_subset = W_E[token_ids_np]  # [n_tokens, d_model]

    # E @ OV @ U
    transformed = E_subset @ OV @ W_U  # [n_tokens, d_vocab]

    per_token = []
    for i, tid in enumerate(token_ids):
        logits = transformed[i]
        self_logit = float(logits[tid])
        rank = int(jnp.sum(logits > logits[tid]))
        per_token.append({
            'token': tid,
            'self_logit': self_logit,
            'self_rank': rank,
            'is_copying': bool(rank < 5),
        })

    mean_rank = sum(t['self_rank'] for t in per_token) / len(per_token)

    return {
        'layer': layer,
        'head': head,
        'per_token': per_token,
        'mean_self_rank': mean_rank,
        'is_copy_head': bool(mean_rank < 10),
    }


def ov_writing_direction(model: HookedTransformer, tokens: jnp.ndarray, layer: int, head: int) -> dict:
    """What direction does the OV circuit write into the residual stream?

    Analyzes the output of value @ W_O for each source position.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]

    v = cache[f'blocks.{layer}.attn.hook_v']  # [seq, n_heads, d_head]
    W_O = model.blocks[layer].attn.W_O[head]  # [d_head, d_model]

    outputs = []
    for s in range(seq_len):
        v_s = v[s, head, :]  # [d_head]
        out = v_s @ W_O  # [d_model]
        outputs.append(out)

    outputs_stack = jnp.stack(outputs)  # [seq, d_model]
    mean_dir = jnp.mean(outputs_stack, axis=0)
    mean_dir = mean_dir / (jnp.linalg.norm(mean_dir) + 1e-10)

    per_source = []
    for s in range(seq_len):
        out = outputs[s]
        norm = float(jnp.linalg.norm(out))
        cos = float(jnp.dot(out, mean_dir) / (norm + 1e-10))
        per_source.append({
            'source': s,
            'token': int(tokens[s]),
            'output_norm': norm,
            'cosine_with_mean': cos,
        })

    return {
        'layer': layer,
        'head': head,
        'per_source': per_source,
        'mean_output_norm': sum(p['output_norm'] for p in per_source) / len(per_source),
    }


def ov_composition_with_next_layer(model: HookedTransformer, layer: int, head: int) -> dict:
    """How does this head's OV output compose with the next layer's QK?

    Measures virtual attention via OV → QK composition.
    """
    n_layers = model.cfg.n_layers
    if layer >= n_layers - 1:
        return {'layer': layer, 'head': head, 'compositions': [], 'error': 'last_layer'}

    W_V = model.blocks[layer].attn.W_V[head]
    W_O = model.blocks[layer].attn.W_O[head]
    OV = W_V @ W_O  # [d_model, d_model]

    next_layer = layer + 1
    n_heads = model.cfg.n_heads

    compositions = []
    for h2 in range(n_heads):
        W_Q = model.blocks[next_layer].attn.W_Q[h2]  # [d_model, d_head]
        W_K = model.blocks[next_layer].attn.W_K[h2]

        # OV → Q path: OV @ W_Q
        ov_q = OV @ W_Q  # [d_model, d_head]
        ov_q_norm = float(jnp.linalg.norm(ov_q))

        # OV → K path: OV @ W_K
        ov_k = OV @ W_K  # [d_model, d_head]
        ov_k_norm = float(jnp.linalg.norm(ov_k))

        compositions.append({
            'next_head': h2,
            'ov_to_q_norm': ov_q_norm,
            'ov_to_k_norm': ov_k_norm,
            'total_composition': ov_q_norm + ov_k_norm,
        })

    compositions.sort(key=lambda x: x['total_composition'], reverse=True)

    return {
        'layer': layer,
        'head': head,
        'next_layer': next_layer,
        'compositions': compositions,
    }


def ov_unembed_alignment(model: HookedTransformer, layer: int, head: int, top_k: int = 10) -> dict:
    """Which vocabulary tokens does the OV circuit most strongly promote/suppress?

    Analyzes the top eigenvectors of OV projected through unembedding.
    """
    W_V = model.blocks[layer].attn.W_V[head]
    W_O = model.blocks[layer].attn.W_O[head]
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    OV = W_V @ W_O  # [d_model, d_model]
    U, S, Vt = jnp.linalg.svd(OV, full_matrices=False)

    # Project top singular vectors through unembedding
    top_vector = U[:, 0]  # [d_model]
    logits = top_vector @ W_U  # [d_vocab]

    promoted = jnp.argsort(logits)[-top_k:][::-1]
    suppressed = jnp.argsort(logits)[:top_k]

    return {
        'layer': layer,
        'head': head,
        'top_sv': float(S[0]),
        'promoted_tokens': [{'token': int(t), 'logit': float(logits[t])} for t in promoted],
        'suppressed_tokens': [{'token': int(t), 'logit': float(logits[t])} for t in suppressed],
    }
