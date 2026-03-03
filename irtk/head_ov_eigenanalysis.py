"""Head OV eigenanalysis: eigenvalue structure of OV circuits."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def ov_eigenspectrum(model: HookedTransformer, layer: int = 0, head: int = 0) -> dict:
    """Singular value spectrum of OV circuit (W_V @ W_O).

    Reveals the effective rank and dominant modes of the OV circuit.
    """
    W_V = model.blocks[layer].attn.W_V[head]  # [d_model, d_head]
    W_O = model.blocks[layer].attn.W_O[head]  # [d_head, d_model]
    OV = W_V @ W_O  # [d_model, d_model]

    svs = jnp.linalg.svd(OV, compute_uv=False)
    svs_norm = svs / jnp.sum(svs).clip(1e-8)
    eff_rank = float(jnp.exp(-jnp.sum(svs_norm * jnp.log(svs_norm.clip(1e-10)))))

    return {
        "layer": layer,
        "head": head,
        "singular_values": [float(s) for s in svs[:10]],
        "effective_rank": eff_rank,
        "top_sv": float(svs[0]),
        "sv_concentration": float(svs[0] / jnp.sum(svs).clip(1e-8)),
    }


def ov_copying_score(model: HookedTransformer, layer: int = 0, head: int = 0) -> dict:
    """How much does the OV circuit copy vs transform?

    A copying OV circuit has OV ≈ identity (high trace relative to norm).
    """
    W_V = model.blocks[layer].attn.W_V[head]
    W_O = model.blocks[layer].attn.W_O[head]
    OV = W_V @ W_O  # [d_model, d_model]

    trace = float(jnp.trace(OV))
    frob_norm = float(jnp.sqrt(jnp.sum(OV ** 2)))
    d = OV.shape[0]
    identity_score = trace / (frob_norm * jnp.sqrt(jnp.array(d, dtype=jnp.float32)).clip(1e-8))

    return {
        "layer": layer,
        "head": head,
        "trace": trace,
        "frobenius_norm": frob_norm,
        "identity_score": float(identity_score),
        "is_copying": float(identity_score) > 0.3,
    }


def ov_unembed_projection(model: HookedTransformer, layer: int = 0, head: int = 0,
                              top_k: int = 5) -> dict:
    """Project OV circuit through unembedding to find promoted tokens."""
    W_V = model.blocks[layer].attn.W_V[head]
    W_O = model.blocks[layer].attn.W_O[head]
    W_U = model.unembed.W_U  # [d_model, d_vocab]
    OV = W_V @ W_O  # [d_model, d_model]

    # OV effect on each vocab direction
    OV_U = OV @ W_U  # [d_model, d_vocab]
    # Mean effect magnitude per vocab item
    effect = jnp.mean(jnp.abs(OV_U), axis=0)  # [d_vocab]

    top_indices = jnp.argsort(-effect)[:top_k]
    top_tokens = [(int(idx), float(effect[idx])) for idx in top_indices]

    return {
        "layer": layer,
        "head": head,
        "top_affected_tokens": top_tokens,
        "mean_effect": float(jnp.mean(effect)),
        "max_effect": float(jnp.max(effect)),
    }


def ov_cross_head_alignment(model: HookedTransformer, layer: int = 0) -> dict:
    """Alignment between OV circuits of different heads in the same layer."""
    n_heads = model.cfg.n_heads
    ov_matrices = []
    for head in range(n_heads):
        W_V = model.blocks[layer].attn.W_V[head]
        W_O = model.blocks[layer].attn.W_O[head]
        ov = (W_V @ W_O).reshape(-1)
        norm = jnp.sqrt(jnp.sum(ov ** 2)).clip(1e-8)
        ov_matrices.append(ov / norm)

    alignment = []
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            cos = float(jnp.sum(ov_matrices[i] * ov_matrices[j]))
            alignment.append({
                "heads": (i, j),
                "cosine": cos,
                "is_aligned": abs(cos) > 0.5,
            })

    return {
        "layer": layer,
        "pairs": alignment,
        "n_aligned": sum(1 for a in alignment if a["is_aligned"]),
    }


def ov_eigenanalysis_summary(model: HookedTransformer) -> dict:
    """Cross-layer OV eigenanalysis summary."""
    per_layer = []
    for layer in range(model.cfg.n_layers):
        ranks = []
        copy_scores = []
        for head in range(model.cfg.n_heads):
            spec = ov_eigenspectrum(model, layer, head)
            copy = ov_copying_score(model, layer, head)
            ranks.append(spec["effective_rank"])
            copy_scores.append(copy["identity_score"])
        per_layer.append({
            "layer": layer,
            "mean_rank": sum(ranks) / len(ranks),
            "mean_copy_score": sum(copy_scores) / len(copy_scores),
        })
    return {"per_layer": per_layer}
