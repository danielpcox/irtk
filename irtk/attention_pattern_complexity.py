"""Attention pattern complexity: entropy, rank, regularity metrics."""

import jax.numpy as jnp
from irtk.hooked_transformer import HookedTransformer


def pattern_entropy_complexity(model: HookedTransformer, tokens: jnp.ndarray,
                                  layer: int = 0) -> dict:
    """Entropy-based complexity of attention patterns per head.

    High entropy = complex/diffuse; low entropy = simple/focused.
    """
    _, cache = model.run_with_cache(tokens)
    pattern = cache[("pattern", layer)]  # [n_heads, seq, seq]
    seq_len = pattern.shape[1]

    per_head = []
    for head in range(model.cfg.n_heads):
        p = pattern[head]  # [seq, seq]
        entropies = []
        for q in range(seq_len):
            row = p[q, :q + 1]
            if row.shape[0] > 0:
                ent = float(-jnp.sum(row * jnp.log(row.clip(1e-10))))
                entropies.append(ent)
        mean_ent = sum(entropies) / max(len(entropies), 1)
        max_ent = float(jnp.log(jnp.array(seq_len, dtype=jnp.float32)))
        per_head.append({
            "head": int(head),
            "mean_entropy": mean_ent,
            "normalized_entropy": mean_ent / max(max_ent, 1e-8),
            "is_complex": mean_ent / max(max_ent, 1e-8) > 0.5,
        })
    return {
        "layer": layer,
        "per_head": per_head,
        "n_complex_heads": sum(1 for h in per_head if h["is_complex"]),
    }


def pattern_rank_complexity(model: HookedTransformer, tokens: jnp.ndarray,
                               layer: int = 0) -> dict:
    """Rank-based complexity of attention patterns per head.

    High effective rank = complex patterns; low rank = simple/structured.
    """
    _, cache = model.run_with_cache(tokens)
    pattern = cache[("pattern", layer)]  # [n_heads, seq, seq]

    per_head = []
    for head in range(model.cfg.n_heads):
        p = pattern[head]  # [seq, seq]
        svs = jnp.linalg.svd(p, compute_uv=False)
        svs_norm = svs / jnp.sum(svs).clip(1e-8)
        eff_rank = float(jnp.exp(-jnp.sum(svs_norm * jnp.log(svs_norm.clip(1e-10)))))
        per_head.append({
            "head": int(head),
            "effective_rank": eff_rank,
            "top_sv": float(svs[0]),
            "sv_ratio": float(svs[0] / svs[-1].clip(1e-8)),
        })
    return {
        "layer": layer,
        "per_head": per_head,
        "mean_rank": sum(h["effective_rank"] for h in per_head) / len(per_head),
    }


def pattern_regularity(model: HookedTransformer, tokens: jnp.ndarray,
                          layer: int = 0, head: int = 0) -> dict:
    """Regularity metrics: how structured/predictable is the attention pattern?

    Measures diagonal dominance, banding, and symmetry.
    """
    _, cache = model.run_with_cache(tokens)
    pattern = cache[("pattern", layer)]
    p = pattern[head]  # [seq, seq]
    seq_len = p.shape[0]

    # Diagonal dominance: attention to self
    diag_mass = float(jnp.mean(jnp.diag(p)))

    # Previous-token dominance: attention to position - 1
    prev_mass = 0.0
    count = 0
    for i in range(1, seq_len):
        prev_mass += float(p[i, i - 1])
        count += 1
    prev_mass = prev_mass / max(count, 1)

    # First-token dominance
    first_mass = float(jnp.mean(p[:, 0]))

    return {
        "layer": layer,
        "head": head,
        "self_attention": diag_mass,
        "prev_token_attention": prev_mass,
        "first_token_attention": first_mass,
        "dominant_pattern": max(
            [("self", diag_mass), ("prev", prev_mass), ("first", first_mass)],
            key=lambda x: x[1]
        )[0],
    }


def pattern_stability_across_positions(model: HookedTransformer, tokens: jnp.ndarray,
                                          layer: int = 0) -> dict:
    """How stable are attention patterns across different query positions?

    High stability = similar patterns regardless of query position.
    """
    _, cache = model.run_with_cache(tokens)
    pattern = cache[("pattern", layer)]  # [n_heads, seq, seq]
    seq_len = pattern.shape[1]

    per_head = []
    for head in range(model.cfg.n_heads):
        p = pattern[head]  # [seq, seq]
        # Compare distributions from different positions (using truncated distributions)
        if seq_len < 3:
            per_head.append({"head": int(head), "stability": 1.0})
            continue

        # Use KL divergence between adjacent positions
        kls = []
        for q in range(1, seq_len):
            # Distribution at position q over keys 0..q
            pq = p[q, :q + 1]
            # Distribution at position q-1 over keys 0..q (extended with zero)
            pq_prev = jnp.concatenate([p[q - 1, :q], jnp.array([0.0])])
            pq_prev = pq_prev / jnp.sum(pq_prev).clip(1e-8)
            kl = float(jnp.sum(pq * jnp.log((pq / pq_prev.clip(1e-10)).clip(1e-10))))
            kls.append(max(kl, 0))

        mean_kl = sum(kls) / max(len(kls), 1)
        per_head.append({
            "head": int(head),
            "mean_kl": mean_kl,
            "stability": 1.0 / (1.0 + mean_kl),
        })
    return {
        "layer": layer,
        "per_head": per_head,
        "mean_stability": sum(h["stability"] for h in per_head) / len(per_head),
    }


def pattern_complexity_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Cross-layer pattern complexity summary."""
    per_layer = []
    for layer in range(model.cfg.n_layers):
        ent = pattern_entropy_complexity(model, tokens, layer)
        rank = pattern_rank_complexity(model, tokens, layer)
        per_layer.append({
            "layer": layer,
            "n_complex_heads": ent["n_complex_heads"],
            "mean_rank": rank["mean_rank"],
        })
    return {"per_layer": per_layer}
