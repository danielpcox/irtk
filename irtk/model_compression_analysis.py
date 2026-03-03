"""Model compression analysis.

Analyze model compressibility: redundant heads/layers, pruning candidates,
effective parameter counts, and compression sensitivity.
"""

import jax
import jax.numpy as jnp


def head_pruning_candidates(model, tokens, top_k=5):
    """Identify attention heads that can be pruned with minimal impact.

    Ranks heads by their output norm (low norm = low contribution).

    Args:
        model: HookedTransformer
        tokens: input token IDs
        top_k: number of pruning candidates

    Returns:
        dict with ranked pruning candidates.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    head_scores = []
    for l in range(n_layers):
        z = cache[f'blocks.{l}.attn.hook_z']  # [seq, n_heads, d_head]
        for h in range(n_heads):
            z_h = z[:, h, :]
            W_O_h = model.blocks[l].attn.W_O[h]
            output = z_h @ W_O_h  # [seq, d_model]
            output_norm = float(jnp.mean(jnp.linalg.norm(output, axis=-1)))
            head_scores.append({
                'layer': l,
                'head': h,
                'output_norm': output_norm,
            })

    head_scores.sort(key=lambda h: h['output_norm'])
    return {
        'pruning_candidates': head_scores[:top_k],
        'all_scores': head_scores,
        'mean_norm': float(jnp.mean(jnp.array([h['output_norm'] for h in head_scores]))),
    }


def layer_pruning_candidates(model, tokens):
    """Identify layers that can be skipped with minimal impact.

    Measures each layer's contribution norm.

    Args:
        model: HookedTransformer
        tokens: input token IDs

    Returns:
        dict with per-layer contribution scores.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    layer_scores = []
    for l in range(n_layers):
        resid_pre = cache[f'blocks.{l}.hook_resid_pre']
        resid_post = cache[f'blocks.{l}.hook_resid_post']
        delta = resid_post - resid_pre
        delta_norm = float(jnp.mean(jnp.linalg.norm(delta, axis=-1)))
        pre_norm = float(jnp.mean(jnp.linalg.norm(resid_pre, axis=-1)))

        layer_scores.append({
            'layer': l,
            'contribution_norm': delta_norm,
            'relative_contribution': delta_norm / max(pre_norm, 1e-10),
        })

    layer_scores.sort(key=lambda l: l['contribution_norm'])
    return {
        'per_layer': layer_scores,
        'most_skippable': layer_scores[0]['layer'] if layer_scores else 0,
        'least_skippable': layer_scores[-1]['layer'] if layer_scores else 0,
    }


def effective_parameter_count(model, threshold=0.01):
    """Estimate the effective number of parameters via weight magnitude.

    Parameters with magnitude below threshold are considered ineffective.

    Args:
        model: HookedTransformer
        threshold: magnitude threshold

    Returns:
        dict with effective parameter counts per component.
    """
    n_layers = model.cfg.n_layers

    total_params = 0
    effective_params = 0
    per_component = {}

    # Embedding
    W_E = model.embed.W_E
    n_total = int(jnp.size(W_E))
    n_eff = int(jnp.sum(jnp.abs(W_E) > threshold))
    total_params += n_total
    effective_params += n_eff
    per_component['embed'] = {'total': n_total, 'effective': n_eff, 'ratio': n_eff / max(n_total, 1)}

    # Unembed
    W_U = model.unembed.W_U
    n_total = int(jnp.size(W_U))
    n_eff = int(jnp.sum(jnp.abs(W_U) > threshold))
    total_params += n_total
    effective_params += n_eff
    per_component['unembed'] = {'total': n_total, 'effective': n_eff, 'ratio': n_eff / max(n_total, 1)}

    # Per-layer
    for l in range(n_layers):
        block = model.blocks[l]
        layer_total = 0
        layer_eff = 0
        for name in ['W_Q', 'W_K', 'W_V', 'W_O']:
            W = getattr(block.attn, name)
            n_t = int(jnp.size(W))
            n_e = int(jnp.sum(jnp.abs(W) > threshold))
            layer_total += n_t
            layer_eff += n_e

        for name in ['W_in', 'W_out']:
            W = getattr(block.mlp, name)
            n_t = int(jnp.size(W))
            n_e = int(jnp.sum(jnp.abs(W) > threshold))
            layer_total += n_t
            layer_eff += n_e

        total_params += layer_total
        effective_params += layer_eff
        per_component[f'layer_{l}'] = {
            'total': layer_total,
            'effective': layer_eff,
            'ratio': layer_eff / max(layer_total, 1),
        }

    return {
        'total_params': total_params,
        'effective_params': effective_params,
        'compression_ratio': effective_params / max(total_params, 1),
        'per_component': per_component,
    }


def weight_low_rank_compressibility(model, layer, target_rank=None):
    """Analyze how compressible weight matrices are via low-rank approximation.

    Args:
        model: HookedTransformer
        layer: layer index
        target_rank: if specified, compute error at this rank

    Returns:
        dict with per-matrix compressibility analysis.
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
        total_energy = float(jnp.sum(S ** 2))
        full_rank = min(W.shape)

        # Compute cumulative energy
        cumulative = jnp.cumsum(S ** 2) / max(total_energy, 1e-10)

        # Rank for 90%/95%/99% energy
        rank_90 = int(jnp.searchsorted(cumulative, 0.9)) + 1
        rank_95 = int(jnp.searchsorted(cumulative, 0.95)) + 1
        rank_99 = int(jnp.searchsorted(cumulative, 0.99)) + 1

        r = {
            'full_rank': full_rank,
            'rank_for_90pct': rank_90,
            'rank_for_95pct': rank_95,
            'rank_for_99pct': rank_99,
            'compression_90': rank_90 / full_rank,
            'compression_95': rank_95 / full_rank,
        }

        if target_rank is not None:
            tr = min(target_rank, len(S))
            energy_retained = float(jnp.sum(S[:tr] ** 2) / max(total_energy, 1e-10))
            r['target_rank'] = tr
            r['energy_at_target'] = energy_retained

        results[name] = r

    return {'layer': layer, 'matrices': results}


def redundancy_score(model, tokens):
    """Compute overall model redundancy score.

    Combines head redundancy, layer similarity, and weight low-rank analysis.

    Returns:
        dict with aggregate redundancy metrics.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Layer similarity (consecutive layers)
    layer_sims = []
    for l in range(n_layers - 1):
        r_curr = cache[f'blocks.{l}.hook_resid_post'].reshape(-1)
        r_next = cache[f'blocks.{l+1}.hook_resid_post'].reshape(-1)
        cos = float(jnp.sum(r_curr * r_next) /
                    jnp.maximum(jnp.linalg.norm(r_curr) * jnp.linalg.norm(r_next), 1e-10))
        layer_sims.append(cos)

    # Within-layer head similarity
    head_sims = []
    for l in range(n_layers):
        z = cache[f'blocks.{l}.attn.hook_z']
        for h1 in range(n_heads):
            for h2 in range(h1 + 1, n_heads):
                o1 = (z[:, h1, :] @ model.blocks[l].attn.W_O[h1]).reshape(-1)
                o2 = (z[:, h2, :] @ model.blocks[l].attn.W_O[h2]).reshape(-1)
                cos = float(jnp.sum(o1 * o2) /
                            jnp.maximum(jnp.linalg.norm(o1) * jnp.linalg.norm(o2), 1e-10))
                head_sims.append(cos)

    mean_layer_sim = float(jnp.mean(jnp.array(layer_sims))) if layer_sims else 0.0
    mean_head_sim = float(jnp.mean(jnp.array(head_sims))) if head_sims else 0.0

    return {
        'layer_similarity': mean_layer_sim,
        'head_similarity': mean_head_sim,
        'redundancy_score': (mean_layer_sim + mean_head_sim) / 2,
        'n_layer_pairs': len(layer_sims),
        'n_head_pairs': len(head_sims),
    }
