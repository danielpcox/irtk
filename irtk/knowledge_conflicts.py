"""Knowledge conflict detection: when facts compete for model capacity.

Tools for detecting and analyzing interference between competing
knowledge representations:
- Conflicting logit directions
- Residual stream tug-of-war between components
- Competing attention patterns
- Knowledge interference localization
- Resolution mechanism identification
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def logit_direction_conflicts(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Find components that push toward conflicting predictions.

    Decomposes the logit into per-component contributions and identifies
    cases where different components promote different tokens.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.
        top_k: Number of top conflicts to return.

    Returns:
        Dict with conflicting component pairs and their target tokens.
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U  # [d_model, d_vocab]

    # Collect all component outputs and their logit contributions
    components = []

    # Embedding
    embed = cache['blocks.0.hook_resid_pre']
    components.append(('embed+pos', np.array(embed[pos])))

    for l in range(model.cfg.n_layers):
        attn_out = cache[f'blocks.{l}.hook_attn_out']
        mlp_out = cache[f'blocks.{l}.hook_mlp_out']
        components.append((f'L{l}_attn', np.array(attn_out[pos])))
        components.append((f'L{l}_mlp', np.array(mlp_out[pos])))

    W_U_np = np.array(W_U)

    # For each component, find which token it most promotes
    comp_preferences = []
    for name, output in components:
        logit_contribs = output @ W_U_np  # [d_vocab]
        top_token = int(np.argmax(logit_contribs))
        top_logit = float(logit_contribs[top_token])
        comp_preferences.append({
            'name': name,
            'top_token': top_token,
            'top_logit': round(top_logit, 4),
            'logit_contribs': logit_contribs,
        })

    # Find conflicts: pairs of components that promote different tokens
    conflicts = []
    for i in range(len(comp_preferences)):
        for j in range(i + 1, len(comp_preferences)):
            ci = comp_preferences[i]
            cj = comp_preferences[j]
            if ci['top_token'] != cj['top_token']:
                # Measure conflict strength
                strength = min(abs(ci['top_logit']), abs(cj['top_logit']))
                conflicts.append({
                    'component_a': ci['name'],
                    'promotes_a': ci['top_token'],
                    'logit_a': ci['top_logit'],
                    'component_b': cj['name'],
                    'promotes_b': cj['top_token'],
                    'logit_b': cj['top_logit'],
                    'conflict_strength': round(strength, 4),
                })

    conflicts.sort(key=lambda x: -x['conflict_strength'])

    return {
        'n_conflicts': len(conflicts),
        'top_conflicts': conflicts[:top_k],
        'component_preferences': [
            {'name': c['name'], 'top_token': c['top_token'], 'top_logit': c['top_logit']}
            for c in comp_preferences
        ],
    }


def residual_tug_of_war(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_token: int,
    pos: int = -1,
) -> dict:
    """Analyze which components push toward/away from a target token.

    Computes the dot product of each component's output with the target
    token's unembedding direction, showing the tug-of-war.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        target_token: Token ID to analyze.
        pos: Position to analyze.

    Returns:
        Dict with per-component contributions (positive = promotes, negative = suppresses).
    """
    _, cache = model.run_with_cache(tokens)
    W_U = model.unembed.W_U
    target_dir = W_U[:, target_token]  # [d_model]

    results = []

    # Embedding contribution
    embed = cache['blocks.0.hook_resid_pre'][pos]
    contrib = float(jnp.dot(embed, target_dir))
    results.append({'component': 'embed+pos', 'contribution': round(contrib, 4)})

    for l in range(model.cfg.n_layers):
        attn_out = cache[f'blocks.{l}.hook_attn_out'][pos]
        mlp_out = cache[f'blocks.{l}.hook_mlp_out'][pos]

        attn_contrib = float(jnp.dot(attn_out, target_dir))
        mlp_contrib = float(jnp.dot(mlp_out, target_dir))

        results.append({'component': f'L{l}_attn', 'contribution': round(attn_contrib, 4)})
        results.append({'component': f'L{l}_mlp', 'contribution': round(mlp_contrib, 4)})

    promoters = [r for r in results if r['contribution'] > 0]
    suppressors = [r for r in results if r['contribution'] < 0]
    promoters.sort(key=lambda x: -x['contribution'])
    suppressors.sort(key=lambda x: x['contribution'])

    total_promote = sum(r['contribution'] for r in promoters)
    total_suppress = sum(r['contribution'] for r in suppressors)

    return {
        'target_token': target_token,
        'promoters': promoters,
        'suppressors': suppressors,
        'total_promotion': round(total_promote, 4),
        'total_suppression': round(total_suppress, 4),
        'net_logit': round(total_promote + total_suppress, 4),
        'all_components': results,
    }


def attention_competition(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    query_pos: int = -1,
    top_k: int = 3,
) -> dict:
    """Analyze competition between positions for attention.

    Identifies cases where multiple positions compete for a head's
    attention at a given query position.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        query_pos: Query position to analyze.
        top_k: Number of competing pairs to return per head.

    Returns:
        Dict with per-head competition analysis.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)

    per_head = []
    for l in range(model.cfg.n_layers):
        pattern = cache[f'blocks.{l}.attn.hook_pattern']  # [n_heads, seq, seq]

        for h in range(model.cfg.n_heads):
            attn_row = np.array(pattern[h, query_pos, :])  # [seq]

            # Sort positions by attention weight
            sorted_pos = np.argsort(attn_row)[::-1]
            top1_weight = float(attn_row[sorted_pos[0]])
            top2_weight = float(attn_row[sorted_pos[1]]) if len(sorted_pos) > 1 else 0.0

            # Competition: ratio of 2nd to 1st
            competition = top2_weight / top1_weight if top1_weight > 1e-10 else 0.0

            # Entropy of attention distribution
            attn_probs = attn_row + 1e-10
            entropy = -float(np.sum(attn_probs * np.log(attn_probs)))

            per_head.append({
                'layer': l,
                'head': h,
                'top_attended': int(sorted_pos[0]),
                'top_weight': round(top1_weight, 4),
                'runner_up': int(sorted_pos[1]) if len(sorted_pos) > 1 else -1,
                'runner_up_weight': round(top2_weight, 4),
                'competition_ratio': round(competition, 4),
                'attention_entropy': round(entropy, 4),
            })

    # Sort by competition (highest competition = most conflicted)
    per_head.sort(key=lambda x: -x['competition_ratio'])

    return {
        'query_position': query_pos,
        'per_head': per_head,
        'most_competitive': per_head[0] if per_head else None,
        'least_competitive': per_head[-1] if per_head else None,
    }


def interference_localization(
    model: HookedTransformer,
    tokens_a: jnp.ndarray,
    tokens_b: jnp.ndarray,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Localize where two inputs' representations interfere.

    Compares activations from two inputs at each component to find
    where their representations diverge most and least.

    Args:
        model: HookedTransformer.
        tokens_a: First token sequence.
        tokens_b: Second token sequence.
        pos: Position to compare.
        top_k: Number of top divergent components.

    Returns:
        Dict with per-component divergence and interference patterns.
    """
    _, cache_a = model.run_with_cache(tokens_a)
    _, cache_b = model.run_with_cache(tokens_b)

    components = []

    for l in range(model.cfg.n_layers):
        for hook_suffix in ['hook_attn_out', 'hook_mlp_out']:
            hook_name = f'blocks.{l}.{hook_suffix}'
            act_a = cache_a[hook_name][pos]
            act_b = cache_b[hook_name][pos]

            diff = act_a - act_b
            diff_norm = float(jnp.linalg.norm(diff))
            norm_a = float(jnp.linalg.norm(act_a))
            norm_b = float(jnp.linalg.norm(act_b))

            # Cosine similarity
            if norm_a > 1e-10 and norm_b > 1e-10:
                cos_sim = float(jnp.dot(act_a, act_b) / (norm_a * norm_b))
            else:
                cos_sim = 0.0

            name = f'L{l}_{"attn" if "attn" in hook_suffix else "mlp"}'
            components.append({
                'component': name,
                'divergence': round(diff_norm, 4),
                'cosine_similarity': round(cos_sim, 4),
                'norm_a': round(norm_a, 4),
                'norm_b': round(norm_b, 4),
            })

    # Sort by divergence (most different first)
    components.sort(key=lambda x: -x['divergence'])

    # Also track residual stream divergence
    resid_divergence = []
    for l in range(model.cfg.n_layers):
        resid_a = cache_a[f'blocks.{l}.hook_resid_post'][pos]
        resid_b = cache_b[f'blocks.{l}.hook_resid_post'][pos]
        d = float(jnp.linalg.norm(resid_a - resid_b))
        resid_divergence.append({'layer': l, 'divergence': round(d, 4)})

    return {
        'per_component': components,
        'top_divergent': components[:top_k],
        'residual_divergence': resid_divergence,
        'max_divergence_component': components[0]['component'] if components else None,
    }
