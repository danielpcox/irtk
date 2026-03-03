"""Attention head interaction analysis.

Tools for understanding how attention heads interact:
- Within-layer head cooperation/competition
- Cross-layer head dependencies
- Head output alignment
- Attention pattern overlap
- Head pair importance
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def within_layer_interaction(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Measure how heads within each layer cooperate or compete.

    Computes pairwise cosine similarity of head output vectors.
    Positive = cooperative, negative = opposing.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.

    Returns:
        Dict with within-layer interaction matrices.
    """
    _, cache = model.run_with_cache(tokens)

    per_layer = []
    for l in range(model.cfg.n_layers):
        z = np.array(cache[f'blocks.{l}.attn.hook_z'][pos])  # [n_heads, d_head]
        W_O = np.array(model.blocks[l].attn.W_O)  # [n_heads, d_head, d_model]

        # Per-head output in d_model space
        head_outputs = []
        for h in range(model.cfg.n_heads):
            out = z[h] @ W_O[h]  # [d_model]
            head_outputs.append(out)

        # Pairwise cosine similarity
        n = len(head_outputs)
        interactions = []
        for i in range(n):
            for j in range(i + 1, n):
                ni = float(np.linalg.norm(head_outputs[i]))
                nj = float(np.linalg.norm(head_outputs[j]))
                if ni > 1e-10 and nj > 1e-10:
                    cos = float(np.dot(head_outputs[i], head_outputs[j]) / (ni * nj))
                else:
                    cos = 0.0
                interactions.append({
                    'head_a': i,
                    'head_b': j,
                    'cosine_similarity': round(cos, 4),
                    'type': 'cooperative' if cos > 0.3 else ('opposing' if cos < -0.3 else 'independent'),
                })

        per_layer.append({
            'layer': l,
            'interactions': interactions,
            'n_cooperative': sum(1 for i in interactions if i['type'] == 'cooperative'),
            'n_opposing': sum(1 for i in interactions if i['type'] == 'opposing'),
            'n_independent': sum(1 for i in interactions if i['type'] == 'independent'),
        })

    return {'per_layer': per_layer}


def cross_layer_alignment(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Measure alignment between heads in different layers.

    Finds which heads in later layers produce outputs aligned with
    earlier heads — potential circuit connections.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.

    Returns:
        Dict with cross-layer head alignment scores.
    """
    _, cache = model.run_with_cache(tokens)

    # Collect per-head output vectors
    head_outputs = {}
    for l in range(model.cfg.n_layers):
        z = np.array(cache[f'blocks.{l}.attn.hook_z'][pos])
        W_O = np.array(model.blocks[l].attn.W_O)
        for h in range(model.cfg.n_heads):
            out = z[h] @ W_O[h]
            head_outputs[(l, h)] = out

    # Cross-layer pairs
    aligned_pairs = []
    for l1 in range(model.cfg.n_layers):
        for l2 in range(l1 + 1, model.cfg.n_layers):
            for h1 in range(model.cfg.n_heads):
                for h2 in range(model.cfg.n_heads):
                    v1 = head_outputs[(l1, h1)]
                    v2 = head_outputs[(l2, h2)]
                    n1 = float(np.linalg.norm(v1))
                    n2 = float(np.linalg.norm(v2))
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos = float(np.dot(v1, v2) / (n1 * n2))
                    else:
                        cos = 0.0

                    if abs(cos) > 0.5:
                        aligned_pairs.append({
                            'head_a': f'L{l1}H{h1}',
                            'head_b': f'L{l2}H{h2}',
                            'cosine_similarity': round(cos, 4),
                            'relationship': 'reinforcing' if cos > 0 else 'canceling',
                        })

    aligned_pairs.sort(key=lambda x: -abs(x['cosine_similarity']))
    return {
        'aligned_pairs': aligned_pairs,
        'n_aligned': len(aligned_pairs),
        'n_reinforcing': sum(1 for p in aligned_pairs if p['relationship'] == 'reinforcing'),
        'n_canceling': sum(1 for p in aligned_pairs if p['relationship'] == 'canceling'),
    }


def attention_pattern_overlap(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layers: Optional[list[int]] = None,
) -> dict:
    """Measure overlap between attention patterns of different heads.

    High overlap means heads attend to similar positions.
    Low overlap means heads attend to complementary positions.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        layers: Layers to analyze (default: all).

    Returns:
        Dict with pattern overlap scores.
    """
    _, cache = model.run_with_cache(tokens)
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    per_layer = []
    for l in layers:
        pattern = np.array(cache[f'blocks.{l}.attn.hook_pattern'])  # [n_heads, seq, seq]

        # Flatten each head's pattern and compute overlap
        flat = pattern.reshape(model.cfg.n_heads, -1)
        overlaps = []
        for i in range(model.cfg.n_heads):
            for j in range(i + 1, model.cfg.n_heads):
                # Jensen-Shannon style: overlap via minimum
                overlap = float(np.sum(np.minimum(flat[i], flat[j])))
                overlaps.append({
                    'head_a': i,
                    'head_b': j,
                    'overlap': round(overlap / len(tokens), 4),
                })

        mean_overlap = float(np.mean([o['overlap'] for o in overlaps])) if overlaps else 0.0
        per_layer.append({
            'layer': l,
            'head_overlaps': overlaps,
            'mean_overlap': round(mean_overlap, 4),
        })

    return {
        'per_layer': per_layer,
        'most_diverse_layer': min(per_layer, key=lambda x: x['mean_overlap'])['layer'] if per_layer else 0,
    }


def head_output_norms(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Measure the output norm of each head to identify dominant vs quiet heads.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.

    Returns:
        Dict with per-head output norms.
    """
    _, cache = model.run_with_cache(tokens)

    per_head = []
    max_norm = 0.0

    for l in range(model.cfg.n_layers):
        z = np.array(cache[f'blocks.{l}.attn.hook_z'][pos])  # [n_heads, d_head]
        W_O = np.array(model.blocks[l].attn.W_O)  # [n_heads, d_head, d_model]

        for h in range(model.cfg.n_heads):
            out = z[h] @ W_O[h]
            norm = float(np.linalg.norm(out))
            max_norm = max(max_norm, norm)
            per_head.append({
                'layer': l,
                'head': h,
                'output_norm': round(norm, 4),
            })

    # Add relative importance
    for h in per_head:
        h['relative_norm'] = round(h['output_norm'] / max_norm, 4) if max_norm > 0 else 0.0

    per_head.sort(key=lambda x: -x['output_norm'])
    return {
        'per_head': per_head,
        'max_norm': round(max_norm, 4),
        'dominant_head': f"L{per_head[0]['layer']}H{per_head[0]['head']}" if per_head else None,
    }


def head_pair_importance(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
    target_token: Optional[int] = None,
) -> dict:
    """Find pairs of heads whose combined output matters most for prediction.

    Looks at head pairs that jointly contribute the most to the target logit.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] token IDs.
        pos: Position to analyze.
        target_token: Token to evaluate (default: top prediction).

    Returns:
        Dict with important head pairs.
    """
    logits = model(tokens)
    _, cache = model.run_with_cache(tokens)

    if target_token is None:
        target_token = int(jnp.argmax(logits[pos]))

    W_U = np.array(model.unembed.W_U[:, target_token])

    # Per-head logit contributions
    head_logits = {}
    for l in range(model.cfg.n_layers):
        z = np.array(cache[f'blocks.{l}.attn.hook_z'][pos])
        W_O = np.array(model.blocks[l].attn.W_O)
        for h in range(model.cfg.n_heads):
            out = z[h] @ W_O[h]
            logit_contrib = float(np.dot(out, W_U))
            head_logits[(l, h)] = logit_contrib

    # Find important pairs (both contribute in same direction)
    pairs = []
    heads = list(head_logits.keys())
    for i in range(len(heads)):
        for j in range(i + 1, len(heads)):
            h1, h2 = heads[i], heads[j]
            combined = abs(head_logits[h1]) + abs(head_logits[h2])
            same_sign = (head_logits[h1] > 0) == (head_logits[h2] > 0)
            pairs.append({
                'head_a': f'L{h1[0]}H{h1[1]}',
                'head_b': f'L{h2[0]}H{h2[1]}',
                'combined_magnitude': round(combined, 4),
                'head_a_logit': round(head_logits[h1], 4),
                'head_b_logit': round(head_logits[h2], 4),
                'cooperative': same_sign,
            })

    pairs.sort(key=lambda x: -x['combined_magnitude'])
    top_pairs = pairs[:10]

    return {
        'target_token': target_token,
        'top_pairs': top_pairs,
        'n_cooperative_top': sum(1 for p in top_pairs if p['cooperative']),
    }
