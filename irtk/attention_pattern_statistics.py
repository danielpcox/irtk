"""Attention pattern statistics: statistical characterization of attention distributions."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def attention_entropy_profile(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Entropy of attention patterns across all heads and positions.

    Low entropy = focused attention. High entropy = diffuse.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    per_head = []
    for layer in range(n_layers):
        patterns = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]
        for head in range(n_heads):
            p = patterns[head]  # [seq, seq]
            entropies = []
            for pos in range(seq_len):
                row = p[pos, :pos + 1]
                ent = -float(jnp.sum(row * jnp.log(row + 1e-10)))
                entropies.append(ent)
            mean_ent = sum(entropies) / len(entropies)
            max_possible = float(jnp.log(jnp.array(seq_len, dtype=jnp.float32)))

            per_head.append({
                'layer': layer,
                'head': head,
                'mean_entropy': mean_ent,
                'max_possible_entropy': max_possible,
                'normalized_entropy': mean_ent / (max_possible + 1e-10),
                'is_focused': bool(mean_ent < max_possible * 0.3),
            })

    return {
        'per_head': per_head,
        'n_focused': sum(1 for h in per_head if h['is_focused']),
    }


def attention_concentration_profile(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How concentrated is attention? Top-k mass analysis.

    Shows whether attention is sharp (few tokens) or diffuse (many).
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    per_head = []
    for layer in range(n_layers):
        patterns = cache[f'blocks.{layer}.attn.hook_pattern']
        for head in range(n_heads):
            p = patterns[head]  # [seq, seq]
            top1_masses = []
            top3_masses = []
            for pos in range(seq_len):
                row = p[pos, :pos + 1]
                sorted_row = jnp.sort(row)[::-1]
                top1_masses.append(float(sorted_row[0]))
                top3 = float(jnp.sum(sorted_row[:min(3, pos + 1)]))
                top3_masses.append(top3)

            per_head.append({
                'layer': layer,
                'head': head,
                'mean_top1_mass': sum(top1_masses) / len(top1_masses),
                'mean_top3_mass': sum(top3_masses) / len(top3_masses),
                'is_sharp': bool(sum(top1_masses) / len(top1_masses) > 0.5),
            })

    return {
        'per_head': per_head,
        'n_sharp': sum(1 for h in per_head if h['is_sharp']),
    }


def attention_positional_bias(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Does attention exhibit positional biases?

    Measures BOS attention, self-attention, and recency bias.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    per_head = []
    for layer in range(n_layers):
        patterns = cache[f'blocks.{layer}.attn.hook_pattern']
        for head in range(n_heads):
            p = patterns[head]  # [seq, seq]

            # BOS attention (how much weight on position 0)
            bos_weights = [float(p[pos, 0]) for pos in range(1, seq_len)]
            mean_bos = sum(bos_weights) / len(bos_weights) if bos_weights else 0.0

            # Self-attention (diagonal)
            self_weights = [float(p[pos, pos]) for pos in range(seq_len)]
            mean_self = sum(self_weights) / len(self_weights)

            # Previous-token attention
            prev_weights = [float(p[pos, pos - 1]) for pos in range(1, seq_len)]
            mean_prev = sum(prev_weights) / len(prev_weights) if prev_weights else 0.0

            per_head.append({
                'layer': layer,
                'head': head,
                'mean_bos_attention': mean_bos,
                'mean_self_attention': mean_self,
                'mean_prev_attention': mean_prev,
                'dominant_bias': max(
                    [('bos', mean_bos), ('self', mean_self), ('prev', mean_prev)],
                    key=lambda x: x[1],
                )[0],
            })

    return {
        'per_head': per_head,
    }


def attention_pattern_stability(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How stable are attention patterns across positions?

    Measures consistency of each head's attention pattern.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    per_head = []
    for layer in range(n_layers):
        patterns = cache[f'blocks.{layer}.attn.hook_pattern']
        for head in range(n_heads):
            p = patterns[head]  # [seq, seq]

            # Compare consecutive rows (padded)
            diffs = []
            for pos in range(2, seq_len):
                row_curr = p[pos, :pos + 1]
                row_prev = p[pos - 1, :pos]
                # Compare on overlapping positions
                overlap = min(pos, pos + 1)
                diff = float(jnp.sum(jnp.abs(row_curr[:overlap] - row_prev[:overlap])))
                diffs.append(diff)

            mean_diff = sum(diffs) / len(diffs) if diffs else 0.0

            per_head.append({
                'layer': layer,
                'head': head,
                'mean_consecutive_diff': mean_diff,
                'is_stable': bool(mean_diff < 0.5),
            })

    return {
        'per_head': per_head,
        'n_stable': sum(1 for h in per_head if h['is_stable']),
    }


def attention_head_diversity(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """How diverse are attention patterns across heads within a layer?

    Low diversity = redundant heads. High diversity = specialized.
    """
    _, cache = model.run_with_cache(tokens)
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    patterns = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]

    # Flatten each head's pattern for comparison
    flat_patterns = []
    for h in range(n_heads):
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        flat = (patterns[h] * mask).reshape(-1)
        flat = flat / (jnp.linalg.norm(flat) + 1e-10)
        flat_patterns.append(flat)

    flat_patterns = jnp.stack(flat_patterns)
    sim_matrix = flat_patterns @ flat_patterns.T

    pairs = []
    total_sim = 0.0
    n_pairs = 0
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            s = float(sim_matrix[i, j])
            pairs.append({'head_a': i, 'head_b': j, 'similarity': s})
            total_sim += abs(s)
            n_pairs += 1

    mean_sim = total_sim / n_pairs if n_pairs > 0 else 0.0

    return {
        'layer': layer,
        'pairs': pairs,
        'mean_abs_similarity': mean_sim,
        'is_diverse': bool(mean_sim < 0.5),
    }
