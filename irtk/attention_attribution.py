"""Attention attribution analysis.

Attribute model behavior to specific attention patterns: knockout effects,
pattern contribution to logits, value decomposition, and position-specific attribution.

References:
    Voita et al. (2019) "Analyzing Multi-Head Self-Attention"
    Clark et al. (2019) "What Does BERT Look At?"
"""

import jax
import jax.numpy as jnp
import numpy as np


def attention_knockout_attribution(model, tokens, metric_fn):
    """Attribute metric to attention heads via knockout (zeroing).

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function mapping logits to scalar metric.

    Returns:
        dict with:
            head_effects: [n_layers, n_heads] metric change when each head is knocked out
            total_attribution: float (sum of all effects)
            top_heads: list of (layer, head, effect)
            bottom_heads: list of (layer, head, effect) (most negative)
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    base_logits = np.array(model(tokens))
    base_metric = metric_fn(base_logits)

    effects = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        for head in range(n_heads):
            hook_key = f"blocks.{layer}.hook_z"
            h = head
            def make_hook(h_idx):
                def hook_fn(x, name):
                    return x.at[:, h_idx, :].set(0.0)
                return hook_fn
            state = HookState(hook_fns={hook_key: make_hook(h)}, cache={})
            abl_logits = np.array(model(tokens, hook_state=state))
            effects[layer, head] = base_metric - metric_fn(abl_logits)

    # Sort
    flat = [(int(l), int(h), float(effects[l, h]))
            for l in range(n_layers) for h in range(n_heads)]
    top = sorted(flat, key=lambda x: -x[2])[:10]
    bottom = sorted(flat, key=lambda x: x[2])[:10]

    return {
        "head_effects": effects,
        "total_attribution": float(np.sum(effects)),
        "top_heads": top,
        "bottom_heads": bottom,
    }


def attention_value_decomposition(model, tokens, layer, pos=-1, top_k=5):
    """Decompose attention output by source position and head.

    For each head, show which source positions contribute most to the output
    at the target position, weighted by attention and value content.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        layer: Layer to analyze.
        pos: Target position.
        top_k: Top source positions per head.

    Returns:
        dict with:
            head_source_contributions: [n_heads, seq_len] contribution per source per head
            top_sources_per_head: dict of head -> list of (source_pos, contribution)
            total_output_norm: float
            head_output_norms: [n_heads] output norm per head
    """
    from irtk.hook_points import HookState

    n_heads = model.cfg.n_heads

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
    v = cache.get(f"blocks.{layer}.attn.hook_v")

    seq_len = len(tokens)

    if pattern is None or v is None:
        return {
            "head_source_contributions": np.zeros((n_heads, seq_len)),
            "top_sources_per_head": {},
            "total_output_norm": 0.0,
            "head_output_norms": np.zeros(n_heads),
        }

    pat = np.array(pattern)  # [n_heads, seq, seq]
    v_arr = np.array(v)  # [seq, n_heads, d_head]
    W_O = np.array(model.blocks[layer].attn.W_O)  # [n_heads, d_head, d_model]

    contribs = np.zeros((n_heads, seq_len))
    head_norms = np.zeros(n_heads)
    top_sources = {}

    for head in range(n_heads):
        # For each source position, compute the contribution
        for src in range(seq_len):
            if src > pos and pos >= 0:
                continue
            # Weighted value contribution
            weighted_v = pat[head, pos, src] * v_arr[src, head]  # [d_head]
            output = weighted_v @ W_O[head]  # [d_model]
            contribs[head, src] = float(np.linalg.norm(output))

        head_norms[head] = float(np.sum(contribs[head]))
        top_idx = np.argsort(-contribs[head])[:top_k]
        top_sources[head] = [(int(i), float(contribs[head, i])) for i in top_idx]

    total_norm = float(np.sum(head_norms))

    return {
        "head_source_contributions": contribs,
        "top_sources_per_head": top_sources,
        "total_output_norm": total_norm,
        "head_output_norms": head_norms,
    }


def position_specific_attribution(model, tokens, metric_fn, target_pos=-1):
    """Attribute metric to attention at specific source positions.

    For each head, ablate attention to each source position and measure the effect.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function mapping logits to scalar metric.
        target_pos: Position whose prediction to analyze.

    Returns:
        dict with:
            position_effects: [n_layers, n_heads, seq_len] effect of zeroing attention to each source
            most_important_positions: dict of (layer, head) -> top source position
            position_summary: [seq_len] total importance of each source position
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    base_logits = np.array(model(tokens))
    base_metric = metric_fn(base_logits)

    effects = np.zeros((n_layers, n_heads, seq_len))
    most_important = {}

    for layer in range(n_layers):
        for head in range(n_heads):
            for src in range(seq_len):
                hook_key = f"blocks.{layer}.attn.hook_pattern"
                l, h, s = layer, head, src
                def make_hook(h_idx, s_idx):
                    def hook_fn(x, name):
                        return x.at[h_idx, :, s_idx].set(0.0)
                    return hook_fn
                state = HookState(hook_fns={hook_key: make_hook(h, s)}, cache={})
                abl_logits = np.array(model(tokens, hook_state=state))
                effects[layer, head, src] = abs(base_metric - metric_fn(abl_logits))

            most_important[(layer, head)] = int(np.argmax(effects[layer, head]))

    pos_summary = np.sum(effects, axis=(0, 1))

    return {
        "position_effects": effects,
        "most_important_positions": most_important,
        "position_summary": pos_summary,
    }


def attention_logit_contribution(model, tokens, pos=-1, top_k=5):
    """Compute each head's direct contribution to output logits.

    Projects each head's output through the unembedding to see which tokens it promotes.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position.
        top_k: Top tokens per head.

    Returns:
        dict with:
            head_logit_contributions: dict of (layer, head) -> [d_vocab] logit vector
            head_top_tokens: dict of (layer, head) -> list of (token, logit)
            total_attn_logit: [d_vocab] sum of all head contributions
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    W_U = np.array(model.unembed.W_U)
    d_vocab = W_U.shape[1]

    head_logits = {}
    head_top = {}
    total = np.zeros(d_vocab)

    for layer in range(n_layers):
        z = cache.get(f"blocks.{layer}.attn.hook_z")
        if z is None:
            continue
        z_arr = np.array(z)
        W_O = np.array(model.blocks[layer].attn.W_O)

        for head in range(n_heads):
            output = z_arr[pos, head] @ W_O[head]  # [d_model]
            logits = output @ W_U  # [d_vocab]
            head_logits[(layer, head)] = logits
            total += logits

            top_idx = np.argsort(-logits)[:top_k]
            head_top[(layer, head)] = [(int(i), float(logits[i])) for i in top_idx]

    return {
        "head_logit_contributions": head_logits,
        "head_top_tokens": head_top,
        "total_attn_logit": total,
    }


def attention_pattern_metric_sensitivity(model, tokens, metric_fn, layer, head):
    """Measure how sensitive a metric is to changes in a specific attention pattern.

    Applies uniform noise to the attention pattern and measures metric variance.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function mapping logits to scalar metric.
        layer: Layer of the head.
        head: Head index.

    Returns:
        dict with:
            base_metric: float
            noise_metrics: list of float (metrics under random perturbation)
            sensitivity: float (std of metrics under noise)
            max_deviation: float
    """
    from irtk.hook_points import HookState

    base_logits = np.array(model(tokens))
    base_metric = float(metric_fn(base_logits))

    noise_metrics = []
    rng = np.random.RandomState(42)

    for trial in range(5):
        noise = rng.randn() * 0.1
        hook_key = f"blocks.{layer}.attn.hook_pattern"
        h = head
        def make_hook(h_idx, n):
            def hook_fn(x, name):
                return x.at[h_idx].set(x[h_idx] + n)
            return hook_fn
        state = HookState(hook_fns={hook_key: make_hook(h, noise)}, cache={})
        abl_logits = np.array(model(tokens, hook_state=state))
        noise_metrics.append(float(metric_fn(abl_logits)))

    sensitivity = float(np.std(noise_metrics))
    max_dev = float(max(abs(m - base_metric) for m in noise_metrics))

    return {
        "base_metric": base_metric,
        "noise_metrics": noise_metrics,
        "sensitivity": sensitivity,
        "max_deviation": max_dev,
    }
