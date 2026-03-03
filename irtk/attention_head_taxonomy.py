"""Attention head taxonomy.

Systematic classification of attention heads by their functional role:
induction, previous-token, copy, backup, inhibition heads.

References:
    Olsson et al. (2022) "In-context Learning and Induction Heads"
    Wang et al. (2023) "Interpretability in the Wild"
"""

import jax
import jax.numpy as jnp
import numpy as np


def induction_head_score(model, tokens):
    """Score each head for induction behavior.

    Induction heads attend to the token after the previous occurrence of the
    current token. Measured by checking if the attention pattern has a
    "shifted duplicate" structure.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len] with repeated tokens.

    Returns:
        dict with:
            scores: [n_layers, n_heads] induction score per head
            top_heads: list of (layer, head, score)
            max_score: float
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    tokens_np = np.array(tokens)
    scores = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is None:
            continue

        pat = np.array(pattern)  # [n_heads, seq, seq]

        for head in range(n_heads):
            # For each position, check if the head attends to the position
            # after the previous occurrence of the same token
            score_sum = 0.0
            count = 0
            for q_pos in range(1, seq_len):
                current_tok = tokens_np[q_pos]
                # Find previous occurrences
                for k_pos in range(q_pos):
                    if tokens_np[k_pos] == current_tok and k_pos + 1 < seq_len:
                        # Induction: attend to k_pos + 1
                        score_sum += float(pat[head, q_pos, k_pos + 1]) if k_pos + 1 <= q_pos else 0.0
                        count += 1
            scores[layer, head] = score_sum / max(count, 1)

    top_heads = []
    for l in range(n_layers):
        for h in range(n_heads):
            if scores[l, h] > 0.01:
                top_heads.append((l, h, float(scores[l, h])))
    top_heads.sort(key=lambda x: -x[2])

    return {
        "scores": scores,
        "top_heads": top_heads[:10],
        "max_score": float(np.max(scores)),
    }


def previous_token_head_score(model, tokens):
    """Score each head for previous-token behavior.

    Previous-token heads attend primarily to the immediately preceding token.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            scores: [n_layers, n_heads] previous-token score per head
            top_heads: list of (layer, head, score)
            max_score: float
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = len(tokens)

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    scores = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
        if pattern is None:
            continue

        pat = np.array(pattern)  # [n_heads, seq, seq]

        for head in range(n_heads):
            # Average attention to position q-1
            prev_attn = 0.0
            count = 0
            for q in range(1, seq_len):
                prev_attn += float(pat[head, q, q - 1])
                count += 1
            scores[layer, head] = prev_attn / max(count, 1)

    top_heads = []
    for l in range(n_layers):
        for h in range(n_heads):
            if scores[l, h] > 0.01:
                top_heads.append((l, h, float(scores[l, h])))
    top_heads.sort(key=lambda x: -x[2])

    return {
        "scores": scores,
        "top_heads": top_heads[:10],
        "max_score": float(np.max(scores)),
    }


def copy_head_score(model, tokens, pos=-1, top_k=5):
    """Score each head for copy behavior.

    Copy heads promote tokens that they attend to into the output prediction.
    Measured by checking if attended-to tokens appear in the head's logit contribution.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        pos: Position to analyze.
        top_k: Number of top tokens to check.

    Returns:
        dict with:
            scores: [n_layers, n_heads] copy score per head
            top_heads: list of (layer, head, score)
            copied_tokens: dict of (layer, head) -> list of copied token IDs
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    W_U = np.array(model.unembed.W_U)
    tokens_np = np.array(tokens)
    input_tokens = set(tokens_np.tolist())

    scores = np.zeros((n_layers, n_heads))
    copied = {}

    for layer in range(n_layers):
        pattern = cache.get(f"blocks.{layer}.attn.hook_pattern")
        z = cache.get(f"blocks.{layer}.hook_z")
        if pattern is None or z is None:
            continue

        pat = np.array(pattern)  # [n_heads, seq, seq]
        z_arr = np.array(z)  # [seq, n_heads, d_head]
        W_O = np.array(model.blocks[layer].attn.W_O)  # [n_heads, d_head, d_model]

        for head in range(n_heads):
            # Head output at pos
            head_out = z_arr[pos, head] @ W_O[head]  # [d_model]
            head_logits = head_out @ W_U
            top_promoted = set(np.argsort(-head_logits)[:top_k].tolist())

            # Check overlap with attended-to tokens
            top_attended = np.argsort(-pat[head, pos])[:top_k]
            attended_tokens = set(tokens_np[top_attended].tolist())

            overlap = top_promoted & attended_tokens & input_tokens
            scores[layer, head] = len(overlap) / max(top_k, 1)
            if overlap:
                copied[(layer, head)] = list(overlap)

    top_heads = []
    for l in range(n_layers):
        for h in range(n_heads):
            if scores[l, h] > 0:
                top_heads.append((l, h, float(scores[l, h])))
    top_heads.sort(key=lambda x: -x[2])

    return {
        "scores": scores,
        "top_heads": top_heads[:10],
        "copied_tokens": copied,
    }


def inhibition_head_score(model, tokens, metric_fn):
    """Score each head for inhibition (negative) behavior.

    Inhibition heads reduce the metric when present (their ablation increases the metric).

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function mapping logits to scalar metric.

    Returns:
        dict with:
            scores: [n_layers, n_heads] inhibition score per head
            top_heads: list of (layer, head, score)
            n_inhibitory: int
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    base_logits = np.array(model(tokens))
    base_metric = metric_fn(base_logits)

    scores = np.zeros((n_layers, n_heads))

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
            change = metric_fn(abl_logits) - base_metric
            # Positive change when ablated = head was reducing metric = inhibitory
            scores[layer, head] = max(0.0, float(change))

    top_heads = []
    for l in range(n_layers):
        for h in range(n_heads):
            if scores[l, h] > 0.01:
                top_heads.append((l, h, float(scores[l, h])))
    top_heads.sort(key=lambda x: -x[2])

    return {
        "scores": scores,
        "top_heads": top_heads[:10],
        "n_inhibitory": int(np.sum(scores > 0.01)),
    }


def head_taxonomy_summary(model, tokens, metric_fn):
    """Classify all heads by their dominant functional role.

    Runs all scoring functions and assigns each head to its highest-scoring category.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len] (should contain repeated tokens for induction).
        metric_fn: Function mapping logits to scalar metric.

    Returns:
        dict with:
            classifications: dict of (layer, head) -> str type
            type_counts: dict of type -> count
            type_distribution: dict of type -> fraction
            head_details: dict of (layer, head) -> dict of all scores
    """
    ind = induction_head_score(model, tokens)
    prev = previous_token_head_score(model, tokens)
    copy = copy_head_score(model, tokens)
    inhib = inhibition_head_score(model, tokens, metric_fn)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    classifications = {}
    details = {}

    for l in range(n_layers):
        for h in range(n_heads):
            scores_map = {
                "induction": float(ind["scores"][l, h]),
                "previous_token": float(prev["scores"][l, h]),
                "copy": float(copy["scores"][l, h]),
                "inhibition": float(inhib["scores"][l, h]),
            }
            details[(l, h)] = scores_map

            max_type = max(scores_map, key=scores_map.get)
            max_score = scores_map[max_type]
            if max_score > 0.01:
                classifications[(l, h)] = max_type
            else:
                classifications[(l, h)] = "unclassified"

    type_counts = {}
    for t in classifications.values():
        type_counts[t] = type_counts.get(t, 0) + 1

    total = n_layers * n_heads
    type_dist = {t: c / total for t, c in type_counts.items()}

    return {
        "classifications": classifications,
        "type_counts": type_counts,
        "type_distribution": type_dist,
        "head_details": details,
    }
