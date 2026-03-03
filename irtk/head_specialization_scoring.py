"""Head specialization scoring: score each head for specific functional roles."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def induction_head_score(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Score each head for induction behavior.

    Induction heads attend to the token after a previous occurrence of the current token.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    per_head = []
    for layer in range(n_layers):
        patterns = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq, seq]
        for head in range(n_heads):
            scores = []
            for pos in range(2, seq_len):
                # Look for previous occurrence of tokens[pos-1]
                for prev in range(pos - 1):
                    if int(tokens[prev]) == int(tokens[pos - 1]) and prev + 1 < pos:
                        # Induction: should attend to prev+1
                        scores.append(float(patterns[head, pos, prev + 1]))

            mean_score = sum(scores) / len(scores) if scores else 0.0
            per_head.append({
                'layer': layer,
                'head': head,
                'induction_score': mean_score,
                'n_opportunities': len(scores),
                'is_induction': bool(mean_score > 0.3 and len(scores) > 0),
            })

    return {
        'per_head': per_head,
        'n_induction': sum(1 for h in per_head if h['is_induction']),
    }


def previous_token_head_score(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Score each head for previous-token attention.

    Previous-token heads consistently attend to position i-1.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    per_head = []
    for layer in range(n_layers):
        patterns = cache[f'blocks.{layer}.attn.hook_pattern']
        for head in range(n_heads):
            prev_weights = []
            for pos in range(1, seq_len):
                prev_weights.append(float(patterns[head, pos, pos - 1]))

            mean_prev = sum(prev_weights) / len(prev_weights) if prev_weights else 0.0
            per_head.append({
                'layer': layer,
                'head': head,
                'prev_token_score': mean_prev,
                'is_prev_token': bool(mean_prev > 0.4),
            })

    return {
        'per_head': per_head,
        'n_prev_token': sum(1 for h in per_head if h['is_prev_token']),
    }


def copy_head_score(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Score each head for copy behavior.

    Copy heads promote the attended token in the output.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    W_U = model.unembed.W_U  # [d_model, d_vocab]

    per_head = []
    for layer in range(n_layers):
        patterns = cache[f'blocks.{layer}.attn.hook_pattern']
        v = cache[f'blocks.{layer}.attn.hook_v']
        W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

        for head in range(n_heads):
            copy_scores = []
            for pos in range(1, seq_len):
                # Most-attended source
                attn_row = patterns[head, pos, :pos + 1]
                src = int(jnp.argmax(attn_row))
                src_token = int(tokens[src])

                # Value through OV and unembed
                v_src = v[src, head, :]  # [d_head]
                output = v_src @ W_O[head]  # [d_model]
                logits = output @ W_U  # [d_vocab]

                # Does it promote the source token?
                rank = int(jnp.sum(logits > logits[src_token]))
                copy_scores.append(rank)

            mean_rank = sum(copy_scores) / len(copy_scores) if copy_scores else float('inf')
            per_head.append({
                'layer': layer,
                'head': head,
                'mean_copy_rank': mean_rank,
                'is_copy': bool(mean_rank < 10),
            })

    return {
        'per_head': per_head,
        'n_copy': sum(1 for h in per_head if h['is_copy']),
    }


def inhibition_head_score(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Score each head for inhibition (negative) behavior.

    Inhibition heads suppress the attended token in the output.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    W_U = model.unembed.W_U

    per_head = []
    for layer in range(n_layers):
        patterns = cache[f'blocks.{layer}.attn.hook_pattern']
        v = cache[f'blocks.{layer}.attn.hook_v']
        W_O = model.blocks[layer].attn.W_O

        for head in range(n_heads):
            suppression_scores = []
            for pos in range(1, seq_len):
                attn_row = patterns[head, pos, :pos + 1]
                src = int(jnp.argmax(attn_row))
                src_token = int(tokens[src])

                v_src = v[src, head, :]
                output = v_src @ W_O[head]
                logits = output @ W_U

                # Negative logit for attended token = suppression
                suppression_scores.append(float(logits[src_token]))

            mean_suppression = sum(suppression_scores) / len(suppression_scores) if suppression_scores else 0.0
            per_head.append({
                'layer': layer,
                'head': head,
                'mean_attended_logit': mean_suppression,
                'is_inhibition': bool(mean_suppression < -0.1),
            })

    return {
        'per_head': per_head,
        'n_inhibition': sum(1 for h in per_head if h['is_inhibition']),
    }


def head_role_summary(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Summarize head roles across all scoring criteria.

    Combines all scores into a single classification per head.
    """
    ind = induction_head_score(model, tokens)
    prev = previous_token_head_score(model, tokens)
    copy = copy_head_score(model, tokens)
    inhib = inhibition_head_score(model, tokens)

    per_head = []
    for i in range(len(ind['per_head'])):
        roles = []
        if ind['per_head'][i]['is_induction']:
            roles.append('induction')
        if prev['per_head'][i]['is_prev_token']:
            roles.append('prev_token')
        if copy['per_head'][i]['is_copy']:
            roles.append('copy')
        if inhib['per_head'][i]['is_inhibition']:
            roles.append('inhibition')

        per_head.append({
            'layer': ind['per_head'][i]['layer'],
            'head': ind['per_head'][i]['head'],
            'roles': roles if roles else ['unclassified'],
            'induction_score': ind['per_head'][i]['induction_score'],
            'prev_token_score': prev['per_head'][i]['prev_token_score'],
            'copy_rank': copy['per_head'][i]['mean_copy_rank'],
            'inhibition_logit': inhib['per_head'][i]['mean_attended_logit'],
        })

    return {
        'per_head': per_head,
    }
