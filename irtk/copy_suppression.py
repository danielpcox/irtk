"""Copy suppression analysis.

Analyzes heads that suppress the copying of tokens from context — the
"negative heads" phenomenon. These heads attend to positions and push the
residual stream away from predicting those tokens, preventing blind copying.

Functions:
- copy_suppression_score: How much a head suppresses the attended token's logit
- find_negative_heads: Scan all heads for copy-suppressing behavior
- suppression_per_attended_token: Attention-weighted suppression per source token
- copy_vs_suppress_decomposition: Decompose heads into copiers vs suppressors
- suppression_circuit_on_ioi: Suppression analysis on IOI-style prompts

References:
    - McDougall et al. (2023) "Copy Suppression: Comprehensively Understanding an
      Attention Head"
    - Wang et al. (2022) "Interpretability in the Wild: GPT-2 IOI"
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def copy_suppression_score(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
    head: int,
    pos: int = -1,
) -> dict:
    """How much a head suppresses the logit of attended-to tokens.

    For the head at (layer, head), computes the logit contribution for
    each attended-to token. Copy suppression means attending to token X
    and producing a negative logit for X.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        layer: Layer index.
        head: Head index.
        pos: Query position (-1 = last).

    Returns:
        Dict with:
            "suppression_scores": [seq_len] suppression magnitude per source position
            "mean_suppression": mean suppression across positions
            "max_suppression_pos": position with strongest suppression
            "is_copy_suppressing": whether the head suppresses on average
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)
    if pos == -1:
        pos = seq_len - 1

    tokens_np = np.array(tokens)

    # Get attention pattern for this head at this position
    pattern_key = f"blocks.{layer}.attn.hook_pattern"
    if pattern_key not in cache.cache_dict:
        return {
            "suppression_scores": np.zeros(seq_len),
            "mean_suppression": 0.0,
            "max_suppression_pos": 0,
            "is_copy_suppressing": False,
        }

    pattern = np.array(cache.cache_dict[pattern_key])  # [n_heads, seq, seq]
    attn_weights = pattern[head, pos, :]  # [seq_len]

    # Get the head's value output: z[head] projected through W_O[head] then W_U
    W_V = np.array(model.blocks[layer].attn.W_V[head])  # [d_model, d_head]
    W_O = np.array(model.blocks[layer].attn.W_O[head])  # [d_head, d_model]
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]

    # OV circuit for this head: maps input -> logit contribution
    OV = W_V @ W_O  # [d_model, d_model]

    # Get residual at each source position (input to this layer's attention)
    resid_key = f"blocks.{layer}.hook_resid_pre"
    if resid_key not in cache.cache_dict:
        resid_key = f"blocks.{layer}.attn.hook_result"  # fallback

    if f"blocks.{layer}.hook_resid_pre" in cache.cache_dict:
        resid = np.array(cache.cache_dict[f"blocks.{layer}.hook_resid_pre"])  # [seq, d_model]
    else:
        resid = np.zeros((seq_len, model.cfg.d_model))

    # For each source position, compute the logit contribution for the token at that position
    suppression = np.zeros(seq_len)
    for src in range(seq_len):
        src_token = int(tokens_np[src])
        # Value vector for this source
        v = resid[src] @ OV  # [d_model]
        # Logit for the source token
        logit_for_src = float(v @ W_U[:, src_token])
        # Weighted by attention
        suppression[src] = -attn_weights[src] * logit_for_src

    mean_supp = float(np.mean(suppression))
    max_pos = int(np.argmax(suppression))

    return {
        "suppression_scores": suppression,
        "mean_suppression": mean_supp,
        "max_suppression_pos": max_pos,
        "is_copy_suppressing": mean_supp > 0,
    }


def find_negative_heads(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    top_k: int = 5,
) -> dict:
    """Scan all heads for copy-suppressing behavior.

    A negative head has an OV circuit that, when attending to token X,
    produces a negative logit contribution for X.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        top_k: Number of top negative heads to return.

    Returns:
        Dict with:
            "negative_scores": [n_layers, n_heads] copy suppression score
            "top_negative_heads": list of (layer, head, score) tuples
            "n_negative_heads": number of heads with positive suppression score
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    tokens_np = np.array(tokens)
    seq_len = len(tokens)

    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]
    scores = np.zeros((n_layers, n_heads))

    for l in range(n_layers):
        pattern_key = f"blocks.{l}.attn.hook_pattern"
        resid_key = f"blocks.{l}.hook_resid_pre"

        if pattern_key not in cache.cache_dict or resid_key not in cache.cache_dict:
            continue

        pattern = np.array(cache.cache_dict[pattern_key])  # [n_heads, seq, seq]
        resid = np.array(cache.cache_dict[resid_key])  # [seq, d_model]

        for h in range(n_heads):
            W_V = np.array(model.blocks[l].attn.W_V[h])  # [d_model, d_head]
            W_O = np.array(model.blocks[l].attn.W_O[h])  # [d_head, d_model]
            OV = W_V @ W_O  # [d_model, d_model]

            total_suppression = 0.0
            count = 0
            for pos in range(1, seq_len):
                for src in range(pos + 1):
                    src_token = int(tokens_np[src])
                    v = resid[src] @ OV
                    logit = float(v @ W_U[:, src_token])
                    total_suppression -= pattern[h, pos, src] * logit
                    count += 1

            scores[l, h] = total_suppression / max(count, 1)

    # Rank
    flat = scores.flatten()
    top_idx = np.argsort(flat)[::-1][:top_k]
    top_heads = []
    for idx in top_idx:
        l = int(idx // n_heads)
        h = int(idx % n_heads)
        top_heads.append((l, h, float(scores[l, h])))

    n_neg = int(np.sum(scores > 0))

    return {
        "negative_scores": scores,
        "top_negative_heads": top_heads,
        "n_negative_heads": n_neg,
    }


def suppression_per_attended_token(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    layer: int,
    head: int,
    pos: int = -1,
) -> dict:
    """Attention-weighted suppression for each source token.

    For a query at `pos`, computes the attention-weighted logit suppression
    of each key position's token.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        layer: Layer index.
        head: Head index.
        pos: Query position (-1 = last).

    Returns:
        Dict with:
            "token_suppression": [seq_len] attention-weighted suppression per source
            "attention_weights": [seq_len] attention from pos to each source
            "most_suppressed_pos": position whose token is most suppressed
            "total_suppression": total logit suppression across all sources
    """
    result = copy_suppression_score(model, tokens, layer, head, pos)
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)
    if pos == -1:
        pos = seq_len - 1

    pattern_key = f"blocks.{layer}.attn.hook_pattern"
    if pattern_key in cache.cache_dict:
        pattern = np.array(cache.cache_dict[pattern_key])
        attn = pattern[head, pos, :]
    else:
        attn = np.zeros(seq_len)

    supp = result["suppression_scores"]
    most_supp = int(np.argmax(supp))
    total = float(np.sum(supp))

    return {
        "token_suppression": supp,
        "attention_weights": attn,
        "most_suppressed_pos": most_supp,
        "total_suppression": total,
    }


def copy_vs_suppress_decomposition(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    target_pos: int = -1,
    target_token: Optional[int] = None,
) -> dict:
    """Decompose all heads into copy-promoting vs copy-suppressing.

    For a target token at target_pos, compute each head's signed logit
    contribution for that token.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        target_pos: Position to analyze (-1 = last).
        target_token: Token to track (None = predicted token).

    Returns:
        Dict with:
            "head_contributions": [n_layers, n_heads] signed logit contribution
            "copy_heads": list of (layer, head) with positive contribution
            "suppress_heads": list of (layer, head) with negative contribution
            "net_effect": sum of all head contributions
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = len(tokens)
    if target_pos == -1:
        target_pos = seq_len - 1

    if target_token is None:
        logits = np.array(model(tokens))
        target_token = int(np.argmax(logits[target_pos]))

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    W_U = np.array(model.unembed.W_U)
    unembed_dir = W_U[:, target_token]  # [d_model]

    contributions = np.zeros((n_layers, n_heads))

    for l in range(n_layers):
        z_key = f"blocks.{l}.attn.hook_z"
        if z_key not in cache.cache_dict:
            continue
        z = np.array(cache.cache_dict[z_key])  # [seq, n_heads, d_head]

        for h in range(n_heads):
            W_O = np.array(model.blocks[l].attn.W_O[h])  # [d_head, d_model]
            head_out = z[target_pos, h] @ W_O  # [d_model]
            contributions[l, h] = float(np.dot(head_out, unembed_dir))

    copy_heads = []
    suppress_heads = []
    for l in range(n_layers):
        for h in range(n_heads):
            if contributions[l, h] > 0:
                copy_heads.append((l, h))
            elif contributions[l, h] < 0:
                suppress_heads.append((l, h))

    return {
        "head_contributions": contributions,
        "copy_heads": copy_heads,
        "suppress_heads": suppress_heads,
        "net_effect": float(np.sum(contributions)),
    }


def suppression_circuit_on_ioi(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    subject_pos: int,
    io_pos: int,
    prediction_pos: int = -1,
) -> dict:
    """Suppression analysis on IOI-style prompts.

    Identifies which heads suppress the subject name at the prediction
    position where the indirect object should be predicted.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        subject_pos: Position of the subject token (to be suppressed).
        io_pos: Position of the indirect object token (to be promoted).
        prediction_pos: Position where prediction happens (-1 = last).

    Returns:
        Dict with:
            "subject_suppression_by_head": [n_layers, n_heads] suppression of subject token
            "io_promotion_by_head": [n_layers, n_heads] promotion of IO token
            "net_effect_by_head": [n_layers, n_heads] promotion - suppression
            "top_suppressors": list of (layer, head, score) for subject suppression
            "subject_token": the subject token being suppressed
            "io_token": the IO token being promoted
    """
    seq_len = len(tokens)
    if prediction_pos == -1:
        prediction_pos = seq_len - 1

    tokens_np = np.array(tokens)
    subject_token = int(tokens_np[subject_pos])
    io_token = int(tokens_np[io_pos])

    # Get contributions to both tokens
    subj_result = copy_vs_suppress_decomposition(
        model, tokens, target_pos=prediction_pos, target_token=subject_token
    )
    io_result = copy_vs_suppress_decomposition(
        model, tokens, target_pos=prediction_pos, target_token=io_token
    )

    subject_supp = -subj_result["head_contributions"]  # negate so positive = suppression
    io_promo = io_result["head_contributions"]
    net = io_promo + subject_supp  # positive = good for IOI task

    # Top suppressors of subject
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    top_supp = []
    for l in range(n_layers):
        for h in range(n_heads):
            top_supp.append((l, h, float(subject_supp[l, h])))
    top_supp.sort(key=lambda x: x[2], reverse=True)
    top_supp = top_supp[:5]

    return {
        "subject_suppression_by_head": subject_supp,
        "io_promotion_by_head": io_promo,
        "net_effect_by_head": net,
        "top_suppressors": top_supp,
        "subject_token": subject_token,
        "io_token": io_token,
    }
