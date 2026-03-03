"""Token influence tracking through the network.

Track how each input token's information flows through attention and
MLP layers to influence the final prediction.
"""

import jax
import jax.numpy as jnp
from irtk.hook_points import HookState


def _run_and_cache(model, tokens):
    """Run model and return activation cache."""
    hook_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=hook_state)
    return hook_state.cache


def token_influence_via_attention(model, tokens, source_pos=0, target_pos=-1):
    """Track how a source token influences the target through attention.

    Returns:
        dict with per_layer list containing:
        - layer: layer index
        - direct_attention: attention weight from target to source
        - per_head_attention: list of per-head attention weights
        - dominant_head: head with highest attention to source
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)

    if target_pos < 0:
        target_pos = seq_len + target_pos

    results = []
    for l in range(n_layers):
        pattern_key = f'blocks.{l}.attn.hook_pattern'
        if pattern_key not in cache:
            continue

        pattern = cache[pattern_key]  # [n_heads, seq_q, seq_k]
        n_heads = pattern.shape[0]

        per_head = []
        for h in range(n_heads):
            attn_weight = float(pattern[h, target_pos, source_pos])
            per_head.append({'head': h, 'attention': attn_weight})

        direct = float(jnp.mean(pattern[:, target_pos, source_pos]))
        dominant = max(per_head, key=lambda x: x['attention'])

        results.append({
            'layer': l,
            'direct_attention': direct,
            'per_head_attention': per_head,
            'dominant_head': dominant['head'],
            'max_attention': dominant['attention'],
        })

    return {
        'source_pos': source_pos,
        'target_pos': target_pos,
        'per_layer': results,
    }


def token_influence_via_residual(model, tokens, source_pos=0, target_pos=-1):
    """Track how a source token influences the target's residual stream.

    Measures the contribution of the source token's embedding to each
    layer's residual at the target position.

    Returns:
        dict with per_layer list containing:
        - layer: layer index
        - influence: cosine similarity between source embed and target residual delta
        - magnitude: norm of influence
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)

    if target_pos < 0:
        target_pos = seq_len + target_pos

    # Source token's initial embedding
    embed_key = 'blocks.0.hook_resid_pre'
    if embed_key not in cache:
        return {'per_layer': []}

    source_embed = cache[embed_key][source_pos]  # [d_model]
    source_norm = jnp.linalg.norm(source_embed)

    results = []
    prev_resid = cache[embed_key][target_pos]

    for l in range(n_layers):
        key = f'blocks.{l}.hook_resid_post'
        if key not in cache:
            continue

        curr_resid = cache[key][target_pos]  # [d_model]
        delta = curr_resid - prev_resid  # what this layer added

        delta_norm = jnp.linalg.norm(delta)
        cos = float(jnp.dot(delta, source_embed) / jnp.maximum(delta_norm * source_norm, 1e-10))
        mag = float(jnp.dot(delta, source_embed) / jnp.maximum(source_norm, 1e-10))

        results.append({
            'layer': l,
            'influence': cos,
            'magnitude': mag,
        })
        prev_resid = curr_resid

    return {
        'source_pos': source_pos,
        'target_pos': target_pos,
        'per_layer': results,
    }


def multi_token_influence(model, tokens, target_pos=-1):
    """Compare influence of all source tokens on the target.

    Returns:
        dict with:
        - per_position: list of per-source-token influence scores
        - most_influential: position with highest cumulative influence
        - influence_entropy: how spread out influence is
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)

    if target_pos < 0:
        target_pos = seq_len + target_pos

    # Aggregate attention weights across layers and heads
    cumulative_influence = jnp.zeros(seq_len)

    for l in range(n_layers):
        pattern_key = f'blocks.{l}.attn.hook_pattern'
        if pattern_key not in cache:
            continue

        pattern = cache[pattern_key]  # [n_heads, seq_q, seq_k]
        # Average across heads
        avg_attn = jnp.mean(pattern[:, target_pos, :], axis=0)  # [seq_k]
        cumulative_influence = cumulative_influence + avg_attn

    # Normalize
    total = jnp.sum(cumulative_influence)
    normalized = cumulative_influence / jnp.maximum(total, 1e-10)

    # Entropy
    normalized_safe = jnp.maximum(normalized, 1e-10)
    entropy = -float(jnp.sum(normalized_safe * jnp.log(normalized_safe)))
    max_entropy = float(jnp.log(jnp.array(seq_len, dtype=jnp.float32)))

    per_position = []
    for p in range(seq_len):
        per_position.append({
            'position': p,
            'influence': float(normalized[p]),
            'raw_influence': float(cumulative_influence[p]),
        })

    most_influential = int(jnp.argmax(normalized))

    return {
        'target_pos': target_pos,
        'per_position': per_position,
        'most_influential': most_influential,
        'influence_entropy': entropy,
        'normalized_entropy': entropy / max(max_entropy, 1e-10),
    }


def influence_path_analysis(model, tokens, source_pos=0, target_pos=-1):
    """Trace the path of influence from source to target through heads.

    Returns:
        dict with per_layer list of per-head influence paths.
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)

    if target_pos < 0:
        target_pos = seq_len + target_pos

    results = []
    for l in range(n_layers):
        pattern_key = f'blocks.{l}.attn.hook_pattern'
        z_key = f'blocks.{l}.attn.hook_z'

        if pattern_key not in cache or z_key not in cache:
            continue

        pattern = cache[pattern_key]  # [n_heads, seq_q, seq_k]
        z = cache[z_key]  # [seq, n_heads, d_head]
        n_heads = pattern.shape[0]
        W_O = model.blocks[l].attn.W_O  # [n_heads, d_head, d_model]

        head_paths = []
        for h in range(n_heads):
            attn_to_source = float(pattern[h, target_pos, source_pos])

            # Value contribution from source through this head
            v_contrib = z[source_pos, h]  # [d_head]
            output = jnp.einsum('h,hm->m', v_contrib, W_O[h])  # [d_model]
            output_norm = float(jnp.linalg.norm(output))

            head_paths.append({
                'head': h,
                'attention_to_source': attn_to_source,
                'output_norm': output_norm,
                'path_strength': attn_to_source * output_norm,
            })

        results.append({
            'layer': l,
            'head_paths': head_paths,
        })

    return {
        'source_pos': source_pos,
        'target_pos': target_pos,
        'per_layer': results,
    }


def influence_on_logits(model, tokens, source_pos=0, target_pos=-1, top_k=5):
    """How does a source token influence the final logit predictions?

    Returns:
        dict with:
        - logit_change_estimate: estimated logit change due to source token
        - top_promoted: tokens promoted by source influence
        - top_demoted: tokens demoted by source influence
    """
    cache = _run_and_cache(model, tokens)
    n_layers = model.cfg.n_layers
    seq_len = len(tokens)

    if target_pos < 0:
        target_pos = seq_len + target_pos

    W_U = model.unembed.W_U  # [d_model, d_vocab]
    b_U = model.unembed.b_U  # [d_vocab]

    # Accumulate per-layer contributions from source
    total_logit_contrib = jnp.zeros(model.cfg.d_vocab)

    for l in range(n_layers):
        pattern_key = f'blocks.{l}.attn.hook_pattern'
        z_key = f'blocks.{l}.attn.hook_z'

        if pattern_key not in cache or z_key not in cache:
            continue

        pattern = cache[pattern_key]  # [n_heads, seq_q, seq_k]
        z = cache[z_key]  # [seq, n_heads, d_head]
        W_O = model.blocks[l].attn.W_O  # [n_heads, d_head, d_model]
        n_heads = pattern.shape[0]

        for h in range(n_heads):
            attn = pattern[h, target_pos, source_pos]
            output = attn * jnp.einsum('h,hm->m', z[source_pos, h], W_O[h])
            logit_contrib = output @ W_U  # [d_vocab]
            total_logit_contrib = total_logit_contrib + logit_contrib

    # Top promoted/demoted
    top_promoted_idx = jnp.argsort(-total_logit_contrib)[:top_k]
    top_demoted_idx = jnp.argsort(total_logit_contrib)[:top_k]

    promoted = [{'token': int(i), 'logit_change': float(total_logit_contrib[i])} for i in top_promoted_idx]
    demoted = [{'token': int(i), 'logit_change': float(total_logit_contrib[i])} for i in top_demoted_idx]

    return {
        'source_pos': source_pos,
        'target_pos': target_pos,
        'top_promoted': promoted,
        'top_demoted': demoted,
        'total_logit_change_norm': float(jnp.linalg.norm(total_logit_contrib)),
    }
