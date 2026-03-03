"""Model health check: quick diagnostic checks for model behavior."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def weight_norm_check(model: HookedTransformer) -> dict:
    """Check weight norms across the model.

    Identifies layers with unusually large or small weights.
    """
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        w_q_norm = float(jnp.linalg.norm(model.blocks[layer].attn.W_Q))
        w_k_norm = float(jnp.linalg.norm(model.blocks[layer].attn.W_K))
        w_v_norm = float(jnp.linalg.norm(model.blocks[layer].attn.W_V))
        w_o_norm = float(jnp.linalg.norm(model.blocks[layer].attn.W_O))
        w_in_norm = float(jnp.linalg.norm(model.blocks[layer].mlp.W_in))
        w_out_norm = float(jnp.linalg.norm(model.blocks[layer].mlp.W_out))

        per_layer.append({
            'layer': layer,
            'W_Q_norm': w_q_norm,
            'W_K_norm': w_k_norm,
            'W_V_norm': w_v_norm,
            'W_O_norm': w_o_norm,
            'W_in_norm': w_in_norm,
            'W_out_norm': w_out_norm,
        })

    embed_norm = float(jnp.linalg.norm(model.embed.W_E))
    unembed_norm = float(jnp.linalg.norm(model.unembed.W_U))

    return {
        'embed_norm': embed_norm,
        'unembed_norm': unembed_norm,
        'per_layer': per_layer,
    }


def activation_range_check(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Check activation ranges through the model.

    Identifies potential overflow/underflow or dead regions.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post']
        resid_max = float(jnp.max(jnp.abs(resid)))
        resid_mean = float(jnp.mean(jnp.abs(resid)))

        mlp_post = cache[f'blocks.{layer}.mlp.hook_post']
        mlp_max = float(jnp.max(jnp.abs(mlp_post)))
        mlp_sparsity = float(jnp.mean(jnp.abs(mlp_post) < 1e-5))

        per_layer.append({
            'layer': layer,
            'resid_max': resid_max,
            'resid_mean': resid_mean,
            'mlp_max': mlp_max,
            'mlp_sparsity': mlp_sparsity,
            'has_large_activations': bool(resid_max > 100),
        })

    return {
        'per_layer': per_layer,
        'n_layers_with_large_acts': sum(1 for p in per_layer if p['has_large_activations']),
    }


def attention_health_check(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Check attention pattern health.

    Identifies degenerate patterns (all mass on one token, uniform, etc).
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[0]

    per_head = []
    for layer in range(n_layers):
        patterns = cache[f'blocks.{layer}.attn.hook_pattern']
        for head in range(n_heads):
            p = patterns[head]
            # Check entropy at last position
            last_row = p[-1, :seq_len]
            entropy = -float(jnp.sum(last_row * jnp.log(last_row + 1e-10)))
            max_weight = float(jnp.max(last_row))

            is_degenerate = bool(max_weight > 0.99 or entropy < 0.01)

            per_head.append({
                'layer': layer,
                'head': head,
                'last_pos_entropy': entropy,
                'last_pos_max_weight': max_weight,
                'is_degenerate': is_degenerate,
            })

    return {
        'per_head': per_head,
        'n_degenerate': sum(1 for h in per_head if h['is_degenerate']),
    }


def prediction_quality_check(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Check prediction quality at each position.

    Measures confidence, entropy, and rank of correct next token.
    """
    logits = model(tokens)  # [seq, d_vocab]
    seq_len = tokens.shape[0]

    W_U = model.unembed.W_U

    per_position = []
    for pos in range(seq_len - 1):
        probs = jax.nn.softmax(logits[pos])
        entropy = -float(jnp.sum(probs * jnp.log(probs + 1e-10)))
        top_prob = float(jnp.max(probs))
        top_token = int(jnp.argmax(probs))

        next_token = int(tokens[pos + 1])
        next_prob = float(probs[next_token])
        next_rank = int(jnp.sum(probs > probs[next_token]))

        per_position.append({
            'position': pos,
            'entropy': entropy,
            'top_prob': top_prob,
            'top_token': top_token,
            'next_token_prob': next_prob,
            'next_token_rank': next_rank,
            'correct': bool(top_token == next_token),
        })

    n_correct = sum(1 for p in per_position if p['correct'])

    return {
        'per_position': per_position,
        'n_correct': n_correct,
        'accuracy': n_correct / len(per_position) if per_position else 0.0,
    }


def residual_growth_check(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Check residual stream norm growth.

    Explosive growth or collapse may indicate problems.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    embed_norm = float(jnp.mean(jnp.linalg.norm(
        cache['hook_embed'] + cache['hook_pos_embed'], axis=-1
    )))

    per_layer = []
    prev_norm = embed_norm
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post']
        curr_norm = float(jnp.mean(jnp.linalg.norm(resid, axis=-1)))
        growth = curr_norm / (prev_norm + 1e-10)

        per_layer.append({
            'layer': layer,
            'mean_norm': curr_norm,
            'growth_factor': growth,
            'is_exploding': bool(growth > 5),
            'is_collapsing': bool(growth < 0.1),
        })
        prev_norm = curr_norm

    return {
        'embed_norm': embed_norm,
        'per_layer': per_layer,
        'final_norm': per_layer[-1]['mean_norm'] if per_layer else embed_norm,
    }
