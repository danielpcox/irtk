"""Token embedding dynamics: how token representations evolve through the model."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def token_identity_evolution(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1) -> dict:
    """How does the model's 'belief' about a token's identity change through layers?

    Projects residual stream at each layer onto the unembedding matrix.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position
    n_layers = model.cfg.n_layers

    W_U = model.unembed.W_U
    input_token = int(tokens[pos])

    per_layer = []
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post'][pos]
        logits = resid @ W_U
        probs = jax.nn.softmax(logits)

        input_prob = float(probs[input_token])
        input_rank = int(jnp.sum(logits > logits[input_token]))
        top_pred = int(jnp.argmax(logits))

        per_layer.append({
            'layer': layer,
            'input_token_prob': input_prob,
            'input_token_rank': input_rank,
            'top_prediction': top_pred,
            'top_prob': float(probs[top_pred]),
            'retains_identity': bool(input_rank < 5),
        })

    return {
        'position': pos,
        'input_token': input_token,
        'per_layer': per_layer,
    }


def embedding_residual_similarity(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How similar is each position's residual to its original embedding?

    Measures identity preservation vs transformation.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    n_layers = model.cfg.n_layers

    per_position = []
    for pos in range(seq_len):
        embed = cache['hook_embed'][pos] + cache['hook_pos_embed'][pos]
        embed_dir = embed / (jnp.linalg.norm(embed) + 1e-10)

        similarities = []
        for layer in range(n_layers):
            resid = cache[f'blocks.{layer}.hook_resid_post'][pos]
            resid_dir = resid / (jnp.linalg.norm(resid) + 1e-10)
            cos = float(jnp.dot(embed_dir, resid_dir))
            similarities.append({
                'layer': layer,
                'cosine_to_embed': cos,
            })

        per_position.append({
            'position': pos,
            'token': int(tokens[pos]),
            'per_layer': similarities,
            'final_similarity': similarities[-1]['cosine_to_embed'] if similarities else 0.0,
        })

    return {
        'per_position': per_position,
    }


def token_mixing_rate(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How quickly do token representations diverge from their initial embeddings?

    Measures the rate of information mixing across positions.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        cosines = []
        for pos in range(seq_len):
            embed = cache['hook_embed'][pos] + cache['hook_pos_embed'][pos]
            resid = cache[f'blocks.{layer}.hook_resid_post'][pos]
            cos = float(jnp.dot(embed, resid) / (jnp.linalg.norm(embed) * jnp.linalg.norm(resid) + 1e-10))
            cosines.append(cos)

        mean_cos = sum(cosines) / len(cosines)
        per_layer.append({
            'layer': layer,
            'mean_embed_similarity': mean_cos,
            'min_similarity': min(cosines),
            'max_similarity': max(cosines),
        })

    return {
        'per_layer': per_layer,
    }


def token_representation_distance(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """Pairwise distances between token representations at a given layer.

    Shows which tokens have similar or different representations.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]

    resids = []
    for pos in range(seq_len):
        resid = cache[f'blocks.{layer}.hook_resid_post'][pos]
        resids.append(resid / (jnp.linalg.norm(resid) + 1e-10))

    pairs = []
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            cos = float(jnp.dot(resids[i], resids[j]))
            pairs.append({
                'pos_a': i,
                'pos_b': j,
                'token_a': int(tokens[i]),
                'token_b': int(tokens[j]),
                'cosine': cos,
            })

    mean_sim = sum(p['cosine'] for p in pairs) / len(pairs) if pairs else 0.0

    return {
        'layer': layer,
        'pairs': pairs,
        'mean_similarity': mean_sim,
    }


def token_prediction_trajectory(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1) -> dict:
    """Track the top-k predictions at each layer for a given position.

    Shows how the prediction forms and stabilizes.
    """
    _, cache = model.run_with_cache(tokens)
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position
    n_layers = model.cfg.n_layers

    W_U = model.unembed.W_U

    per_layer = []
    prev_top = None
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post'][pos]
        logits = resid @ W_U
        probs = jax.nn.softmax(logits)

        top_5 = jnp.argsort(logits)[-5:][::-1]
        top_predictions = [{'token': int(t), 'prob': float(probs[t])} for t in top_5]

        current_top = int(top_5[0])
        changed = prev_top is not None and current_top != prev_top
        prev_top = current_top

        per_layer.append({
            'layer': layer,
            'top_predictions': top_predictions,
            'top_changed': bool(changed),
        })

    return {
        'position': pos,
        'per_layer': per_layer,
        'n_changes': sum(1 for p in per_layer if p['top_changed']),
    }
