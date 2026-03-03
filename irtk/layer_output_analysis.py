"""Layer output analysis: characterize each layer's overall contribution."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def layer_output_decomposition(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Decompose each layer's output into attention and MLP contributions.

    Shows the balance and interaction at each layer.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    per_layer = []
    for layer in range(n_layers):
        attn_out = cache[f'blocks.{layer}.hook_attn_out']  # [seq, d_model]
        mlp_out = cache[f'blocks.{layer}.hook_mlp_out']
        combined = attn_out + mlp_out

        attn_norm = float(jnp.mean(jnp.linalg.norm(attn_out, axis=-1)))
        mlp_norm = float(jnp.mean(jnp.linalg.norm(mlp_out, axis=-1)))
        combined_norm = float(jnp.mean(jnp.linalg.norm(combined, axis=-1)))

        # Alignment
        cos = float(jnp.mean(
            jnp.sum(attn_out * mlp_out, axis=-1) /
            (jnp.linalg.norm(attn_out, axis=-1) * jnp.linalg.norm(mlp_out, axis=-1) + 1e-10)
        ))

        # Efficiency
        efficiency = combined_norm / (attn_norm + mlp_norm + 1e-10)

        per_layer.append({
            'layer': layer,
            'attn_norm': attn_norm,
            'mlp_norm': mlp_norm,
            'combined_norm': combined_norm,
            'alignment': cos,
            'efficiency': efficiency,
            'is_cooperative': cos > 0,
        })

    return {
        'per_layer': per_layer,
        'n_cooperative': sum(1 for p in per_layer if p['is_cooperative']),
    }


def layer_prediction_change(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1) -> dict:
    """How does each layer change the prediction?

    Tracks the logit of the final prediction through layers.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    # Final prediction
    final_resid = cache[f'blocks.{n_layers - 1}.hook_resid_post']
    final_logits = final_resid[pos] @ W_U + b_U
    target = int(jnp.argmax(final_logits))

    per_layer = []
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post']
        logits = resid[pos] @ W_U + b_U
        target_logit = float(logits[target])
        target_rank = int(jnp.sum(logits > logits[target]))
        top_pred = int(jnp.argmax(logits))

        per_layer.append({
            'layer': layer,
            'target_logit': target_logit,
            'target_rank': target_rank,
            'top_prediction': top_pred,
            'matches_final': top_pred == target,
        })

    return {
        'position': pos,
        'target_token': target,
        'per_layer': per_layer,
    }


def layer_residual_growth(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Analyze residual stream growth at each layer.

    Separates growth into magnitude and direction changes.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    embed = cache['hook_embed'] + cache['hook_pos_embed']
    prev = embed

    per_layer = []
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post']
        delta = resid - prev

        prev_norm = float(jnp.mean(jnp.linalg.norm(prev, axis=-1)))
        curr_norm = float(jnp.mean(jnp.linalg.norm(resid, axis=-1)))
        delta_norm = float(jnp.mean(jnp.linalg.norm(delta, axis=-1)))

        # Angle between delta and residual
        cos_delta_resid = float(jnp.mean(
            jnp.sum(delta * prev, axis=-1) /
            (jnp.linalg.norm(delta, axis=-1) * jnp.linalg.norm(prev, axis=-1) + 1e-10)
        ))

        per_layer.append({
            'layer': layer,
            'prev_norm': prev_norm,
            'curr_norm': curr_norm,
            'delta_norm': delta_norm,
            'relative_delta': delta_norm / (prev_norm + 1e-10),
            'delta_residual_alignment': cos_delta_resid,
        })
        prev = resid

    return {
        'per_layer': per_layer,
    }


def layer_information_content(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """Measure information content at each layer via prediction entropy.

    Low entropy = more information concentrated in top predictions.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    per_layer = []
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post']  # [seq, d_model]
        logits = resid @ W_U + b_U  # [seq, d_vocab]
        probs = jax.nn.softmax(logits, axis=-1)
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)  # [seq]
        mean_entropy = float(jnp.mean(entropy))
        max_probs = jnp.max(probs, axis=-1)
        mean_conf = float(jnp.mean(max_probs))

        per_layer.append({
            'layer': layer,
            'mean_entropy': mean_entropy,
            'mean_confidence': mean_conf,
        })

    # Entropy reduction from first to last
    if per_layer:
        reduction = per_layer[0]['mean_entropy'] - per_layer[-1]['mean_entropy']
    else:
        reduction = 0.0

    return {
        'per_layer': per_layer,
        'entropy_reduction': reduction,
        'sharpens_prediction': reduction > 0,
    }


def layer_uniqueness_score(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How unique is each layer's contribution?

    Compares each layer's output direction to all other layers.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    # Get each layer's delta (attn + MLP contribution)
    deltas = []
    for layer in range(n_layers):
        attn_out = cache[f'blocks.{layer}.hook_attn_out']
        mlp_out = cache[f'blocks.{layer}.hook_mlp_out']
        combined = attn_out + mlp_out
        mean_dir = jnp.mean(combined, axis=0)
        mean_dir = mean_dir / (jnp.linalg.norm(mean_dir) + 1e-10)
        deltas.append(mean_dir)

    deltas = jnp.stack(deltas)  # [n_layers, d_model]
    sim_matrix = deltas @ deltas.T

    per_layer = []
    for i in range(n_layers):
        sims = []
        for j in range(n_layers):
            if i != j:
                sims.append(float(sim_matrix[i, j]))
        mean_sim = sum(abs(s) for s in sims) / len(sims) if sims else 0
        max_sim = max(abs(s) for s in sims) if sims else 0

        per_layer.append({
            'layer': i,
            'mean_abs_similarity': mean_sim,
            'max_abs_similarity': max_sim,
            'uniqueness_score': 1.0 - mean_sim,
            'is_unique': mean_sim < 0.3,
        })

    return {
        'per_layer': per_layer,
        'n_unique': sum(1 for p in per_layer if p['is_unique']),
    }
