"""Token prediction stability: how stable are predictions across layers, positions, and perturbations."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def prediction_layer_stability(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1) -> dict:
    """How stable is the top prediction across layers?

    Tracks when the model commits to its final prediction.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position

    W_U = model.unembed.W_U  # [d_model, d_vocab]
    b_U = model.unembed.b_U  # [d_vocab]

    final_resid = cache[f'blocks.{n_layers - 1}.hook_resid_post']
    final_logits = final_resid[pos] @ W_U + b_U
    final_pred = int(jnp.argmax(final_logits))

    per_layer = []
    first_commit = n_layers
    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post']
        logits = resid[pos] @ W_U + b_U
        pred = int(jnp.argmax(logits))
        probs = jax.nn.softmax(logits)
        conf = float(probs[pred])
        matches_final = pred == final_pred

        if matches_final and first_commit == n_layers:
            # Check if it stays committed
            stays = True
            for future_layer in range(layer + 1, n_layers):
                future_resid = cache[f'blocks.{future_layer}.hook_resid_post']
                future_logits = future_resid[pos] @ W_U + b_U
                if int(jnp.argmax(future_logits)) != final_pred:
                    stays = False
                    break
            if stays:
                first_commit = layer

        per_layer.append({
            'layer': layer,
            'prediction': pred,
            'confidence': conf,
            'matches_final': matches_final,
        })

    return {
        'position': pos,
        'final_prediction': final_pred,
        'commit_layer': first_commit,
        'per_layer': per_layer,
        'is_early_commit': first_commit <= n_layers // 2,
    }


def prediction_position_consistency(model: HookedTransformer, tokens: jnp.ndarray) -> dict:
    """How consistent is the model's confidence across positions?

    Some positions may be much harder to predict than others.
    """
    logits = model(tokens)
    seq_len = tokens.shape[0]

    per_position = []
    confidences = []
    for pos in range(seq_len):
        probs = jax.nn.softmax(logits[pos])
        top = int(jnp.argmax(probs))
        conf = float(probs[top])
        entropy = float(-jnp.sum(probs * jnp.log(probs + 1e-10)))

        per_position.append({
            'position': pos,
            'top_prediction': top,
            'confidence': conf,
            'entropy': entropy,
        })
        confidences.append(conf)

    mean_conf = sum(confidences) / len(confidences)
    std_conf = (sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)) ** 0.5

    return {
        'per_position': per_position,
        'mean_confidence': mean_conf,
        'std_confidence': std_conf,
        'is_uniform': std_conf < 0.1,
    }


def prediction_token_competition(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1, top_k: int = 5) -> dict:
    """Analyze competition between top-k token predictions.

    Shows how close the race is between candidate tokens.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    # Get final top-k tokens
    final_resid = cache[f'blocks.{n_layers - 1}.hook_resid_post']
    final_logits = final_resid[pos] @ W_U + b_U
    final_probs = jax.nn.softmax(final_logits)
    top_tokens = jnp.argsort(final_logits)[-top_k:][::-1]

    # Track each candidate through layers
    per_token = {}
    for t_idx in range(top_k):
        t = int(top_tokens[t_idx])
        per_token[t] = {
            'token': t,
            'final_rank': t_idx,
            'final_probability': float(final_probs[t]),
            'per_layer': [],
        }

    for layer in range(n_layers):
        resid = cache[f'blocks.{layer}.hook_resid_post']
        logits = resid[pos] @ W_U + b_U
        probs = jax.nn.softmax(logits)
        sorted_tokens = jnp.argsort(logits)[::-1]

        for t_idx in range(top_k):
            t = int(top_tokens[t_idx])
            rank = int(jnp.where(sorted_tokens == t, size=1)[0][0])
            per_token[t]['per_layer'].append({
                'layer': layer,
                'logit': float(logits[t]),
                'probability': float(probs[t]),
                'rank': rank,
            })

    # Margin between top-1 and top-2
    top1_logit = float(final_logits[top_tokens[0]])
    top2_logit = float(final_logits[top_tokens[1]]) if top_k > 1 else 0.0
    margin = top1_logit - top2_logit

    return {
        'position': pos,
        'per_token': list(per_token.values()),
        'margin': margin,
        'is_decisive': margin > 1.0,
    }


def prediction_component_attribution(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1) -> dict:
    """Which components are responsible for the final prediction?

    Decomposes the logit of the top token by component.
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
    target_token = int(jnp.argmax(final_logits))
    target_dir = W_U[:, target_token]  # [d_model]

    # Embedding contribution
    embed = cache['hook_embed'][pos] + cache['hook_pos_embed'][pos]
    embed_contrib = float(jnp.dot(embed, target_dir))

    components = [{'component': 'embed', 'logit_contribution': embed_contrib}]

    for layer in range(n_layers):
        attn_out = cache[f'blocks.{layer}.hook_attn_out'][pos]
        mlp_out = cache[f'blocks.{layer}.hook_mlp_out'][pos]

        attn_contrib = float(jnp.dot(attn_out, target_dir))
        mlp_contrib = float(jnp.dot(mlp_out, target_dir))

        components.append({
            'component': f'L{layer}_attn',
            'logit_contribution': attn_contrib,
        })
        components.append({
            'component': f'L{layer}_mlp',
            'logit_contribution': mlp_contrib,
        })

    components.sort(key=lambda x: abs(x['logit_contribution']), reverse=True)
    total = sum(c['logit_contribution'] for c in components)

    return {
        'position': pos,
        'target_token': target_token,
        'target_logit': float(final_logits[target_token]),
        'components': components,
        'total_attribution': total,
        'top_component': components[0]['component'],
    }


def prediction_flip_sensitivity(model: HookedTransformer, tokens: jnp.ndarray, position: int = -1) -> dict:
    """How sensitive is the prediction to small perturbations?

    Adds noise to the final residual and checks if prediction flips.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[0]
    pos = position if position >= 0 else seq_len + position

    W_U = model.unembed.W_U
    b_U = model.unembed.b_U

    final_resid = cache[f'blocks.{n_layers - 1}.hook_resid_post']
    clean_logits = final_resid[pos] @ W_U + b_U
    clean_pred = int(jnp.argmax(clean_logits))
    clean_conf = float(jax.nn.softmax(clean_logits)[clean_pred])

    resid_norm = float(jnp.linalg.norm(final_resid[pos]))
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]

    per_noise = []
    for noise_frac in noise_levels:
        key = jax.random.PRNGKey(int(noise_frac * 1000))
        noise = jax.random.normal(key, final_resid[pos].shape) * resid_norm * noise_frac
        perturbed = final_resid[pos] + noise
        noisy_logits = perturbed @ W_U + b_U
        noisy_pred = int(jnp.argmax(noisy_logits))
        noisy_conf = float(jax.nn.softmax(noisy_logits)[noisy_pred])

        per_noise.append({
            'noise_fraction': noise_frac,
            'prediction': noisy_pred,
            'confidence': noisy_conf,
            'flipped': noisy_pred != clean_pred,
        })

    first_flip = None
    for entry in per_noise:
        if entry['flipped']:
            first_flip = entry['noise_fraction']
            break

    return {
        'position': pos,
        'clean_prediction': clean_pred,
        'clean_confidence': clean_conf,
        'per_noise_level': per_noise,
        'first_flip_noise': first_flip,
        'is_robust': first_flip is None or first_flip > 0.1,
    }
