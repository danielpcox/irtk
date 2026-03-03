"""MLP memory analysis: analyze MLPs as key-value memory stores."""

import jax
import jax.numpy as jnp
from irtk import HookedTransformer


def mlp_key_value_decomposition(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """Decompose MLP as key-value memory: W_in rows are keys, W_out columns are values.

    Analyzes which keys are activated and what values they retrieve.
    """
    _, cache = model.run_with_cache(tokens)

    pre_key = f'blocks.{layer}.mlp.hook_pre'
    post_key = f'blocks.{layer}.mlp.hook_post'

    pre_acts = cache[pre_key]   # [seq, d_mlp]
    post_acts = cache[post_key]  # [seq, d_mlp]

    seq_len = pre_acts.shape[0]
    d_mlp = pre_acts.shape[1]

    # Key activation statistics
    mean_pre = jnp.mean(jnp.abs(pre_acts), axis=0)  # [d_mlp]
    mean_post = jnp.mean(post_acts, axis=0)  # [d_mlp]

    # How many neurons are active (post > 0)
    active_mask = post_acts > 0  # [seq, d_mlp]
    active_per_position = jnp.sum(active_mask, axis=-1)  # [seq]
    active_fraction = float(jnp.mean(active_per_position) / d_mlp)

    # Per-position activation density
    per_position = []
    for pos in range(seq_len):
        n_active = int(active_per_position[pos])
        mean_activation = float(jnp.mean(post_acts[pos][active_mask[pos]])) if n_active > 0 else 0.0
        per_position.append({
            'position': pos,
            'n_active_neurons': n_active,
            'active_fraction': n_active / d_mlp,
            'mean_activation': mean_activation,
        })

    return {
        'layer': layer,
        'n_neurons': d_mlp,
        'mean_active_fraction': active_fraction,
        'per_position': per_position,
    }


def mlp_retrieval_pattern(model: HookedTransformer, tokens: jnp.ndarray, layer: int, top_k: int = 10) -> dict:
    """What does the MLP retrieve (write to residual stream) at each position?

    Analyzes the MLP output direction and its vocabulary effect.
    """
    _, cache = model.run_with_cache(tokens)

    mlp_out_key = f'blocks.{layer}.hook_mlp_out'
    mlp_out = cache[mlp_out_key]  # [seq, d_model]
    seq_len = mlp_out.shape[0]

    W_U = model.unembed.W_U  # [d_model, d_vocab]

    per_position = []
    for pos in range(seq_len):
        output = mlp_out[pos]
        output_norm = float(jnp.linalg.norm(output))

        # Project to vocabulary
        vocab_logits = output @ W_U  # [d_vocab]
        top_promoted = jnp.argsort(vocab_logits)[-top_k:][::-1]
        top_suppressed = jnp.argsort(vocab_logits)[:top_k]

        per_position.append({
            'position': pos,
            'output_norm': output_norm,
            'top_promoted': [int(t) for t in top_promoted],
            'top_suppressed': [int(t) for t in top_suppressed],
            'max_logit': float(vocab_logits[top_promoted[0]]),
            'min_logit': float(vocab_logits[top_suppressed[0]]),
        })

    return {
        'layer': layer,
        'per_position': per_position,
        'mean_output_norm': float(jnp.mean(jnp.linalg.norm(mlp_out, axis=-1))),
    }


def mlp_storage_capacity(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """Estimate MLP storage capacity: effective rank and utilization.

    Higher effective rank = more independent directions being used.
    """
    _, cache = model.run_with_cache(tokens)

    post_key = f'blocks.{layer}.mlp.hook_post'
    post_acts = cache[post_key]  # [seq, d_mlp]
    d_mlp = post_acts.shape[1]

    # Active neurons across all positions
    active_ever = jnp.any(post_acts > 0, axis=0)  # [d_mlp]
    n_ever_active = int(jnp.sum(active_ever))

    # Effective rank of activations via SVD
    centered = post_acts - jnp.mean(post_acts, axis=0)
    svd_vals = jnp.linalg.svd(centered, compute_uv=False)
    svd_vals = svd_vals / (jnp.sum(svd_vals) + 1e-10)
    effective_rank = float(jnp.exp(-jnp.sum(svd_vals * jnp.log(svd_vals + 1e-10))))

    # Neuron reuse: how many positions activate each neuron
    active_mask = post_acts > 0
    reuse_counts = jnp.sum(active_mask, axis=0)  # [d_mlp]
    mean_reuse = float(jnp.mean(reuse_counts[active_ever]))

    return {
        'layer': layer,
        'n_neurons': d_mlp,
        'n_ever_active': n_ever_active,
        'utilization': n_ever_active / d_mlp,
        'effective_rank': effective_rank,
        'mean_neuron_reuse': mean_reuse,
    }


def mlp_input_selectivity(model: HookedTransformer, tokens: jnp.ndarray, layer: int, top_k: int = 10) -> dict:
    """Which neurons are most input-selective (respond to specific positions)?

    High selectivity = neuron activates strongly at few positions.
    """
    _, cache = model.run_with_cache(tokens)

    post_key = f'blocks.{layer}.mlp.hook_post'
    post_acts = cache[post_key]  # [seq, d_mlp]
    seq_len, d_mlp = post_acts.shape

    # Selectivity for each neuron: 1 - (entropy / max_entropy)
    act_positive = jnp.maximum(post_acts, 0)
    act_sums = jnp.sum(act_positive, axis=0) + 1e-10  # [d_mlp]
    act_probs = act_positive / act_sums[None, :]  # [seq, d_mlp]
    entropies = -jnp.sum(act_probs * jnp.log(act_probs + 1e-10), axis=0)  # [d_mlp]
    max_entropy = float(jnp.log(seq_len))
    selectivities = 1.0 - entropies / (max_entropy + 1e-10)

    # Top-k most selective
    mean_acts = jnp.mean(post_acts, axis=0)
    active_mask = mean_acts > 0

    top_idx = jnp.argsort(selectivities * active_mask.astype(jnp.float32))[-top_k:][::-1]

    per_neuron = []
    for idx in top_idx:
        idx_int = int(idx)
        peak_pos = int(jnp.argmax(post_acts[:, idx_int]))
        per_neuron.append({
            'neuron_idx': idx_int,
            'selectivity': float(selectivities[idx_int]),
            'peak_position': peak_pos,
            'peak_activation': float(post_acts[peak_pos, idx_int]),
            'mean_activation': float(mean_acts[idx_int]),
        })

    n_selective = int(jnp.sum((selectivities > 0.5) & active_mask))

    return {
        'layer': layer,
        'per_neuron': per_neuron,
        'n_selective': n_selective,
        'mean_selectivity': float(jnp.mean(selectivities[active_mask])) if jnp.any(active_mask) else 0.0,
    }


def mlp_write_read_alignment(model: HookedTransformer, tokens: jnp.ndarray, layer: int) -> dict:
    """How well does the MLP's write (output) align with what later layers read?

    Compares MLP output direction with the residual stream's subsequent changes.
    """
    _, cache = model.run_with_cache(tokens)
    n_layers = model.cfg.n_layers

    mlp_out_key = f'blocks.{layer}.hook_mlp_out'
    mlp_out = cache[mlp_out_key]  # [seq, d_model]

    # Compare with subsequent layer residual changes
    per_target = []
    for target_layer in range(layer + 1, n_layers):
        resid_pre = cache[f'blocks.{target_layer}.hook_resid_pre']
        resid_post = cache[f'blocks.{target_layer}.hook_resid_post']
        change = resid_post - resid_pre  # [seq, d_model]

        # Cosine alignment per position, then average
        mlp_normed = mlp_out / (jnp.linalg.norm(mlp_out, axis=-1, keepdims=True) + 1e-10)
        change_normed = change / (jnp.linalg.norm(change, axis=-1, keepdims=True) + 1e-10)
        alignment = jnp.sum(mlp_normed * change_normed, axis=-1)  # [seq]
        mean_alignment = float(jnp.mean(alignment))

        per_target.append({
            'target_layer': target_layer,
            'mean_alignment': mean_alignment,
            'is_read': mean_alignment > 0.1,
        })

    n_readers = sum(1 for p in per_target if p['is_read'])

    return {
        'source_layer': layer,
        'per_target': per_target,
        'n_reading_layers': n_readers,
        'mean_downstream_alignment': sum(p['mean_alignment'] for p in per_target) / len(per_target) if per_target else 0.0,
    }
