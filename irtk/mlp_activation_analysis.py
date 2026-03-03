"""MLP activation analysis.

Deep analysis of MLP neuron activations: distributions, dead/saturated neurons,
neuron-token correlations, and activation sparsity profiles.

References:
    Gurnee et al. (2023) "Finding Neurons in a Haystack"
    Bricken et al. (2023) "Towards Monosemanticity"
"""

import jax
import jax.numpy as jnp
import numpy as np


def mlp_activation_distribution(model, tokens):
    """Analyze activation distributions in each MLP layer.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            mean_activation: [n_layers] mean activation magnitude per layer
            std_activation: [n_layers] activation std per layer
            max_activation: [n_layers] max activation per layer
            sparsity: [n_layers] fraction of near-zero activations per layer
            kurtosis: [n_layers] kurtosis (peakedness) per layer
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    means = np.zeros(n_layers)
    stds = np.zeros(n_layers)
    maxs = np.zeros(n_layers)
    sparsity = np.zeros(n_layers)
    kurt = np.zeros(n_layers)

    for layer in range(n_layers):
        key = f"blocks.{layer}.mlp.hook_post"
        act = cache.get(key)
        if act is None:
            continue

        a = np.array(act).flatten()
        means[layer] = float(np.mean(np.abs(a)))
        stds[layer] = float(np.std(a))
        maxs[layer] = float(np.max(np.abs(a)))
        sparsity[layer] = float(np.mean(np.abs(a) < 0.01))

        # Kurtosis
        if stds[layer] > 1e-10:
            centered = a - np.mean(a)
            kurt[layer] = float(np.mean(centered ** 4) / (stds[layer] ** 4) - 3)

    return {
        "mean_activation": means,
        "std_activation": stds,
        "max_activation": maxs,
        "sparsity": sparsity,
        "kurtosis": kurt,
    }


def dead_neuron_analysis(model, tokens_list, threshold=0.01):
    """Identify dead neurons (never activate above threshold).

    Args:
        model: HookedTransformer model.
        tokens_list: List of input token arrays to test.
        threshold: Activation threshold for "alive".

    Returns:
        dict with:
            dead_fraction: [n_layers] fraction of dead neurons per layer
            dead_neurons: dict of layer -> list of dead neuron indices
            total_dead: int
            total_neurons: int
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    d_mlp = model.cfg.d_model * 4  # standard MLP expansion

    # Track max activation per neuron
    max_acts = {}
    for layer in range(n_layers):
        max_acts[layer] = np.zeros(d_mlp)

    for tokens in tokens_list:
        cache_state = HookState(hook_fns={}, cache={})
        model(tokens, hook_state=cache_state)
        cache = cache_state.cache

        for layer in range(n_layers):
            key = f"blocks.{layer}.mlp.hook_post"
            act = cache.get(key)
            if act is not None:
                a = np.array(act)  # [seq_len, d_mlp]
                layer_max = np.max(np.abs(a), axis=0)
                actual_d = min(len(layer_max), d_mlp)
                max_acts[layer][:actual_d] = np.maximum(max_acts[layer][:actual_d], layer_max[:actual_d])

    dead_frac = np.zeros(n_layers)
    dead_neurons = {}
    total_dead = 0

    for layer in range(n_layers):
        dead = np.where(max_acts[layer] < threshold)[0]
        dead_frac[layer] = len(dead) / d_mlp
        dead_neurons[layer] = dead.tolist()
        total_dead += len(dead)

    return {
        "dead_fraction": dead_frac,
        "dead_neurons": dead_neurons,
        "total_dead": total_dead,
        "total_neurons": n_layers * d_mlp,
    }


def neuron_token_correlation(model, tokens, layer, top_k=5):
    """Correlate neuron activations with input tokens.

    For each neuron, find which input tokens cause the highest activation.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        layer: MLP layer to analyze.
        top_k: Top neurons to report.

    Returns:
        dict with:
            neuron_activations: [d_mlp, seq_len] activation per neuron per position
            top_neurons: list of (neuron_idx, max_activation, max_position)
            position_means: [seq_len] mean activation per position
    """
    from irtk.hook_points import HookState

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    key = f"blocks.{layer}.mlp.hook_post"
    act = cache.get(key)

    if act is None:
        seq_len = len(tokens)
        return {
            "neuron_activations": np.zeros((0, seq_len)),
            "top_neurons": [],
            "position_means": np.zeros(seq_len),
        }

    a = np.array(act)  # [seq_len, d_mlp]
    activations = a.T  # [d_mlp, seq_len]

    # Top neurons by max activation
    max_per_neuron = np.max(np.abs(activations), axis=1)
    top_idx = np.argsort(-max_per_neuron)[:top_k]
    top_neurons = []
    for idx in top_idx:
        max_pos = int(np.argmax(np.abs(activations[idx])))
        top_neurons.append((int(idx), float(max_per_neuron[idx]), max_pos))

    pos_means = np.mean(np.abs(a), axis=1)

    return {
        "neuron_activations": activations,
        "top_neurons": top_neurons,
        "position_means": pos_means,
    }


def activation_sparsity_profile(model, tokens):
    """Profile activation sparsity across layers and positions.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            layer_sparsity: [n_layers] sparsity per layer
            position_sparsity: [n_layers, seq_len] sparsity per position per layer
            mean_active_neurons: [n_layers] average active neurons per position
            effective_width: [n_layers] estimated effective MLP width
    """
    from irtk.hook_points import HookState

    n_layers = model.cfg.n_layers
    seq_len = len(tokens)

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    layer_sp = np.zeros(n_layers)
    pos_sp = np.zeros((n_layers, seq_len))
    mean_active = np.zeros(n_layers)
    eff_width = np.zeros(n_layers)

    for layer in range(n_layers):
        key = f"blocks.{layer}.mlp.hook_post"
        act = cache.get(key)
        if act is None:
            continue

        a = np.array(act)  # [seq_len, d_mlp]
        d_mlp = a.shape[1]

        # Per position sparsity
        for p in range(seq_len):
            pos_sp[layer, p] = float(np.mean(np.abs(a[p]) < 0.01))

        layer_sp[layer] = float(np.mean(np.abs(a) < 0.01))
        active_per_pos = np.sum(np.abs(a) >= 0.01, axis=1)
        mean_active[layer] = float(np.mean(active_per_pos))

        # Effective width via L1/L2 norm ratio
        norms_l1 = np.sum(np.abs(a), axis=1)
        norms_l2 = np.sqrt(np.sum(a ** 2, axis=1))
        ratios = norms_l1 / (norms_l2 + 1e-10)
        eff_width[layer] = float(np.mean(ratios ** 2))

    return {
        "layer_sparsity": layer_sp,
        "position_sparsity": pos_sp,
        "mean_active_neurons": mean_active,
        "effective_width": eff_width,
    }


def neuron_logit_attribution(model, tokens, layer, pos=-1, top_k=5):
    """Attribute logit changes to individual MLP neurons.

    Projects each neuron's contribution through the output weight and unembedding.

    Args:
        model: HookedTransformer model.
        tokens: Input token IDs [seq_len].
        layer: MLP layer.
        pos: Position.
        top_k: Top neurons and tokens.

    Returns:
        dict with:
            neuron_logit_effects: [d_mlp, d_vocab] logit effect per neuron per token
            top_neuron_token_pairs: list of (neuron, token, logit)
            neuron_total_effects: [d_mlp] total logit magnitude per neuron
    """
    from irtk.hook_points import HookState

    cache_state = HookState(hook_fns={}, cache={})
    model(tokens, hook_state=cache_state)
    cache = cache_state.cache

    key = f"blocks.{layer}.mlp.hook_post"
    act = cache.get(key)

    W_out = np.array(model.blocks[layer].mlp.W_out)  # [d_mlp, d_model]
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]

    if act is None:
        d_mlp = W_out.shape[0]
        d_vocab = W_U.shape[1]
        return {
            "neuron_logit_effects": np.zeros((d_mlp, d_vocab)),
            "top_neuron_token_pairs": [],
            "neuron_total_effects": np.zeros(d_mlp),
        }

    a = np.array(act[pos])  # [d_mlp]
    d_mlp = len(a)
    d_vocab = W_U.shape[1]

    # Per-neuron logit contribution: activation * W_out_row @ W_U
    neuron_logits = np.zeros((d_mlp, d_vocab))
    for n in range(d_mlp):
        neuron_logits[n] = a[n] * (W_out[n] @ W_U)

    total_effects = np.sum(np.abs(neuron_logits), axis=1)

    # Top pairs
    flat = neuron_logits.flatten()
    top_flat_idx = np.argsort(-np.abs(flat))[:top_k * 2]
    pairs = []
    for idx in top_flat_idx:
        n, t = np.unravel_index(idx, neuron_logits.shape)
        pairs.append((int(n), int(t), float(neuron_logits[n, t])))

    return {
        "neuron_logit_effects": neuron_logits,
        "top_neuron_token_pairs": pairs,
        "neuron_total_effects": total_effects,
    }
