"""Model comparison analysis.

Compare two models structurally and behaviorally: weight distances,
activation divergence, prediction agreement, and circuit similarity.
"""

import jax
import jax.numpy as jnp
import numpy as np


def weight_distance(model_a, model_b):
    """Compare weight matrices between two models.

    Args:
        model_a: First HookedTransformer model.
        model_b: Second HookedTransformer model.

    Returns:
        dict with:
            layer_distances: dict of component_name -> L2 distance
            total_distance: float (sum of all distances)
            max_distance_component: str
            relative_distances: dict of component_name -> distance / norm_a
    """
    n_layers = model_a.cfg.n_layers

    distances = {}
    relative = {}

    # Embeddings
    w_a = np.array(model_a.embed.W_E)
    w_b = np.array(model_b.embed.W_E)
    d = float(np.linalg.norm(w_a - w_b))
    n = float(np.linalg.norm(w_a))
    distances["embed"] = d
    relative["embed"] = d / n if n > 1e-10 else 0.0

    # Unembed
    w_a = np.array(model_a.unembed.W_U)
    w_b = np.array(model_b.unembed.W_U)
    d = float(np.linalg.norm(w_a - w_b))
    n = float(np.linalg.norm(w_a))
    distances["unembed"] = d
    relative["unembed"] = d / n if n > 1e-10 else 0.0

    for layer in range(n_layers):
        for comp, getter in [
            (f"attn_W_Q_L{layer}", lambda m, l=layer: np.array(m.blocks[l].attn.W_Q)),
            (f"attn_W_K_L{layer}", lambda m, l=layer: np.array(m.blocks[l].attn.W_K)),
            (f"attn_W_V_L{layer}", lambda m, l=layer: np.array(m.blocks[l].attn.W_V)),
            (f"attn_W_O_L{layer}", lambda m, l=layer: np.array(m.blocks[l].attn.W_O)),
            (f"mlp_W_in_L{layer}", lambda m, l=layer: np.array(m.blocks[l].mlp.W_in)),
            (f"mlp_W_out_L{layer}", lambda m, l=layer: np.array(m.blocks[l].mlp.W_out)),
        ]:
            w_a = getter(model_a)
            w_b = getter(model_b)
            d = float(np.linalg.norm(w_a - w_b))
            n = float(np.linalg.norm(w_a))
            distances[comp] = d
            relative[comp] = d / n if n > 1e-10 else 0.0

    total = sum(distances.values())
    max_comp = max(distances, key=distances.get) if distances else ""

    return {
        "layer_distances": distances,
        "total_distance": total,
        "max_distance_component": max_comp,
        "relative_distances": relative,
    }


def activation_divergence(model_a, model_b, tokens):
    """Compare activations between two models on the same input.

    Args:
        model_a: First model.
        model_b: Second model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            layer_divergence: [n_layers] L2 norm of activation difference per layer
            cosine_similarity: [n_layers] cosine similarity per layer
            max_divergence_layer: int
            logit_divergence: float (L2 distance of output logits)
    """
    from irtk.hook_points import HookState

    n_layers = model_a.cfg.n_layers

    cache_a = HookState(hook_fns={}, cache={})
    cache_b = HookState(hook_fns={}, cache={})

    logits_a = np.array(model_a(tokens, hook_state=cache_a))
    logits_b = np.array(model_b(tokens, hook_state=cache_b))

    divergence = np.zeros(n_layers)
    cosine = np.zeros(n_layers)

    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        ra = cache_a.cache.get(key)
        rb = cache_b.cache.get(key)
        if ra is not None and rb is not None:
            a = np.array(ra).flatten()
            b = np.array(rb).flatten()
            divergence[layer] = float(np.linalg.norm(a - b))
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na > 1e-10 and nb > 1e-10:
                cosine[layer] = float(np.dot(a, b) / (na * nb))

    logit_div = float(np.linalg.norm(logits_a - logits_b))
    max_div_layer = int(np.argmax(divergence))

    return {
        "layer_divergence": divergence,
        "cosine_similarity": cosine,
        "max_divergence_layer": max_div_layer,
        "logit_divergence": logit_div,
    }


def prediction_agreement(model_a, model_b, tokens_list, top_k=1):
    """Compare predictions between two models across inputs.

    Args:
        model_a: First model.
        model_b: Second model.
        tokens_list: List of input token arrays.
        top_k: Number of top predictions to compare.

    Returns:
        dict with:
            agreement_rate: float (fraction of inputs with same top-1 prediction)
            top_k_overlap: float (mean overlap of top-k predictions)
            kl_divergences: list of float (KL(a||b) per input)
            mean_kl: float
    """
    agreements = 0
    overlaps = []
    kls = []

    for tokens in tokens_list:
        logits_a = np.array(model_a(tokens))[-1]
        logits_b = np.array(model_b(tokens))[-1]

        # Top-1
        pred_a = int(np.argmax(logits_a))
        pred_b = int(np.argmax(logits_b))
        if pred_a == pred_b:
            agreements += 1

        # Top-k overlap
        top_a = set(np.argsort(-logits_a)[:top_k].tolist())
        top_b = set(np.argsort(-logits_b)[:top_k].tolist())
        overlap = len(top_a & top_b) / top_k
        overlaps.append(overlap)

        # KL divergence
        probs_a = np.exp(logits_a - logits_a.max())
        probs_a = probs_a / probs_a.sum()
        probs_b = np.exp(logits_b - logits_b.max())
        probs_b = probs_b / probs_b.sum()
        kl = float(np.sum(probs_a * np.log((probs_a + 1e-10) / (probs_b + 1e-10))))
        kls.append(kl)

    n = max(len(tokens_list), 1)

    return {
        "agreement_rate": agreements / n,
        "top_k_overlap": float(np.mean(overlaps)),
        "kl_divergences": kls,
        "mean_kl": float(np.mean(kls)),
    }


def attention_pattern_comparison(model_a, model_b, tokens):
    """Compare attention patterns between two models.

    Args:
        model_a: First model.
        model_b: Second model.
        tokens: Input token IDs [seq_len].

    Returns:
        dict with:
            pattern_distances: [n_layers, n_heads] L2 distance per head
            pattern_cosine: [n_layers, n_heads] cosine similarity per head
            most_different_head: tuple (layer, head)
            mean_distance: float
    """
    from irtk.hook_points import HookState

    n_layers = model_a.cfg.n_layers
    n_heads = model_a.cfg.n_heads

    cache_a = HookState(hook_fns={}, cache={})
    cache_b = HookState(hook_fns={}, cache={})
    model_a(tokens, hook_state=cache_a)
    model_b(tokens, hook_state=cache_b)

    distances = np.zeros((n_layers, n_heads))
    cosines = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        key = f"blocks.{layer}.attn.hook_pattern"
        pa = cache_a.cache.get(key)
        pb = cache_b.cache.get(key)
        if pa is not None and pb is not None:
            for head in range(n_heads):
                a = np.array(pa[head]).flatten()
                b = np.array(pb[head]).flatten()
                distances[layer, head] = float(np.linalg.norm(a - b))
                na = np.linalg.norm(a)
                nb = np.linalg.norm(b)
                if na > 1e-10 and nb > 1e-10:
                    cosines[layer, head] = float(np.dot(a, b) / (na * nb))

    flat_idx = np.argmax(distances.flatten())
    most_diff = np.unravel_index(flat_idx, distances.shape)

    return {
        "pattern_distances": distances,
        "pattern_cosine": cosines,
        "most_different_head": (int(most_diff[0]), int(most_diff[1])),
        "mean_distance": float(np.mean(distances)),
    }


def component_importance_comparison(model_a, model_b, tokens, metric_fn):
    """Compare component importance rankings between two models.

    Args:
        model_a: First model.
        model_b: Second model.
        tokens: Input token IDs [seq_len].
        metric_fn: Function mapping logits to scalar metric.

    Returns:
        dict with:
            importance_a: dict of component -> importance
            importance_b: dict of component -> importance
            rank_correlation: float (Spearman correlation of importance rankings)
            biggest_rank_changes: list of (component, rank_a, rank_b)
    """
    from irtk.hook_points import HookState

    n_layers = model_a.cfg.n_layers

    def get_importance(model):
        base = metric_fn(np.array(model(tokens)))
        imp = {}
        for layer in range(n_layers):
            for comp_type, hook_key in [("attn", f"blocks.{layer}.hook_attn_out"),
                                         ("mlp", f"blocks.{layer}.hook_mlp_out")]:
                name = f"{comp_type}_L{layer}"
                state = HookState(hook_fns={hook_key: lambda x, n: jnp.zeros_like(x)}, cache={})
                abl = metric_fn(np.array(model(tokens, hook_state=state)))
                imp[name] = abs(float(base - abl))
        return imp

    imp_a = get_importance(model_a)
    imp_b = get_importance(model_b)

    # Rank correlation
    components = sorted(imp_a.keys())
    ranks_a = np.argsort(np.argsort([-imp_a[c] for c in components]))
    ranks_b = np.argsort(np.argsort([-imp_b[c] for c in components]))

    n = len(components)
    if n > 1:
        d_sq = np.sum((ranks_a - ranks_b) ** 2)
        rank_corr = 1 - 6 * d_sq / (n * (n ** 2 - 1))
    else:
        rank_corr = 1.0

    # Biggest rank changes
    changes = [(components[i], int(ranks_a[i]), int(ranks_b[i]))
               for i in range(n)]
    changes.sort(key=lambda x: -abs(x[1] - x[2]))

    return {
        "importance_a": imp_a,
        "importance_b": imp_b,
        "rank_correlation": float(rank_corr),
        "biggest_rank_changes": changes[:10],
    }
