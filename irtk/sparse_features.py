"""Sparse feature interpretation tools.

Analyze individual SAE features to understand what they represent,
how they interact, and how they affect model behavior:
- feature_activation_examples: Find top-activating token contexts
- feature_to_feature_correlation: Co-activation between features
- feature_circuit: Trace which upstream components drive a feature
- feature_token_bias: Which vocab tokens a feature responds to via W_E
- feature_downstream_effect: What output tokens a feature promotes/suppresses
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer
from irtk.sae import SparseAutoencoder


def feature_activation_examples(
    sae: SparseAutoencoder,
    model: HookedTransformer,
    token_sequences: list,
    feature_idx: int,
    hook_name: str,
    k: int = 20,
) -> list[dict]:
    """Find the top-k token contexts that maximally activate a feature.

    Args:
        sae: Trained SparseAutoencoder.
        model: HookedTransformer.
        token_sequences: List of token arrays to scan.
        feature_idx: Which feature to analyze.
        hook_name: Hook point where the SAE was trained.
        k: Number of top examples to return.

    Returns:
        List of dicts sorted by activation (descending), each with:
        - "prompt_idx": which token sequence
        - "position": sequence position
        - "activation": feature activation value
        - "tokens": the full token sequence
    """
    results = []
    for prompt_idx, tokens in enumerate(token_sequences):
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)
        if hook_name not in cache.cache_dict:
            continue
        acts = cache.cache_dict[hook_name]  # [seq_len, d_model]
        feat_acts = sae.encode(acts)  # [seq_len, n_features]
        feat_col = np.array(feat_acts[:, feature_idx])

        for pos in range(len(feat_col)):
            results.append({
                "prompt_idx": prompt_idx,
                "position": int(pos),
                "activation": float(feat_col[pos]),
                "tokens": np.array(tokens),
            })

    results.sort(key=lambda x: x["activation"], reverse=True)
    return results[:k]


def feature_to_feature_correlation(
    sae: SparseAutoencoder,
    model: HookedTransformer,
    token_sequences: list,
    feature_a: int,
    feature_b: int,
    hook_name: str,
) -> dict:
    """Measure co-activation between two features across a dataset.

    Args:
        sae: Trained SparseAutoencoder.
        model: HookedTransformer.
        token_sequences: List of token arrays.
        feature_a: First feature index.
        feature_b: Second feature index.
        hook_name: Hook point where the SAE was trained.

    Returns:
        Dict with:
        - "correlation": Pearson correlation between activations
        - "co_activation_rate": fraction of positions where both fire
        - "conditional_a_given_b": P(a active | b active)
        - "conditional_b_given_a": P(b active | a active)
    """
    all_a = []
    all_b = []

    for tokens in token_sequences:
        tokens = jnp.array(tokens)
        _, cache = model.run_with_cache(tokens)
        if hook_name not in cache.cache_dict:
            continue
        acts = cache.cache_dict[hook_name]
        feat_acts = np.array(sae.encode(acts))
        all_a.append(feat_acts[:, feature_a])
        all_b.append(feat_acts[:, feature_b])

    if not all_a:
        return {"correlation": 0.0, "co_activation_rate": 0.0,
                "conditional_a_given_b": 0.0, "conditional_b_given_a": 0.0}

    a = np.concatenate(all_a)
    b = np.concatenate(all_b)

    # Pearson correlation
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        corr = 0.0
    else:
        corr = float(np.corrcoef(a, b)[0, 1])

    # Co-activation
    a_active = a > 0
    b_active = b > 0
    both = np.sum(a_active & b_active)
    total = len(a)

    co_rate = float(both / max(total, 1))
    cond_a_b = float(both / max(np.sum(b_active), 1))
    cond_b_a = float(both / max(np.sum(a_active), 1))

    return {
        "correlation": corr,
        "co_activation_rate": co_rate,
        "conditional_a_given_b": cond_a_b,
        "conditional_b_given_a": cond_b_a,
    }


def feature_circuit(
    sae: SparseAutoencoder,
    model: HookedTransformer,
    tokens: jnp.ndarray,
    feature_idx: int,
    hook_name: str,
    threshold: float = 0.1,
) -> dict:
    """Trace which upstream heads/MLPs causally drive a feature.

    Ablates each attention and MLP output in turn and measures
    the change in feature activation.

    Args:
        sae: Trained SparseAutoencoder.
        model: HookedTransformer.
        tokens: Input tokens.
        feature_idx: Feature to trace.
        hook_name: Hook point where the SAE was trained.
        threshold: Minimum relative effect to include in circuit.

    Returns:
        Dict with:
        - "clean_activation": mean feature activation on clean input
        - "head_effects": dict of (layer, head) -> effect
        - "mlp_effects": dict of layer -> effect
        - "circuit_components": list of (component, effect) above threshold
    """
    tokens = jnp.array(tokens)

    # Clean activation
    _, clean_cache = model.run_with_cache(tokens)
    if hook_name not in clean_cache.cache_dict:
        return {"clean_activation": 0.0, "head_effects": {}, "mlp_effects": {},
                "circuit_components": []}

    clean_acts = clean_cache.cache_dict[hook_name]
    clean_feat = float(np.mean(np.array(sae.encode(clean_acts)[:, feature_idx])))

    head_effects = {}
    mlp_effects = {}

    # Parse hook_name to find which layer it's at
    # Only ablate layers upstream of the hook
    hook_layer = model.cfg.n_layers  # default: check all
    parts = hook_name.split(".")
    for i, p in enumerate(parts):
        if p == "blocks" and i + 1 < len(parts) and parts[i + 1].isdigit():
            hook_layer = int(parts[i + 1]) + 1
            break

    for layer in range(min(hook_layer, model.cfg.n_layers)):
        # Ablate attention
        attn_hook = f"blocks.{layer}.hook_attn_out"

        def zero_attn(x, name):
            return jnp.zeros_like(x)

        try:
            logits = model.run_with_hooks(tokens, fwd_hooks=[(attn_hook, zero_attn)])
            # Re-get activations at the hook point after ablation
            # We need to run with cache, but run_with_hooks doesn't return cache
            # Instead, re-run with hook that captures
            captured = {}
            def capture_hook(x, name):
                captured["act"] = x
                return x
            model.run_with_hooks(tokens, fwd_hooks=[
                (attn_hook, zero_attn),
                (hook_name, capture_hook),
            ])
            if "act" in captured:
                abl_feat = float(np.mean(np.array(sae.encode(captured["act"])[:, feature_idx])))
                effect = abl_feat - clean_feat
                for head in range(model.cfg.n_heads):
                    head_effects[(layer, head)] = effect / model.cfg.n_heads
        except Exception:
            pass

        # Ablate MLP
        mlp_hook = f"blocks.{layer}.hook_mlp_out"

        def zero_mlp(x, name):
            return jnp.zeros_like(x)

        try:
            captured = {}
            def capture_hook2(x, name):
                captured["act"] = x
                return x
            model.run_with_hooks(tokens, fwd_hooks=[
                (mlp_hook, zero_mlp),
                (hook_name, capture_hook2),
            ])
            if "act" in captured:
                abl_feat = float(np.mean(np.array(sae.encode(captured["act"])[:, feature_idx])))
                mlp_effects[layer] = abl_feat - clean_feat
        except Exception:
            pass

    # Collect circuit components above threshold
    circuit = []
    for (l, h), eff in head_effects.items():
        if abs(eff) > threshold * max(abs(clean_feat), 1e-10):
            circuit.append((f"L{l}H{h}", float(eff)))
    for l, eff in mlp_effects.items():
        if abs(eff) > threshold * max(abs(clean_feat), 1e-10):
            circuit.append((f"L{l}_MLP", float(eff)))

    circuit.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "clean_activation": clean_feat,
        "head_effects": {f"L{l}H{h}": float(e) for (l, h), e in head_effects.items()},
        "mlp_effects": {f"L{l}": float(e) for l, e in mlp_effects.items()},
        "circuit_components": circuit,
    }


def feature_token_bias(
    sae: SparseAutoencoder,
    model: HookedTransformer,
    feature_idx: int,
    k: int = 20,
) -> dict:
    """Find which vocabulary tokens most strongly activate a feature via embeddings.

    Computes W_E @ feature_direction to see which tokens are most aligned.

    Args:
        sae: Trained SparseAutoencoder.
        model: HookedTransformer.
        feature_idx: Feature to analyze.
        k: Number of top/bottom tokens to return.

    Returns:
        Dict with:
        - "top_positive": [(token_id, score), ...] most activating tokens
        - "top_negative": [(token_id, score), ...] most suppressed
        - "all_scores": [d_vocab] scores for all tokens
    """
    # Feature direction = encoder column (transposed from W_enc)
    feat_dir = np.array(sae.W_enc[:, feature_idx])  # [d_model]

    # Project through embedding matrix
    W_E = np.array(model.embed.W_E)  # [d_vocab, d_model]
    scores = W_E @ feat_dir  # [d_vocab]

    top_pos_idx = np.argsort(scores)[::-1][:k]
    top_neg_idx = np.argsort(scores)[:k]

    return {
        "top_positive": [(int(i), float(scores[i])) for i in top_pos_idx],
        "top_negative": [(int(i), float(scores[i])) for i in top_neg_idx],
        "all_scores": scores,
    }


def feature_downstream_effect(
    sae: SparseAutoencoder,
    model: HookedTransformer,
    tokens: jnp.ndarray,
    feature_idx: int,
    hook_name: str,
    k: int = 20,
) -> dict:
    """Measure a feature's contribution to output logits via ablation.

    Ablates the feature's contribution and measures the logit change.

    Args:
        sae: Trained SparseAutoencoder.
        model: HookedTransformer.
        tokens: Input tokens.
        feature_idx: Feature to analyze.
        hook_name: Hook point where the SAE operates.
        k: Number of top promoted/suppressed tokens.

    Returns:
        Dict with:
        - "top_promoted": [(token_id, logit_change), ...] tokens promoted by feature
        - "top_suppressed": [(token_id, logit_change), ...] tokens suppressed
        - "logit_diff_norm": L2 norm of logit change
    """
    tokens = jnp.array(tokens)

    # Clean logits
    clean_logits = np.array(model(tokens))

    # Ablate the feature: zero out its contribution in the activation
    _, cache = model.run_with_cache(tokens)
    if hook_name not in cache.cache_dict:
        return {"top_promoted": [], "top_suppressed": [], "logit_diff_norm": 0.0}

    clean_act = cache.cache_dict[hook_name]
    feat_acts = sae.encode(clean_act)  # [seq, n_features]

    # Reconstruct with feature zeroed
    zeroed_feat_acts = feat_acts.at[:, feature_idx].set(0.0)
    # Contribution of this feature
    feat_contribution = sae.decode(feat_acts) - sae.decode(zeroed_feat_acts)

    def subtract_feature(x, name):
        return x - feat_contribution

    ablated_logits = np.array(model.run_with_hooks(
        tokens, fwd_hooks=[(hook_name, subtract_feature)]
    ))

    # Logit difference at last position
    diff = clean_logits[-1] - ablated_logits[-1]  # positive = promoted by feature
    norm = float(np.linalg.norm(diff))

    top_promoted_idx = np.argsort(diff)[::-1][:k]
    top_suppressed_idx = np.argsort(diff)[:k]

    return {
        "top_promoted": [(int(i), float(diff[i])) for i in top_promoted_idx],
        "top_suppressed": [(int(i), float(diff[i])) for i in top_suppressed_idx],
        "logit_diff_norm": norm,
    }
