"""Internal confidence analysis.

Probes how models represent their own epistemic uncertainty internally
in the residual stream, distinct from output-level entropy. A model may
have low output entropy (confident prediction) while having high internal
uncertainty (residual stream still exploring alternatives), or vice versa.

Functions:
- confidence_direction: Extract a linear direction predicting confidence
- internal_vs_output_confidence_gap: Gap between internal and output confidence
- confidence_accumulation_profile: How confidence builds through layers
- uncertainty_decomposition: Decompose residual variance into uncertainty axes
- self_consistency_probe: Are internal representations consistent across paraphrases

References:
    - Kadavath et al. (2022) "Language Models (Mostly) Know What They Know"
    - Burns et al. (2022) "Discovering Latent Knowledge in Language Models"
    - Slobodkin et al. (2023) "On Internal Confidence Signals"
"""

from typing import Optional, Callable

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def confidence_direction(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    correct_token: int,
    hook_name: str,
    pos: int = -1,
) -> dict:
    """Extract a direction in residual stream that predicts output confidence.

    Uses the difference between the residual representation and its projection
    onto the unembedding direction for the correct token to estimate a
    confidence direction.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        correct_token: The expected next token.
        hook_name: Hook point to analyze.
        pos: Token position (-1 = last).

    Returns:
        Dict with:
            "direction": [d_model] confidence direction vector
            "confidence_score": projection of residual onto this direction
            "output_prob": probability assigned to correct_token
            "direction_norm": norm of the confidence direction
    """
    _, cache = model.run_with_cache(tokens)
    logits = model(tokens)

    if hook_name not in cache.cache_dict:
        return {
            "direction": np.zeros(model.cfg.d_model),
            "confidence_score": 0.0,
            "output_prob": 0.0,
            "direction_norm": 0.0,
        }

    resid = cache.cache_dict[hook_name]  # [seq, d_model]
    resid_vec = np.array(resid[pos])  # [d_model]

    # Output probability for correct token
    last_logits = logits[pos]
    probs = np.array(jax.nn.softmax(last_logits))
    output_prob = float(probs[correct_token])

    # Unembedding direction for correct token
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]
    correct_dir = W_U[:, correct_token]
    correct_dir_norm = correct_dir / (np.linalg.norm(correct_dir) + 1e-10)

    # Confidence direction: component of residual along unembedding direction
    # weighted by how much of the logit this position contributes
    conf_dir = correct_dir_norm * output_prob
    conf_score = float(np.dot(resid_vec, correct_dir_norm))

    return {
        "direction": conf_dir,
        "confidence_score": conf_score,
        "output_prob": output_prob,
        "direction_norm": float(np.linalg.norm(conf_dir)),
    }


def internal_vs_output_confidence_gap(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    correct_token: int,
    hook_names: Optional[list] = None,
    pos: int = -1,
) -> dict:
    """Measure gap between internal confidence and output probability at each layer.

    At each layer, computes:
    - Internal confidence: projection onto unembedding direction (logit lens)
    - Output probability: softmax of the final logits

    The gap reveals where the model "knows" but hasn't yet committed.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        correct_token: The expected next token.
        hook_names: Hook points to analyze. Defaults to all resid_post.
        pos: Token position (-1 = last).

    Returns:
        Dict with:
            "layer_internal_conf": [n_layers] internal confidence scores
            "output_prob": final output probability for correct_token
            "gaps": [n_layers] internal_conf - output_prob
            "max_gap_layer": layer with largest gap
    """
    _, cache = model.run_with_cache(tokens)
    logits = model(tokens)

    if hook_names is None:
        hook_names = [f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)]

    # Output probability
    probs = np.array(jax.nn.softmax(logits[pos]))
    output_prob = float(probs[correct_token])

    # Unembedding direction
    W_U = np.array(model.unembed.W_U)  # [d_model, d_vocab]
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') else np.zeros(W_U.shape[1])

    internal_confs = []
    for name in hook_names:
        if name not in cache.cache_dict:
            internal_confs.append(0.0)
            continue

        resid = np.array(cache.cache_dict[name][pos])  # [d_model]
        # Logit lens: project through unembedding
        layer_logits = resid @ W_U + b_U
        layer_probs = np.exp(layer_logits - np.max(layer_logits))
        layer_probs = layer_probs / np.sum(layer_probs)
        internal_confs.append(float(layer_probs[correct_token]))

    internal_confs = np.array(internal_confs)
    gaps = internal_confs - output_prob

    max_gap_layer = int(np.argmax(np.abs(gaps))) if len(gaps) > 0 else 0

    return {
        "layer_internal_conf": internal_confs,
        "output_prob": output_prob,
        "gaps": gaps,
        "max_gap_layer": max_gap_layer,
    }


def confidence_accumulation_profile(
    model: HookedTransformer,
    token_sequences: list,
    correct_tokens: list,
    hook_names: Optional[list] = None,
    pos: int = -1,
) -> dict:
    """Compute how confidence builds across layers, averaged over examples.

    Args:
        model: HookedTransformer.
        token_sequences: List of [seq_len] token arrays.
        correct_tokens: List of correct next-token indices.
        hook_names: Hook points to analyze. Defaults to all resid_post.
        pos: Token position (-1 = last).

    Returns:
        Dict with:
            "mean_confidence": [n_layers] mean internal confidence per layer
            "std_confidence": [n_layers] std across examples
            "monotonicity": fraction of layers where confidence increases
            "final_mean_prob": mean output probability across examples
    """
    if hook_names is None:
        hook_names = [f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)]

    if not token_sequences:
        n = len(hook_names)
        return {
            "mean_confidence": np.zeros(n),
            "std_confidence": np.zeros(n),
            "monotonicity": 0.0,
            "final_mean_prob": 0.0,
        }

    all_confs = []
    final_probs = []

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') else np.zeros(W_U.shape[1])

    for tokens, correct in zip(token_sequences, correct_tokens):
        _, cache = model.run_with_cache(tokens)
        logits = model(tokens)

        probs = np.array(jax.nn.softmax(logits[pos]))
        final_probs.append(float(probs[correct]))

        layer_confs = []
        for name in hook_names:
            if name not in cache.cache_dict:
                layer_confs.append(0.0)
                continue
            resid = np.array(cache.cache_dict[name][pos])
            layer_logits = resid @ W_U + b_U
            layer_probs = np.exp(layer_logits - np.max(layer_logits))
            layer_probs = layer_probs / np.sum(layer_probs)
            layer_confs.append(float(layer_probs[correct]))

        all_confs.append(layer_confs)

    all_confs = np.array(all_confs)
    mean_conf = np.mean(all_confs, axis=0)
    std_conf = np.std(all_confs, axis=0)

    # Monotonicity: fraction of consecutive layer pairs where confidence increases
    if len(mean_conf) >= 2:
        diffs = np.diff(mean_conf)
        monotonicity = float(np.mean(diffs >= 0))
    else:
        monotonicity = 1.0

    return {
        "mean_confidence": mean_conf,
        "std_confidence": std_conf,
        "monotonicity": monotonicity,
        "final_mean_prob": float(np.mean(final_probs)),
    }


def uncertainty_decomposition(
    model: HookedTransformer,
    token_sequences: list,
    hook_name: str,
    pos: int = -1,
    top_k: int = 5,
) -> dict:
    """Decompose residual stream variance into uncertainty-related directions.

    Performs PCA on the residual representations and correlates each
    component with output entropy.

    Args:
        model: HookedTransformer.
        token_sequences: List of [seq_len] token arrays.
        hook_name: Hook point to analyze.
        pos: Token position (-1 = last).
        top_k: Number of top components to analyze.

    Returns:
        Dict with:
            "uncertainty_axes": [top_k, d_model] principal directions
            "explained_variance": [top_k] variance explained by each axis
            "entropy_correlations": [top_k] correlation with output entropy
            "total_variance": total variance in representations
    """
    if not token_sequences:
        d = model.cfg.d_model
        return {
            "uncertainty_axes": np.zeros((top_k, d)),
            "explained_variance": np.zeros(top_k),
            "entropy_correlations": np.zeros(top_k),
            "total_variance": 0.0,
        }

    representations = []
    entropies = []

    for tokens in token_sequences:
        _, cache = model.run_with_cache(tokens)
        logits = model(tokens)

        if hook_name in cache.cache_dict:
            resid = np.array(cache.cache_dict[hook_name][pos])
            representations.append(resid)

            # Output entropy
            probs = np.array(jax.nn.softmax(logits[pos]))
            entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
            entropies.append(entropy)

    if len(representations) < 2:
        d = model.cfg.d_model
        return {
            "uncertainty_axes": np.zeros((top_k, d)),
            "explained_variance": np.zeros(top_k),
            "entropy_correlations": np.zeros(top_k),
            "total_variance": 0.0,
        }

    X = np.array(representations)  # [n_samples, d_model]
    entropies = np.array(entropies)

    # Center
    X_centered = X - np.mean(X, axis=0, keepdims=True)

    # PCA via SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    total_var = float(np.sum(S ** 2))

    k = min(top_k, len(S))
    axes = Vt[:k]  # [k, d_model]
    variance = S[:k] ** 2

    # Correlation of each component with entropy
    projections = X_centered @ axes.T  # [n_samples, k]
    correlations = np.zeros(k)
    for i in range(k):
        if np.std(projections[:, i]) > 1e-8 and np.std(entropies) > 1e-8:
            corr = np.corrcoef(projections[:, i], entropies)[0, 1]
            correlations[i] = corr if not np.isnan(corr) else 0.0

    # Pad if fewer than top_k components
    if k < top_k:
        d = model.cfg.d_model
        axes = np.concatenate([axes, np.zeros((top_k - k, d))], axis=0)
        variance = np.concatenate([variance, np.zeros(top_k - k)])
        correlations = np.concatenate([correlations, np.zeros(top_k - k)])

    return {
        "uncertainty_axes": axes,
        "explained_variance": variance,
        "entropy_correlations": correlations,
        "total_variance": total_var,
    }


def self_consistency_probe(
    model: HookedTransformer,
    tokens_a: jnp.ndarray,
    tokens_b: jnp.ndarray,
    hook_name: str,
    pos: int = -1,
) -> dict:
    """Measure consistency of internal representations between two inputs.

    Computes how similar the residual stream representations are for two
    inputs (e.g., paraphrases of the same question). High similarity suggests
    the model has a stable internal belief regardless of surface form.

    Args:
        model: HookedTransformer.
        tokens_a: [seq_len] first input.
        tokens_b: [seq_len] second input (e.g., paraphrase).
        hook_name: Hook point to compare.
        pos: Token position (-1 = last).

    Returns:
        Dict with:
            "cosine_similarity": cosine similarity of residual representations
            "output_agreement": whether top prediction matches
            "kl_divergence": KL divergence between output distributions
            "consistency_score": combined consistency metric (0-1)
    """
    _, cache_a = model.run_with_cache(tokens_a)
    _, cache_b = model.run_with_cache(tokens_b)
    logits_a = model(tokens_a)
    logits_b = model(tokens_b)

    # Residual similarity
    if hook_name in cache_a.cache_dict and hook_name in cache_b.cache_dict:
        resid_a = np.array(cache_a.cache_dict[hook_name][pos])
        resid_b = np.array(cache_b.cache_dict[hook_name][pos])
        cos_sim = float(np.dot(resid_a, resid_b) / (
            np.linalg.norm(resid_a) * np.linalg.norm(resid_b) + 1e-10
        ))
    else:
        cos_sim = 0.0

    # Output agreement
    probs_a = np.array(jax.nn.softmax(logits_a[pos]))
    probs_b = np.array(jax.nn.softmax(logits_b[pos]))
    output_agreement = bool(np.argmax(probs_a) == np.argmax(probs_b))

    # KL divergence
    kl = float(np.sum(probs_a * np.log((probs_a + 1e-10) / (probs_b + 1e-10))))

    # Combined score
    consistency = (cos_sim + 1.0) / 2.0  # Map [-1,1] to [0,1]
    if output_agreement:
        consistency = min(1.0, consistency + 0.1)

    return {
        "cosine_similarity": cos_sim,
        "output_agreement": output_agreement,
        "kl_divergence": kl,
        "consistency_score": consistency,
    }
