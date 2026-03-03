"""Concept erasure and direction analysis.

Tools for finding, erasing, and amplifying concept directions in
activation space:
- find_concept_direction: Extract a concept direction from contrasting prompts
- erase_concept: Project out a concept direction during forward pass
- amplify_concept: Scale up a concept's component in activations
- concept_sensitivity: Measure how much logits change when a concept is erased
- leace: LEAst-squares Concept Erasure (closed-form optimal erasure)
- concept_spectrum: Find multiple concept directions via PCA on differences
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def find_concept_direction(
    model: HookedTransformer,
    positive_tokens: list[jnp.ndarray],
    negative_tokens: list[jnp.ndarray],
    hook_name: str,
    pos: int = -1,
    normalize: bool = True,
) -> np.ndarray:
    """Find a linear concept direction from contrasting prompts.

    Computes mean(positive_activations) - mean(negative_activations) at the
    specified hook point. This is the simplest form of concept direction
    extraction (mean difference).

    Args:
        model: HookedTransformer.
        positive_tokens: Token sequences representing the positive concept.
        negative_tokens: Token sequences representing the negative concept.
        hook_name: Hook point to collect activations from.
        pos: Position to take activations from (-1 for last).
        normalize: If True, return a unit vector.

    Returns:
        [d_model] concept direction vector.
    """
    def _collect(token_sequences):
        acts = []
        for tokens in token_sequences:
            _, cache = model.run_with_cache(tokens)
            if hook_name in cache.cache_dict:
                acts.append(np.array(cache.cache_dict[hook_name][pos]))
        if not acts:
            return np.zeros(model.cfg.d_model)
        return np.mean(acts, axis=0)

    direction = _collect(positive_tokens) - _collect(negative_tokens)

    if normalize:
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm

    return direction


def erase_concept(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    hook_name: str,
    direction: jnp.ndarray,
) -> jnp.ndarray:
    """Run the model with a concept direction projected out of activations.

    For each position, removes the component along the concept direction:
        x' = x - (x . d) * d

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        hook_name: Hook point to erase the concept at.
        direction: [d_model] concept direction (should be unit norm).

    Returns:
        [seq_len, d_vocab] logits with the concept erased.
    """
    d = jnp.array(direction)
    d = d / (jnp.linalg.norm(d) + 1e-10)

    def erase_hook(x, name):
        # x: [seq_len, d_model]
        projections = x @ d  # [seq_len]
        return x - projections[:, None] * d[None, :]

    return model.run_with_hooks(tokens, fwd_hooks=[(hook_name, erase_hook)])


def amplify_concept(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    hook_name: str,
    direction: jnp.ndarray,
    alpha: float = 2.0,
) -> jnp.ndarray:
    """Run the model with a concept direction amplified in activations.

    Scales the component along the concept direction by alpha:
        x' = x + (alpha - 1) * (x . d) * d

    When alpha=0, this erases the concept. When alpha=1, no change.
    When alpha=2, the concept component is doubled.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        hook_name: Hook point to amplify the concept at.
        direction: [d_model] concept direction (should be unit norm).
        alpha: Amplification factor. 0=erase, 1=identity, 2=double.

    Returns:
        [seq_len, d_vocab] logits with the concept amplified.
    """
    d = jnp.array(direction)
    d = d / (jnp.linalg.norm(d) + 1e-10)

    def amplify_hook(x, name):
        projections = x @ d  # [seq_len]
        return x + (alpha - 1.0) * projections[:, None] * d[None, :]

    return model.run_with_hooks(tokens, fwd_hooks=[(hook_name, amplify_hook)])


def concept_sensitivity(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    hook_name: str,
    direction: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Measure how sensitive model predictions are to a concept direction.

    Compares clean logits vs logits with the concept erased. Returns
    various difference metrics.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        hook_name: Hook point to erase the concept at.
        direction: [d_model] concept direction.
        pos: Position to analyze (-1 for last).

    Returns:
        Dict with:
        - "logit_diff_l2": L2 distance between clean and erased logit vectors
        - "logit_diff_max": Max absolute change in any logit
        - "top_token_change": Change in the top token's logit
        - "kl_divergence": KL(clean || erased) on probability distributions
        - "top_promoted": List of (token_id, logit_change) for tokens most promoted by erasure
        - "top_suppressed": List of (token_id, logit_change) for tokens most suppressed
    """
    clean_logits = model(tokens)
    erased_logits = erase_concept(model, tokens, hook_name, direction)

    clean = np.array(clean_logits[pos])
    erased = np.array(erased_logits[pos])
    diff = erased - clean

    # KL divergence: sum p * log(p/q) where p=clean, q=erased
    clean_probs = np.exp(clean - np.max(clean))
    clean_probs = clean_probs / clean_probs.sum()
    erased_probs = np.exp(erased - np.max(erased))
    erased_probs = erased_probs / erased_probs.sum()
    kl = np.sum(clean_probs * np.log(clean_probs / (erased_probs + 1e-10) + 1e-10))

    # Top promoted/suppressed
    sorted_idx = np.argsort(diff)
    top_k = min(10, len(diff))
    top_promoted = [(int(sorted_idx[-(i + 1)]), float(diff[sorted_idx[-(i + 1)]]))
                    for i in range(top_k)]
    top_suppressed = [(int(sorted_idx[i]), float(diff[sorted_idx[i]]))
                      for i in range(top_k)]

    top_token = int(np.argmax(clean))

    return {
        "logit_diff_l2": float(np.linalg.norm(diff)),
        "logit_diff_max": float(np.max(np.abs(diff))),
        "top_token_change": float(diff[top_token]),
        "kl_divergence": float(max(0, kl)),
        "top_promoted": top_promoted,
        "top_suppressed": top_suppressed,
    }


def leace(
    activations: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """LEAst-squares Concept Erasure (LEACE).

    Computes the optimal linear projection that removes all information
    about the binary labels from the activations, in a least-squares sense.

    LEACE finds the direction(s) that a linear probe would use to classify
    the labels, then returns a projection matrix that erases those directions.

    Reference: Belrose et al., "LEACE: Perfect linear concept erasure in
    closed form" (2023).

    Args:
        activations: [n_samples, d_model] activation matrix.
        labels: [n_samples] binary labels (0 or 1).

    Returns:
        Dict with:
        - "projection": [d_model, d_model] projection matrix P such that
          P @ x erases the concept. Use as: erased = activations @ P.T
        - "concept_direction": [d_model] the primary concept direction erased
        - "explained_variance": Fraction of total variance along the concept direction
    """
    activations = np.array(activations, dtype=np.float64)
    labels = np.array(labels).ravel()

    n = len(labels)
    mask_0 = labels == 0
    mask_1 = labels == 1
    n0 = mask_0.sum()
    n1 = mask_1.sum()

    if n0 == 0 or n1 == 0:
        d = activations.shape[1]
        return {
            "projection": np.eye(d),
            "concept_direction": np.zeros(d),
            "explained_variance": 0.0,
        }

    # Class means
    mu_0 = activations[mask_0].mean(axis=0)
    mu_1 = activations[mask_1].mean(axis=0)

    # Between-class direction
    delta = mu_1 - mu_0
    delta_norm = np.linalg.norm(delta)

    if delta_norm < 1e-10:
        d = activations.shape[1]
        return {
            "projection": np.eye(d),
            "concept_direction": np.zeros(d),
            "explained_variance": 0.0,
        }

    direction = delta / delta_norm

    # Within-class covariance
    centered = activations - activations.mean(axis=0)
    total_var = np.sum(centered ** 2) / n

    # Variance explained by the concept direction
    concept_var = np.sum((activations @ direction) ** 2) / n
    # Actually compute the between-class variance contribution
    overall_mean = activations.mean(axis=0)
    between_var = (n0 * np.sum((mu_0 - overall_mean) ** 2) +
                   n1 * np.sum((mu_1 - overall_mean) ** 2)) / n

    explained = between_var / (total_var + 1e-10) if total_var > 1e-10 else 0.0

    # Projection matrix: P = I - d @ d.T
    d_model = activations.shape[1]
    projection = np.eye(d_model) - np.outer(direction, direction)

    return {
        "projection": projection.astype(np.float32),
        "concept_direction": direction.astype(np.float32),
        "explained_variance": float(np.clip(explained, 0, 1)),
    }


def concept_spectrum(
    model: HookedTransformer,
    positive_tokens: list[jnp.ndarray],
    negative_tokens: list[jnp.ndarray],
    hook_name: str,
    k: int = 5,
    pos: int = -1,
) -> dict:
    """Find multiple concept directions via PCA on activation differences.

    For each positive/negative pair, computes activation differences.
    Then applies PCA to find the top-k directions that capture the most
    variance in these differences. This reveals whether a "concept" is
    distributed across multiple orthogonal directions.

    Args:
        model: HookedTransformer.
        positive_tokens: Token sequences for the positive concept.
        negative_tokens: Token sequences for the negative concept.
        hook_name: Hook point to analyze.
        k: Number of concept directions to return.
        pos: Position to take activations from (-1 for last).

    Returns:
        Dict with:
        - "directions": [k, d_model] top concept directions
        - "explained_variance": [k] fraction of variance per direction
        - "singular_values": [k] singular values
    """
    # Collect activations for both groups
    def _collect_all(token_sequences):
        acts = []
        for tokens in token_sequences:
            _, cache = model.run_with_cache(tokens)
            if hook_name in cache.cache_dict:
                acts.append(np.array(cache.cache_dict[hook_name][pos]))
        return np.array(acts) if acts else None

    pos_acts = _collect_all(positive_tokens)
    neg_acts = _collect_all(negative_tokens)

    if pos_acts is None or neg_acts is None:
        d = model.cfg.d_model
        return {
            "directions": np.zeros((k, d)),
            "explained_variance": np.zeros(k),
            "singular_values": np.zeros(k),
        }

    # Compute pairwise differences (all positive - all negative)
    # Use broadcasted differences
    n_pos, d = pos_acts.shape
    n_neg = neg_acts.shape[0]
    n_diffs = min(n_pos * n_neg, 200)  # cap to avoid memory issues

    if n_pos <= n_neg:
        # Match each positive with a random negative
        diffs = []
        for i in range(n_pos):
            for j in range(min(n_neg, n_diffs // n_pos + 1)):
                diffs.append(pos_acts[i] - neg_acts[j % n_neg])
    else:
        diffs = []
        for j in range(n_neg):
            for i in range(min(n_pos, n_diffs // n_neg + 1)):
                diffs.append(pos_acts[i % n_pos] - neg_acts[j])

    diffs = np.array(diffs[:n_diffs])

    # Center the differences
    diffs_centered = diffs - diffs.mean(axis=0)

    # SVD
    k = min(k, min(diffs_centered.shape))
    U, S, Vh = np.linalg.svd(diffs_centered, full_matrices=False)

    total_var = np.sum(S ** 2)
    explained = (S[:k] ** 2) / (total_var + 1e-10)

    return {
        "directions": Vh[:k].astype(np.float32),
        "explained_variance": explained[:k].astype(np.float32),
        "singular_values": S[:k].astype(np.float32),
    }
