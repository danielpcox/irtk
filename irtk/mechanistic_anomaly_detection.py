"""Mechanistic anomaly detection: identify unusual computation patterns.

Builds baseline activation profiles from reference data and detects when
a model's internal computation deviates from its typical patterns on
novel or adversarial inputs. Useful for safety auditing.

Functions:
- build_activation_profile: Build baseline statistics from reference inputs
- detect_pathway_anomalies: Score how anomalous a new input's computation is
- find_trojan_signatures: Scan for sparse, localized anomalous activation patterns
- compare_surface_vs_internals: Detect internal-output prediction divergence
- cluster_computational_strategies: Group inputs by internal computation similarity

References:
    - Casper et al. (2024) "Latent Adversarial Training"
    - Christiano (2022) "Eliciting Latent Knowledge"
    - Sun et al. (2024) "Massive Activations in Large Language Models"
"""

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from irtk.hooked_transformer import HookedTransformer


def build_activation_profile(
    model: HookedTransformer,
    reference_inputs: Sequence[jnp.ndarray],
    hook_names: Optional[list] = None,
) -> dict:
    """Build baseline activation statistics from reference inputs.

    Computes mean, variance, and covariance of activations at each
    specified hook point over a set of reference inputs.

    Args:
        model: HookedTransformer.
        reference_inputs: List of [seq_len] token arrays for the reference distribution.
        hook_names: Hook names to profile. Defaults to all resid_post hooks.

    Returns:
        Dict with:
            "means": dict mapping hook_name -> [d_model] mean activation per position (last)
            "stds": dict mapping hook_name -> [d_model] std per dimension
            "covariances": dict mapping hook_name -> [d_model, d_model] covariance matrix
            "n_samples": number of reference inputs used
            "hook_names": list of profiled hook names
    """
    n_layers = model.cfg.n_layers
    if hook_names is None:
        hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]

    # Collect activations
    all_acts = {name: [] for name in hook_names}

    for tokens in reference_inputs:
        _, cache = model.run_with_cache(tokens)
        for name in hook_names:
            if name in cache.cache_dict:
                act = np.array(cache.cache_dict[name][-1])  # last position
                all_acts[name].append(act)

    means = {}
    stds = {}
    covariances = {}

    for name in hook_names:
        if all_acts[name]:
            acts_array = np.stack(all_acts[name])  # [n_samples, d_model]
            means[name] = np.mean(acts_array, axis=0)
            stds[name] = np.std(acts_array, axis=0)
            if acts_array.shape[0] > 1:
                covariances[name] = np.cov(acts_array.T)
            else:
                covariances[name] = np.eye(acts_array.shape[1])
        else:
            d = model.cfg.d_model
            means[name] = np.zeros(d)
            stds[name] = np.ones(d)
            covariances[name] = np.eye(d)

    return {
        "means": means,
        "stds": stds,
        "covariances": covariances,
        "n_samples": len(reference_inputs),
        "hook_names": hook_names,
    }


def detect_pathway_anomalies(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    profile: dict,
) -> dict:
    """Score how anomalous a new input's computation is vs the baseline profile.

    Uses Mahalanobis distance and z-scores at each profiled layer to
    detect deviations from typical computation patterns.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens to evaluate.
        profile: Activation profile from build_activation_profile.

    Returns:
        Dict with:
            "layer_anomaly_scores": dict mapping hook_name -> anomaly score
            "max_anomaly_layer": hook name with highest anomaly
            "total_anomaly_score": sum of all layer anomaly scores
            "dimension_z_scores": dict mapping hook_name -> [d_model] per-dim z-scores
    """
    _, cache = model.run_with_cache(tokens)

    layer_scores = {}
    dim_z_scores = {}

    for name in profile["hook_names"]:
        if name in cache.cache_dict:
            act = np.array(cache.cache_dict[name][-1])
            mean = profile["means"][name]
            std = profile["stds"][name]

            # Per-dimension z-scores
            z = np.abs(act - mean) / (std + 1e-10)
            dim_z_scores[name] = z

            # Mahalanobis-like score (using diagonal approximation for efficiency)
            score = float(np.mean(z ** 2))
            layer_scores[name] = score
        else:
            layer_scores[name] = 0.0
            dim_z_scores[name] = np.zeros(model.cfg.d_model)

    max_layer = max(layer_scores, key=layer_scores.get) if layer_scores else ""
    total = float(sum(layer_scores.values()))

    return {
        "layer_anomaly_scores": layer_scores,
        "max_anomaly_layer": max_layer,
        "total_anomaly_score": total,
        "dimension_z_scores": dim_z_scores,
    }


def find_trojan_signatures(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    profile: dict,
    sparsity_threshold: float = 3.0,
) -> dict:
    """Scan for sparse, localized anomalous activations (trojan signatures).

    Looks for a small number of dimensions with very high z-scores while
    most dimensions remain normal — a pattern consistent with implanted
    backdoor circuits.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        profile: Activation profile from build_activation_profile.
        sparsity_threshold: z-score threshold for "anomalous" dimensions.

    Returns:
        Dict with:
            "suspicious_dimensions": dict mapping hook_name -> list of (dim, z_score)
            "sparsity_ratios": dict mapping hook_name -> fraction of anomalous dims
            "max_z_score": highest z-score found across all layers
            "trojan_risk_score": combined risk score (0 = normal, higher = more suspicious)
    """
    anomalies = detect_pathway_anomalies(model, tokens, profile)

    suspicious = {}
    sparsity_ratios = {}
    max_z = 0.0

    for name in profile["hook_names"]:
        z = anomalies["dimension_z_scores"].get(name, np.zeros(1))

        anomalous_dims = np.where(z > sparsity_threshold)[0]
        suspicious[name] = [(int(d), float(z[d])) for d in anomalous_dims]

        ratio = float(len(anomalous_dims) / (len(z) + 1e-10))
        sparsity_ratios[name] = ratio

        if len(z) > 0:
            max_z = max(max_z, float(np.max(z)))

    # Trojan risk: high max z-score but low sparsity (few dims affected)
    mean_sparsity = float(np.mean(list(sparsity_ratios.values()))) if sparsity_ratios else 0.0
    risk = max_z * (1.0 - mean_sparsity) if mean_sparsity < 0.5 else 0.0

    return {
        "suspicious_dimensions": suspicious,
        "sparsity_ratios": sparsity_ratios,
        "max_z_score": max_z,
        "trojan_risk_score": risk,
    }


def compare_surface_vs_internals(
    model: HookedTransformer,
    tokens: jnp.ndarray,
    pos: int = -1,
) -> dict:
    """Detect divergence between internal predictions and final output.

    Projects each layer's representation to output space and compares
    with the actual final output, detecting cases where the model's
    trajectory and conclusion disagree.

    Args:
        model: HookedTransformer.
        tokens: [seq_len] input tokens.
        pos: Token position (-1 = last).

    Returns:
        Dict with:
            "layer_predictions": [n_layers] predicted token at each layer
            "final_prediction": final output token
            "agreement_fraction": fraction of layers agreeing with final output
            "disagreement_layers": list of layers that predict differently
            "trajectory_consistency": mean cosine similarity of consecutive layer logits
    """
    _, cache = model.run_with_cache(tokens)
    logits = model(tokens)
    n_layers = model.cfg.n_layers

    W_U = np.array(model.unembed.W_U)
    b_U = np.array(model.unembed.b_U) if hasattr(model.unembed, 'b_U') else np.zeros(W_U.shape[1])

    final_pred = int(np.argmax(np.array(logits[pos])))

    layer_preds = []
    layer_logit_vecs = []
    for l in range(n_layers):
        key = f"blocks.{l}.hook_resid_post"
        if key in cache.cache_dict:
            resid = np.array(cache.cache_dict[key][pos])
            layer_logits = resid @ W_U + b_U
            layer_preds.append(int(np.argmax(layer_logits)))
            layer_logit_vecs.append(layer_logits)
        else:
            layer_preds.append(-1)
            layer_logit_vecs.append(np.zeros(W_U.shape[1]))

    agrees = sum(1 for p in layer_preds if p == final_pred)
    agreement = agrees / max(n_layers, 1)
    disagreement = [l for l, p in enumerate(layer_preds) if p != final_pred]

    # Trajectory consistency
    if len(layer_logit_vecs) >= 2:
        cosines = []
        for i in range(len(layer_logit_vecs) - 1):
            a, b = layer_logit_vecs[i], layer_logit_vecs[i + 1]
            cs = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
            cosines.append(cs)
        consistency = float(np.mean(cosines))
    else:
        consistency = 1.0

    return {
        "layer_predictions": layer_preds,
        "final_prediction": final_pred,
        "agreement_fraction": agreement,
        "disagreement_layers": disagreement,
        "trajectory_consistency": consistency,
    }


def cluster_computational_strategies(
    model: HookedTransformer,
    input_set: Sequence[jnp.ndarray],
    hook_name: Optional[str] = None,
    n_clusters: int = 3,
) -> dict:
    """Group inputs by internal computation similarity.

    Clusters inputs based on their activation trajectories, revealing
    when the model uses qualitatively different algorithms.

    Args:
        model: HookedTransformer.
        input_set: List of [seq_len] token arrays.
        hook_name: Hook to use for clustering. Defaults to last resid_post.
        n_clusters: Number of clusters to form.

    Returns:
        Dict with:
            "cluster_assignments": [n_inputs] cluster index for each input
            "cluster_sizes": [n_clusters] number of inputs per cluster
            "cluster_centroids": [n_clusters, d_model] cluster centers
            "within_cluster_variance": [n_clusters] variance within each cluster
            "between_cluster_variance": total between-cluster variance
    """
    n_layers = model.cfg.n_layers
    if hook_name is None:
        hook_name = f"blocks.{n_layers - 1}.hook_resid_post"

    # Collect activation vectors
    acts = []
    for tokens in input_set:
        _, cache = model.run_with_cache(tokens)
        if hook_name in cache.cache_dict:
            act = np.array(cache.cache_dict[hook_name][-1])
            acts.append(act)
        else:
            acts.append(np.zeros(model.cfg.d_model))

    acts = np.stack(acts)  # [n_inputs, d_model]
    n_inputs = len(acts)
    n_clusters = min(n_clusters, n_inputs)

    # Simple k-means clustering
    # Initialize centroids with k-means++
    rng = np.random.RandomState(42)
    centroids = [acts[rng.randint(n_inputs)]]
    for _ in range(1, n_clusters):
        dists = np.array([min(np.sum((a - c) ** 2) for c in centroids) for a in acts])
        probs = dists / (np.sum(dists) + 1e-10)
        idx = rng.choice(n_inputs, p=probs)
        centroids.append(acts[idx])
    centroids = np.stack(centroids)

    # Run k-means iterations
    for _ in range(20):
        # Assign
        dists = np.array([[np.sum((a - c) ** 2) for c in centroids] for a in acts])
        assignments = np.argmin(dists, axis=1)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            mask = assignments == k
            if np.sum(mask) > 0:
                new_centroids[k] = np.mean(acts[mask], axis=0)
            else:
                new_centroids[k] = centroids[k]
        centroids = new_centroids

    # Compute statistics
    sizes = np.array([int(np.sum(assignments == k)) for k in range(n_clusters)])
    within_var = np.zeros(n_clusters)
    for k in range(n_clusters):
        mask = assignments == k
        if np.sum(mask) > 1:
            within_var[k] = float(np.mean(np.var(acts[mask], axis=0)))

    global_mean = np.mean(acts, axis=0)
    between_var = float(np.sum([sizes[k] * np.sum((centroids[k] - global_mean) ** 2)
                                for k in range(n_clusters)]) / (n_inputs + 1e-10))

    return {
        "cluster_assignments": assignments,
        "cluster_sizes": sizes,
        "cluster_centroids": centroids,
        "within_cluster_variance": within_var,
        "between_cluster_variance": between_var,
    }
