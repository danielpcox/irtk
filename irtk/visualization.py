"""Visualization tools for mechanistic interpretability.

Provides publication-quality plotting functions for:
- Attention patterns (single head, all heads, head summaries)
- Logit lens (predictions evolving through layers)
- Residual stream analysis (norms, PCA)
- Neuron activation heatmaps
- Token-colored text display
- Logit attribution bar charts
"""

from typing import Optional, Sequence

import numpy as np
import jax.numpy as jnp

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
except ImportError:
    raise ImportError("matplotlib is required for visualization: pip install matplotlib")


def _to_numpy(x) -> np.ndarray:
    """Convert JAX arrays to numpy for matplotlib."""
    if isinstance(x, jnp.ndarray):
        return np.array(x)
    return np.asarray(x)


def _token_labels(tokens, tokenizer) -> list[str]:
    """Convert token IDs to readable string labels."""
    labels = []
    for t in tokens:
        s = tokenizer.decode([int(t)])
        # Make whitespace visible
        s = s.replace("\n", "\\n")
        labels.append(s)
    return labels


# ─── Attention Patterns ────────────────────────────────────────────────────────


def plot_attention_pattern(
    pattern: np.ndarray,
    tokens: Optional[Sequence] = None,
    tokenizer=None,
    title: str = "",
    ax=None,
    cmap: str = "Blues",
    vmin: float = 0.0,
    vmax: Optional[float] = None,
):
    """Plot a single attention pattern as a heatmap.

    Args:
        pattern: [seq_q, seq_k] attention weights.
        tokens: Token IDs for axis labels.
        tokenizer: HF tokenizer for decoding token labels.
        title: Plot title.
        ax: Matplotlib axes (creates new figure if None).
        cmap: Colormap name.
        vmin/vmax: Color scale bounds.

    Returns:
        The axes object.
    """
    pattern = _to_numpy(pattern)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    if vmax is None:
        vmax = float(pattern.max())

    im = ax.imshow(pattern, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xlabel("Key (source)")
    ax.set_ylabel("Query (destination)")
    if title:
        ax.set_title(title)

    if tokens is not None and tokenizer is not None:
        labels = _token_labels(tokens, tokenizer)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_attention_heads(
    patterns: np.ndarray,
    tokens: Optional[Sequence] = None,
    tokenizer=None,
    layer: Optional[int] = None,
    cols: int = 4,
    figsize_per_head: tuple = (3.5, 3),
    cmap: str = "Blues",
):
    """Plot attention patterns for all heads in a layer.

    Args:
        patterns: [n_heads, seq_q, seq_k] attention weights.
        tokens: Token IDs for axis labels.
        tokenizer: HF tokenizer.
        layer: Layer index (for title).
        cols: Number of columns in grid.
        figsize_per_head: (width, height) per subplot.
        cmap: Colormap.

    Returns:
        Figure and axes array.
    """
    patterns = _to_numpy(patterns)
    n_heads = patterns.shape[0]
    rows = (n_heads + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * figsize_per_head[0], rows * figsize_per_head[1]),
    )
    axes_flat = np.array(axes).flatten() if n_heads > 1 else [axes]

    labels = _token_labels(tokens, tokenizer) if tokens is not None and tokenizer is not None else None

    for h in range(n_heads):
        ax = axes_flat[h]
        im = ax.imshow(patterns[h], cmap=cmap, vmin=0, aspect="auto")
        prefix = f"L{layer}" if layer is not None else ""
        ax.set_title(f"{prefix}H{h}", fontsize=9)
        if labels is not None:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=5)
            ax.set_yticklabels(labels, fontsize=5)

    # Turn off unused axes
    for h in range(n_heads, len(axes_flat)):
        axes_flat[h].set_visible(False)

    layer_str = f" Layer {layer}" if layer is not None else ""
    fig.suptitle(f"Attention Patterns:{layer_str}, All Heads", fontsize=12)
    fig.tight_layout()
    return fig, axes


def plot_head_summary(
    scores: np.ndarray,
    title: str = "Head Scores",
    xlabel: str = "Head",
    ylabel: str = "Layer",
    cmap: str = "YlOrRd",
    vmin: float = 0.0,
    vmax: Optional[float] = None,
    annotate: bool = True,
    figsize: tuple = (12, 6),
):
    """Plot a layer x head heatmap (e.g., induction scores, ablation results).

    Args:
        scores: [n_layers, n_heads] score matrix.
        title: Plot title.
        cmap: Colormap.
        annotate: Whether to add text annotations.
        figsize: Figure size.

    Returns:
        Figure and axes.
    """
    scores = _to_numpy(scores)
    n_layers, n_heads = scores.shape

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(scores, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)])

    if annotate:
        thresh = (scores.max() + scores.min()) / 2 if vmax is None else (vmax + vmin) / 2
        for l in range(n_layers):
            for h in range(n_heads):
                color = "white" if scores[l, h] > thresh else "black"
                ax.text(
                    h, l, f"{scores[l, h]:.2f}",
                    ha="center", va="center", fontsize=6, color=color,
                )

    plt.colorbar(im, ax=ax, label=title)
    fig.tight_layout()
    return fig, ax


# ─── Logit Lens ───────────────────────────────────────────────────────────────


def plot_logit_lens(
    per_layer_logits: np.ndarray,
    tokens: Sequence,
    tokenizer,
    top_k: int = 5,
    figsize: tuple = (14, 8),
    title: str = "Logit Lens: Predictions at Each Layer",
):
    """Plot a logit lens heatmap showing how predictions evolve through layers.

    Args:
        per_layer_logits: [n_layers, seq_len, d_vocab] logits at each layer.
        tokens: Input token IDs.
        tokenizer: HF tokenizer for decoding.
        top_k: Show top-k predictions in hover text.
        figsize: Figure size.
        title: Plot title.

    Returns:
        Figure and axes.
    """
    per_layer_logits = _to_numpy(per_layer_logits)
    n_layers, seq_len, d_vocab = per_layer_logits.shape
    tok_labels = _token_labels(tokens, tokenizer)

    # For each layer and position, get the probability of the correct next token
    correct_token_probs = np.zeros((n_layers, seq_len - 1))
    for layer in range(n_layers):
        log_probs = per_layer_logits[layer, :-1, :]  # predict next token
        probs = np.exp(log_probs - log_probs.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        for pos in range(seq_len - 1):
            next_token = int(tokens[pos + 1])
            correct_token_probs[layer, pos] = probs[pos, next_token]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(correct_token_probs, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xlabel("Position (predicting next token)")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    ax.set_xticks(range(seq_len - 1))
    ax.set_xticklabels(
        [f"{tok_labels[i]}->{tok_labels[i+1]}" for i in range(seq_len - 1)],
        rotation=90, fontsize=7,
    )
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)])

    plt.colorbar(im, ax=ax, label="P(correct next token)")
    fig.tight_layout()
    return fig, ax


# ─── Residual Stream ──────────────────────────────────────────────────────────


def plot_residual_norms(
    cache,
    tokens: Optional[Sequence] = None,
    tokenizer=None,
    figsize: tuple = (12, 4),
):
    """Plot how residual stream norms evolve through the model.

    Args:
        cache: ActivationCache from run_with_cache().
        tokens: Token IDs for legend.
        tokenizer: HF tokenizer.
        figsize: Figure size.

    Returns:
        Figure and axes.
    """
    model = cache.model
    n_layers = model.cfg.n_layers

    positions_norms = []
    labels = []

    embed = _to_numpy(cache["hook_embed"])
    if "hook_pos_embed" in cache:
        embed = embed + _to_numpy(cache["hook_pos_embed"])
    positions_norms.append(np.linalg.norm(embed, axis=-1))
    labels.append("embed")

    for l in range(n_layers):
        resid = _to_numpy(cache[("resid_post", l)])
        positions_norms.append(np.linalg.norm(resid, axis=-1))
        labels.append(f"L{l}")

    fig, ax = plt.subplots(figsize=figsize)
    means = [n.mean() for n in positions_norms]
    ax.bar(range(len(means)), means, color="steelblue")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean L2 Norm")
    ax.set_title("Residual Stream Norm Through the Model")
    fig.tight_layout()
    return fig, ax


# ─── Logit Attribution ────────────────────────────────────────────────────────


def plot_logit_attribution(
    attrs: np.ndarray,
    labels: list[str],
    target_token: Optional[str] = None,
    figsize: tuple = (12, 5),
    title: Optional[str] = None,
):
    """Plot a bar chart of logit attributions from residual stream decomposition.

    Args:
        attrs: [n_components] logit contribution from each component.
        labels: Component labels.
        target_token: The token being attributed (for title).
        figsize: Figure size.
        title: Custom title.

    Returns:
        Figure and axes.
    """
    attrs = _to_numpy(attrs)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#2ca02c" if a > 0 else "#d62728" for a in attrs]
    ax.bar(range(len(attrs)), attrs, color=colors, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    short_labels = [l.replace("blocks.", "L").replace(".hook_", " ") for l in labels]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Logit contribution")
    ax.axhline(y=0, color="black", linewidth=0.5)

    if title:
        ax.set_title(title)
    elif target_token:
        ax.set_title(f"Logit Attribution for '{target_token}'")

    fig.tight_layout()
    return fig, ax


# ─── Neuron Activations ──────────────────────────────────────────────────────


def plot_neuron_activations(
    activations: np.ndarray,
    tokens: Optional[Sequence] = None,
    tokenizer=None,
    n_neurons: int = 50,
    title: str = "Neuron Activations",
    figsize: tuple = (14, 5),
    cmap: str = "viridis",
):
    """Plot a heatmap of neuron activations across tokens.

    Args:
        activations: [seq_len, d_mlp] neuron activations.
        tokens: Token IDs.
        tokenizer: HF tokenizer.
        n_neurons: Number of neurons to show.
        title: Plot title.
        figsize: Figure size.
        cmap: Colormap.

    Returns:
        Figure and axes.
    """
    activations = _to_numpy(activations)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(activations[:, :n_neurons], cmap=cmap, aspect="auto")
    ax.set_xlabel(f"Neuron index (first {n_neurons})")
    ax.set_ylabel("Token position")
    ax.set_title(title)

    if tokens is not None and tokenizer is not None:
        labels = _token_labels(tokens, tokenizer)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)

    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig, ax


# ─── Patching Results ────────────────────────────────────────────────────────


def plot_patching_heatmap(
    results: np.ndarray,
    title: str = "Activation Patching Results",
    xlabel: str = "Head",
    ylabel: str = "Layer",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotate: bool = True,
    figsize: tuple = (14, 8),
    clean_value: Optional[float] = None,
    corrupted_value: Optional[float] = None,
):
    """Plot patching or ablation results as a heatmap.

    Works with output from patch_by_head(), ablate_heads(), etc.

    Args:
        results: [n_layers, n_heads] metric values after patching/ablation.
        title: Plot title.
        cmap: Colormap (RdBu_r works well for patching).
        annotate: Add value text.
        clean_value: Optional clean baseline (shown as green dashed line in colorbar).
        corrupted_value: Optional corrupted baseline.

    Returns:
        Figure and axes.
    """
    results = _to_numpy(results)
    n_layers, n_heads = results.shape

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(results, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)])

    if annotate and n_heads <= 16:
        thresh = np.median(results)
        for l in range(n_layers):
            for h in range(n_heads):
                color = "white" if abs(results[l, h] - thresh) > (results.max() - results.min()) / 4 else "black"
                ax.text(
                    h, l, f"{results[l, h]:.2f}",
                    ha="center", va="center", fontsize=6, color=color,
                )

    cbar = plt.colorbar(im, ax=ax)
    if clean_value is not None:
        cbar.ax.axhline(y=clean_value, color="green", linewidth=2, linestyle="--")
    if corrupted_value is not None:
        cbar.ax.axhline(y=corrupted_value, color="red", linewidth=2, linestyle="--")

    fig.tight_layout()
    return fig, ax


def plot_layer_patching(
    results: np.ndarray,
    clean_value: Optional[float] = None,
    corrupted_value: Optional[float] = None,
    title: str = "Activation Patching by Layer",
    figsize: tuple = (10, 5),
):
    """Plot layer-level patching results as a bar chart.

    Args:
        results: [n_layers] metric values after patching each layer.
        clean_value: Clean baseline value.
        corrupted_value: Corrupted baseline value.

    Returns:
        Figure and axes.
    """
    results = _to_numpy(results)
    n_layers = len(results)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(n_layers), results, color="steelblue", alpha=0.8)

    if clean_value is not None:
        ax.axhline(y=clean_value, color="green", linestyle="--", alpha=0.7, label=f"Clean ({clean_value:.2f})")
    if corrupted_value is not None:
        ax.axhline(y=corrupted_value, color="red", linestyle="--", alpha=0.7, label=f"Corrupted ({corrupted_value:.2f})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Metric Value")
    ax.set_title(title)
    ax.set_xticks(range(n_layers))
    ax.grid(True, alpha=0.3, axis="y")
    if clean_value is not None or corrupted_value is not None:
        ax.legend()

    fig.tight_layout()
    return fig, ax


# ─── Probe and SAE Results ──────────────────────────────────────────────────


def plot_probe_accuracy_by_layer(
    accuracies: Sequence[float],
    title: str = "Linear Probe Accuracy by Layer",
    chance_level: float = 0.5,
    figsize: tuple = (10, 5),
):
    """Plot probe accuracy across layers.

    Args:
        accuracies: List of accuracy values, one per layer.
        title: Plot title.
        chance_level: Random baseline accuracy.

    Returns:
        Figure and axes.
    """
    fig, ax = plt.subplots(figsize=figsize)
    layers = range(len(accuracies))
    ax.plot(layers, accuracies, "o-", markersize=6, color="steelblue")
    ax.axhline(y=chance_level, color="gray", linestyle="--", alpha=0.5, label=f"Chance ({chance_level:.1%})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(0, len(accuracies), max(1, len(accuracies) // 12)))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_sae_training(
    train_losses: Sequence[float],
    recon_losses: Sequence[float],
    l1_losses: Sequence[float],
    l0_sparsities: Sequence[float],
    title: str = "SAE Training",
    figsize: tuple = (12, 8),
):
    """Plot SAE training curves in a 2x2 grid.

    Args:
        train_losses: Total loss per epoch.
        recon_losses: Reconstruction loss per epoch.
        l1_losses: L1 loss per epoch.
        l0_sparsities: Average L0 per epoch.

    Returns:
        Figure and axes.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    axes[0, 0].plot(train_losses)
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(recon_losses)
    axes[0, 1].set_title("Reconstruction Loss (MSE)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(l1_losses)
    axes[1, 0].set_title("L1 Loss (Sparsity)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(l0_sparsities)
    axes[1, 1].set_title("L0 Sparsity (avg active features)")
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Epoch")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig, axes


def plot_composition_scores(
    scores: np.ndarray,
    n_heads: int,
    composition_type: str = "QK",
    figsize: tuple = (10, 8),
):
    """Plot head-to-head composition scores.

    Args:
        scores: [n_total_heads, n_total_heads] composition score matrix.
        n_heads: Number of heads per layer.
        composition_type: "QK" or "OV" (for title).

    Returns:
        Figure and axes.
    """
    scores = _to_numpy(scores)
    total = scores.shape[0]
    n_layers = total // n_heads

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(scores, cmap="viridis", aspect="auto")
    ax.set_xlabel("Destination Head")
    ax.set_ylabel("Source Head")
    ax.set_title(f"{composition_type} Composition Scores")

    # Draw layer boundary lines
    for i in range(1, n_layers):
        pos = i * n_heads - 0.5
        ax.axhline(y=pos, color="white", linewidth=0.5, alpha=0.5)
        ax.axvline(x=pos, color="white", linewidth=0.5, alpha=0.5)

    # Label layers on edges
    for i in range(n_layers):
        mid = i * n_heads + n_heads / 2 - 0.5
        ax.text(-1, mid, f"L{i}", ha="right", va="center", fontsize=8)
        ax.text(mid, -1, f"L{i}", ha="center", va="bottom", fontsize=8)

    plt.colorbar(im, ax=ax, label="Composition Score")
    fig.tight_layout()
    return fig, ax


# ─── Token Display ────────────────────────────────────────────────────────────


def color_tokens(
    tokens: Sequence,
    values: np.ndarray,
    tokenizer,
    cmap: str = "RdBu",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: tuple = (14, 1.5),
    title: str = "",
):
    """Display tokens colored by a per-token value (e.g., loss, attribution).

    Args:
        tokens: Token IDs.
        values: [seq_len] values to color by.
        tokenizer: HF tokenizer.
        cmap: Colormap name.
        vmin/vmax: Color scale bounds.
        figsize: Figure size.
        title: Plot title.

    Returns:
        Figure and axes.
    """
    values = _to_numpy(values)
    labels = _token_labels(tokens, tokenizer)

    if vmin is None:
        vmin = float(values.min())
    if vmax is None:
        vmax = float(values.max())

    colormap = plt.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, len(labels))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])

    x = 0
    for i, (label, val) in enumerate(zip(labels, values)):
        color = colormap(norm(val))
        width = 1
        rect = Rectangle((x, 0), width, 1, facecolor=color, edgecolor="white", linewidth=0.5)
        ax.add_patch(rect)
        ax.text(
            x + width / 2, 0.5, label,
            ha="center", va="center", fontsize=8,
            color="white" if norm(val) > 0.6 or norm(val) < 0.4 else "black",
        )
        x += width

    if title:
        ax.set_title(title)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.3)

    fig.tight_layout()
    return fig, ax


# ─── Token Attribution Display ────────────────────────────────────────────────


def plot_token_attribution(
    tokens: Sequence,
    scores: np.ndarray,
    tokenizer=None,
    title: str = "Token Attribution",
    cmap: str = "Reds",
    figsize: tuple = (14, 2),
    show_values: bool = True,
):
    """Display tokens colored by attribution scores with value annotations.

    Unlike color_tokens, this is designed specifically for attribution
    outputs (gradient*input, integrated gradients, etc.), with a
    single-color scale and optional score labels.

    Args:
        tokens: Token IDs or string labels.
        scores: [seq_len] attribution scores (higher = more important).
        tokenizer: HF tokenizer for decoding (optional if tokens are strings).
        title: Plot title.
        cmap: Colormap (single-direction, e.g., Reds, Oranges).
        figsize: Figure size.
        show_values: Whether to show numeric scores on each token.

    Returns:
        Figure and axes.
    """
    scores = _to_numpy(scores)
    if tokenizer is not None:
        labels = _token_labels(tokens, tokenizer)
    else:
        labels = [str(t) for t in tokens]

    vmin = 0.0
    vmax = float(scores.max()) if scores.max() > 0 else 1.0

    colormap = plt.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, len(labels))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])

    for i, (label, val) in enumerate(zip(labels, scores)):
        color = colormap(norm(val))
        rect = Rectangle((i, 0), 1, 1, facecolor=color, edgecolor="white", linewidth=0.5)
        ax.add_patch(rect)

        # Choose readable text color
        brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = "white" if brightness < 0.5 else "black"

        if show_values:
            ax.text(i + 0.5, 0.65, label, ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold")
            ax.text(i + 0.5, 0.3, f"{val:.3f}", ha="center", va="center",
                    fontsize=6, color=text_color)
        else:
            ax.text(i + 0.5, 0.5, label, ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold")

    ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.3)

    fig.tight_layout()
    return fig, ax


def plot_causal_tracing(
    tracing_result: dict,
    title: str = "Causal Tracing",
    figsize: tuple = (12, 5),
):
    """Plot causal tracing results showing recovery at each layer.

    Designed to visualize the output of attention_utils.causal_tracing().

    Args:
        tracing_result: Dict with keys "clean", "corrupted", "restored_resid",
            "restored_attn", "restored_mlp".
        title: Plot title.
        figsize: Figure size.

    Returns:
        Figure and axes.
    """
    clean = tracing_result["clean"]
    corrupted = tracing_result["corrupted"]
    restored_resid = _to_numpy(tracing_result["restored_resid"])
    restored_attn = _to_numpy(tracing_result["restored_attn"])
    restored_mlp = _to_numpy(tracing_result["restored_mlp"])

    n_layers = len(restored_resid)
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(layers, restored_resid, "o-", label="Restore resid", markersize=6)
    ax.plot(layers, restored_attn, "s-", label="Restore attn", markersize=6)
    ax.plot(layers, restored_mlp, "^-", label="Restore MLP", markersize=6)

    ax.axhline(y=clean, color="green", linestyle="--", alpha=0.7, label=f"Clean ({clean:.3f})")
    ax.axhline(y=corrupted, color="red", linestyle="--", alpha=0.7, label=f"Corrupted ({corrupted:.3f})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Metric Value")
    ax.set_title(title)
    ax.set_xticks(layers)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


def plot_prediction_trajectory(
    trajectory: list,
    tokenizer=None,
    target_token: Optional[int] = None,
    title: str = "Prediction Trajectory Across Layers",
    figsize: tuple = (12, 6),
):
    """Plot how top predictions change across layers.

    Designed to visualize output of residual_stream.token_prediction_trajectory().

    Args:
        trajectory: List of [top-k predictions at each layer].
            Each prediction is (token_id, probability).
        tokenizer: HF tokenizer for decoding token names.
        target_token: If given, highlight this token's probability.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Figure and axes.
    """
    n_layers = len(trajectory)
    k = len(trajectory[0]) if trajectory else 0

    fig, ax = plt.subplots(figsize=figsize)

    # Track probabilities of the top tokens at the final layer
    if k > 0:
        final_top_tokens = [tok_id for tok_id, _ in trajectory[-1]]
        for rank, tok_id in enumerate(final_top_tokens):
            probs_across_layers = []
            for layer_preds in trajectory:
                prob = 0.0
                for t_id, p in layer_preds:
                    if t_id == tok_id:
                        prob = p
                        break
                probs_across_layers.append(prob)

            label = f"#{rank+1}"
            if tokenizer is not None:
                label = tokenizer.decode([tok_id])
            ax.plot(range(n_layers), probs_across_layers, "o-", label=label, markersize=5)

    # Highlight target token if given
    if target_token is not None:
        target_probs = []
        for layer_preds in trajectory:
            prob = 0.0
            for t_id, p in layer_preds:
                if t_id == target_token:
                    prob = p
                    break
            target_probs.append(prob)

        target_label = f"target ({target_token})"
        if tokenizer is not None:
            target_label = tokenizer.decode([target_token])
        ax.plot(range(n_layers), target_probs, "k*-", label=f"Target: {target_label}",
                markersize=10, linewidth=2)

    layer_labels = ["embed"] + [f"L{i}" for i in range(n_layers - 1)]
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layer_labels[:n_layers])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig, ax
