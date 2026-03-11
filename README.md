# IRTK — Interpretability Research Toolkit

A comprehensive mechanistic interpretability library built on [JAX](https://github.com/jax-ml/jax) and [Equinox](https://github.com/patrick-kidger/equinox). Originally inspired by [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), IRTK provides 170+ analysis modules for understanding transformer internals — from basic logit lenses and activation patching to circuit discovery, sparse autoencoders, causal scrubbing, and representation engineering.

Initially vibe-coded by Opus, so YMMV. PRs welcome.

## Installation

```bash
pip install irtk-jax
```

For Apple Silicon GPU acceleration via [jax-mps](https://github.com/danielpcox/jax-mps):

```bash
pip install irtk-jax[mps]
```

### Development setup

```bash
git clone git@github.com:danielpcox/irtk.git
cd irtk
uv sync --extra dev
```

**Requirements:** Python 3.11+.

## Quick start

```python
import irtk

# Load a pretrained model (GPT-2, GPT-Neo, GPT-NeoX, LLaMA, Mistral)
model = irtk.HookedTransformer.from_pretrained("gpt2")

# Run with activation caching
logits, cache = model.run_with_cache(tokens)

# Logit lens — decode residual stream at each layer
irtk.logit_lens.logit_lens(model, tokens, cache)

# Activation patching
irtk.patching.activation_patch(model, clean_tokens, corrupt_tokens, cache)

# Circuit analysis
irtk.circuits.direct_logit_attribution(model, tokens, cache)

# Attention head taxonomy
irtk.head_analysis.find_induction_heads(model, tokens, cache)
```

## Architecture

IRTK models are functional and immutable [Equinox modules](https://docs.kidger.site/equinox/):

- **Unbatched components** — individual components operate on single examples; use `jax.vmap` for batches.
- **HookState** — a plain Python object (not a PyTree) threaded through `__call__` for activation caching and intervention.
- **TransformerLens weight shapes** — `W_Q`, `W_K`, `W_V` are `[n_heads, d_model, d_head]` for direct per-head access.
- **Functional weight loading** — HuggingFace weights are loaded via `eqx.tree_at` chains (no mutation).

### Core classes

| Class | Description |
|---|---|
| `HookedTransformer` | Main model class with hook points at every intermediate computation |
| `HookedTransformerConfig` | Model configuration (dimensions, heads, layers, vocab, etc.) |
| `ActivationCache` | Dictionary-like cache of intermediate activations |
| `FactoredMatrix` | Efficient representation of `A @ B` without materializing the product |
| `HookPoint` | Attachment point for activation access and intervention |

### Supported model families

| Family | Architectures |
|---|---|
| GPT-2 | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl` |
| GPT-Neo | `EleutherAI/gpt-neo-125M`, etc. |
| GPT-NeoX | `EleutherAI/gpt-neox-20b`, `EleutherAI/pythia-*` |
| LLaMA / Mistral | LLaMA 2, LLaMA 3, Mistral 7B (with GatedMLP, RoPE) |

## Module reference

IRTK's 170+ modules cover the full spectrum of mechanistic interpretability research. Every module is importable from the top-level package (`irtk.<module>`).

### Probing & representation analysis

| Module | Key functions |
|---|---|
| `probes` | Linear/regression probes, `train_linear_probe` |
| `activation_probing` | Multiclass probe, nonlinear probe, concept localization |
| `sparse_probing` | L1 probes, sparse concept directions, feature selection |
| `probing_dynamics` | Probe accuracy by layer, emergence threshold, selectivity |
| `distributed_reps` | Representation rank, cross-layer concept tracking |
| `embeddings` | Embedding similarity, PCA, analogies, clustering, W_E/W_U alignment |
| `embedding_dynamics` | Identity decay, semantic drift, context mixing |

### Logit lens & prediction

| Module | Key functions |
|---|---|
| `logit_lens` | Standard logit lens, tuned lens |
| `logit_lens_variants` | Contrastive, causal, residual contribution lens |
| `prediction_entropy` | Layer prediction entropy, commit depth, surprisal |
| `prediction_geometry` | Vocabulary trajectory, sharpening rate |
| `prediction_confidence` | Confidence profile, layerwise evolution |
| `logit_dynamics` | Logit flips, prediction stability, commitment timing |
| `token_prediction_analysis` | Confidence, surprisal, difficulty, rank trajectory |

### Activation patching & ablation

| Module | Key functions |
|---|---|
| `patching` | Activation patching, ablation, path patching |
| `activation_patching_variants` | Denoising/noising patching, mean/resample ablation |
| `ablation_study` | Layer-by-layer ablation, head importance, position sensitivity |
| `token_level_ablation` | Per-token knockout, necessity/sufficiency, minimal set |
| `token_erasure` | Erasure effects, pairwise interaction, layerwise |

### Circuit analysis & discovery

| Module | Key functions |
|---|---|
| `circuits` | OV/QK circuits, composition scores, direct logit attribution |
| `circuit_discovery` | ACDC-style edge attribution, iterative pruning |
| `circuit_evaluation` | Faithfulness, completeness, minimality, circuit IoU |
| `circuit_motifs` | Skip trigram, negative mover, backup circuits |
| `causal_scrubbing` | Causal scrub, interchange intervention, path patching |
| `causal_abstraction` | Interchange intervention, DAS, multi-variable alignment |
| `path_expansion` | Path enumeration, virtual weights, contribution matrix |
| `subnetwork_analysis` | Circuit extraction, faithfulness, greedy search |
| `induction_circuit_analysis` | Full path tracing, matching scores, copy verification |

### Attention analysis

| Module | Key functions |
|---|---|
| `attention_utils` | Entropy, similarity, causal tracing, attention flow |
| `attention_surgery` | Knockout, pattern patching, force attention |
| `attention_rollout` | Rollout, flow, effective attention, per-head rollout |
| `attention_composition` | QK/V composition scores, path tracing, virtual patterns |
| `head_analysis` | Induction/prev-token head finding, scoring, classification |
| `attention_head_taxonomy` | Induction, copy, inhibition scoring, full taxonomy |
| `attention_sparsity` | Entropy profile, sparse/dense heads, window analysis |
| `attention_motif_discovery` | SVD-based motif extraction, function inference |
| `attention_sink_analysis` | Sink identification, magnitude tracking, removal impact |
| `attention_distance` | Mean distance, local/global heads, receptive field |

### Sparse autoencoders & features

| Module | Key functions |
|---|---|
| `sae` | SparseAutoencoder, training, feature analysis |
| `sparse_features` | Feature activation examples, feature circuits |
| `sae_feature_steering` | SAE feature-level steering, clamping |
| `transcoder` | Transcoder class, MLP feature logit attribution |
| `cross_coder` | CrossCoder, joint training, shared vs specific features |
| `feature_geometry` | Dictionary geometry: splitting, absorption, universality |
| `polysemantic_features` | Polysemanticity score, context clusters |
| `attribution_graphs` | Feature-to-feature computation graphs, pruning |

### Steering & intervention

| Module | Key functions |
|---|---|
| `steering` | Add/subtract/compute steering vectors, `steer_generation` |
| `representation_engineering` | RepE reading vectors, control vectors, suppression curves |
| `activation_surgery` | Clamp, scale, project, rotate, replace activations |
| `layerwise_intervention` | Activation addition, direction intervention |
| `intervention_effects` | Scaling sensitivity, knockout recovery, transferability |

### Attribution & gradients

| Module | Key functions |
|---|---|
| `gradients` | Grad x input, integrated gradients, Jacobian |
| `feature_attribution` | Token-to-neuron, logit decomposition, cross-layer |
| `principled_attribution` | Shapley values, KernelSHAP, interaction index |
| `logit_attribution_decomposition` | Full logit decomposition, promoted/demoted tokens |
| `residual_attribution` | Per-component contribution, interference, decomposition |

### MLP & neuron analysis

| Module | Key functions |
|---|---|
| `neurons` | Neuron stats, top activating tokens, neuron-to-logit |
| `mlp_decomposition` | Neuron contributions, knowledge storage, nonlinearity |
| `mlp_knowledge_editing` | Fact localization, rank-one edit, side effects |
| `mlp_gating_analysis` | Gate patterns, selectivity, contribution decomposition |
| `knowledge` | Knowledge neurons, causal tracing, fact editing vectors |
| `model_editing` | ROME-style fact editing, rank-one MLP updates |

### Geometry & similarity

| Module | Key functions |
|---|---|
| `geometry` | CKA, subspace overlap, intrinsic dimensionality |
| `activation_geometry` | Manifold dimension, clustering, curvature, norms |
| `residual_stream` | Cosine to unembed, norm tracking, prediction trajectory |
| `superposition` | Feature directions, interference, dimensionality |
| `low_rank` | Weight SVD, effective rank, spectrum similarity |
| `representation_similarity` | CKA, component similarity, drift |

### Model comparison & cross-model

| Module | Key functions |
|---|---|
| `model_diff` | Weight diff, activation diff, logit diff on dataset |
| `model_comparison` | Weight distance, prediction agreement, importance ranking |
| `cross_model_alignment` | CKA correspondence, head matching, circuit universality |
| `behavior_alignment` | Multi-model comparison, interpretability transfer |

### Weight analysis

| Module | Key functions |
|---|---|
| `weight_processing` | Fold LayerNorm, center writing weights, center unembed |
| `weight_importance` | Fisher information, magnitude pruning, lottery ticket mask |
| `weight_structure` | Spectral analysis, parameter utilization, weight norms |
| `weight_decomposition` | SVD, clustering, spectral analysis, norm distribution |

### Training & development

| Module | Key functions |
|---|---|
| `training` | Algorithmic datasets, `train_tiny_model`, checkpoint analysis |
| `training_dynamics` | Phase transitions, grokking, circuit formation |
| `developmental_interpretability` | Crystallization, grokking dynamics |

### Safety & robustness

| Module | Key functions |
|---|---|
| `safety_relevant_features` | Refusal direction, deception detection, alignment circuits |
| `memorization_detection` | Memorization scores, extractability, trigger localization |
| `mechanistic_anomaly_detection` | Activation profiles, trojan signatures |
| `robustness_perturbation` | Weight noise tolerance, mode connectivity |
| `counterfactual` | Contrastive activation diff, necessity/sufficiency |

### Visualization

| Module | Key functions |
|---|---|
| `visualization` | 19 plot functions: attention patterns, logit attribution, causal tracing, prediction trajectory, and more |

### Generation

| Module | Key functions |
|---|---|
| `generation` | `generate`, `generate_with_cache`, top-k/nucleus sampling |

## Notebooks

The `notebooks/` directory contains 350+ Jupyter notebooks with worked examples and API demos covering every module.

**Start here:** `00_getting_started.ipynb` — a complete mechinterp investigation walkthrough from loading a model through logit lens, activation patching, and circuit analysis.

To run notebooks:

```bash
uv run jupyter lab notebooks/
```

If you don't have Jupyter installed yet:

```bash
uv add --dev jupyterlab
uv run jupyter lab notebooks/
```

**Recommended reading order:**

| Notebook | Topic |
|---|---|
| `00_getting_started` | End-to-end mechinterp workflow |
| `01_api_cheatsheet` | Quick reference for every module |
| `02_transformer_anatomy` | Understanding transformer components |
| `03_logit_lens` | Logit lens and tuned lens |
| `04_ioi_patching` | Activation patching on IOI |
| `05_linear_probes` | Training linear probes |
| `06_sparse_autoencoders` | SAE feature discovery |
| `07_circuit_analysis` | OV/QK circuits and composition |
| `08_gradient_interpretability` | Gradient-based attribution |
| `09_automatic_circuit_discovery` | ACDC-style circuit finding |

The remaining 340+ notebooks cover individual modules in depth — each module in the [module reference](#module-reference) above has a corresponding notebook.

## Testing

```bash
uv run pytest                       # run full suite (4000+ tests)
uv run pytest irtk/tests/ -x -q     # quick smoke test, stop on first failure
```

## License

Apache 2.0
