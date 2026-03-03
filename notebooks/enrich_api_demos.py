#!/usr/bin/env python3
"""
Enrich IRTK API demo notebooks (12-332) with "Why This Matters" motivation cells.

Inserts a markdown cell after cell 0 (the title) containing:
- 2-5 sentence motivation explaining research value
- Key paper references with real URLs

Safe to run multiple times (skips notebooks that already have the cell).
"""

import json
import os
import re
import sys

# =============================================================================
# Theme definitions with references
# =============================================================================

REFS = {
    "framework": '[Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) — Foundational framework for attention head and residual stream analysis',
    "induction": '[Olsson et al. (2022) "In-context Learning and Induction Heads"](https://arxiv.org/abs/2209.11895) — Discovers induction heads and training phase transitions',
    "ioi": '[Wang et al. (2023) "Interpretability in the Wild: IOI"](https://arxiv.org/abs/2211.00593) — Detailed circuit analysis of indirect object identification',
    "acdc": '[Conmy et al. (2023) "Towards Automated Circuit Discovery"](https://arxiv.org/abs/2304.14997) — Automated methods for circuit finding via edge pruning',
    "geva": '[Geva et al. (2021) "Transformer Feed-Forward Layers Are Key-Value Memories"](https://arxiv.org/abs/2012.14913) — Shows MLPs function as key-value memory stores',
    "bills": '[Bills et al. (2023) "Language Models Can Explain Neurons"](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) — Automated neuron interpretation methods',
    "monosemantic": '[Bricken et al. (2023) "Towards Monosemanticity"](https://transformer-circuits.pub/2023/monosemantic-features/index.html) — Sparse autoencoders find interpretable features',
    "sae_interp": '[Cunningham et al. (2023) "Sparse Autoencoders Find Highly Interpretable Features"](https://arxiv.org/abs/2309.08600) — SAE features in larger language models',
    "logit_lens": '[nostalgebraist (2020) "interpreting GPT: the logit lens"](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) — Project intermediate residuals through the unembedding',
    "tuned_lens": '[Belrose et al. (2023) "Eliciting Latent Predictions"](https://arxiv.org/abs/2303.08112) — Learned affine probes improve on the logit lens',
    "probing": '[Belinkov (2022) "Probing Classifiers"](https://arxiv.org/abs/2102.12452) — Methodology for probing neural representations',
    "activation_addition": '[Turner et al. (2023) "Activation Addition"](https://arxiv.org/abs/2308.10248) — Steering model behavior by adding vectors to activations',
    "iti": '[Li et al. (2023) "Inference-Time Intervention"](https://arxiv.org/abs/2306.03341) — Targeted activation interventions for truthfulness',
    "cka": '[Kornblith et al. (2019) "Similarity of Neural Network Representations"](https://arxiv.org/abs/1905.00414) — CKA for comparing representations across models and layers',
    "rome": '[Meng et al. (2022) "Locating and Editing Factual Associations"](https://arxiv.org/abs/2202.05262) — ROME: rank-one model editing for fact modification',
    "causal_tracing": '[Meng et al. (2022) "Locating and Editing Factual Associations"](https://arxiv.org/abs/2202.05262) — Causal tracing to locate factual knowledge',
    "repe": '[Zou et al. (2023) "Representation Engineering"](https://arxiv.org/abs/2310.01405) — Reading and controlling model behavior via representations',
    "shapley": '[Sundararajan & Najmi (2020) "The Many Shapley Values"](https://arxiv.org/abs/1908.08474) — Principled attribution using game-theoretic methods',
}


# =============================================================================
# Module-specific motivations
# Each entry: module_name -> (motivation_text, [ref_keys])
# =============================================================================

MOTIVATIONS = {
    # --- Superposition & features (12) ---
    "superposition": (
        "Models represent more features than they have dimensions by encoding them as nearly-orthogonal directions — a phenomenon called superposition. Understanding superposition is essential because it explains why individual neurons are polysemantic and motivates the search for better decomposition methods like sparse autoencoders.",
        ["monosemantic", "framework"],
    ),
    # --- Training (13, 19, 63) ---
    "training_tiny_models": (
        "Training small models from scratch lets you study how circuits form during learning. By controlling the training data and observing weight evolution, you can identify phase transitions (like the sudden emergence of induction heads) and test hypotheses about what drives circuit formation.",
        ["induction"],
    ),
    "training_dynamics": (
        "Training dynamics analysis reveals when and how specific capabilities emerge during learning. Key phenomena like grokking (delayed generalization) and phase transitions (sudden capability jumps) provide clues about the relationship between memorization, generalization, and circuit formation.",
        ["induction"],
    ),
    "developmental_interpretability": (
        "Developmental interpretability studies how circuits crystallize during training — when do specific heads specialize, when do induction circuits form, and what triggers phase transitions? This connects training dynamics to the static circuits found in fully-trained models.",
        ["induction"],
    ),
    # --- Head analysis (14) ---
    "head_analysis": (
        "Attention heads are the primary information-routing mechanism in transformers. Classifying heads by function (previous-token, induction, copying, inhibition) is the first step toward understanding what algorithm the model implements for any given task.",
        ["framework", "induction"],
    ),
    # --- Token embeddings (15) ---
    "token_embeddings": (
        "The embedding space defines the model's initial representation of tokens. Analyzing embedding geometry — nearest neighbors, analogies, and alignment with the unembedding — reveals what semantic structure the model starts with before any attention or MLP processing.",
        ["framework"],
    ),
    # --- Representation geometry (16) ---
    "representation_geometry": (
        "The geometry of internal representations — measured via CKA, intrinsic dimensionality, and subspace overlap — reveals how models organize information across layers. This is key for understanding whether different models learn similar representations and how representations transform through the network.",
        ["cka", "framework"],
    ),
    # --- Knowledge (17) ---
    "knowledge_analysis": (
        "Knowledge neurons are MLP neurons that store specific factual associations. Locating where facts are stored (via causal tracing) and understanding how MLPs recall them is essential for fact editing, understanding hallucination, and measuring what a model actually knows vs. guesses.",
        ["geva", "causal_tracing"],
    ),
    # --- Concept erasure (18) ---
    "concept_erasure": (
        "Concept erasure removes specific information (like gender or sentiment) from representations while preserving other capabilities. This tests what information is linearly accessible and provides a controlled way to study how models encode and use sensitive attributes.",
        ["probing"],
    ),
    # --- Feature attribution (20) ---
    "feature_attribution": (
        "Feature attribution decomposes model predictions into contributions from individual tokens, neurons, and directions. This bridges the gap between input-level explanations (which tokens matter?) and mechanistic understanding (which internal components produce the prediction?).",
        ["framework"],
    ),
    # --- Circuit evaluation (21) ---
    "circuit_evaluation": (
        "Once you find a circuit, how good is it? Circuit evaluation measures faithfulness (does the circuit reproduce the full model's behavior?), completeness (does it capture all relevant computation?), and minimality (are there unnecessary components?). These metrics are essential for rigorous circuit claims.",
        ["ioi", "acdc"],
    ),
    # --- Attention surgery (22) ---
    "attention_surgery": (
        "Attention surgery lets you directly modify attention patterns — knocking out specific heads, forcing attention to particular positions, or patching patterns between runs. These interventions establish causal claims about what attention heads actually do.",
        ["framework", "ioi"],
    ),
    # --- Model surgery (23) ---
    "model_surgery": (
        "Model surgery tools let you transplant components between models, zero out specific heads or layers, and compare head behavior across architectures. This enables controlled experiments that isolate the contribution of individual components.",
        ["framework"],
    ),
    # --- Token interactions (24) ---
    "token_interactions": (
        "Token interactions measure how pairs of input tokens jointly influence predictions — capturing synergies (tokens that matter more together than apart) and redundancies (tokens with overlapping contributions). This reveals the combinatorial structure of model computations.",
        ["framework"],
    ),
    # --- Model editing (25) ---
    "model_editing": (
        "ROME-style model editing modifies specific factual associations by making rank-one updates to MLP weights. Understanding how this works requires knowing where facts are stored and how MLP layers function as key-value memories.",
        ["rome", "geva"],
    ),
    # --- Attention rollout (26) ---
    "attention_rollout": (
        "Attention rollout aggregates attention across layers to estimate the total information flow from input tokens to output positions. Unlike single-layer patterns, rollout accounts for the compositional nature of multi-layer attention.",
        ["framework"],
    ),
    # --- Causal scrubbing (27) ---
    "causal_scrubbing": (
        "Causal scrubbing is a rigorous method for testing whether a hypothesized circuit fully explains a model's behavior. By replacing non-circuit activations with those from unrelated inputs, it tests whether the circuit is sufficient — a stronger claim than correlation.",
        ["ioi", "acdc"],
    ),
    # --- Low rank (28) ---
    "low_rank": (
        "Low-rank analysis reveals the effective dimensionality of weight matrices and attention heads. Many components have surprisingly low rank, meaning they compute in a lower-dimensional subspace than their full capacity would allow. This constrains what computations are possible.",
        ["framework"],
    ),
    # --- Interp benchmarks (29) ---
    "interp_benchmarks": (
        "Interpretability benchmarks provide standardized metrics for evaluating explanations: logit difference, KL divergence, loss recovered, and ablation effect size. Consistent metrics make it possible to compare findings across studies and validate circuit claims.",
        ["ioi"],
    ),
    # --- Sparse features (30) ---
    "sparse_features": (
        "Sparse feature analysis examines the features discovered by sparse autoencoders — their activation patterns, correlations with other features, and downstream effects on predictions. This is how you determine whether SAE features are genuinely interpretable.",
        ["monosemantic", "sae_interp"],
    ),
    # --- Distributed reps (31) ---
    "distributed_reps": (
        "Distributed representation analysis probes how concepts are encoded across multiple dimensions rather than single neurons. Linear probes, representation rank, and cross-layer tracking reveal how information is distributed and transformed through the network.",
        ["probing", "framework"],
    ),
    # --- Ablation study (32) ---
    "ablation_study": (
        "Systematic ablation studies remove components one at a time (or in combination) to measure their importance. Layer-by-layer ablation, head importance matrices, and double-ablation interaction tests reveal which components are necessary and how they depend on each other.",
        ["ioi", "framework"],
    ),
    # --- Patchscopes (33) ---
    "patchscopes": (
        "Patchscopes inspect what information is encoded at specific positions by patching activations into different contexts. This extends the logit lens idea: instead of just projecting through the unembedding, you can test what a representation 'means' by placing it in a context designed to extract that meaning.",
        ["logit_lens", "framework"],
    ),
    # --- Transcoder (34) ---
    "transcoder": (
        "Transcoders learn to map between MLP input and output activations, providing a sparse decomposition of what each MLP layer computes. Unlike SAEs (which decompose a single activation), transcoders capture the input-output transformation, revealing MLP features as computational primitives.",
        ["monosemantic", "geva"],
    ),
    # --- Prediction entropy (35) ---
    "prediction_entropy": (
        "Prediction entropy at each layer measures how 'decided' the model is about its output. Tracking when entropy drops reveals commitment points — the layers where the model transitions from uncertainty to confidence. Early commitment on easy tokens vs. late commitment on hard ones reflects the model's internal difficulty assessment.",
        ["logit_lens"],
    ),
    # --- Principled attribution (36) ---
    "principled_attribution": (
        "Shapley values and KernelSHAP provide game-theoretically principled attribution scores that satisfy desirable axioms (efficiency, symmetry, null player). While more expensive than gradient-based methods, they give unambiguous answers about component importance and can detect interaction effects.",
        ["shapley"],
    ),
    # --- Weight importance (37) ---
    "weight_importance": (
        "Weight importance analysis identifies which parameters matter most for model behavior. Fisher information, magnitude pruning, and lottery ticket analysis reveal the sparse substructure within dense networks — often a small fraction of weights carry most of the model's capability.",
        ["framework"],
    ),
    # --- Compositional structure (38) ---
    "compositional_structure": (
        "Compositional structure analysis identifies how the model decomposes complex computations into modular subroutines. Information bottleneck analysis, subroutine clustering, and algorithmic decomposition reveal the model's computational architecture at a higher level than individual heads.",
        ["acdc", "ioi"],
    ),
    # --- Counterfactual (39) ---
    "counterfactual": (
        "Counterfactual analysis asks: what's the minimal change to the input that would flip the model's prediction? Contrastive activation differences, necessity/sufficiency scores, and minimal-change token identification establish tight causal connections between inputs and outputs.",
        ["ioi"],
    ),
    # --- Auto interp (40) ---
    "auto_interp": (
        "Automated interpretability tools generate labels and summaries for heads and neurons based on their activation patterns and output effects. While not a replacement for manual analysis, they help prioritize which components to investigate and provide a starting point for understanding large models.",
        ["bills"],
    ),
    # --- Polysemantic features (41) ---
    "polysemantic_features": (
        "Polysemanticity analysis quantifies how much each neuron responds to multiple unrelated concepts. Context clustering, activation decomposition, and interference matrices reveal the degree of feature overlap — directly measuring the problem that sparse autoencoders aim to solve.",
        ["monosemantic"],
    ),
    # --- Attribution graphs (42) ---
    "attribution_graphs": (
        "Attribution graphs trace feature-to-feature computation across layers, revealing the directed graph of how information flows from input features through intermediate computations to output features. This provides a circuit-level view of model computation with explicit paths and interaction strengths.",
        ["acdc", "framework"],
    ),
    # --- Representation engineering (43) ---
    "representation_engineering": (
        "Representation engineering (RepE) extracts 'reading vectors' that detect specific concepts in activations, and 'control vectors' that steer model behavior when added to the residual stream. This bridges interpretability and control — understanding what the model represents enables targeted interventions.",
        ["repe", "activation_addition"],
    ),
    # --- Context sensitivity (44) ---
    "context_sensitivity": (
        "Context sensitivity analysis measures how a position's representation changes as the surrounding context varies. This reveals whether heads operate locally (nearby tokens only) or globally, how the effective context window compares to the architectural maximum, and how in-context learning unfolds.",
        ["framework", "induction"],
    ),
    # --- Influence functions (45) ---
    "influence_functions": (
        "Influence functions attribute model behavior back to specific training examples — identifying which training data points most influenced a prediction. This connects mechanistic understanding (which components produce the output) to data attribution (which data taught the model this behavior).",
        ["framework"],
    ),
    # --- Mechanistic hypotheses (46) ---
    "mechanistic_hypotheses": (
        "Mechanistic hypothesis tools help automate the scientific process of interpretability: forming hypotheses about what circuits do, testing them against model behavior, and refining them based on evidence. This supports scaling interpretability beyond manual analysis.",
        ["ioi", "acdc"],
    ),
    # --- Behavior alignment (47) ---
    "behavior_alignment": (
        "Multi-model comparison reveals whether different architectures learn the same algorithms. Head correspondence, circuit alignment, and interpretability transfer tests address the universality question: are the circuits we find in GPT-2 also present in other models?",
        ["induction", "framework"],
    ),
    # --- Feature geometry (48) ---
    "feature_geometry": (
        "SAE dictionary geometry analyzes the structure of learned feature dictionaries — how features split, absorb each other, or form interaction graphs. Understanding feature geometry is essential for evaluating whether SAEs find the 'true' features of a model or artifacts of the training procedure.",
        ["monosemantic", "sae_interp"],
    ),
    # --- ICL mechanisms (49) ---
    "icl_mechanisms": (
        "In-context learning (ICL) mechanisms reveal how models learn from examples in the prompt without weight updates. Task vectors, induction head identification, and implicit gradient descent tests probe whether ICL uses the same circuits as few-shot learning or implements a fundamentally different algorithm.",
        ["induction"],
    ),
    # --- Cross coder (50) ---
    "cross_coder": (
        "CrossCoders train sparse autoencoders jointly on activations from two models (e.g., base and finetuned), discovering shared features and model-specific features. This is a powerful tool for understanding what changes during finetuning at the feature level.",
        ["monosemantic"],
    ),
    # --- Information flow (51) ---
    "information_flow": (
        "Information flow analysis measures how much information is transmitted, compressed, or created at each layer using entropy and mutual information. This reveals the model's information processing pipeline: which layers compress, which transform, and where bottlenecks occur.",
        ["framework"],
    ),
    # --- Circuit discovery (52) ---
    "circuit_discovery": (
        "Automated circuit discovery (ACDC-style) uses iterative edge pruning and subnetwork probing to find minimal circuits for specific behaviors. This scales beyond manual circuit analysis by systematically testing which connections between components are necessary.",
        ["acdc", "ioi"],
    ),
    # --- Token position attribution (53) ---
    "token_position_attribution": (
        "Token-position attribution traces which input positions contribute to the prediction at each layer. Position-specific gradient analysis and flow tracking reveal the model's information routing — which tokens are read from and when their information reaches the output.",
        ["framework"],
    ),
    # --- SAE feature steering (54) ---
    "sae_feature_steering": (
        "SAE feature steering enables fine-grained behavioral control by clamping or scaling individual SAE features. Unlike residual stream steering (which modifies a direction in a polysemantic space), feature-level interventions target specific interpretable features, enabling precise and predictable effects.",
        ["monosemantic", "activation_addition"],
    ),
    # --- Backup detection (55) ---
    "backup_detection": (
        "Backup head detection identifies redundant circuits that compensate when primary components are ablated. Understanding circuit redundancy is crucial for evaluating ablation results: if a 'backup' activates after knockout, the original ablation may underestimate the component's normal role.",
        ["ioi"],
    ),
    # --- Binding analysis (56) ---
    "binding_analysis": (
        "Entity-attribute binding analysis studies how models track which properties belong to which entities in context. This is fundamental to language understanding — the model must correctly bind 'red' to 'car' and 'blue' to 'house' in complex sentences.",
        ["framework", "ioi"],
    ),
    # --- Probing dynamics (57) ---
    "probing_dynamics": (
        "Probing dynamics tracks how probe accuracy changes across layers, revealing where specific information emerges, peaks, and potentially disappears. Emergence thresholds, calibration, and selectivity measurements characterize the model's layer-by-layer information processing.",
        ["probing"],
    ),
    # --- Cross model alignment (58) ---
    "cross_model_alignment": (
        "Cross-model alignment measures whether different architectures learn corresponding representations. CKA correspondence, head matching, and circuit universality tests address a core question: do our interpretability findings generalize across models, or are they artifacts of specific architectures?",
        ["cka", "induction"],
    ),
    # --- Internal confidence (59) ---
    "internal_confidence": (
        "Internal confidence analysis finds directions in activation space that predict how confident the model is in its output, even when that confidence doesn't match the output probabilities. Gaps between internal confidence and output calibration reveal when models 'know they don't know.'",
        ["framework"],
    ),
    # --- Norm mechanics (60) ---
    "norm_mechanics": (
        "LayerNorm is not just a normalization step — it actively shapes which features get amplified and which get suppressed. Understanding LayerNorm's feature scaling, directionality bias, and gradient flow effects is essential because every component's output passes through it.",
        ["framework"],
    ),
    # --- Semantic saturation (61) ---
    "semantic_saturation": (
        "Semantic saturation analysis identifies where in the network new information stops being added. Information saturation, redundant layers, and stabilization points reveal which layers are doing useful work and which are largely passing information through — informing both pruning and circuit analysis.",
        ["framework"],
    ),
    # --- MLP decomposition (62) ---
    "mlp_decomposition": (
        "MLP decomposition breaks down the MLP's computation into individual neuron contributions, feature directions, and knowledge storage patterns. Understanding how nonlinearities (GELU/ReLU) interact with the key-value structure reveals the mechanisms of MLP computation at a fine-grained level.",
        ["geva"],
    ),
    # --- Mechanistic anomaly detection (64) ---
    "mechanistic_anomaly_detection": (
        "Mechanistic anomaly detection identifies unusual activation patterns that may indicate trojans, backdoors, or unexpected behaviors. By establishing baseline activation profiles and flagging deviations, this connects interpretability directly to model safety and security.",
        ["framework"],
    ),
    # --- Outlier dimensions (65) ---
    "outlier_dimensions": (
        "Outlier dimension analysis identifies the 'massive activations' phenomenon where a small number of residual stream dimensions have disproportionately large values. Understanding attention sinks, dimension utilization, and the impact of removing outliers reveals structural properties of transformer representations.",
        ["framework"],
    ),
    # --- Robustness perturbation (66) ---
    "robustness_perturbation": (
        "Robustness analysis measures how sensitive model behavior is to weight noise, identifying critical parameters whose perturbation has outsized effects. This connects to pruning (which parameters can be removed?) and understanding which weights are load-bearing vs. redundant.",
        ["framework"],
    ),
    # --- Memorization detection (67) ---
    "memorization_detection": (
        "Memorization detection identifies when the model is reproducing training data verbatim rather than generalizing. Memorization scores, extractability metrics, and trigger localization are essential for understanding privacy risks and the boundary between memorization and generalization.",
        ["framework"],
    ),
    # --- Sequence dynamics (68) ---
    "sequence_dynamics": (
        "Sequence dynamics analysis reveals how models handle positional patterns: repetition detection, long-range dependencies, position bias, and length effects. Understanding these dynamics is crucial for explaining why models succeed or fail on specific sequence structures.",
        ["induction", "framework"],
    ),
    # --- Activation statistics (69) ---
    "activation_statistics": (
        "Activation statistics — moments, kurtosis, normality, sparsity, and multimodality — provide a quantitative profile of how each layer processes information. Unusual statistics (e.g., extreme kurtosis or bimodal distributions) flag components worth investigating in detail.",
        ["framework"],
    ),
    # --- Logit dynamics (70) ---
    "logit_dynamics": (
        "Logit dynamics tracks how the model's prediction changes layer by layer — when do logit flips occur, when does the prediction stabilize, and which components cause the decisive changes? This extends the logit lens from a static snapshot to a dynamic view of prediction formation.",
        ["logit_lens"],
    ),
    # --- Multi-token prediction (71) ---
    "multi_token_prediction": (
        "Multi-token prediction analysis probes whether models plan ahead — can intermediate representations predict not just the next token but tokens further in the future? Planning horizon and next-k accuracy measurements reveal whether models implement genuine lookahead or process one token at a time.",
        ["framework"],
    ),
    # --- Copy suppression (72) ---
    "copy_suppression": (
        "Copy suppression heads actively prevent the model from copying tokens that appear in the input but shouldn't be predicted. Understanding these 'negative' heads — which tokens they suppress and when — is essential for understanding tasks like IOI where the model must distinguish between repeated and non-repeated names.",
        ["ioi", "framework"],
    ),
    # --- Function vectors (73) ---
    "function_vectors": (
        "Function vectors capture the representation of a task (like 'capitalize' or 'translate to French') as a direction in activation space. These vectors can transfer tasks between contexts, providing evidence that models represent tasks and inputs in separable subspaces.",
        ["repe"],
    ),
    # --- Prediction geometry (74) ---
    "prediction_geometry": (
        "Prediction geometry tracks the trajectory of the residual stream through vocabulary space as it approaches the final prediction. Sharpening rate, unembedding alignment, and prediction decomposition reveal the geometric process by which diffuse representations converge to sharp predictions.",
        ["logit_lens", "framework"],
    ),
    # --- Layer composition (75) ---
    "layer_composition": (
        "Layer composition analysis measures how each layer's contribution interacts with previous layers — do they add independent information, reinforce existing predictions, or interfere? Understanding composition patterns (additive, multiplicative, corrective) reveals the model's computational strategy.",
        ["framework"],
    ),
    # --- Attention motif discovery (76) ---
    "attention_motif_discovery": (
        "Attention motif discovery uses SVD to extract recurring attention patterns across different inputs. These motifs — like 'attend to previous token' or 'attend to separator' — represent the functional building blocks from which complex behaviors are composed.",
        ["framework", "induction"],
    ),
    # --- Feature deletion sensitivity (77) ---
    "feature_deletion_sensitivity": (
        "Feature deletion sensitivity measures how much removing individual features (neurons, directions, or SAE features) impacts predictions. Deletion cascades and interaction tests reveal which features are independently important vs. which only matter in combination.",
        ["framework"],
    ),
    # --- Path expansion (78) ---
    "path_expansion": (
        "Path expansion enumerates and quantifies the computational paths through the network — from specific input tokens through attention heads and MLPs to specific output logits. This makes the implicit computation graph of a transformer explicit and measurable.",
        ["framework", "ioi"],
    ),
    # --- Activation patching variants (79) ---
    "activation_patching_variants": (
        "Different patching methods (denoising, noising, mean ablation, resample ablation) make different assumptions about what counts as a 'null' baseline. Understanding which variant to use and how results differ is essential for drawing valid causal conclusions from patching experiments.",
        ["ioi"],
    ),
    # --- Vocabulary dynamics (80) ---
    "vocabulary_dynamics": (
        "Vocabulary dynamics analyzes the structure of the embedding and unembedding spaces — alignment between W_E and W_U, frequency bias, isotropy, and neighborhood structure. These properties constrain what the model can represent and predict at the token level.",
        ["framework"],
    ),
    # --- Gradient flow (81) ---
    "gradient_flow": (
        "Gradient flow analysis traces how learning signals propagate backward through the network. Gradient norms, saturation, and LayerNorm effects reveal which components receive strong training signals and which are in gradient dead zones — connecting architecture to trainability.",
        ["framework"],
    ),
    # --- Attention pattern analysis (82) ---
    "attention_pattern_analysis": (
        "Systematic attention pattern analysis — entropy, positional bias, sparsity, cross-head similarity, and automatic classification — provides a comprehensive profile of how attention operates across all heads. This is the foundation for identifying specialized heads and understanding information routing.",
        ["framework", "induction"],
    ),
    # --- Logit attribution decomposition (83) ---
    "logit_attribution_decomposition": (
        "Full logit decomposition attributes the final logit of any token to specific attention heads, MLP layers, and positions. Cumulative buildup plots show how the prediction assembles layer by layer, revealing which components promote and which suppress each prediction.",
        ["framework", "logit_lens"],
    ),
    # --- Residual dynamics (84) ---
    "residual_dynamics": (
        "Residual dynamics tracks how the residual stream changes through the network — drift analysis, signal-to-noise decomposition, and projection tracking reveal whether information is being refined, corrupted, or transformed. This is key to understanding the residual stream as a shared communication channel.",
        ["framework", "logit_lens"],
    ),
    # --- Intervention effects (85) ---
    "intervention_effects": (
        "Intervention effect analysis measures the downstream consequences of modifying activations — scaling sensitivity, direction sweeps, knockout recovery, and transferability. This characterizes the causal structure of the network: not just which components matter, but how sensitive behavior is to the magnitude and direction of changes.",
        ["activation_addition", "iti"],
    ),
    # --- Weight structure (86) ---
    "weight_structure": (
        "Weight structure analysis examines the spectral properties, parameter utilization, and internal organization of weight matrices. The structure of weights constrains what computations are possible, and spectral analysis can reveal low-rank structure, symmetries, and functional specialization.",
        ["framework"],
    ),
    # --- Token prediction analysis (87) ---
    "token_prediction_analysis": (
        "Token prediction analysis examines model confidence, surprisal, difficulty, and agreement across positions. Position-level prediction quality metrics reveal which parts of the input are easy vs. hard for the model, and how prediction difficulty correlates with internal computation.",
        ["framework"],
    ),
    # --- Subnetwork analysis (88) ---
    "subnetwork_analysis": (
        "Subnetwork analysis extracts minimal circuits that faithfully reproduce model behavior on specific tasks. Faithfulness, minimality, and comparison metrics let you evaluate whether a discovered subnetwork truly captures the relevant computation or misses important components.",
        ["acdc", "ioi"],
    ),
    # --- Activation geometry (89) ---
    "activation_geometry": (
        "Activation geometry measures the manifold structure of internal representations — intrinsic dimensionality, representation similarity, clustering, and curvature. These geometric properties reveal how the model organizes information in its high-dimensional activation space.",
        ["cka", "framework"],
    ),
    # --- Feature circuits (90) ---
    "feature_circuits": (
        "Feature circuits trace how SAE features propagate and compose across layers — a feature-level analog of circuit analysis. Composition scores, path attribution, and interaction matrices reveal the computational graph in terms of interpretable features rather than opaque activations.",
        ["monosemantic", "acdc"],
    ),
    # --- Attention head probing (91) ---
    "attention_head_probing": (
        "Attention head probing uses targeted probes to understand what information each head encodes in its output. Positional probes, identity probes, and specialization measurements characterize heads from the perspective of what they compute, complementing pattern-based analysis.",
        ["probing", "framework"],
    ),
    # --- Representation bottleneck (92) ---
    "representation_bottleneck": (
        "Representation bottleneck analysis identifies where the network compresses information — where capacity is limited and what information is lost. Compression metrics, capacity estimates, and flow bottleneck detection reveal the information-theoretic constraints on model computation.",
        ["framework"],
    ),
    # --- Causal abstraction (93) ---
    "causal_abstraction": (
        "Causal abstraction tests whether a high-level algorithm (like 'compare two numbers') is faithfully implemented by a neural circuit. Interchange interventions and distributed alignment search (DAS) provide rigorous methods for mapping between algorithmic descriptions and neural implementations.",
        ["ioi"],
    ),
    # --- Embedding dynamics (94) ---
    "embedding_dynamics": (
        "Embedding dynamics tracks how token representations evolve across layers — identity decay (how quickly the original token embedding is overwritten), semantic drift, and context mixing. This reveals the transformation from token-level to contextual representations.",
        ["framework"],
    ),
    # --- Attention flow analysis (95) ---
    "attention_flow_analysis": (
        "Attention flow analysis traces information paths through the network by composing attention patterns across layers. Flow paths, bottleneck detection, and source attribution reveal the multi-layer routing of information that single-layer attention patterns cannot capture.",
        ["framework"],
    ),
    # --- Layer pruning analysis (96) ---
    "layer_pruning_analysis": (
        "Layer pruning analysis identifies which layers can be removed with minimal impact on model performance. Skip analysis, progressive pruning, and criticality measurements reveal the effective depth of the model — often much less than the architectural depth.",
        ["framework"],
    ),
    # --- Token erasure (97) ---
    "token_erasure": (
        "Token erasure measures the effect of removing each input token on the model's prediction, establishing necessity and sufficiency for individual tokens. Erasure curves and pairwise interactions reveal which tokens are essential and how they cooperate to produce the output.",
        ["framework"],
    ),
    # --- Computation graph (98) ---
    "computation_graph": (
        "Computation graph analysis makes the implicit dataflow of a transformer explicit — tracing dependencies between components, identifying critical paths, and measuring interaction strengths. This provides a structural view of how information flows through the network.",
        ["acdc", "framework"],
    ),
    # --- Activation surgery (99) ---
    "activation_surgery": (
        "Activation surgery provides precise tools for modifying internal activations: clamping, scaling, projecting, rotating, and replacing. These fine-grained interventions enable controlled experiments that go beyond simple ablation to test specific mechanistic hypotheses.",
        ["activation_addition", "framework"],
    ),
}

# =============================================================================
# Theme-based motivation templates for modules not explicitly listed above
# =============================================================================

THEME_TEMPLATES = {
    "attention": {
        "motivation": "Attention {specific} reveals how heads route information between positions. Understanding attention mechanics is central to reverse-engineering transformer circuits, as attention patterns determine what information each component can access and transform.",
        "refs": ["framework", "induction"],
    },
    "mlp": {
        "motivation": "MLP {specific} provides tools for understanding how feed-forward layers transform and store information. Since MLPs have been shown to function as key-value memories, understanding their internal mechanics is essential for locating knowledge and understanding factual recall.",
        "refs": ["geva", "bills"],
    },
    "circuit": {
        "motivation": "Circuit {specific} helps identify and characterize the specific subnetworks responsible for model behaviors. Finding circuits — the algorithms models actually implement — is the core goal of mechanistic interpretability.",
        "refs": ["ioi", "acdc"],
    },
    "residual": {
        "motivation": "Residual stream {specific} characterizes the shared information bus that all transformer components read from and write to. Because the residual stream carries all inter-component communication, understanding its structure is fundamental to understanding the model as a whole.",
        "refs": ["framework", "logit_lens"],
    },
    "feature": {
        "motivation": "Feature {specific} investigates the interpretable directions that models use internally. Understanding features — the natural units of neural computation — is essential for moving beyond neuron-level analysis to a true understanding of what models represent.",
        "refs": ["monosemantic", "sae_interp"],
    },
    "probe": {
        "motivation": "Probing {specific} tests what information is linearly accessible in model representations. Linear probes provide a principled way to measure whether specific concepts (syntax, semantics, factual knowledge) are encoded at each layer.",
        "refs": ["probing"],
    },
    "steer": {
        "motivation": "Steering and intervention {specific} enables controlled modification of model behavior by adding, removing, or modifying specific activation components. These causal interventions go beyond correlation to establish what internal representations actually do.",
        "refs": ["activation_addition", "iti"],
    },
    "gradient": {
        "motivation": "Gradient {specific} traces how training signals and attribution scores flow through the network. Understanding gradient structure reveals which components are most trainable, which carry the strongest attribution signals, and how LayerNorm affects learning.",
        "refs": ["framework"],
    },
    "weight": {
        "motivation": "Weight {specific} examines the structure and statistics of model parameters. The structure of weights constrains what computations a component can perform, and spectral analysis can reveal functional specialization, redundancy, and low-rank structure.",
        "refs": ["framework"],
    },
    "logit": {
        "motivation": "Logit {specific} analyzes how the model's output predictions form and change across layers. Understanding logit formation — which components contribute what to the final prediction — is central to explaining model behavior.",
        "refs": ["logit_lens", "tuned_lens"],
    },
    "model": {
        "motivation": "Model {specific} provides tools for systematic analysis and comparison of transformer internals. These diagnostics help identify unexpected behaviors, compare architectures, and build a comprehensive picture of how the model processes information.",
        "refs": ["framework"],
    },
    "token": {
        "motivation": "Token {specific} tracks how individual token representations evolve through the network. Understanding token-level dynamics reveals how context is integrated, when predictions form, and how different positions interact to produce the output.",
        "refs": ["framework"],
    },
    "layer": {
        "motivation": "Layer {specific} characterizes what each layer contributes to the model's computation. Understanding layer-level organization — which layers are critical, which are redundant, and how they specialize — is essential for both interpretability and efficient model design.",
        "refs": ["framework"],
    },
    "embedding": {
        "motivation": "Embedding {specific} examines the structure of the model's token representation spaces. The embedding and unembedding matrices define the model's interface with language, and their geometry constrains what semantic relationships the model can represent.",
        "refs": ["framework"],
    },
    "prediction": {
        "motivation": "Prediction {specific} analyzes how the model arrives at its output predictions. Tracking prediction formation across layers reveals the computational process — when the model commits to an answer, what changes its mind, and how confident it is.",
        "refs": ["logit_lens", "framework"],
    },
    "head": {
        "motivation": "Head {specific} examines the behavior and function of individual attention heads. Since each head has a specific computational role (routing information, copying tokens, inhibiting predictions), characterizing heads is fundamental to understanding transformer circuits.",
        "refs": ["framework", "induction"],
    },
    "cross": {
        "motivation": "Cross-component {specific} measures how information transfers between different parts of the model. Understanding cross-component interactions reveals the collaborative computation that produces emergent model capabilities.",
        "refs": ["framework"],
    },
    "representation": {
        "motivation": "Representation {specific} characterizes the structure of internal model representations. Understanding how models organize information — which concepts are linearly separable, how representations cluster, and how they change across layers — is central to interpretability.",
        "refs": ["probing", "cka"],
    },
    "activation": {
        "motivation": "Activation {specific} characterizes the patterns and statistics of internal model activations. Activation structure reveals how the model processes information — which components are active, how sparse the computation is, and what geometric patterns emerge.",
        "refs": ["framework"],
    },
    "safety": {
        "motivation": "Safety-relevant {specific} connects interpretability directly to AI safety by identifying internal representations of safety-critical behaviors. Locating refusal directions, deception signatures, and alignment-relevant circuits enables monitoring and intervention for safer AI systems.",
        "refs": ["framework", "repe"],
    },
    "qk": {
        "motivation": "QK circuit {specific} analyzes the query-key interaction that determines attention patterns. Understanding what features drive attention — positional vs. content-based, local vs. global — reveals how the model decides which information to route where.",
        "refs": ["framework"],
    },
    "ov": {
        "motivation": "OV circuit {specific} analyzes the value-output projection that determines what information flows through attention. Understanding what each head copies when it attends to a position reveals the head's computational role in the model's circuits.",
        "refs": ["framework"],
    },
    "unembed": {
        "motivation": "Unembedding {specific} examines how the model's final representation is projected into vocabulary space to produce predictions. The unembedding matrix is the lens through which all internal computation becomes observable, making its structure fundamental to interpretability.",
        "refs": ["framework", "logit_lens"],
    },
    "intervention": {
        "motivation": "Intervention {specific} provides tools for controlled modifications of model internals. Causal interventions are the gold standard for mechanistic claims — they establish not just what correlates with behavior, but what actually causes it.",
        "refs": ["activation_addition", "ioi"],
    },
    "causal": {
        "motivation": "Causal {specific} applies rigorous causal inference methods to understanding model internals. Causal methods go beyond correlation to establish which components are genuinely responsible for specific behaviors, providing the strongest form of mechanistic evidence.",
        "refs": ["ioi", "acdc"],
    },
    "knowledge": {
        "motivation": "Knowledge {specific} investigates where and how models store factual information. Locating knowledge in specific components (neurons, layers, directions) is essential for understanding hallucination, enabling fact editing, and assessing what a model truly knows.",
        "refs": ["geva", "causal_tracing"],
    },
    "sparse": {
        "motivation": "Sparse {specific} investigates representations that are sparse — where only a few components are active for any given input. Sparsity is a key property of interpretable representations, as it enables individual features to be studied in isolation.",
        "refs": ["monosemantic"],
    },
    "composition": {
        "motivation": "Composition {specific} measures how components interact across layers to implement complex computations. Understanding composition — how one head's output feeds into another's input — is essential for tracing multi-step algorithms in transformer circuits.",
        "refs": ["framework", "induction"],
    },
    "norm": {
        "motivation": "Norm {specific} analyzes the scale and magnitude of internal representations. Representation norms carry meaningful signal — they indicate component importance, reveal outlier dimensions, and constrain what information can be transmitted through the residual stream.",
        "refs": ["framework"],
    },
    "path": {
        "motivation": "Path {specific} traces specific computational pathways through the network. Path-level analysis reveals how information flows from specific input tokens through specific components to specific output predictions, providing the most detailed view of model computation.",
        "refs": ["framework", "acdc"],
    },
    "ablation": {
        "motivation": "Ablation {specific} measures the effect of removing or zeroing specific components. Ablation is the simplest causal intervention: if removing a component changes behavior, that component is necessary. Systematic ablation reveals the importance hierarchy of model components.",
        "refs": ["ioi", "framework"],
    },
    "scaling": {
        "motivation": "Scaling {specific} investigates how model properties change with size. Understanding scaling trends — does a circuit become more or less important in larger models? — is essential for determining whether interpretability findings generalize beyond the specific model studied.",
        "refs": ["induction"],
    },
    "fallback": {
        "motivation": "This module provides tools for analyzing model internals. Understanding the internal mechanisms of transformer models is the core goal of mechanistic interpretability research — enabling us to move from treating models as black boxes to understanding the algorithms they implement.",
        "refs": ["framework"],
    },
}


def classify_module(module_name: str) -> str:
    """Classify a module name into a theme using keyword matching."""
    name = module_name.lower()

    # Specific matches first
    if "qk_" in name or name.startswith("qk") or "key_query" in name:
        return "qk"
    if "ov_" in name or name.startswith("ov"):
        return "ov"
    if "unembed" in name:
        return "unembed"
    if "safety" in name or "anomaly" in name:
        return "safety"
    if "knowledge" in name and "edit" not in name:
        return "knowledge"
    if "causal" in name:
        return "causal"
    if "ablation" in name:
        return "ablation"
    if "path" in name and "patch" not in name:
        return "path"
    if "composition" in name or "composit" in name:
        return "composition"
    if "norm_" in name or name.endswith("_norm") or "layernorm" in name:
        return "norm"
    if "sparse" in name and "feature" not in name:
        return "sparse"
    if "sae" in name or "cross_coder" in name or "transcoder" in name:
        return "feature"
    if any(w in name for w in ["attention", "attn", "head_", "_head"]):
        return "attention"
    if "head" in name and "attention" not in name:
        return "head"
    if any(w in name for w in ["mlp", "neuron", "gating", "gate_"]):
        return "mlp"
    if any(w in name for w in ["circuit", "motif"]):
        return "circuit"
    if any(w in name for w in ["residual", "resid"]):
        return "residual"
    if any(w in name for w in ["feature", "polysemantic", "superposition"]):
        return "feature"
    if any(w in name for w in ["probe", "probing"]):
        return "probe"
    if any(w in name for w in ["steer", "intervention", "surgery"]):
        return "steer" if "surgery" not in name else "intervention"
    if any(w in name for w in ["gradient", "grad_"]):
        return "gradient"
    if any(w in name for w in ["weight", "parameter"]):
        return "weight"
    if any(w in name for w in ["logit"]):
        return "logit"
    if any(w in name for w in ["model_comp", "cross_model", "behavior_alignment"]):
        return "model"
    if any(w in name for w in ["scaling"]):
        return "scaling"
    if any(w in name for w in ["token"]):
        return "token"
    if any(w in name for w in ["layer"]):
        return "layer"
    if any(w in name for w in ["embedding", "embed"]):
        return "embedding"
    if any(w in name for w in ["prediction", "predict"]):
        return "prediction"
    if any(w in name for w in ["representation", "similarity"]):
        return "representation"
    if any(w in name for w in ["activation", "activ"]):
        return "activation"
    if any(w in name for w in ["cross_"]):
        return "cross"
    if any(w in name for w in ["model", "dashboard", "diagnostic", "benchmark", "hypothesis"]):
        return "model"
    return "fallback"


def make_specific_phrase(module_name: str, theme: str) -> str:
    """Generate a specific phrase from module name for template insertion."""
    name = module_name.replace("_analysis", "").replace("_profiling", "")
    # Remove theme keyword to avoid "Attention attention distance"
    theme_words = {
        "attention": ["attention", "attn"],
        "mlp": ["mlp"],
        "circuit": ["circuit"],
        "residual": ["residual", "resid"],
        "feature": ["feature"],
        "probe": ["probe", "probing"],
        "steer": ["steer", "steering"],
        "gradient": ["gradient", "grad"],
        "weight": ["weight"],
        "logit": ["logit"],
        "token": ["token"],
        "layer": ["layer"],
        "embedding": ["embedding", "embed"],
        "prediction": ["prediction", "predict"],
        "head": ["head"],
        "activation": ["activation"],
        "representation": ["representation"],
        "ov": ["ov"],
        "qk": ["qk"],
        "norm": ["norm"],
        "causal": ["causal"],
        "ablation": ["ablation"],
        "path": ["path"],
        "composition": ["composition"],
        "safety": ["safety"],
        "knowledge": ["knowledge"],
        "sparse": ["sparse"],
        "model": ["model"],
        "cross": ["cross"],
        "unembed": ["unembed"],
        "scaling": ["scaling"],
    }
    words_to_strip = theme_words.get(theme, [])
    parts = name.split("_")
    parts = [p for p in parts if p not in words_to_strip]
    name = " ".join(parts) if parts else name.replace("_", " ")
    return name.strip()


def get_motivation(module_name: str) -> str:
    """Get the full motivation markdown cell content for a module."""
    # Check explicit motivations first
    if module_name in MOTIVATIONS:
        text, ref_keys = MOTIVATIONS[module_name]
    else:
        # Use theme-based template
        theme = classify_module(module_name)
        template = THEME_TEMPLATES.get(theme, THEME_TEMPLATES["fallback"])
        specific = make_specific_phrase(module_name, theme)
        text = template["motivation"].format(specific=specific)
        ref_keys = template["refs"]

    # Build reference section
    ref_lines = []
    for key in ref_keys:
        if key in REFS:
            ref_lines.append(f"- {REFS[key]}")

    refs_section = ""
    if ref_lines:
        refs_section = "\n\n**Key references:**\n" + "\n".join(ref_lines)

    return f"## Why This Matters\n\n{text}{refs_section}"


def enrich_notebook(filepath: str) -> bool:
    """Insert a 'Why This Matters' cell after cell 0. Returns True if modified."""
    with open(filepath, "r") as f:
        nb = json.load(f)

    # Check if already enriched
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            content = "".join(cell.get("source", []))
            if "Why This Matters" in content:
                return False

    # Extract module name from filename
    basename = os.path.basename(filepath)
    match = re.match(r"\d+_(.+)\.ipynb", basename)
    if not match:
        return False
    module_name = match.group(1)

    # Generate motivation
    motivation = get_motivation(module_name)

    # Build source as list of lines (notebook format)
    lines = motivation.split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]

    # Create new cell
    new_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }

    # Insert after cell 0
    nb["cells"].insert(1, new_cell)

    # Write back
    with open(filepath, "w") as f:
        json.dump(nb, f, indent=1)

    return True


def main():
    notebooks_dir = os.path.dirname(os.path.abspath(__file__))

    # Find all numbered notebooks from 12-332
    notebooks = []
    for fname in sorted(os.listdir(notebooks_dir)):
        match = re.match(r"(\d+)_(.+)\.ipynb", fname)
        if match:
            num = int(match.group(1))
            if 12 <= num <= 332:
                notebooks.append(os.path.join(notebooks_dir, fname))

    print(f"Found {len(notebooks)} notebooks to process (12-332)")

    enriched = 0
    skipped = 0
    errors = 0

    for nb_path in notebooks:
        basename = os.path.basename(nb_path)
        try:
            if enrich_notebook(nb_path):
                enriched += 1
                print(f"  [+] {basename}")
            else:
                skipped += 1
                print(f"  [=] {basename} (already has 'Why This Matters')")
        except Exception as e:
            errors += 1
            print(f"  [!] {basename}: {e}")

    print(f"\nDone: {enriched} enriched, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
