"""Model name registry: aliases to canonical HuggingFace model names."""

MODEL_ALIASES: dict[str, str] = {
    # GPT-2
    "gpt2": "openai-community/gpt2",
    "gpt2-small": "openai-community/gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
    "gpt2-large": "openai-community/gpt2-large",
    "gpt2-xl": "openai-community/gpt2-xl",
    # GPT-Neo
    "gpt-neo-125m": "EleutherAI/gpt-neo-125m",
    "gpt-neo-125M": "EleutherAI/gpt-neo-125m",
    "gpt-neo-1.3b": "EleutherAI/gpt-neo-1.3B",
    "gpt-neo-1.3B": "EleutherAI/gpt-neo-1.3B",
    "gpt-neo-2.7b": "EleutherAI/gpt-neo-2.7B",
    "gpt-neo-2.7B": "EleutherAI/gpt-neo-2.7B",
    # Pythia (GPT-NeoX)
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "pythia-12b": "EleutherAI/pythia-12b",
    # Pythia deduped
    "pythia-70m-deduped": "EleutherAI/pythia-70m-deduped",
    "pythia-160m-deduped": "EleutherAI/pythia-160m-deduped",
    "pythia-410m-deduped": "EleutherAI/pythia-410m-deduped",
    "pythia-1b-deduped": "EleutherAI/pythia-1b-deduped",
    "pythia-1.4b-deduped": "EleutherAI/pythia-1.4b-deduped",
    # LLaMA
    "llama-7b": "meta-llama/Llama-2-7b-hf",
    "llama-13b": "meta-llama/Llama-2-13b-hf",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
    # Mistral
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
}


def get_official_model_name(model_name: str) -> str:
    """Resolve a model alias to its canonical HuggingFace name.

    Args:
        model_name: Model name or alias.

    Returns:
        Canonical HuggingFace model name.
    """
    # Check aliases (case-insensitive for convenience)
    lower = model_name.lower()
    for alias, official in MODEL_ALIASES.items():
        if lower == alias.lower():
            return official
    # If not in aliases, assume it's already a HF model name
    return model_name
