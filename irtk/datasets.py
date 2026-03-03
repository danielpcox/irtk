"""Dataset utilities for mechanistic interpretability experiments.

Provides:
- Tokenization helpers (text <-> tokens with any HF tokenizer)
- Common mechinterp datasets (IOI, repeated sequences, etc.)
- Batch preparation utilities
"""

from typing import Optional, Sequence
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from transformers import AutoTokenizer


def get_tokenizer(model_name: str = "openai-community/gpt2") -> AutoTokenizer:
    """Get a HuggingFace tokenizer.

    Args:
        model_name: HF model name or path.

    Returns:
        AutoTokenizer instance.
    """
    return AutoTokenizer.from_pretrained(model_name)


def to_tokens(
    text: str | list[str],
    tokenizer=None,
    model_name: str = "openai-community/gpt2",
    prepend_bos: bool = False,
) -> jnp.ndarray:
    """Tokenize text into a JAX array of token IDs.

    Args:
        text: A string or list of strings to tokenize.
        tokenizer: HF tokenizer (will create one from model_name if None).
        model_name: Model name for creating tokenizer.
        prepend_bos: Whether to prepend BOS token.

    Returns:
        [seq_len] array for a single string, or list of arrays for multiple.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model_name)

    if isinstance(text, str):
        token_ids = tokenizer.encode(text)
        if prepend_bos and hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
            token_ids = [tokenizer.bos_token_id] + token_ids
        return jnp.array(token_ids)
    else:
        return [to_tokens(t, tokenizer, prepend_bos=prepend_bos) for t in text]


def to_string(
    tokens: jnp.ndarray | list[int],
    tokenizer=None,
    model_name: str = "openai-community/gpt2",
) -> str:
    """Decode token IDs back to a string.

    Args:
        tokens: Token IDs (JAX array, numpy array, or list).
        tokenizer: HF tokenizer.
        model_name: Model name for creating tokenizer.

    Returns:
        Decoded string.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model_name)
    if isinstance(tokens, jnp.ndarray):
        tokens = [int(t) for t in tokens]
    elif isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()
    return tokenizer.decode(tokens)


def to_single_token(
    text: str,
    tokenizer=None,
    model_name: str = "openai-community/gpt2",
) -> int:
    """Get the single token ID for a string. Raises if it's not a single token.

    Args:
        text: String that should be exactly one token.
        tokenizer: HF tokenizer.
        model_name: Model name for creating tokenizer.

    Returns:
        Integer token ID.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model_name)
    tokens = tokenizer.encode(text)
    if len(tokens) != 1:
        raise ValueError(
            f"'{text}' tokenizes to {len(tokens)} tokens ({tokens}), not 1."
        )
    return tokens[0]


def to_token_labels(
    tokens: jnp.ndarray | list[int],
    tokenizer=None,
    model_name: str = "openai-community/gpt2",
) -> list[str]:
    """Convert token IDs to readable string labels.

    Args:
        tokens: Token IDs.
        tokenizer: HF tokenizer.
        model_name: Model name for creating tokenizer.

    Returns:
        List of decoded token strings.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model_name)
    return [tokenizer.decode([int(t)]) for t in tokens]


# ─── Common Mechinterp Datasets ──────────────────────────────────────────────


@dataclass
class IOIDataset:
    """Indirect Object Identification dataset.

    Generates prompts of the form:
      "When [Name1] and [Name2] went to the store, [Name1] gave a gift to"
    where the model should predict [Name2].

    This is the classic mechinterp dataset from Wang et al. (2023).
    """
    clean_prompts: list[str]
    corrupted_prompts: list[str]
    answer_tokens: list[int]     # correct IO token at each position
    wrong_tokens: list[int]      # S token (the wrong answer)
    io_names: list[str]          # indirect object names
    s_names: list[str]           # subject names


# Names commonly used in IOI experiments (single-token in GPT-2)
IOI_NAMES = [
    " Mary", " John", " Alice", " Bob", " Charlie", " David",
    " Emma", " James", " Sarah", " Michael", " Lisa", " Tom",
    " Anna", " Mark", " Kate", " Daniel",
]

IOI_TEMPLATES = [
    "When{name1} and{name2} went to the store,{name1} gave a gift to",
    "When{name1} and{name2} went to the park,{name1} gave a bottle of water to",
    "When{name1} and{name2} went to the restaurant,{name1} gave the bill to",
    "When{name1} and{name2} had dinner,{name1} passed the salt to",
    "After{name1} and{name2} finished the meeting,{name1} sent an email to",
    "When{name1} and{name2} got home,{name1} gave the keys to",
]


def make_ioi_dataset(
    n_prompts: int = 50,
    tokenizer=None,
    model_name: str = "openai-community/gpt2",
    seed: int = 42,
) -> IOIDataset:
    """Generate an IOI (Indirect Object Identification) dataset.

    Creates prompt pairs (clean/corrupted) for studying the IOI circuit.
    In clean prompts, the repeated name is the subject (S) and the other is
    the indirect object (IO). In corrupted prompts, the IO name is replaced
    with a random different name.

    Args:
        n_prompts: Number of prompts to generate.
        tokenizer: HF tokenizer.
        model_name: Model name for tokenizer.
        seed: Random seed.

    Returns:
        IOIDataset with clean prompts, corrupted prompts, and answer tokens.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model_name)

    rng = np.random.RandomState(seed)

    clean_prompts = []
    corrupted_prompts = []
    answer_tokens = []
    wrong_tokens = []
    io_names = []
    s_names = []

    for i in range(n_prompts):
        # Pick two different names
        name_indices = rng.choice(len(IOI_NAMES), size=2, replace=False)
        s_name = IOI_NAMES[name_indices[0]]    # subject (repeated)
        io_name = IOI_NAMES[name_indices[1]]   # indirect object (answer)

        # Pick a template
        template = IOI_TEMPLATES[rng.randint(len(IOI_TEMPLATES))]

        # Clean prompt: S appears twice, IO appears once
        clean = template.format(name1=s_name, name2=io_name)

        # Corrupted prompt: replace IO with a random different name
        corrupt_idx = rng.choice(
            [j for j in range(len(IOI_NAMES)) if j not in name_indices]
        )
        corrupt_name = IOI_NAMES[corrupt_idx]
        corrupted = template.format(name1=s_name, name2=corrupt_name)

        # Get answer token IDs
        io_token = tokenizer.encode(io_name)
        s_token = tokenizer.encode(s_name)
        if len(io_token) == 1 and len(s_token) == 1:
            clean_prompts.append(clean)
            corrupted_prompts.append(corrupted)
            answer_tokens.append(io_token[0])
            wrong_tokens.append(s_token[0])
            io_names.append(io_name.strip())
            s_names.append(s_name.strip())

    return IOIDataset(
        clean_prompts=clean_prompts,
        corrupted_prompts=corrupted_prompts,
        answer_tokens=answer_tokens,
        wrong_tokens=wrong_tokens,
        io_names=io_names,
        s_names=s_names,
    )


def make_repeated_tokens(
    seq_half: int = 50,
    vocab_min: int = 1000,
    vocab_max: int = 40000,
    bos_token: int = 50256,
    seed: int = 42,
) -> jnp.ndarray:
    """Generate a repeated random token sequence for induction head testing.

    Creates: [BOS] + [random tokens] + [random tokens]

    Args:
        seq_half: Number of random tokens per half.
        vocab_min: Minimum token ID to use.
        vocab_max: Maximum token ID to use.
        bos_token: BOS token ID.
        seed: Random seed.

    Returns:
        [1 + 2*seq_half] array of token IDs.
    """
    rng = np.random.RandomState(seed)
    random_tokens = rng.randint(vocab_min, vocab_max, size=seq_half)
    tokens = np.concatenate([[bos_token], random_tokens, random_tokens])
    return jnp.array(tokens)


def make_greater_than_dataset(
    n_prompts: int = 50,
    tokenizer=None,
    model_name: str = "openai-community/gpt2",
    seed: int = 42,
) -> dict:
    """Generate prompts for studying numerical comparison (greater-than).

    Creates prompts like "The year was 17XX. The next year was 17" where the
    model should predict a year >= XX.

    Args:
        n_prompts: Number of prompts.
        tokenizer: HF tokenizer.
        model_name: Model name.
        seed: Random seed.

    Returns:
        Dict with 'prompts', 'years', 'tokens' keys.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model_name)

    rng = np.random.RandomState(seed)
    prompts = []
    years = []

    for _ in range(n_prompts):
        year = rng.randint(10, 90)
        prompt = f"The year was 17{year:02d}. The next year was 17"
        prompts.append(prompt)
        years.append(year)

    tokens_list = [jnp.array(tokenizer.encode(p)) for p in prompts]
    return {"prompts": prompts, "years": years, "tokens": tokens_list}
