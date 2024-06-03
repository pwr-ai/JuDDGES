from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List

import numpy as np
import torch
from joblib import Memory
from jsonlines import jsonlines
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from juddges.settings import CACHE_DIR

util_memory = Memory(CACHE_DIR, verbose=0)


def save_jsonl(records: Iterable[Dict[str, Any]], out: Path | str, mode="w") -> None:
    """Save a list of dictionaries to a jsonl file."""
    with jsonlines.open(out, mode=mode) as writer:
        writer.write_all(records)


def read_jsonl(path: Path | str) -> Generator[Dict[str, Any], None, None]:
    """Read a jsonl file and yield dictionaries."""
    with jsonlines.open(path) as reader:
        yield from reader


def path_safe_udate() -> str:
    """Generate a unique timestamp string for file naming.

    Returns:
        str: A string with the current date and time in the %Y%m%d_%H%M%Sf%f format
    """
    return datetime.now().strftime("%Y%m%d_%H%M%Sf%f")


@util_memory.cache(ignore=["model"])
def _model_call(
    model: torch.nn.Module | PreTrainedTokenizer | PreTrainedModel, model_name: str, *args, **kwargs
) -> Any:
    return model(*args, **kwargs)


@torch.no_grad()
def get_texts_embeddings(
    texts: Iterable[str],
    model_name: str = "allegro/herbert-base-cased",
    batch_size: int = 128,
    device: str = "cuda",
    max_length: int = 512,
) -> List[List[float]]:
    """
    Processes texts in batches to calculate embeddings for each, handling texts longer than the model's max token limit
    by breaking them into parts, each not exceeding the max_length, then averaging these embeddings.

    Args:
    texts (Iterable[str]): Iterable of text strings to process.
    batch_size (int): Number of texts to process in each batch.
    device (str): Device to perform the computation on.
    max_length (int): Maximum number of tokens for the model.

    Returns:
    List[List[float]]: List of averaged embeddings in Python list format.
    """
    embedder = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    all_embeddings = []
    with tqdm(total=len(texts)) as pbar:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = []

            for text in batch_texts:
                # Tokenize text and split into chunks
                tokens = _model_call(
                    tokenizer, model_name, text, add_special_tokens=True, truncation=False
                ).input_ids
                chunks = [tokens[j : j + max_length] for j in range(0, len(tokens), max_length)]

                # Initialize tensor to accumulate embeddings
                embeddings_accum = torch.zeros(
                    (len(chunks), embedder.config.hidden_size), device=device
                )

                for index, chunk in enumerate(chunks):
                    # Prepare inputs
                    input_ids = torch.tensor([chunk], device=device)

                    # Get model output
                    output = _model_call(embedder, model_name, input_ids)
                    embeddings = output.pooler_output

                    # Accumulate embeddings
                    embeddings_accum[index] = embeddings.squeeze()

                # Average the embeddings and add to batch_embeddings
                average_embeddings = embeddings_accum.mean(dim=0)
                batch_embeddings.append(average_embeddings.cpu().tolist())
                pbar.update()

            # Extend all_embeddings with the batch results
            all_embeddings.extend(batch_embeddings)

    return all_embeddings


@torch.no_grad()
def get_texts_sentiment(
    texts: List[str],
    model_name: str = "Voicelab/herbert-base-cased-sentiment",
    batch_size: int = 128,
    device: str = "cuda",
    max_length: int = 512,
) -> List[List[float]]:
    """
    Processes texts in batches to calculate sentiment scores for each, handling texts longer than the model's max token limit
    by breaking them into parts if necessary.

    Args:
    texts (List[str]): List of text strings to process.
    batch_size (int): Number of texts to process in each batch.
    device (str): Device to perform the computation on.
    max_length (int): Maximum number of tokens for the model.

    Returns:
    List[List[float]]: List of sentiment scores
    """
    sent_tokenizer = AutoTokenizer.from_pretrained(model_name)
    sent_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    sentiment_scores = []
    with tqdm(total=len(texts)) as pbar:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_scores = []
            for text in batch_texts:
                # Split text into chunks if too long
                chunks = [text[j : j + max_length] for j in range(0, len(text), max_length)]
                chunk_scores = []
                for chunk in chunks:
                    encoding = _model_call(
                        sent_tokenizer,
                        model_name,
                        [chunk],
                        add_special_tokens=True,
                        return_token_type_ids=True,
                        truncation=True,
                        padding="max_length",
                        return_attention_mask=True,
                        return_tensors="pt",
                    ).to(device)
                    output = _model_call(sent_model, model_name, **encoding)
                    logits = output.logits[0].to("cpu").detach().numpy()
                    chunk_scores.append(logits)
                # Average scores across chunks for a single text
                batch_scores.append(np.mean(chunk_scores, axis=0))
                pbar.update()

            sentiment_scores.extend(batch_scores)

    return sentiment_scores


@torch.no_grad()
def get_texts_formality(
    texts: List[str],
    model_name: str = "s-nlp/xlmr_formality_classifier",
    batch_size: int = 128,
    device: str = "cuda",
    max_length: int = 512,
) -> List[List[float]]:
    """
    Processes texts in batches to calculate formality scores for each, handling texts longer than the model's max token limit
    by breaking them into parts if necessary.

    Args:
    texts (List[str]): List of text strings to process.
    batch_size (int): Number of texts to process in each batch.
    device (str): Device to perform the computation on.
    max_length (int): Maximum number of tokens for the model.

    Returns:
    List[List[float]]: List of formality scores
    """
    formality_tokenizer = AutoTokenizer.from_pretrained(model_name)
    formality_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    formality_scores = []
    with tqdm(total=len(texts)) as pbar:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_scores = []
            for text in batch_texts:
                # Split text into chunks if too long
                chunks = [text[j : j + max_length] for j in range(0, len(text), max_length)]
                chunk_scores = []
                for chunk in chunks:
                    encoding = _model_call(
                        formality_tokenizer,
                        model_name,
                        [chunk],
                        add_special_tokens=True,
                        return_token_type_ids=True,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    ).to(device)
                    output = _model_call(formality_model, model_name, **encoding)
                    logits = output.logits.softmax(dim=1)[0].to("cpu").detach().numpy()
                    chunk_scores.append(logits)
                # Average scores across chunks for a single text
                batch_scores.append(np.mean(chunk_scores, axis=0))
                pbar.update()

            formality_scores.extend(batch_scores)

    return formality_scores
