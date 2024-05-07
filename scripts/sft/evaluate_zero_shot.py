import json
import math
from pathlib import Path
import time
from datasets import load_dataset

from loguru import logger
import torch
from tqdm import tqdm
import typer

from juddges.metrics.info_extraction import evaluate_extraction
from juddges.models.factory import get_model
from juddges.preprocessing.text_encoder import EvalEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLM = "meta-llama/Meta-Llama-3-8B-Instruct"
BATCH_SIZE = 1  # for now llama-3 doesn't work with padding, hence with batch_size > 1 also
MAX_NEW_TOKENS = 250
MAX_LENGTH = 2_048


@torch.no_grad()
def main(
    llm: str = typer.Option(LLM),
    output_file: Path = typer.Option(...),
    metrics_file: Path = typer.Option(...),
    batch_size: int = typer.Option(BATCH_SIZE),
    max_length: int = typer.Option(MAX_LENGTH),
    max_new_tokens: int = typer.Option(MAX_NEW_TOKENS),
    device: str = typer.Option(DEVICE),
):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("JuDDGES/pl-court-instruct")
    logger.info("Loading model...")

    model_pack = get_model(llm, device=device)
    model, tokenizer = model_pack.model, model_pack.tokenizer

    enable_padding = batch_size > 1
    encoder = EvalEncoder(
        tokenizer=tokenizer,
        max_length=max_length,
        enable_padding=enable_padding,
    )
    encoded_ds = ds["test"].map(
        encoder,
        batched=False,
        num_proc=10,
    )
    encoded_ds.set_format("torch")

    results = []
    num_batches = math.ceil(encoded_ds.num_rows / batch_size)

    with tqdm(encoded_ds.iter(batch_size=batch_size), total=num_batches) as pbar:
        for item in pbar:
            model_inputs = item["tokens"].view(batch_size, -1).to(DEVICE)
            input_length = model_inputs.size(1)

            start_time = time.time()
            generated_ids = model.generate(
                model_inputs,
                max_new_tokens=max_new_tokens,
                **model_pack.generate_kwargs,
            )
            duration = time.time() - start_time

            decoded = tokenizer.batch_decode(
                generated_ids[:, input_length:], skip_special_tokens=True
            )
            results.extend(
                [{"answer": ans, "gold": gold} for ans, gold in zip(decoded, item["output"])]
            )

            pbar.set_postfix_str(f"{generated_ids.numel() / duration: 0.2f} tok/sec")

    with open(output_file, "w") as f:
        json.dump(results, f, indent="\t")

    res = evaluate_extraction(results)
    with open(metrics_file, "w") as file:
        json.dump(res, file, indent="\t")


if __name__ == "__main__":
    typer.run(main)
