import json
import math
from pathlib import Path
import time
from datasets import load_dataset

from loguru import logger
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)
import typer

from juddges.metrics.info_extraction import evaluate_extraction
from juddges.preprocessing.context_truncator import ContextTruncator


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLM = "meta-llama/Meta-Llama-3-8B-Instruct"
BATCH_SIZE = 1
MAX_NEW_TOKENS = 250
MAX_LENGTH = 2_048


@torch.no_grad()
def main(
    output_file: Path = typer.Option(...),
    metrics_file: Path = typer.Option(...),
    batch_size: int = typer.Option(BATCH_SIZE),
):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("JuDDGES/pl-court-instruct")
    logger.info("Loading model...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        LLM,
        quantization_config=quantization_config,
        device_map=DEVICE,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(LLM, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    terminators: list[int] = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    enable_padding = batch_size > 1
    encoder = Encoder(tokenizer=tokenizer, max_length=MAX_LENGTH, enable_padding=enable_padding)
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
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=terminators,
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


class Encoder:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int, enable_padding: bool):
        self.tokenizer = tokenizer
        self.truncator = ContextTruncator(tokenizer, max_length, use_output=False)
        self.max_length = max_length
        self.enable_padding = enable_padding

    def __call__(self, item: dict[str, str]):
        truncated_context = self.truncator(
            item["prompt"],
            item["context"],
            item["output"],
        )
        input_message = item["prompt"].format(context=truncated_context)
        input_chat = [{"role": "user", "content": input_message}]
        encoded = self.tokenizer.apply_chat_template(
            input_chat,
            add_generation_prompt=True,
            padding=self.enable_padding,
            max_length=self.max_length,
            truncation=False,
        )
        return {"tokens": encoded}


if __name__ == "__main__":
    typer.run(main)
