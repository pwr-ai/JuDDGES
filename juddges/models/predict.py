import time

from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from juddges.models.factory import ModelForGeneration


def predict_with_llm(
    model_pack: ModelForGeneration,
    dataset: Dataset,
    batch_size: int,
    num_proc: int,
    verbose: bool = True,
) -> list[str]:
    """Generates LLM predictions for a given dataset

    Args:
        llm (AutoModelForCausalLM): LLM
        dataset (Dataset): dataset to make prediction for, should be tokenized and has input_ids field

    Returns:
        list[str]: List of generated texts with preserved order
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_proc,
        pin_memory=(num_proc > 1),
        shuffle=False,
    )

    model_outputs = []

    model = model_pack.model
    tokenizer = model_pack.tokenizer
    device = next(model.parameters()).device

    with tqdm(dataloader, disable=not verbose) as pbar:
        for batch in pbar:
            input_ids = batch["input_ids"].view(batch_size, -1)
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            input_length = input_ids.size(1)

            start_time = time.time()
            generated_ids = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                **model_pack.generate_kwargs,
            )
            duration = time.time() - start_time

            decoded = tokenizer.batch_decode(
                generated_ids[:, input_length:],
                skip_special_tokens=True,
            )
            model_outputs.extend(decoded)

            pbar.set_postfix_str(f"{generated_ids.numel() / duration: 0.2f} tok/sec")

    return model_outputs
