import json
import os
import sys
from pprint import pformat

import hydra
import torch
from datasets import Dataset, load_dataset
from loguru import logger
from omegaconf import DictConfig
from transformers import set_seed

from juddges.config import PredictInfoExtractionConfig
from juddges.llm.factory import ModelForGeneration, get_llm
from juddges.llm.predict import predict_with_llm
from juddges.preprocessing.context_truncator import ContextTruncator
from juddges.preprocessing.formatters import ConversationFormatter
from juddges.preprocessing.text_encoder import TokenizerEncoder
from juddges.settings import CONFIG_PATH
from juddges.utils.config import resolve_config
from juddges.utils.misc import save_yaml, sort_dataset_by_input_length

torch.set_float32_matmul_precision("high")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_PROC = int(os.getenv("NUM_PROC", 1))

if NUM_PROC > 1:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="predict.yaml")
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    """Performs inference on a given dataset using given model.
    The outputs are saved to a file in the following JSONL format:
    [
        {
            "answer": str,
            "gold": str
        },
        ...
    ]
    """
    config = PredictInfoExtractionConfig(**resolve_config(cfg))
    logger.info(f"config:\n{pformat(config.model_dump())}")

    try:
        config.output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        logger.error(
            f"Output directory {config.output_dir} already exists. Remove stale outputs first."
        )
        sys.exit(1)

    ds = load_dataset(config.dataset.name, split="test")
    ds, reverse_sort_idx = sort_dataset_by_input_length(ds, config.dataset.context_field)
    logger.info("Loading model...")

    model_pack = get_llm(config.llm, device_map=config.device_map)

    assert not any(key in model_pack.generate_kwargs for key in config.generate_kwargs.keys())
    model_pack.generate_kwargs |= config.generate_kwargs

    model_pack.model.eval()
    # model_pack.model.compile()  # might cause libcuda.so not found error

    if config.llm.batch_size > 1 and config.llm.padding is False:
        raise ValueError("Padding has to be enabled if batch size > 1.")

    gold_outputs = [item["output"] for item in ds]

    ds = prepare_and_save_dataset_for_prediction(dataset=ds, config=config, model_pack=model_pack)

    set_seed(config.random_seed)
    model_outputs = predict_with_llm(
        model_pack=model_pack,
        dataset=ds,
        batch_size=config.llm.batch_size,
        num_proc=NUM_PROC,
        verbose=True,
    )
    results = [
        {"answer": ans, "gold": gold_ans} for ans, gold_ans in zip(model_outputs, gold_outputs)
    ]
    results = [results[i] for i in reverse_sort_idx]

    with open(config.predictions_file, "w") as f:
        json.dump(results, f, indent="\t", ensure_ascii=False)

    save_yaml(config.model_dump(), config.config_file)

    logger.info(f"Successfully finished prediction, saved outcomes to {config.output_dir}")


def prepare_and_save_dataset_for_prediction(
    dataset: Dataset,
    config: PredictInfoExtractionConfig,
    model_pack: ModelForGeneration,
) -> Dataset:
    max_input_length = config.get_max_input_length_accounting_for_output(
        model_pack.model.config.max_position_embeddings
    )
    context_truncator = ContextTruncator(
        prompt_without_context=config.prompt.render(context=""),
        tokenizer=model_pack.tokenizer,
        max_length=max_input_length,
    )
    dataset = dataset.map(
        lambda x: context_truncator(context=x[config.dataset.context_field]),
        batched=False,
        desc="Truncating context",
        num_proc=NUM_PROC,
    )

    formatter = ConversationFormatter(
        tokenizer=model_pack.tokenizer,
        prompt=config.prompt,
        dataset_context_field=config.dataset.context_field,
        dataset_output_field=None,
        use_output=False,
    )
    cols_to_remove = set(dataset.column_names).difference(
        set(["num_truncated_tokens", "truncated_ratio"])
    )
    dataset = dataset.map(
        formatter,
        batched=False,
        remove_columns=cols_to_remove,
        desc="Formatting dataset",
        num_proc=NUM_PROC,
    )

    logger.info(f"Saving dataset to {config.dataset_file}")
    logger.info(f"Example item:\n{pformat(dataset[0])}")
    dataset.to_json(config.dataset_file, force_ascii=False)

    tokenizer_encoder = TokenizerEncoder(
        final_input_field=formatter.FINAL_INPUT_FIELD,
        tokenizer=model_pack.tokenizer,
    )
    dataset.set_transform(tokenizer_encoder)

    return dataset


if __name__ == "__main__":
    main()
