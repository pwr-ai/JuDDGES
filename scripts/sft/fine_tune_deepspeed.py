"""
SFT script based on https://huggingface.co/blog/unsloth-trl
The script is purposed to run on a single node with multiple GPUs.
"""

import os
from pathlib import Path
from pprint import pformat

import hydra
import numpy as np
from accelerate import PartialState
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from loguru import logger
from omegaconf import DictConfig
from peft.tuners.lora.config import LoraConfig
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
)
from trl import SFTConfig, SFTTrainer

from juddges.config import FineTuningConfig
from juddges.llm.factory import get_llm
from juddges.preprocessing.context_truncator import ContextTruncator
from juddges.preprocessing.formatters import ConversationFormatter
from juddges.settings import CONFIG_PATH
from juddges.utils.config import resolve_config

NUM_PROC = int(os.getenv("NUM_PROC", 1))

state = PartialState()


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="peft_fine_tuning.yaml")
def main(cfg: DictConfig) -> None:
    cfg_dict = resolve_config(cfg)
    config = FineTuningConfig(**cfg_dict)

    if state.is_main_process:
        logger.info(f"config:\n{pformat(config.model_dump())}")

    if config.training_args["report_to"] == "wandb":
        os.environ["WANDB_ENTITY"] = config.wandb_entity
        os.environ["WANDB_PROJECT"] = config.wandb_project

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with state.local_main_process_first():
        model_pack = get_llm(
            config.llm,
            use_cache=False,
        )

        dataset = load_dataset(config.dataset.name, split="train", num_proc=NUM_PROC)
        dataset = prepare_dataset(
            dataset=dataset,
            tokenizer=model_pack.tokenizer,
            config=config,
            num_proc=NUM_PROC,
        )

    trainer = get_trainer(
        model=model_pack.model,
        tokenizer=model_pack.tokenizer,
        dataset=dataset,
        config=config,
        num_proc=NUM_PROC,
    )
    trainer.train()
    trainer.save_model()

    state.wait_for_everyone()


def prepare_dataset(
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
    tokenizer: PreTrainedTokenizer,
    config: FineTuningConfig,
    num_proc: int | None,
) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    if config.training_args.get("packing", False) and config.truncate_context:
        raise ValueError("Truncating context shouldn't be enabled when packing is enabled.")

    if config.truncate_context:
        context_truncator = ContextTruncator(
            prompt_without_context=config.prompt.render(context=""),
            tokenizer=tokenizer,
            max_length=config.max_context_size,
        )
        dataset = dataset.map(
            lambda x: context_truncator(
                context=x[config.dataset.context_field],
                output=x[config.dataset.output_field],
            ),
            desc="Truncating context",
            num_proc=num_proc,
        )
        _log_truncation_stats(dataset)

    formatter = ConversationFormatter(
        config=config,
    )
    dataset = dataset.map(
        formatter,
        remove_columns=dataset.column_names,
        desc="Formatting dataset",
        num_proc=num_proc,
    )

    return dataset


def get_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
    config: FineTuningConfig,
    num_proc: int | None,
) -> Trainer:
    sft_config = SFTConfig(
        max_seq_length=config.max_context_size,
        dataset_num_proc=num_proc,
        **config.training_args,
    )

    if config.use_peft:
        logger.info(f"Using PEFT with config: {config.peft_args}")
        peft_config = LoraConfig(**config.peft_args)
    else:
        logger.info("Full fine-tuning, without PEFT")
        peft_config = None

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    return trainer


def _log_truncation_stats(dataset: Dataset) -> None:
    mean_num_truncated_tokens = np.mean(dataset["num_truncated_tokens"]).item()
    std_num_truncated_tokens = np.std(dataset["num_truncated_tokens"]).item()
    mean_truncated_ratio = np.mean(dataset["truncated_ratio"]).item()
    std_truncated_ratio = np.std(dataset["truncated_ratio"]).item()
    logger.info(
        f"Average number of truncated tokens: {mean_num_truncated_tokens:.2f} ± {std_num_truncated_tokens:.2f}"
    )
    logger.info(f"Average truncated ratio: {mean_truncated_ratio:.2f} ± {std_truncated_ratio:.2f}")


main()
