"""
SFT script based on https://huggingface.co/blog/unsloth-trl
"""

import os
from pathlib import Path
from pprint import pformat

import hydra
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
from juddges.preprocessing.formatters import format_to_conversations
from juddges.settings import CONFIG_PATH
from juddges.utils.config import resolve_config
from juddges.utils.formatters import format_to_conversations

NUM_PROC = int(os.getenv("NUM_PROC", 1))


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="fine_tuning.yaml")
def main(cfg: DictConfig) -> None:
    cfg_dict = resolve_config(cfg)
    logger.info(f"config:\n{pformat(cfg_dict)}")
    config = FineTuningConfig(**cfg_dict)

    if config.training_args["report_to"] == "wandb":
        os.environ["WANDB_ENTITY"] = config.wandb_entity
        os.environ["WANDB_PROJECT"] = config.wandb_project

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(config.dataset.name, split="train", num_proc=NUM_PROC)
    model_pack = get_llm(
        config.llm,
        device_map={"": PartialState().process_index},
    )
    model_pack.model.config.use_cache = False

    dataset = prepare_dataset(
        dataset=dataset,
        dataset_prompt_field=config.dataset.prompt_field,
        dataset_context_field=config.dataset.context_field,
        dataset_output_field=config.dataset.output_field,
        truncate_context=config.truncate_context,
        tokenizer=model_pack.tokenizer,
        max_length=config.max_context_size,
        num_proc=NUM_PROC,
    )

    trainer = get_trainer(
        model_pack.model,
        model_pack.tokenizer,
        dataset,
        config,
        NUM_PROC,
    )
    trainer.train()
    trainer.save_model()


def prepare_dataset(
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
    dataset_prompt_field: str,
    dataset_context_field: str,
    dataset_output_field: str,
    truncate_context: bool,
    tokenizer: PreTrainedTokenizer,
    max_length: int | None,
    num_proc: int | None,
) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    if truncate_context:
        assert max_length is not None
        truncator = ContextTruncator(tokenizer, max_length)

        dataset = dataset.map(
            lambda x: {
                "context": truncator(
                    x[dataset_prompt_field], x[dataset_context_field], x[dataset_output_field]
                )
            },
            desc="Truncating context",
            num_proc=num_proc,
        )

    dataset = dataset.map(
        lambda x: format_to_conversations(
            x, dataset_prompt_field, dataset_context_field, dataset_output_field
        ),
        remove_columns=dataset.column_names,
        desc="Formatting to chat",
    )
    dataset = dataset.map(
        lambda x: {"messages": tokenizer.apply_chat_template(x["messages"], tokenize=False)},
        desc="Applying chat template",
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
        dataset_text_field="messages",
        max_seq_length=config.max_context_size,
        dataset_num_proc=num_proc,
        **config.training_args,
    )

    if config.use_peft and config.llm.use_unsloth:
        from unsloth import FastLanguageModel

        model = FastLanguageModel.get_peft_model(
            model,
            **config.peft_args,
        )
        peft_config = None
    elif config.use_peft:
        peft_config = LoraConfig(**config.peft_args)
    else:
        peft_config = None

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    return trainer


main()
