"""
SFT script based on https://huggingface.co/blog/unsloth-trl
"""

import os
from pathlib import Path

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
from openai import BaseModel
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from trl import SFTTrainer
from unsloth import FastLanguageModel

from juddges.config import DatasetConfig, LLMConfig
from juddges.data.datasets.utils import create_chat
from juddges.preprocessing.context_truncator import ContextTruncator
from juddges.settings import CONFIG_PATH
from juddges.utils.config import resolve_config

NUM_PROC = int(os.getenv("NUM_PROC", 1))


class FineTuningConfig(BaseModel, extra="forbid"):
    model: LLMConfig
    dataset: DatasetConfig
    batch_size: int
    epochs: int
    output_dir: Path
    run_name: str
    wandb_entity: str
    wandb_project: str
    truncate_context: bool


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="fine_tuning.yaml")
def main(cfg: DictConfig) -> None:
    cfg_dict = resolve_config(cfg)
    logger.info(f"config:\n{cfg_dict}")
    config = FineTuningConfig(**cfg_dict)

    os.environ["WANDB_ENTITY"] = config.wandb_entity
    os.environ["WANDB_PROJECT"] = config.wandb_project
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(config.dataset.name, NUM_PROC)
    model, tokenizer = get_model_and_tokenizer(config.model.name, config.model.max_seq_length)

    dataset = prepare_dataset(
        dataset=dataset,
        dataset_prompt_field=config.dataset.prompt_field,
        dataset_context_field=config.dataset.context_field,
        dataset_output_field=config.dataset.output_field,
        truncate_context=config.truncate_context,
        tokenizer=tokenizer,
        max_length=config.model.max_seq_length,
        num_proc=NUM_PROC,
    )

    trainer = get_trainer(
        model,
        tokenizer,
        dataset,
        config,
        NUM_PROC,
    )
    trainer.train()
    trainer.save_model()


def get_dataset(
    dataset_name: str,
    num_proc: int,
) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    return load_dataset(dataset_name, split="train", num_proc=num_proc)


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
        lambda x: create_chat(x, dataset_prompt_field, dataset_context_field, dataset_output_field),
        remove_columns=dataset.column_names,
        desc="Formatting to chat",
    )
    dataset = dataset.map(
        lambda x: {"messages": tokenizer.apply_chat_template(x["messages"], tokenize=False)},
        desc="Applying chat template",
    )
    return dataset


def get_model_and_tokenizer(
    model_name: str,
    max_seq_length: int,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    device_string = PartialState().process_index
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        device_map=device_string,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing=True,
        random_state=3407,
        max_seq_length=max_seq_length,
    )
    tokenizer.padding_side = "right"  # to prevent warnings
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
    config: FineTuningConfig,
    num_proc: int | None,
) -> Trainer:
    # TODO: Move all hparams to config and use SFTConfig due to deprecation of kwargs
    args = TrainingArguments(
        run_name=config.run_name,  # run name for the experiment
        output_dir=str(config.output_dir),  # directory to save and repository id
        num_train_epochs=config.epochs,  # number of training epochs
        per_device_train_batch_size=config.batch_size,  # batch size per device during training
        gradient_accumulation_steps=3,  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,  # use gradient checkpointing to save memory
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # necessary when train on multiple-GPU
        optim="adamw_8bit",  # use fused adamw optimizer
        logging_steps=1,  # log every 1 step
        save_strategy="steps",  # save checkpoint every epoch
        save_steps=100,
        bf16=True,
        learning_rate=2e-4,  # learning rate, based on QLoRA paper
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",  # use constant learning rate scheduler
        push_to_hub=False,  # push model to hub
        report_to="wandb",  # report metrics to tensorboard
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        dataset_text_field="messages",
        max_seq_length=config.model.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
        dataset_num_proc=num_proc,
    )

    return trainer


main()
