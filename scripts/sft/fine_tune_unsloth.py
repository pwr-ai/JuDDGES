"""
SFT script based on https://huggingface.co/blog/unsloth-trl
"""

import os
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from openai import BaseModel
from trl import SFTTrainer
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
)

from juddges.config import DatasetConfig, LLMConfig
from juddges.data.datasets.utils import create_chat
from juddges.defaults import FINE_TUNING_DATASETS_PATH, CONFIG_PATH

from datasets import (
    load_dataset,
    DatasetDict,
    Dataset,
    IterableDatasetDict,
    IterableDataset,
    load_from_disk,
)

import torch
from transformers import TrainingArguments

from juddges.preprocessing.context_truncator import ContextTruncator
from unsloth import FastLanguageModel

NUM_PROC = int(os.getenv("NUM_PROC", 1))


class FineTuningConfig(BaseModel, extra="forbid"):
    model: LLMConfig
    dataset: DatasetConfig
    output_dir: Path
    run_name: str
    wandb_entity: str
    wandb_project: str
    truncate_context: bool


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="fine_tuning.yaml")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    logger.info(f"config:\n{cfg}")
    config = FineTuningConfig(**cfg)

    os.environ["WANDB_ENTITY"] = config.wandb_entity
    os.environ["WANDB_PROJECT"] = config.wandb_project
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(config.dataset.dataset_name, NUM_PROC)
    model, tokenizer = get_model_and_tokenizer(config.model.llm, config.model.max_seq_length)

    dataset = prepare_dataset(
        dataset=dataset,
        dataset_prompt_field=config.dataset.prompt_field,
        dataset_context_field=config.dataset.context_field,
        dataset_output_field=config.dataset.output_field,
        truncate_context=config.truncate_context,
        tokenizer=tokenizer,
        max_length=config.model.max_seq_length,
        num_proc=config.num_proc,
    )

    trainer = get_trainer(
        model,
        tokenizer,
        config.model.max_seq_length,
        dataset,
        "messages",
        output_dir,
        config.run_name,
        NUM_PROC,
    )
    trainer.train()
    trainer.save_model()


def get_dataset(
    dataset_name: str,
    num_proc: int,
) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    if dataset_name == "dummy":
        dataset = load_from_disk(FINE_TUNING_DATASETS_PATH / "dummy_dataset")
    else:
        dataset = load_dataset(dataset_name, split="train", num_proc=num_proc)
    return dataset


def prepare_dataset(
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
    dataset_prompt_field: str,
    dataset_context_field: str,
    dataset_output_field: str,
    truncate_context: bool,
    tokenizer: PreTrainedTokenizer | None,
    max_length: int | None,
    num_proc: int | None,
) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    if truncate_context:
        assert tokenizer is not None
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
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
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
    max_seq_length: int,
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
    dataset_text_field: str,
    output_dir: Path,
    run_name: str | None,
    num_proc: int | None,
) -> Trainer:
    args = TrainingArguments(
        run_name=run_name,  # run name for the experiment
        output_dir=str(output_dir),  # directory to save and repository id
        num_train_epochs=3,  # number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        gradient_accumulation_steps=3,  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,  # use gradient checkpointing to save memory
        optim="adamw_8bit",  # use fused adamw optimizer
        logging_steps=1,  # log every 1 step
        save_strategy="steps",  # save checkpoint every epoch
        bf16=(not torch.cuda.is_bf16_supported()),
        fp16=torch.cuda.is_bf16_supported(),
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
        dataset_text_field=dataset_text_field,
        max_seq_length=max_seq_length,
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
