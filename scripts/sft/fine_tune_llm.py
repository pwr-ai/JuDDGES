import os
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from openai import BaseModel
from peft.tuners.lora.config import LoraConfig
from trl import SFTTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
)

from juddges.config import DatasetConfig, LLMConfig
from juddges.data.datasets.utils import create_chat
from juddges.settings import CONFIG_PATH

from datasets import (
    load_dataset,
    DatasetDict,
    Dataset,
    IterableDatasetDict,
    IterableDataset,
)

import torch
from transformers import TrainingArguments

from juddges.preprocessing.context_truncator import ContextTruncator

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

    dataset = get_dataset(config.dataset.name, NUM_PROC)
    model, tokenizer = get_model_and_tokenizer(config.model.name, config.model.tokenizer_name)

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

    peft_config = get_peft_config()
    trainer = get_trainer(
        model,
        tokenizer,
        config.model.max_seq_length,
        peft_config,
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
    return load_dataset(dataset_name, split="train", num_proc=num_proc)


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
    model_name: str, tokenizer_name: str
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.padding_side = "right"  # to prevent warnings
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_peft_config() -> LoraConfig:
    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=6,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    return peft_config


def get_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    peft_config: LoraConfig,
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
        optim="adamw_torch_fused",  # use fused adamw optimizer
        logging_steps=1,  # log every 1 step
        save_strategy="steps",  # save checkpoint every epoch
        bf16=True,  # use bfloat16 precision
        tf32=True,  # use tf32 precision
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
        peft_config=peft_config,
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
