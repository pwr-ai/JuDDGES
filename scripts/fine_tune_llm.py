import torch
from peft.tuners.lora.config import LoraConfig
from trl import SFTTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
    Trainer,
)

from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset

from transformers import TrainingArguments


def main() -> None:
    dataset = get_dataset()
    model, tokenizer = get_model_and_tokenizer()
    peft_config = get_peft_config()
    trainer = get_trainer(model, tokenizer, peft_config, dataset)
    trainer.train()
    trainer.save_model()


def get_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    # Load Dolly Dataset.
    dataset = load_dataset("philschmid/dolly-15k-oai-style", split="train")
    return dataset


def get_model_and_tokenizer() -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    # Hugging Face model id
    model_id = "google/gemma-7b"
    tokenizer_id = "philschmid/gemma-tokenizer-chatml"

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.padding_side = "right"  # to prevent warnings

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
    peft_config: LoraConfig,
    dataset: DatasetDict | Dataset | IterableDatasetDict | IterableDataset,
) -> Trainer:
    args = TrainingArguments(
        output_dir="gemma-7b-dolly-chatml",  # directory to save and repository id
        num_train_epochs=3,  # number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        gradient_accumulation_steps=2,  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,  # use gradient checkpointing to save memory
        optim="adamw_torch_fused",  # use fused adamw optimizer
        logging_steps=10,  # log every 10 steps
        save_strategy="epoch",  # save checkpoint every epoch
        bf16=True,  # use bfloat16 precision
        tf32=True,  # use tf32 precision
        learning_rate=2e-4,  # learning rate, based on QLoRA paper
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",  # use constant learning rate scheduler
        push_to_hub=False,  # push model to hub
        report_to="tensorboard",  # report metrics to tensorboard
    )

    max_seq_length = 1512  # max sequence length for model and packing of the dataset

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
    )

    return trainer


main()
