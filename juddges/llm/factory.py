import warnings
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger
from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from juddges.config import LLMConfig

LLAMA_3_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

PHI_4_MODELS = [
    "microsoft/Phi-4",
    "microsoft/Phi-4-mini-instruct",
]

MISTRAL_MODELS = [
    "mistralai/Mistral-Nemo-Instruct-2407",
    "CYFRAGOVPL/PLLuM-12B-instruct",
]

BIELIK_MODELS = [
    "speakleash/Bielik-11B-v2.3-Instruct",
]


@dataclass
class ModelForGeneration:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    generate_kwargs: dict[str, Any]


def get_llm(llm_config: LLMConfig, **kwargs: Any) -> ModelForGeneration:
    if llm_config.name in LLAMA_3_MODELS:
        return get_llama_3(llm_config, **kwargs)
    elif llm_config.name in PHI_4_MODELS or llm_config.name in BIELIK_MODELS:
        return get_llm_with_default_setup(llm_config, **kwargs)
    elif llm_config.name in MISTRAL_MODELS:
        return get_mistral(llm_config, **kwargs)
    else:
        raise ValueError(f"Model: {llm_config} not yet handled or doesn't exists.")


def get_llama_3(llm_config: LLMConfig, **kwargs: Any) -> ModelForGeneration:
    model, tokenizer = get_llm_tokenizer(llm_config, **kwargs)
    tokenizer.padding_side = llm_config.padding_side
    tokenizer.pad_token = tokenizer.eos_token
    terminators: list[int] = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    return ModelForGeneration(
        model=model,
        tokenizer=tokenizer,
        generate_kwargs={"eos_token_id": terminators, "pad_token_id": tokenizer.eos_token_id},
    )


def get_mistral(llm_config: LLMConfig, **kwargs: Any) -> ModelForGeneration:
    model, tokenizer = get_llm_tokenizer(llm_config, **kwargs)
    tokenizer.padding_side = llm_config.padding_side
    tokenizer.pad_token = tokenizer.eos_token

    return ModelForGeneration(
        model=model,
        tokenizer=tokenizer,
        generate_kwargs={"pad_token_id": tokenizer.eos_token_id},
    )


def get_llm_with_default_setup(llm_config: LLMConfig, **kwargs: Any) -> ModelForGeneration:
    model, tokenizer = get_llm_tokenizer(llm_config, **kwargs)
    tokenizer.padding_side = llm_config.padding_side

    return ModelForGeneration(
        model=model,
        tokenizer=tokenizer,
        generate_kwargs={"pad_token_id": tokenizer.eos_token_id},
    )


def get_llm_tokenizer(
    llm_config: LLMConfig,
    **kwargs: Any,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if llm_config.use_unsloth:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=llm_config.name,
            max_seq_length=llm_config.max_seq_length,
            dtype=None,
            load_in_4bit=llm_config.use_4bit,
            **kwargs,
        )
    else:
        if llm_config.use_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        if torch.cuda.is_available():
            kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(
            llm_config.name,
            torch_dtype="auto",
            **kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(llm_config.name)

    if llm_config.should_load_adapter:
        logger.info(f"Loading adapter from {llm_config.adapter_path}")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Found missing adapter keys while loading the checkpoint*",
            )
            model = PeftModelForCausalLM.from_pretrained(
                model=model,
                model_id=llm_config.adapter_path_or_last_ckpt_path,
            )
            model = model.merge_and_unload(safe_merge=True)

    return model, tokenizer
