from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger
from peft import PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from juddges.config import LLMConfig


@dataclass
class ModelForGeneration:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    generate_kwargs: dict[str, Any]


def get_model(llm_config: LLMConfig, **kwargs: Any) -> ModelForGeneration:
    if "llama" in llm_config.name.lower():
        return get_llama_3(llm_config, **kwargs)
    elif llm_config.name.lower().startswith("speakleash/bielik-11b-v2"):
        return get_model_with_default_setup(llm_config, **kwargs)
    elif any(mistral_model in llm_config.name.lower() for mistral_model in ("mistral", "bielik")):
        return get_mistral(llm_config, **kwargs)
    elif any(llama_2_model in llm_config.name.lower() for llama_2_model in ("trurl", "qra")):
        return get_model_with_default_setup(llm_config, **kwargs)
    else:
        raise ValueError(f"Model: {llm_config} not yet handled or doesn't exists.")


def get_llama_3(llm_config: LLMConfig, **kwargs: Any) -> ModelForGeneration:
    model, tokenizer = _get_model_tokenizer(llm_config, **kwargs)
    tokenizer.padding_side = llm_config.padding_side
    tokenizer.pad_token = tokenizer.eos_token
    terminators: list[int] = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    return ModelForGeneration(
        model=model,
        tokenizer=tokenizer,
        generate_kwargs={"eos_token_id": terminators, "pad_token_id": tokenizer.eos_token_id},
    )


def get_mistral(llm_config: LLMConfig, **kwargs: Any) -> ModelForGeneration:
    model, tokenizer = _get_model_tokenizer(llm_config, **kwargs)
    tokenizer.padding_side = llm_config.padding_side
    tokenizer.pad_token = tokenizer.eos_token

    return ModelForGeneration(
        model=model,
        tokenizer=tokenizer,
        generate_kwargs={"pad_token_id": tokenizer.eos_token_id},
    )


def get_model_with_default_setup(llm_config: LLMConfig, **kwargs: Any) -> ModelForGeneration:
    model, tokenizer = _get_model_tokenizer(llm_config, **kwargs)
    tokenizer.padding_side = llm_config.padding_side

    return ModelForGeneration(
        model=model,
        tokenizer=tokenizer,
        generate_kwargs={"pad_token_id": tokenizer.eos_token_id},
    )


def _get_model_tokenizer(
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

    if llm_config.adapter_path is not None:
        logger.info(f"Loading adapter from {llm_config.adapter_path}")
        model = PeftModelForCausalLM.from_pretrained(model, llm_config.adapter_path)
        model = model.merge_and_unload(safe_merge=True)

    return model, tokenizer
