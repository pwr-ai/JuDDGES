from dataclasses import dataclass
from typing import Any
from peft import PeftModel
import torch
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
    elif "mistral" in llm_config.name.lower():
        return get_mistral(llm_config, **kwargs)
    else:
        raise ValueError(f"Model: {llm_config} not yet handled or doesn't exists.")


def get_llama_3(llm_config: LLMConfig, device_map: str) -> ModelForGeneration:
    model, tokenizer = _get_model_tokenizer(llm_config, device_map)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    terminators: list[int] = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    return ModelForGeneration(
        model=model,
        tokenizer=tokenizer,
        generate_kwargs={"eos_token_id": terminators, "pad_token_id": tokenizer.eos_token_id},
    )


def get_mistral(llm_config: LLMConfig, device_map: str) -> ModelForGeneration:
    model, tokenizer = _get_model_tokenizer(llm_config, device_map)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    return ModelForGeneration(
        model=model,
        tokenizer=tokenizer,
        generate_kwargs={"pad_token_id": tokenizer.eos_token_id},
    )


def _get_model_tokenizer(
    llm_config: LLMConfig, device: str
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if llm_config.use_unsloth:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=llm_config.name,
            max_seq_length=llm_config.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            llm_config.name,
            quantization_config=quantization_config,
            device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(llm_config.name)

    if llm_config.adapter_path is not None:
        model = PeftModel.from_pretrained(model, llm_config.adapter_path)

    return model, tokenizer
