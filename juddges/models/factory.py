from dataclasses import dataclass
from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class ModelForGeneration:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    generate_kwargs: dict[str, Any]


def get_model(llm_name: str, **kwargs) -> ModelForGeneration:
    if llm_name.startswith("meta-llama"):
        return get_llama_3(llm_name, **kwargs)
    elif llm_name.startswith("mistralai"):
        return get_mistral(llm_name, **kwargs)
    else:
        raise ValueError(f"Model: {llm_name} not yet handled or doesn't exists.")


def get_llama_3(llm_name: str, device: str) -> ModelForGeneration:
    assert llm_name.startswith("meta-llama")
    model, tokenizer = _get_model_tokenizer(llm_name, device)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    terminators: list[int] = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    return ModelForGeneration(
        model=model,
        tokenizer=tokenizer,
        generate_kwargs={"eos_token_id": terminators, "pad_token_id": tokenizer.eos_token_id},
    )


def get_mistral(llm_name: str, device: str) -> ModelForGeneration:
    assert llm_name.startswith("mistralai")
    model, tokenizer = _get_model_tokenizer(llm_name, device)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    return ModelForGeneration(
        model=model,
        tokenizer=tokenizer,
        generate_kwargs={"pad_token_id": tokenizer.eos_token_id},
    )


def _get_model_tokenizer(llm_name: str, device: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        quantization_config=quantization_config,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    return model, tokenizer
