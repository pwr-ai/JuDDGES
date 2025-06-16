import json
import os
import sys
from pprint import pformat

import hydra
import torch
from datasets import Dataset
from loguru import logger
from omegaconf import DictConfig
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from juddges.config import PredictInfoExtractionConfig
from juddges.data.dataset_factory import get_dataset
from juddges.preprocessing.context_truncator import ContextTruncator
from juddges.preprocessing.formatters import ConversationFormatter
from juddges.settings import CONFIG_PATH
from juddges.utils.config import resolve_config
from juddges.utils.misc import save_yaml

torch.set_float32_matmul_precision("high")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_PROC = int(os.getenv("NUM_PROC", 1))
NUM_GPUS = torch.cuda.device_count()


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="predict.yaml")
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    config = PredictInfoExtractionConfig(**resolve_config(cfg))
    logger.info(f"config:\n{pformat(config.model_dump())}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.predictions_file.exists():
        logger.error(
            f"Output file {config.predictions_file} already exists. Remove stale outputs first."
        )
        sys.exit(1)

    ds = get_dataset(config.dataset.name, config.split)

    llm = LLM(
        model=config.llm.name,
        dtype="bfloat16",
        enable_lora=True,
        max_num_seqs=config.llm.batch_size,
        tensor_parallel_size=NUM_GPUS,
        seed=config.random_seed,
        enable_prefix_caching=True,
    )

    if config.llm.should_load_adapter:
        logger.info("Loading LoRA adapter")
        lora_request = LoRARequest(
            lora_name="sft_adapter",
            lora_int_id=1,
            lora_path=str(config.llm.adapter_path_or_first_ckpt_path),
        )
    else:
        lora_request = None

    ds = prepare_and_save_dataset_for_prediction(
        dataset=ds,
        config=config,
        llm=llm,
    )

    params = SamplingParams(
        max_tokens=config.generate_kwargs.pop("max_new_tokens"),
        temperature=config.generate_kwargs.pop("temperature"),
        enable_prefix_caching=True,
    )
    assert config.generate_kwargs.pop("do_sample")
    assert not config.generate_kwargs

    outputs = llm.chat(
        messages=ds[ConversationFormatter.MESSAGES_FIELD],
        sampling_params=params,
        lora_request=lora_request,
        add_generation_prompt=True,
    )
    results = [
        {"answer": ans.outputs[0].text, "gold": gold} for ans, gold in zip(outputs, ds["output"])
    ]

    save_yaml(config.model_dump(), config.config_file)

    with open(config.predictions_file, "w") as f:
        json.dump(results, f, indent="\t", ensure_ascii=False)


def prepare_and_save_dataset_for_prediction(
    dataset: Dataset,
    config: PredictInfoExtractionConfig,
    llm: LLM,
) -> Dataset:
    tokenizer = llm.get_tokenizer()
    max_input_length = config.get_max_input_length_accounting_for_output(
        llm.llm_engine.model_config.max_model_len
    )
    context_truncator = ContextTruncator(
        prompt_without_context=config.prompt.render(context=""),
        tokenizer=tokenizer,
        max_length=max_input_length,
    )
    dataset = dataset.map(
        lambda x: context_truncator(context=x[config.dataset.context_field]),
        batched=False,
        desc="Truncating context",
        num_proc=NUM_PROC,
    )

    formatter = ConversationFormatter(
        tokenizer=None,
        prompt=config.prompt,
        dataset_context_field=config.dataset.context_field,
        dataset_output_field=None,
        use_output=False,
        format_as_chat=False,
    )
    cols_to_remove = set(dataset.column_names).difference(
        set(["num_truncated_tokens", "truncated_ratio", "output"])
    )
    dataset = dataset.map(
        formatter,
        batched=False,
        remove_columns=cols_to_remove,
        desc="Formatting dataset",
        num_proc=NUM_PROC,
    )

    logger.info(f"Saving dataset to {config.dataset_file}")
    dataset.to_json(config.dataset_file, force_ascii=False)

    return dataset


if __name__ == "__main__":
    main()
