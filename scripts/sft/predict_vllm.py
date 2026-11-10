import os
import sys
import time
from pathlib import Path
from pprint import pformat

import hydra
import torch
from datasets import Dataset
from loguru import logger
from omegaconf import DictConfig
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

try:  # vLLM >= 0.11
    from vllm.sampling_params import StructuredOutputsParams
except ImportError:  # vLLM 0.9.x
    StructuredOutputsParams = None
    from vllm.sampling_params import GuidedDecodingParams

from juddges.config import PredictInfoExtractionConfig
from juddges.data.dataset_factory import get_dataset
from juddges.preprocessing.context_truncator import ContextTruncator
from juddges.preprocessing.formatters import ConversationFormatter
from juddges.settings import CONFIG_PATH
from juddges.utils.config import resolve_config
from juddges.utils.json_schema import ie_schema_to_json_schema
from juddges.utils.misc import save_yaml
from juddges.utils.run_record import write_json_atomic, write_run_record

torch.set_float32_matmul_precision("high")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_PROC = int(os.getenv("NUM_PROC", 1))
NUM_GPUS = torch.cuda.device_count()

logger.info(f"NUM_PROC: {NUM_PROC}")
logger.info(f"NUM_GPUS: {NUM_GPUS}, CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}")


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="predict_vllm.yaml")
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    start_time = time.time()
    config = PredictInfoExtractionConfig(**resolve_config(cfg))
    logger.info(f"config:\n{pformat(config.model_dump())}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.predictions_file.exists():
        logger.error(
            f"Output file {config.predictions_file} already exists. Remove stale outputs first."
        )
        sys.exit(1)

    ds = get_dataset(config.dataset.name, config.split)
    if config.max_docs is not None:
        ds = ds.select(range(min(config.max_docs, len(ds))))
        logger.warning(f"max_docs={config.max_docs}: using only the first {len(ds)} documents")

    assert config.generate_kwargs.keys()
    json_schema = ie_schema_to_json_schema(config.ie_schema) if config.guided_json else None
    if StructuredOutputsParams is not None:
        constraint = {
            "structured_outputs": StructuredOutputsParams(json=json_schema) if json_schema else None
        }
    else:
        constraint = {
            "guided_decoding": GuidedDecodingParams(json=json_schema) if json_schema else None
        }
    params = SamplingParams(**config.generate_kwargs, seed=config.random_seed, **constraint)

    llm = LLM(
        model=config.llm.name,
        dtype="bfloat16",
        enable_lora=config.llm.should_load_adapter,
        max_lora_rank=config.max_lora_rank,
        max_num_seqs=config.llm.batch_size,
        pipeline_parallel_size=NUM_GPUS if config.parallel == "pipeline" else 1,
        tensor_parallel_size=NUM_GPUS if config.parallel == "tensor" else 1,
        seed=config.random_seed,
        enable_prefix_caching=True,
        max_model_len=config.max_model_len,
        quantization=config.quantization,
        gpu_memory_utilization=config.gpu_memory_utilization,
    )

    adapter_path = None
    if config.llm.should_load_adapter:
        adapter_path = config.llm.adapter_path_or_first_ckpt_path
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        lora_request = LoRARequest(
            lora_name="sft_adapter",
            lora_int_id=1,
            lora_path=str(adapter_path),
        )
    else:
        lora_request = None

    ds = prepare_and_save_dataset_for_prediction(
        dataset=ds,
        config=config,
        llm=llm,
    )

    outputs = llm.chat(
        messages=ds[ConversationFormatter.MESSAGES_FIELD],
        sampling_params=params,
        lora_request=lora_request,
        add_generation_prompt=True,
        chat_template_kwargs=config.llm.chat_template_kwargs,
    )

    results = [
        {
            "answer": ans.outputs[0].text,
            "gold": gold,
            "finish_reason": ans.outputs[0].finish_reason,
            "num_input_tokens": len(ans.prompt_token_ids),
            "num_output_tokens": len(ans.outputs[0].token_ids),
        }
        for ans, gold in zip(outputs, ds["output"])
    ]

    num_length = sum(r["finish_reason"] == "length" for r in results)
    logger.info(
        f"finish_reason=length: {num_length}/{len(results)} ({num_length / max(len(results), 1):.1%})"
    )

    save_yaml(config.model_dump(), config.config_file)
    # run record first, predictions.json last: its presence marks a complete run
    write_run_record(
        output_dir=config.output_dir,
        config=config.model_dump(),
        seed=config.random_seed,
        model_name=config.llm.name,
        split_files={config.split: Path(config.dataset.name) / f"{config.split}.json"},
        start_time=start_time,
        adapter_path=adapter_path,
        extra={"guided_decoding_backend": _guided_decoding_backend(llm, config.guided_json)},
    )

    logger.info(f"Saving results to {config.predictions_file}")
    write_json_atomic(results, config.predictions_file, indent="\t", ensure_ascii=False)


def _guided_decoding_backend(llm: LLM, guided_json: bool) -> str | None:
    if not guided_json:
        return None
    engine = llm.llm_engine
    decoding_config = getattr(engine, "decoding_config", None)
    if decoding_config is not None:
        return getattr(decoding_config, "guided_decoding_backend", None)
    structured = getattr(getattr(engine, "vllm_config", None), "structured_outputs_config", None)
    return getattr(structured, "backend", None)


def prepare_and_save_dataset_for_prediction(
    dataset: Dataset,
    config: PredictInfoExtractionConfig,
    llm: LLM,
) -> Dataset:
    tokenizer = llm.get_tokenizer()
    max_input_length = config.get_max_input_length_accounting_for_output(
        config.max_model_len or llm.llm_engine.model_config.max_model_len
    )
    logger.info(f"Max input length: {max_input_length}")

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
    _log_truncation_stats(dataset)

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


def _log_truncation_stats(dataset: Dataset) -> None:
    num_truncated_docs = sum(bool(x) for x in dataset["num_truncated_tokens"])
    if num_truncated_docs > 0:
        avg_num_truncated_tokens = sum(dataset["num_truncated_tokens"]) / num_truncated_docs
        avg_truncated_ratio = sum(dataset["truncated_ratio"]) / num_truncated_docs
    else:
        avg_num_truncated_tokens = 0
        avg_truncated_ratio = 0

    logger.info(f"Number of truncated docs: {num_truncated_docs}/{len(dataset)}")
    logger.info(f"Average number of truncated tokens: {avg_num_truncated_tokens:0.3f}")
    logger.info(f"Average truncated ratio: {avg_truncated_ratio:0.3f}")


if __name__ == "__main__":
    main()
