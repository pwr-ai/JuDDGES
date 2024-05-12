import json
from importlib import import_module
from pathlib import Path
from collections import defaultdict

import typer
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client
from langsmith.schemas import DataType
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from juddges.data.models import LangCode, SyntheticQAPairs, QAGenerationJudgementMetadata
from juddges.data.qa_pairs_json_parser import QAPairsJsonParser
from juddges.data.utils import path_safe_udate, read_jsonl, save_jsonl
from juddges.exception import LanguageMismatchError
from juddges.prompts.technique import PromptingTechnique
from juddges.settings import prepare_langchain_cache, SAMPLE_DATA_PATH

prepare_langchain_cache()


def main(
    judgements_fpath: str = typer.Option(
        default=str(SAMPLE_DATA_PATH / "judgements_sample50_20240427_220002f908780.jsonl"),
        help="Dumped `judgements` collection file path",
    ),
    prompt_template_libpath: str = typer.Option(
        default="juddges.prompts.synthetic_qa.GEN_QA_BASELINE_PROMPT",
        help="Prompt variable module path",
    ),
    prompting_technique: PromptingTechnique = typer.Option(
        default=PromptingTechnique.STANDARD.value,
        show_choices=True,
        help="Prompting technique",
    ),
    out_smith_dataset: str = typer.Option(default=None, help="LangSmith dataset name"),
    out: str = typer.Option(default=None, help="Output file path"),
    hf_model: str = typer.Option(
        help="Hugging Face model name or path",
        # default="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
        # default="microsoft/Phi-3-mini-128k-instruct",
        default="TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ",
    ),
    max_input_length: int = typer.Option(
        # default=32_000,  # TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
        # default=128_000,  # microsoft/Phi-3-mini-128k-instruct
        default=3551,  # TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ
        help="Maximum number of tokens of the prompt and the context",
    ),
    lang: LangCode = typer.Option(
        default=LangCode.POLISH.value,
        show_choices=True,
        help="Language code (ISO 639-1) of the given data",
    ),
    max_new_tokens: int = typer.Option(
        default=512, help="Maximum number of tokens in generated text"
    ),
    temperature: float = typer.Option(default=0.7, help="Temperature for text generation"),
    top_p: float = typer.Option(default=0.95, help="Top-p for text generation"),
    top_k: int = typer.Option(default=40, help="Top-k for text generation"),
    repetition_penalty: float = typer.Option(
        default=1.1, help="Repetition penalty for text generation"
    ),
) -> None:
    ts_suffix = path_safe_udate()
    parent_module_path, prompt_template_varname = prompt_template_libpath.rsplit(".", maxsplit=1)
    prompt_src = import_module(parent_module_path)
    prompt_template = getattr(prompt_src, prompt_template_varname)

    if out is None:
        from juddges.settings import PL_JUDGEMENTS_SYNTH_QA_PATH

        default_fname = (
            f"judgements_sample50_synth_qa__{prompting_technique.value}__{ts_suffix}.jsonl"
        )
        out = str(PL_JUDGEMENTS_SYNTH_QA_PATH / default_fname)
        logger.warning(f"Output file path not provided. Using the default `out`: {out}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Setting up LangSmith client...")
    smith = Client()

    logger.info("Creating dataset...")
    dataset_default_name = f"judgements_sample50_synth_qa__{prompting_technique.value}"
    dataset = smith.create_dataset(
        dataset_name=out_smith_dataset if out_smith_dataset is not None else dataset_default_name,
        data_type=DataType.kv,
        description=(
            f"Synthetic QA pairs generated from `{judgements_fpath}` with `{hf_model}`"
            f" model and `{prompting_technique.value}` prompting technique"
        ),
    )

    qa_parser = QAPairsJsonParser(pydantic_object=SyntheticQAPairs)
    logger.debug(f"{qa_parser.get_format_instructions()=}")

    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={
            "format_instructions": qa_parser.get_format_instructions(),
            "format_md_ext": "json",
            "context_metadata_schema": QAGenerationJudgementMetadata.schema_json(),
        },
    )

    logger.info("Loading Hugging Face model and tokenizer...")

    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        trust_remote_code=True,
        device_map="auto",
    )
    logger.debug(f"{model.device=}")

    tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=True)
    text_gen_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    hf_pipeline = HuggingFacePipeline(pipeline=text_gen_pipeline)

    logger.info("Setting up generation pipeline...")
    gen_chain = prompt | hf_pipeline | qa_parser

    logger.info("Generating QA pairs from provided collection data...")
    generation_raport = defaultdict(int)
    for judgement in tqdm(list(read_jsonl(judgements_fpath))):
        # FIXME: change to batch processing
        lang_mismatch = False
        context_metadata = QAGenerationJudgementMetadata(
            type_=judgement["type"],
            excerpt=judgement["excerpt"],
            chairman=judgement["chairman"],
            decision=judgement["decision"],
            judges=judgement["judges"],
            legal_bases=judgement["legalBases"],
            publisher=judgement["publisher"],
            recorder=judgement["recorder"],
            references=judgement["references"],
            reviser=judgement["reviser"],
            theme_phrases=judgement["themePhrases"],
        )
        try:
            chain_input = {
                "lang_code": lang.value,
                "context_metadata": context_metadata.dict(),
                "context": judgement["text"],
            }
            qa_pairs = gen_chain.invoke(chain_input)

            dto = SyntheticQAPairs(**qa_pairs)
            logger.debug(f"{dto.dict()=}")
            dto.test(language=lang)
        except LanguageMismatchError as e:
            logger.warning(
                f"Language mismatch for {judgement['_id']=}\n{e}."
                " Metadata `language` property will be set to `None`."
            )
            generation_raport[f"warning__{e.__class__.__name__}"] += 1
            lang_mismatch = True
        except Exception as e:
            logger.error(f"QA unsuccessful generation for {judgement['_id']=}\n{e}")
            generation_raport[f"failure__{e.__class__.__name__}"] += 1
            continue

        logger.debug(f"Creating example for {judgement['_id']=}...")
        example_metadata = {
            "judgement_id": judgement["_id"],
            "hf_model": hf_model,
            "prompt_template": prompt_template,
            "prompting_technique": prompting_technique.value,
            "pairs_count": len(dto),
            "language": lang.value if not lang_mismatch else None,
        }
        smith.create_example(
            inputs=chain_input,
            outputs=dto.dict(),
            dataset_id=dataset.id,
            metadata=example_metadata,
        )

        logger.debug(f"Appending example to `{out}`")
        local_example = {**dto.dict(), "metadata": example_metadata}
        save_jsonl(records=[local_example], out=out, mode="a")

        generation_raport["success"] += 1

    logger.info(f"Generation report:\n{json.dumps(generation_raport, indent=2)}")
    report_out = Path(out).parent / f"gen_report__{Path(out).stem}.json"
    with open(report_out, "w") as f:
        json.dump(generation_raport, f, indent=2)


if __name__ == "__main__":
    typer.run(main)
    logger.success("Done!")
