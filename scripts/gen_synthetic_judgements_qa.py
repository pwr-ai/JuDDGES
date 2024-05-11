import json
from importlib import import_module
from pathlib import Path

import typer
from auto_gptq import exllama_set_max_input_length
from dotenv import load_dotenv
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client
from langsmith.schemas import DataType
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from juddges.data.models import LangCode, SyntheticLegisQAPairs
from juddges.data.qa_pairs_json_parser import QAPairsJsonParser
from juddges.data.utils import path_safe_udate, read_jsonl
from juddges.exception import LanguageMismatchError
from juddges.prompts.technique import PromptingTechnique

load_dotenv("secrets.env", verbose=True)


def main(
    judgements_fpath: str = typer.Option(
        default=None, help="Dumped `judgements` collection file path"
    ),
    prompt_template_libpath: str = typer.Option(
        default=None,
        help="Prompt variable module path, ex. `juddges.prompts.synthetic_qa.GEN_QA_BASELINE_PROMPT`",
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
        help="Language code (ISO 639) of the given data",
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
    if judgements_fpath is None:
        from juddges.settings import SAMPLE_DATA_PATH

        judgements_fpath = str(
            SAMPLE_DATA_PATH / "judgements_sample50_20240427_220002f908780.jsonl"
        )
        logger.warning(
            "Dumped `judgements` collection file path not provided."
            f" Using the default `judgements` path: {judgements_fpath}"
        )

    if prompt_template_libpath is None:
        prompt_template_libpath = "juddges.prompts.synthetic_qa.GEN_QA_BASELINE_PROMPT"
        logger.warning(
            f"Prompt variable module path not provided. Using the default: `{prompt_template_libpath}`"
        )
    parent_module_path, prompt_template_varname = prompt_template_libpath.rsplit(".", maxsplit=1)
    prompt_src = import_module(parent_module_path)
    prompt_template = getattr(prompt_src, prompt_template_varname)

    if out is None:
        from juddges.settings import PL_JUDGEMENTS_SYNTH_QA_PATH

        default_fname = f"judgements_synth_qa__{ts_suffix}.jsonl"
        out = str(PL_JUDGEMENTS_SYNTH_QA_PATH / default_fname)
        logger.warning(f"Output file path not provided. Using the default `out`: {out}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Setting up LangSmith client...")
    smith = Client()

    logger.info("Creating dataset...")
    dataset_default_name = f"judgements_synth_qa__{ts_suffix}"
    dataset = smith.create_dataset(
        dataset_name=out_smith_dataset if out_smith_dataset is not None else dataset_default_name,
        data_type=DataType.kv,
    )

    qa_parser = QAPairsJsonParser(pydantic_object=SyntheticLegisQAPairs)
    logger.debug(f"{qa_parser.get_format_instructions()=}")

    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={"format_instructions": qa_parser.get_format_instructions()},
    )

    logger.info("Loading Hugging Face model and tokenizer...")

    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        trust_remote_code=True,
        device_map="auto",
    )
    model = exllama_set_max_input_length(model, max_input_length=max_input_length)
    logger.debug(f"{model.device=}")

    tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=True)
    text_gen_pipeline = pipeline(
        "text-generation",
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
    generation_raport = {k: 0 for k in ["success", "failure", "warning"]}
    for judgement in tqdm(list(read_jsonl(judgements_fpath))):
        # FIXME: change to batch processing
        try:
            chain_input = {
                "language": lang.value,
                "context": judgement["text"],
                "format_md_ext": "json",
            }
            qa_pairs = gen_chain.invoke(chain_input)

            dto = SyntheticLegisQAPairs(**qa_pairs)
            logger.debug(f"{dto.dict()=}")
            dto.test(language=lang)
        except LanguageMismatchError as e:
            logger.warning(f"Language mismatch for {judgement['_id']=}\n{e}")
            generation_raport["warning"] += 1
        except Exception as e:
            logger.error(f"QA unsuccessful generation for {judgement['_id']=}\n{e}")
            generation_raport["failure"] += 1
            continue

        logger.debug(f"Creating example for {judgement['_id']=}...")
        smith.create_example(
            inputs=chain_input,
            outputs=dto.dict(),
            dataset_id=dataset.id,
            metadata={
                "judgement_id": judgement["_id"],
                "prompt_template": prompt_template,
                "prompting_technique": prompting_technique,
                "pairs_count": len(dto.questions),
                "language": lang.value,
            },
        )

        generation_raport["success"] += 1

    logger.info(f"Generation report:\n{json.dumps(generation_raport, indent=2)}")


if __name__ == "__main__":
    typer.run(main)
    logger.success("Done!")
