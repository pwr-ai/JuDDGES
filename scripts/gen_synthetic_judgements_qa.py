import json
from importlib import import_module
from pathlib import Path
from typing import List

import typer
from auto_gptq import exllama_set_max_input_length
from dotenv import load_dotenv
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langdetect import detect as lang_detect
from langsmith import Client
from langsmith.schemas import DataType
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from juddges.data import LangCode
from juddges.data.qa_pairs_json_parser import QAPairsJsonParser
from juddges.data.synthetic.generation_prompt import GEN_QA_COT_PROMPT
from juddges.data.utils import path_safe_udate, read_jsonl
from juddges.prompts.technique import PromptingTechnique

load_dotenv("secrets.env", verbose=True)


class LanguageMismatchError(Exception):
    pass


class SyntheticLegisQAPairs(BaseModel):
    questions: List[str] = Field(description="List of generated questions")
    answers: List[str] = Field(description="List of generated answers")

    def test_empty(self) -> None:
        assert len(self.questions) > 0, "At least one question should be generated"

    def test_equal_length(self) -> None:
        assertion_msg = "Number of questions and answers should be equal"
        assert len(self.questions) == len(self.answers), assertion_msg

    def test_unique_questions(self) -> None:
        assertion_msg = "Questions should be unique"
        assert len(set(self.questions)) == len(self.questions), assertion_msg

    def test_language(self, lang: LangCode) -> None:
        msg = "{smth} should match context language" + f" ({lang.name}/{lang.value})"
        assert lang_detect("\n".join(self.questions)) == lang.value, LanguageMismatchError(
            msg.format(smth="Questions")
        )
        assert lang_detect("\n".join(self.answers)) == lang.value, LanguageMismatchError(
            msg.format(smth="Answers")
        )

    def test(self, language: LangCode) -> None:
        self.test_empty()
        self.test_equal_length()
        self.test_unique_questions()
        self.test_language(language)


def main(
    judgements_fpath: str = typer.Option(
        default=None, help="Dumped `judgements` collection file path"
    ),
    prompt_template_libpath: str = typer.Option(
        default=None,
        help="Prompt variable module path, ex. `juddges.data.synthetic.generation_prompt.GEN_QA_BASELINE_PROMPT`",
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
        help="Maximum number of tokens in input text",
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

        judgements_fpath = SAMPLE_DATA_PATH / "judgements_sample50_20240427_220002f908780.jsonl"
        logger.warning(
            "Dumped `judgements` collection file path not provided."
            f" Using the default `judgements` path: {judgements_fpath}"
        )

    if prompt_template_libpath is None:
        prompt_template_libpath = "juddges.data.synthetic.generation_prompt.GEN_QA_BASELINE_PROMPT"
        logger.warning(
            f"Prompt variable module path not provided. Using the default: `{prompt_template_libpath}`"
        )
    parent_module_path, prompt_template_varname = prompt_template_libpath.rsplit(".", maxsplit=1)
    try:
        prompt_src = import_module(parent_module_path)
    except ImportError:
        logger.error(f"Failed to import prompt template from `{prompt_template_libpath}`")
        raise
    else:
        prompt_template = getattr(prompt_src, prompt_template_varname)

    if out is None:
        from juddges.settings import PL_JUDGEMENTS_SYNTH_QA_PATH

        default_fname = f"judgements_synth_qa__{ts_suffix}.jsonl"
        out = PL_JUDGEMENTS_SYNTH_QA_PATH / default_fname
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
                "prompting_technique": PromptingTechnique.CHAIN_OF_THOUGHT.value,
                "prompt_template": GEN_QA_COT_PROMPT,
                "pairs_count": len(dto.questions),
                "language": lang.value,
            },
        )

        generation_raport["success"] += 1

    logger.info(f"Generation report:\n{json.dumps(generation_raport, indent=2)}")


if __name__ == "__main__":
    typer.run(main)
    logger.success("Done!")
