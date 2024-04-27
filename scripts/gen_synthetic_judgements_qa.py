import json
from pathlib import Path
from typing import List

import typer
from dotenv import load_dotenv
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langdetect import detect as lang_detect
from langsmith import Client
from langsmith.schemas import DataType
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from juddges.data import LangCode
from juddges.data.qa_pairs_json_parser import QAPairsJsonParser
from juddges.data.synthetic.generation_prompt import GEN_QA_COT_PROMPT
from juddges.data.utils import path_safe_udate, read_jsonl
from juddges.prompts.technique import PromptingTechnique

load_dotenv("secrets.env", verbose=True)


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
        assert lang_detect("\n".join(self.questions)) == lang.value, msg.format(smth="Questions")
        assert lang_detect("\n".join(self.answers)) == lang.value, msg.format(smth="Answers")

    def test(self, language: LangCode) -> None:
        self.test_empty()
        self.test_equal_length()
        self.test_unique_questions()
        self.test_language(language)


def main(
    judgements_fpath: str = typer.Option(
        default=None, help="Dumped `judgements` collection file path"
    ),
    out_smith_dataset: str = typer.Option(default=None, help="Smith dataset name"),
    out: str = typer.Option(default=None, help="Output file path"),
    hf_model: str = typer.Option(
        help="Hugging Face model name or path",
        default="TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ",
    ),
    max_input_length: int = typer.Option(
        default=3551, help="Maximum number of tokens in input text"
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
        # FIXME
        default_judgements_fpath = (
            "/app/data/datasets/pl/judgements_sample50_20240427_220002f908780.jsonl"
        )
        logger.warning(
            "Dumped `judgements` collection file path not provided."
            f" Using the default `judgements` path: {default_judgements_fpath}"
        )
        judgements_fpath = default_judgements_fpath

    if out is None:
        # FIXME
        default_out_dir = Path(f"/app/data/datasets/{lang.value}/qa/generated")
        default_out_dir.mkdir(parents=True, exist_ok=True)
        default_fname = f"judgements_synth_qa__{ts_suffix}.jsonl"
        default_fpath = Path(default_out_dir) / default_fname
        logger.warning(f"Output file path not provided. Using the default `out`: {default_fpath}")
        out = default_fpath

    smith = Client()

    dataset_default_name = f"judgements_synth_qa__{ts_suffix}"
    dataset = smith.create_dataset(
        dataset_name=out_smith_dataset if out_smith_dataset is not None else dataset_default_name,
        data_type=DataType.kv,
    )

    qa_parser = QAPairsJsonParser(pydantic_object=SyntheticLegisQAPairs)
    logger.debug(f"{qa_parser.get_format_instructions()=}")

    prompt = ChatPromptTemplate.from_template(
        template=GEN_QA_COT_PROMPT,
        partial_variables={"format_instructions": qa_parser.get_format_instructions()},
    )

    # For example: revision="gptq-4bit-32g-actorder_True"
    model = AutoModelForCausalLM.from_pretrained(
        hf_model, device_map="auto", trust_remote_code=True, revision="main"
    )
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

    gen_chain = prompt | hf_pipeline | qa_parser
    logger.info("Generating QA pairs from provided collection data...")

    generation_raport = {k: 0 for k in ["success", "failure", "skipped"]}
    for judgement in read_jsonl(judgements_fpath):
        # FIXME: change to batch processing
        num_text_tokens = tokenizer.encode(judgement["text"], return_tensors="pt").shape[1]
        if num_text_tokens > 0.95 * max_input_length:
            logger.warning(
                f"Skipping {judgement['_id']=} due to `max_input_length`"
                f" > {max_input_length} ({num_text_tokens})..."
            )
            generation_raport["skipped"] += 1
            continue

        try:
            chain_input = {
                "language": lang.value,
                "context": judgement["text"],
                "format_md_ext": "json",
            }
            qa_pairs = gen_chain.invoke(chain_input)

            dto = SyntheticLegisQAPairs(**qa_pairs)
            dto.test(language=lang)
        except Exception as e:
            logger.error(f"QA unsuccessful generation for {judgement['_id']=}\n{e}")
            generation_raport["failure"] += 1
            continue

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
