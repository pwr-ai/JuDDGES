import json
from typing import List

import typer
from dotenv import load_dotenv
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from juddges.data.qa_pairs_json_parser import QAPairsJsonParser
from juddges.data.synthetic.generation_prompt import JUDGEMENTS_QA_COT_PROMPT_V1
from juddges.data.utils import read_jsonl

load_dotenv("secrets.env", verbose=True)


class SyntheticLegisQAPairs(BaseModel):
    questions: List[str] = Field(description="List of generated questions")
    answers: List[str] = Field(description="List of generated answers")

    def test_empty(self) -> None:
        assert len(self.questions) > 0, "At least one question should be generated"

    def test_equal_length(self) -> None:
        assertion_msg = "Number of questions and answers should be equal"
        assert len(self.questions) == len(self.answers), assertion_msg

    def test_q_duplicates(self) -> None:
        assertion_msg = "Questions should be unique"
        assert len(set(self.questions)) == len(self.questions), assertion_msg

    def test_a_duplicates(self) -> None:
        assertion_msg = "Answers should be unique"
        assert len(set(self.answers)) == len(self.answers), assertion_msg

    def test_duplicates(self) -> None:
        self.test_q_duplicates()
        self.test_a_duplicates()

    def test(self) -> None:
        self.test_empty()
        self.test_equal_length()
        self.test_duplicates()


def main(
    judgements_fpath: str = typer.Option(
        default=None, help="Dumped `judgements` collection file path"
    ),
    out: str = typer.Option(default=None, help="Output file path"),
    hf_model: str = typer.Option(
        help="Hugging Face model name or path",
        default="TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ",
    ),
    max_input_length: int = typer.Option(
        default=3551, help="Maximum number of tokens in input text"
    ),
) -> None:
    if judgements_fpath is None:
        # FIXME
        default_judgements_fpath = (
            "/app/data/datasets/pl/judgements_sample10_20240427_094707f595590.jsonl"
        )
        logger.warning(
            "Dumped `judgements` collection file path not provided."
            f" Using the default `judgements` path: {default_judgements_fpath}"
        )
        judgements_fpath = default_judgements_fpath

    if out is None:
        # FIXME
        default_out = "/app/data/datasets/pl/synthetic_judgements_qa.jsonl"
        logger.warning("Output file path not provided. Using the default `out`: {default_out}")
        out = default_out

    qa_parser = QAPairsJsonParser(pydantic_object=SyntheticLegisQAPairs)
    logger.debug(f"{qa_parser.get_format_instructions()=}")

    prompt = ChatPromptTemplate.from_template(
        template=JUDGEMENTS_QA_COT_PROMPT_V1,
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
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1,
    )
    hf_pipeline = HuggingFacePipeline(pipeline=text_gen_pipeline)

    gen_chain = prompt | hf_pipeline | qa_parser

    logger.info("Generating QA pairs from provided collection data...")
    for judgement in read_jsonl(judgements_fpath):
        num_text_tokens = tokenizer.encode(judgement["text"], return_tensors="pt").shape[1]
        if num_text_tokens > 0.95 * max_input_length:
            logger.warning(
                f"Skipping judgement with id: {judgement['_id']} due to text"
                f"length > {max_input_length} ({num_text_tokens})..."
            )
            continue

        chain_input = {"context": judgement["text"], "format_md_ext": "json"}
        qa_pairs = gen_chain.invoke(chain_input)
        logger.debug(json.dumps(qa_pairs, indent=2, ensure_ascii=False))

        dto = SyntheticLegisQAPairs(**qa_pairs)
        dto.test()

        break


if __name__ == "__main__":
    typer.run(main)
