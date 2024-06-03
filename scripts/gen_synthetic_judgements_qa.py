import json
import os
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Tuple

import deepl
import typer
import yaml
from joblib import Memory
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.schemas import DataType
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from juddges.data.models import LangCode, QAGenerationJudgementMetadata, SyntheticQAPairs
from juddges.data.qa_pairs_json_parser import QAPairsJsonParser
from juddges.data.utils import path_safe_udate, read_jsonl, save_jsonl
from juddges.exception import LanguageMismatchError
from juddges.prompts.technique import PromptingTechnique
from juddges.settings import (
    CACHE_DIR,
    SAMPLE_DATA_PATH,
    prepare_langchain_cache,
)

prepare_langchain_cache()
deepl.http_client.user_agent = os.environ.get("DEEPL_USER_AGENT", None)
translator_memory = Memory(CACHE_DIR, verbose=0)  # FIXME: change to non-local; postgres?


@translator_memory.cache(ignore=["translator"])
def translate_text(translator: deepl.Translator, text: str, target_lang: str) -> str:
    return translator.translate_text(text, target_lang=target_lang).text


def translate_qa_pairs(
    translator: deepl.Translator, qa_pairs: SyntheticQAPairs, target_lang: LangCode
) -> SyntheticQAPairs:
    translated_questions = [
        translate_text(translator, question, target_lang=target_lang.value)
        for question in qa_pairs.questions
    ]
    translated_answers = [
        translate_text(translator, answer, target_lang=target_lang.value)
        for answer in qa_pairs.answers
    ]
    return SyntheticQAPairs(questions=translated_questions, answers=translated_answers)


def handle_lang_mismatch(
    translator: deepl.Translator,
    context_id: str,
    qa_pairs: SyntheticQAPairs,
    target_lang: LangCode,
) -> Tuple[List[str], SyntheticQAPairs | None]:
    catched_errors = []
    try:
        dto = translate_qa_pairs(translator, qa_pairs, target_lang=target_lang)
    except Exception as e:
        logger.error(f"SyntheticQAPairs translation failed for {context_id=}\n{e}")
        catched_errors.append(e.__class__.__name__)
        return catched_errors, None
    else:
        try:
            logger.debug(f"Translated SyntheticQAPairs: {dto.dict()=}")
            dto.test(language=target_lang)
        except LanguageMismatchError as e:
            logger.error(f"Language mismatch after translation for {context_id=}\n{e}")
            catched_errors.append(e.__class__.__name__)
            return catched_errors, None
        except Exception as e:
            logger.error(f"Unexpected error after translation for {context_id=}\n{e}")
            catched_errors.append(e.__class__.__name__)
            return catched_errors, None
        else:
            logger.info(f"SyntheticQAPairs translated successfully for {context_id=}")
    return catched_errors, dto


def main(
    openai_model: str = typer.Option(
        default="gpt-3.5-turbo",
        help="OpenAI model name",
    ),
    model_cfg: str = typer.Option(
        default=None,
        help="Path to the model .yaml configuration file",
    ),
    judgements_fpath: str = typer.Option(
        default=str(SAMPLE_DATA_PATH / "judgements_sample50_20240427_220002f908780.jsonl"),
        help="Dumped `judgements` collection file path",
    ),
    prompt_template_libpath: str = typer.Option(
        default="juddges.prompts.synthetic_qa.GEN_QA_COT_PROMPT",
        help="Prompt variable module path",
    ),
    prompting_technique: PromptingTechnique = typer.Option(
        default=PromptingTechnique.CHAIN_OF_THOUGHT.value,
        show_choices=True,
        help="Prompting technique",
    ),
    out_smith_dataset: str = typer.Option(default=None, help="LangSmith dataset name"),
    out: str = typer.Option(default=None, help="Output file path"),
    lang: LangCode = typer.Option(
        default=LangCode.POLISH.value,
        show_choices=True,
        help="Language code (ISO 639-1) of the given data",
    ),
    translate: bool = typer.Option(
        default=True,
        help=(
            "Translate generated data to context language `lang`."
            " Requires DEEPL_AUTH_KEY environment variable."
        ),
    ),
) -> None:
    if translate:
        translator = deepl.Translator(os.environ["DEEPL_AUTH_KEY"])

    if model_cfg is not None:
        with open(model_cfg) as f:
            model_cfg_: Dict[str, Any] = yaml.safe_load(f)
    elif openai_model is not None:
        model_cfg_ = {"model": openai_model}
    else:
        logger.error("Provide either `openai_model` or `model_cfg`. Aborting...")
        raise typer.Abort()

    parent_module_path, prompt_template_varname = prompt_template_libpath.rsplit(".", maxsplit=1)
    prompt_src = import_module(parent_module_path)
    prompt_template = getattr(prompt_src, prompt_template_varname)

    ts_suffix = path_safe_udate()
    if out is None:
        from juddges.settings import PL_JUDGEMENTS_SYNTH_QA_PATH

        default_fname = f"judgements_sample50_synth_qa__{prompting_technique.value}__{model_cfg_['model']}__{ts_suffix}.jsonl"
        out = str(PL_JUDGEMENTS_SYNTH_QA_PATH / default_fname)
        logger.warning(f"Output file path not provided. Using the default `out`: {out}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Setting up LangSmith client...")
    smith = Client()

    logger.info("Creating dataset...")
    dataset_default_name = "judgements_sample50_synth_qa__{prompting_technique}__{model}".format(
        prompting_technique=prompting_technique.value,
        model=model_cfg_["model"].replace("/", "_"),
    )
    dataset = smith.create_dataset(
        dataset_name=(out_smith_dataset if out_smith_dataset is not None else dataset_default_name),
        data_type=DataType.kv,
        description=(
            f"Synthetic QA pairs generated from `{judgements_fpath}` with `{model_cfg_['model']}`"
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

    if model_cfg is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg_["model"],
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=model_cfg_.get("torch_dtype", None),
            **model_cfg_.get("kwargs", {}),
        )
        logger.debug(f"{model.device=}")

        tokenizer = AutoTokenizer.from_pretrained(model_cfg_["model"], use_fast=True)
        text_gen_pipeline = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            **model_cfg_["generate_kwargs"],
        )
        llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
    elif openai_model is not None:
        llm = ChatOpenAI(model=model_cfg_["model"], api_key=os.environ["OPENAI_API_KEY"])

    logger.info("Setting up generation pipeline...")
    gen_chain = prompt | llm | qa_parser

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
        except Exception as e:
            logger.error(f"QA unsuccessful generation for {judgement['_id']=}\n{e}")
            generation_raport[f"failure__{e.__class__.__name__}"] += 1
            continue

        try:
            logger.debug(f"{qa_pairs=}")
            dto = SyntheticQAPairs(**qa_pairs)
        except Exception as e:
            logger.error(
                f"SyntheticQAPairs data object creation failed for {judgement['_id']=}\n{e}"
            )
            continue
        else:
            try:
                dto.test(language=lang)
            except LanguageMismatchError as e:
                logger.warning(f"Language mismatch for {judgement['_id']=}\n{e}.")
                lang_mismatch = True
            except Exception as e:
                logger.error(f"Failed quality tests of generated data for {judgement['_id']=}\n{e}")
                continue

        if lang_mismatch:
            if translate:
                lang_mismatch_errors, dto = handle_lang_mismatch(
                    translator, context_id=judgement["_id"], qa_pairs=dto, target_lang=lang
                )
                if dto is None:
                    for error in lang_mismatch_errors:
                        generation_raport[f"failure__{error}"] += 1
                    continue
            else:
                logger.warning(
                    f"Ignoring language mismatch between context and generated data for {judgement['_id']=}"
                )

        logger.debug(f"Creating example for {judgement['_id']=}...")
        example_metadata = {
            "judgement_id": judgement["_id"],
            "model": model_cfg_["model"],
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
