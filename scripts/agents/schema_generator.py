import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from juddges.agents.schema_generator import SchemaGenerator

MODEL_NAME = "llama3.1"
PROMPT_DIR = Path("configs/prompt/schema_construction")

load_dotenv(".env")

API_KEY = os.getenv("SELFHOSTED_API_KEY")
API_URL = os.getenv("SELFHOSTED_API_URL")


def load_prompts() -> dict[str, str]:
    prompt_names = [
        "problem_definer_helper",
        "problem_definer",
        "schema_refiner",
        "schema_assessment",
        "schema_generator",
    ]
    prompts = {}
    for prompt_config_file in prompt_names:
        with open(PROMPT_DIR / f"{prompt_config_file}.yaml", "r") as f:
            prompt_config = yaml.safe_load(f)
        prompts.update(prompt_config)
    return prompts


def main() -> None:
    prompts = load_prompts()

    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=API_URL,
        api_key=API_KEY,
        temperature=0.7,
        max_tokens=32_000,
        top_p=0.8,
        presence_penalty=1.5,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
    )
    schema_system = SchemaGenerator(
        llm,
        prompts["problem_definer_helper_prompt"],
        prompts["problem_definer_prompt"],
        prompts["schema_generator_prompt"],
        prompts["schema_assessment_prompt"],
        prompts["schema_refiner_prompt"],
    )

    print("🚀 DEMO 1: Schema Generation for Legal Judgments")
    print("=" * 60)

    # Example 1: Generate schema from natural language description
    schema_system.stream_graph_updates(
        "Generate schema to extract information from judgments about drug abuse, "
        "focus on the defendant's age and the amount of the fine."
    )

    print("\n" + "=" * 60)
    print("✅ Schema generation demo completed!")


if __name__ == "__main__":
    main()
