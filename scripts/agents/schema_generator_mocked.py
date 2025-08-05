import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from juddges.agents.schema_generator import AgentState, SchemaGenerator

MODEL_NAME = "llama3.1"
PROMPT_DIR = Path("configs/prompt/schema_construction")
CASES_DIR = Path("data/agents/cases")

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


def load_cases() -> dict[str, dict[str, str]]:
    cases = {}
    for case_file in CASES_DIR.glob("*.yaml"):
        with open(case_file, "r") as f:
            case = yaml.safe_load(f)
        cases[case_file.stem] = case
    return cases


def create_mock_problem_helper(problem_help_text: str):
    """Create a mock function that returns predefined problem help text."""

    def mock_problem_helper(state: AgentState) -> dict[str, Any]:
        response = AIMessage(content=problem_help_text)
        return {"messages": [response], "problem_help": problem_help_text}

    return mock_problem_helper


def create_mock_user_feedback(user_feedback_text: str):
    """Create a mock function that returns predefined user feedback."""

    def mock_user_feedback(state: AgentState) -> dict[str, Any]:
        return {
            "user_feedback": user_feedback_text,
            "messages": [HumanMessage(content=user_feedback_text)],
        }

    return mock_user_feedback


def main() -> None:
    prompts = load_prompts()
    cases = load_cases()

    print("📁 Available cases:")
    for case_name in cases.keys():
        print(f"  - {case_name}")
    print()

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

    # Demo each case with mocked agents
    for case_name, case_data in cases.items():
        print(f"🚀 DEMO: Schema Generation for {case_name.replace('_', ' ').title()}")
        print("=" * 60)

        # Create the schema system normally
        schema_system = SchemaGenerator(
            llm,
            prompts["problem_definer_helper_prompt"],
            prompts["problem_definer_prompt"],
            prompts["schema_generator_prompt"],
            prompts["schema_assessment_prompt"],
            prompts["schema_refiner_prompt"],
        )

        # Replace only the specific agents that have YAML data
        if "problem_help" in case_data:
            schema_system.problem_definer_helper = create_mock_problem_helper(
                case_data["problem_help"]
            )

        if "user_feedback" in case_data:
            schema_system.human_feedback = create_mock_user_feedback(case_data["user_feedback"])

        # Recompile the graph with the mocked nodes
        schema_system.graph = schema_system.build_graph()

        # Use the user_input from the case data
        user_input = case_data.get("user_input", "Generate a schema for legal documents")
        schema_system.stream_graph_updates(user_input)

        print("\n" + "=" * 60)
        print(f"✅ Schema generation demo for {case_name} completed!")
        print("\n" * 2)


if __name__ == "__main__":
    main()
