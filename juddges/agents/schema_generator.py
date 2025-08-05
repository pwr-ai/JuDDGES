import operator
from typing import Annotated, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# State management for the multi-agent system
class AgentState(TypedDict):
    """State shared across all agents in the schema processing workflow."""

    messages: Annotated[list, add_messages]  # Chat messages between agents
    user_input: str  # Original user request
    problem_help: str | None  # output from problem definer helper agent
    user_feedback: str | None  # feedback for the problem definer helper agent
    problem_definition: str | None  # problem definition
    current_schema: dict[str, Any] | None  # Current schema being processed
    refinement_rounds: Annotated[int, operator.add]  # Number of refinement iterations
    assessment_result: dict[str, Any] | None  # Quality assessment results


class ProblemDefinerHelperAgent:
    """Routes user requests to the appropriate processing agent."""

    def __init__(self, llm, prompt) -> None:
        self.parser = StrOutputParser()
        prompt = PromptTemplate.from_template(prompt)
        self.chain = prompt | llm

    def __call__(self, state: AgentState) -> dict[str, Any]:
        response = self.chain.invoke({"user_input": state["user_input"]})
        parsed_response = self.parser.parse(response.content)
        return {"messages": [response], "problem_help": parsed_response}


class ProblemDefinerAgent:
    """Generates new schemas from natural language descriptions."""

    def __init__(self, llm, prompt) -> None:
        self.parser = StrOutputParser()
        prompt = PromptTemplate.from_template(prompt)
        self.chain = prompt | llm

    def __call__(self, state: AgentState) -> dict[str, Any]:
        user_input = state["user_input"]
        problem_help = state["problem_help"]
        user_feedback = state["user_feedback"]

        response = self.chain.invoke(
            {"user_input": user_input, "problem_help": problem_help, "user_feedback": user_feedback}
        )
        parsed_response = self.parser.parse(response.content)

        update_dict = {"messages": [response], "problem_definition": parsed_response}
        return update_dict


class SchemaGeneratorAgent:
    """Generates new schemas from natural language descriptions."""

    def __init__(self, llm, prompt) -> None:
        self.parser = JsonOutputParser()
        prompt = PromptTemplate.from_template(prompt)
        self.chain = prompt | llm

    def __call__(self, state: AgentState) -> dict[str, Any]:
        user_input = state["user_input"]
        problem_help = state["problem_help"]
        user_feedback = state["user_feedback"]
        problem_definition = state["problem_definition"]

        response = self.chain.invoke(
            {
                "user_input": user_input,
                "problem_help": problem_help,
                "user_feedback": user_feedback,
                "problem_definition": problem_definition,
            }
        )
        parsed_response = self.parser.parse(response.content)

        update_dict = {"messages": [response]}
        if parsed_response.get("is_generated", False):
            update_dict["current_schema"] = parsed_response.get("schema")

        return update_dict


class SchemaAssessmentAgent:
    """Evaluates schema quality against multiple criteria."""

    def __init__(self, llm, prompt) -> None:
        self.parser = JsonOutputParser()
        prompt = PromptTemplate.from_template(prompt)
        self.chain = prompt | llm

    def __call__(self, state: AgentState) -> dict[str, Any]:
        user_input = state["user_input"]
        problem_help = state["problem_help"]
        user_feedback = state["user_feedback"]
        problem_definition = state["problem_definition"]
        current_schema = state["current_schema"]

        response = self.chain.invoke(
            {
                "user_input": user_input,
                "problem_help": problem_help,
                "user_feedback": user_feedback,
                "problem_definition": problem_definition,
                "current_schema": current_schema,
            }
        )
        parsed_response = self.parser.parse(response.content)
        return {"messages": [response], "assessment_result": parsed_response}


class SchemaRefinerAgent:
    """Improves existing schemas based on quality criteria."""

    def __init__(self, llm, prompt) -> None:
        self.parser = JsonOutputParser()
        prompt = PromptTemplate.from_template(prompt)
        self.chain = prompt | llm

    def __call__(self, state: AgentState) -> dict[str, Any]:
        user_input = state["user_input"]
        problem_help = state["problem_help"]
        user_feedback = state["user_feedback"]
        problem_definition = state["problem_definition"]
        current_schema = state["current_schema"]
        assessment_result = state["assessment_result"]

        response = self.chain.invoke(
            {
                "user_input": user_input,
                "problem_help": problem_help,
                "user_feedback": user_feedback,
                "problem_definition": problem_definition,
                "current_schema": current_schema,
                "assessment_result": assessment_result,
            }
        )
        parsed_response = self.parser.parse(response.content)
        refinement_rounds = state.get("refinement_rounds", 0)
        update_dict = {"messages": [response], "refinement_rounds": refinement_rounds + 1}
        if parsed_response.get("is_refined", False):
            update_dict["current_schema"] = parsed_response.get("schema")
        return update_dict


def route_after_assessment(state: AgentState) -> str:
    """Route to refiner if needs refinement and under max rounds, otherwise END."""
    assessment = state.get("assessment_result", {})
    refinement_rounds = state.get("refinement_rounds", 0)

    needs_refinement = assessment.get("needs_refinement", False)
    max_rounds_reached = refinement_rounds >= 5

    if refinement_rounds < 2:
        return "llm_schema_refiner"

    if needs_refinement and not max_rounds_reached:
        return "llm_schema_refiner"

    if max_rounds_reached:
        print("⚠️ Maximum refinement rounds (5) reached, ending process")

    return END


# def human_feedback(state):
#     feedback = input("Please provide feedback: ")
#     return {"user_feedback": feedback, "messages": [HumanMessage(content=feedback)]}


class HumanFeedback:
    """Human feedback node."""

    def __call__(self, state: AgentState) -> dict[str, Any]:
        feedback = input("Please provide feedback: ")
        return {"user_feedback": feedback, "messages": [HumanMessage(content=feedback)]}


class SchemaGenerator:
    """Multi-agent system for generating, refining, and assessing legal schemas."""

    def __init__(
        self,
        llm: BaseChatModel,
        prompt_problem_definer_helper: str,
        prompt_problem_definer: str,
        prompt_schema_generator: str,
        prompt_schema_assessment: str,
        prompt_schema_refiner: str,
    ) -> None:
        # Initialize all agents
        self.problem_definer_helper = ProblemDefinerHelperAgent(llm, prompt_problem_definer_helper)
        self.human_feedback = HumanFeedback()
        self.problem_definer = ProblemDefinerAgent(llm, prompt_problem_definer)
        self.schema_generator = SchemaGeneratorAgent(llm, prompt_schema_generator)
        self.schema_assessment = SchemaAssessmentAgent(llm, prompt_schema_assessment)
        self.schema_refiner = SchemaRefinerAgent(llm, prompt_schema_refiner)

        # # Build graph
        # self.graph_builder = StateGraph(AgentState)

        # # Add nodes
        # self.graph_builder.add_node("llm_problem_definer_helper", self.problem_definer_helper)
        # self.graph_builder.add_node("user_feedback_node", human_feedback)
        # self.graph_builder.add_node("llm_problem_definer", self.problem_definer)
        # self.graph_builder.add_node("llm_schema_generator", self.schema_generator)
        # self.graph_builder.add_node("llm_schema_assessment", self.schema_assessment)
        # self.graph_builder.add_node("llm_schema_refiner", self.schema_refiner)

        # # Add edges
        # self.graph_builder.add_edge(START, "llm_problem_definer_helper")
        # self.graph_builder.add_edge("llm_problem_definer_helper", "user_feedback_node")
        # self.graph_builder.add_edge("user_feedback_node", "llm_problem_definer")
        # self.graph_builder.add_edge("llm_problem_definer", "llm_schema_generator")
        # self.graph_builder.add_edge("llm_schema_generator", "llm_schema_assessment")
        # self.graph_builder.add_edge("llm_schema_refiner", "llm_schema_assessment")

        # # Route from assessment to refiner or END
        # self.graph_builder.add_conditional_edges(
        #     "llm_schema_assessment",
        #     route_after_assessment,
        #     {
        #         "llm_schema_refiner": "llm_schema_refiner",
        #         END: END,
        #     },
        # )

        self.graph = self.build_graph()

    def build_graph(self):
        graph_builder = StateGraph(AgentState)

        graph_builder.add_node("llm_problem_definer_helper", self.problem_definer_helper)
        graph_builder.add_node("user_feedback_node", self.human_feedback)
        graph_builder.add_node("llm_problem_definer", self.problem_definer)
        graph_builder.add_node("llm_schema_generator", self.schema_generator)
        graph_builder.add_node("llm_schema_assessment", self.schema_assessment)
        graph_builder.add_node("llm_schema_refiner", self.schema_refiner)

        # Add edges
        graph_builder.add_edge(START, "llm_problem_definer_helper")
        graph_builder.add_edge("llm_problem_definer_helper", "user_feedback_node")
        graph_builder.add_edge("user_feedback_node", "llm_problem_definer")
        graph_builder.add_edge("llm_problem_definer", "llm_schema_generator")
        graph_builder.add_edge("llm_schema_generator", "llm_schema_assessment")
        graph_builder.add_edge("llm_schema_refiner", "llm_schema_assessment")

        graph_builder.add_conditional_edges(
            "llm_schema_assessment",
            route_after_assessment,
            {
                "llm_schema_refiner": "llm_schema_refiner",
                END: END,
            },
        )

        return graph_builder.compile()

    def stream_graph_updates(self, user_input: str, current_schema: dict = None) -> None:
        print("👤 Human:")
        print(user_input)
        print("-" * 50)
        """Process user input through the multi-agent workflow and display results."""
        initial_state = AgentState(
            messages=[],
            user_input=user_input,
            problem_help=None,
            user_feedback=None,
            problem_definition=None,
            current_schema=current_schema,
            refinement_rounds=0,
            assessment_result=None,
        )

        for event in self.graph.stream(initial_state):
            for value in event.values():
                # Get the full content from the latest message
                full_content = value["messages"][-1].content

                # Print with proper formatting to avoid truncation
                if value["messages"][-1].type == "human":
                    print("👤 Human:")
                else:
                    print("🤖 Assistant:")
                print(full_content)
                print("-" * 50)

    def get_complete_results(self, user_input: str, current_schema: dict = None) -> dict:
        """Process user input and return complete results without display truncation."""
        initial_state = AgentState(
            messages=[],
            user_input=user_input,
            problem_help=None,
            user_feedback=None,
            problem_definition=None,
            current_schema=current_schema,
            refinement_rounds=0,
            assessment_result=None,
        )

        # Run the complete workflow
        final_state = self.graph.invoke(initial_state)
        print(final_state)

        # Extract and return all results
        return {
            "final_schema": final_state.get("current_schema"),
            "assessment_result": final_state.get("assessment_result"),
            "refinement_rounds": final_state.get("refinement_rounds", 0),
            "all_messages": [msg.content for msg in final_state.get("messages", [])],
        }
