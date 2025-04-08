from langchain_openai import ChatOpenAI
from loguru import logger
from tqdm.auto import tqdm

from juddges.evaluation.eval_structured import StructuredEvaluatorBase
from juddges.evaluation.parse import EMPTY_ANSWER

# TODO: might be a configurable prompt in future
# Credit: https://github.com/openai/evals
PROMPT_PL = """
You are comparing the extracted information from a submission to the expert-provided information on a given text in Polish. Here is the data:
[BEGIN DATA]
************
[Expert Extraction]: {gold}
************
[Submission Extraction]: {answer}
************
[END DATA]

Compare the factual content of the extracted information with the expert-provided information. Ignore any minor differences in style, grammar, punctuation, or abbreviations.
The extracted information may either be a subset or superset of the expert extraction, or it may conflict with it. Determine which case applies. Assess the extraction by selecting one of the following options:
(Subset) The extracted information is a subset, i.e., contains part of the expert-provided information and is fully consistent with it.
(Superset) The extracted information is a superset, i.e., contains all and some extra information of the expert-provided information and is fully consistent with it.
(Correct) The extracted information contains all the same details as the expert-provided information.
(Disagreement) There is a disagreement, either full or partial, between the extracted information and the expert-provided information.

Format your answer as only a single word in parentheses, e.g., "(Superset)".
"""

PROMPT_EN = """
You are comparing the extracted information from a submission to the expert-provided information. Here is the data:
[BEGIN DATA]
************
[Expert Extraction]: {gold}
************
[Submission Extraction]: {answer}
************
[END DATA]

Compare the factual content of the extracted information with the expert-provided information. Ignore any minor differences in style, grammar, punctuation, or abbreviations.
The extracted information may either be a subset or superset of the expert extraction, or it may conflict with it. Determine which case applies. Assess the extraction by selecting one of the following options:
(Subset) The extracted information is a subset, i.e., contains part of the expert-provided information and is fully consistent with it.
(Superset) The extracted information is a superset, i.e., contains all and some extra information of the expert-provided information and is fully consistent with it.
(Correct) The extracted information contains all the same details as the expert-provided information.
(Disagreement) There is a disagreement, either full or partial, between the extracted information and the expert-provided information.

Format your answer as only a single word in parentheses, e.g., "(Superset)".
"""

PROMPTS = {"pl": PROMPT_PL, "en": PROMPT_EN}

INVALID_JUDGMENT = "(non-evaluable)"
CORRECT_JUDGEMENT = "(Correct)"
MISSING_ANSWER = "(empty-answer)"
allowed_answers = [
    "(Subset)",
    "(Superset)",
    CORRECT_JUDGEMENT,
    "(Disagreement)",
    INVALID_JUDGMENT,
    MISSING_ANSWER,
]


class StructuredLLMJudgeEvaluator(StructuredEvaluatorBase):
    """Evaluation pipeline which, whenver justified, queries LLM for a judgement.
    The pipeline works as follows for each extracted field:
        1. Comparing the gold and predicted answers:
            - If the predicted answer is the same as the gold, it is treated as correct.
            - Otherwise, the LLM is queried for a judgement.
        2. Aggregating the results and computing accuracy.
    Returns dictionary formatted as {"<field_name>": {"accuracy": <float_value>}}.
    """

    def __init__(self, client: ChatOpenAI, prompt: str):
        super().__init__(name="llm_as_judge", num_proc=1)
        self.client = client
        self.prompt = prompt

    def evaluate(
        self,
        preds: dict[str, list[str]],
        golds: dict[str, list[str]],
    ) -> dict[str, dict[str, float]]:
        """Evaluates information extraction by computing metrics per each field."""
        # todo: change to asyncio.gather when https://github.com/langchain-ai/langchain/issues/23904 is resolved
        return {
            field: self.evalute_answers(preds=preds[field], golds=golds[field], field_name=field)
            for field in tqdm(golds.keys(), desc="Fields")
        }

    def evalute_answers(
        self,
        preds: list[str],
        golds: list[str],
        field_name: str,
    ) -> dict[str, float]:
        assert len(golds) == len(preds)
        llm_assessments = [
            self.evaluate_single_answer(ans_pred, ans_gold)
            for ans_gold, ans_pred in tqdm(
                zip(golds, preds), desc=f"Evaluating {field_name}", total=len(golds), leave=False
            )
        ]
        results_summary = self.answers_to_metrics(llm_assessments)
        return results_summary

    def evaluate_single_answer(self, ans_pred: str, ans_gold: str) -> str:
        """Assesses single answer either by comparing raw string or using LLM-as-judge otherwise."""
        if ans_pred == EMPTY_ANSWER:
            return MISSING_ANSWER
        elif ans_pred == ans_gold:
            return CORRECT_JUDGEMENT
        else:
            response = self.client.invoke(self.prompt.format(gold=ans_gold, answer=ans_pred))

            if response is not None:
                return response.content
            else:
                logger.warning(f"Empty API response for: {(ans_gold, ans_pred)}")
                return INVALID_JUDGMENT

    @staticmethod
    def answers_to_metrics(llm_assessments: list[str]) -> dict[str, float]:
        results_summary = dict.fromkeys(allowed_answers, 0)
        for judgement in llm_assessments:
            try:
                results_summary[judgement] += 1
            except KeyError:
                results_summary[INVALID_JUDGMENT] += 1

        return {name: val / len(llm_assessments) for name, val in results_summary.items()}


# Example usage:
if __name__ == "__main__":
    import pandas as pd
    from dotenv import dotenv_values
    from langchain_openai import ChatOpenAI

    OPENAI_API_KEY = dotenv_values()["OPENAI_API_KEY"]

    client = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model_name="gpt-4o-mini",
        temperature=0.0,
    )

    evaluator = StructuredLLMJudgeEvaluator(client)

    preds = {
        "name": ["John", "Jane"],
        "surname": ["Doe", "Doe"],
        "age": ["29", "30"],
        "birth_date": ["1993-01-01", "1993-01-01"],
        "graduation_date": ["2015-06-01", "2015-06-06"],
    }
    golds = {
        "name": ["John", "John"],
        "surname": ["Doe", "Doe"],
        "age": ["29", "30"],
        "birth_date": ["1993-01-01", "1993-01-01"],
        "graduation_date": ["2015-06-01", "2015-06-01"],
    }

    metrics = evaluator.evaluate(preds=preds, golds=golds)
    print(pd.DataFrame.from_dict(metrics).transpose().to_markdown())
