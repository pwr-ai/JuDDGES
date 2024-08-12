import yaml
from openai import OpenAI
from tqdm.auto import tqdm

from juddges.evaluation.eval_structured import StructuredEvaluatorBase
from juddges.evaluation.parse import EMPTY_ANSWER

# TODO: might be a configurable prompt in future
# Credit: https://github.com/openai/evals
PROMPT = """
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

    def __init__(self, oai_client: OpenAI, model_name: str):
        super().__init__(name="llm_as_judge")
        self.oai_client = oai_client
        self.model_name = model_name

    def evaluate(
        self,
        preds: dict[str, list[str]],
        golds: dict[str, list[str]],
    ) -> dict[str, dict[str, float]]:
        """Evaluates information extraction by computing metrics per each field."""
        return {
            field: self.compute_metrics(preds[field], golds[field])
            for field in tqdm(golds.keys(), desc="Fields")
        }

    def compute_metrics(self, preds: list[str], golds: list[str]) -> dict[str, float]:
        """Assesses single field prediction either by comparing raw string or using LLM-as-judge otherwise."""
        assert len(golds) == len(preds)
        llm_assessments = []
        num_llm_evals = 0
        enum_golds_preds = enumerate(zip(golds, preds))
        for i, (ans_gold, ans_pred) in (
            pbar := tqdm(enum_golds_preds, total=len(golds), leave=False, desc="Evaluating")
        ):
            if ans_pred == EMPTY_ANSWER:
                llm_assessments.append(MISSING_ANSWER)
            elif ans_pred == ans_gold:
                llm_assessments.append(CORRECT_JUDGEMENT)
            else:
                num_llm_evals += 1
                # TODO: Further can be improved with asynchronous requests
                response = self.oai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": PROMPT.format(gold=ans_gold, answer=ans_pred)}
                    ],
                    temperature=0.0,
                    n=1,
                )

                response_msg = response.choices[0].message.content
                llm_assessments.append(response_msg)

            pbar.set_postfix({"llm_calls": f"{num_llm_evals}/{i + 1}"})

        results_summary = self.answers_to_metrics(llm_assessments)

        return results_summary

    @staticmethod
    def answers_to_metrics(llm_assessments: list[str]) -> dict[str, float]:
        results_summary = dict.fromkeys(allowed_answers, 0)
        for judgement in llm_assessments:
            try:
                results_summary[judgement] += 1
            except KeyError:
                results_summary[INVALID_JUDGMENT] += 1

        return {name: val / len(llm_assessments) for name, val in results_summary.items()}


if __name__ == "__main__":
    from dotenv import load_dotenv
    from langsmith.wrappers import wrap_openai
    from openai import OpenAI

    load_dotenv()

    oai_client = wrap_openai(OpenAI())
    model = "gpt-3.5-turbo"

    evaluator = StructuredLLMJudgeEvaluator(oai_client, model)

    results = [
        {
            "gold": yaml.dump(
                {
                    "name": "John",
                    "surname": "Doe",
                    "age": 29,
                    "birth_date": "1993-01-01",
                    "graduation_date": "2015-06-01",
                }
            ),
            "answer": yaml.dump(
                {
                    "name": "John",
                    "surname": "Does",
                    "age": 30,
                    "birth_date": "1993-01-01",
                    "graduation_date": "2015-06-06",
                }
            ),
        }
    ]

    metrics = evaluator.evaluate(results)
    print(metrics)
