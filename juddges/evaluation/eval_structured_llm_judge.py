import yaml
from openai import OpenAI
from tqdm.auto import tqdm

from juddges.evaluation.eval_structured import StructuredEvaluatorBase
from juddges.evaluation.parse import EMPTY_ANSWER

PROMPT = """
You are professional assistant comparing a submitted Answer to a Reference written in Polish.
Assess correctness of the Answer with one of the following options:
- (Subset) The submitted Answer is a subset, i.e., contains part of information of the Reference and is fully consistent with it.
- (Superset) The submitted Answer is a superset, i.e., contains all and some extra information of the Reference and is fully consistent with it.
- (Correct) The submitted Answer contains all the same details as the Reference.
- (Disagreement) There is a disagreement, either full or partial, between the submitted Answer and the Reference.

[BEGIN DATA]
************
[Reference]: {gold}
************
[Answer]: {answer}
************
[END DATA]

Format your judgment as only a single word in parentheses, e.g., "(Superset)"
"""

INVALID_JUDGMENT = "(non-evaluable)"
allowed_answers = [
    "(Subset)",
    "(Superset)",
    "(Correct)",
    "(Disagreement)",
    INVALID_JUDGMENT,
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
        assert len(golds) == len(preds)
        results = []
        num_llm_evals = 0
        iter_golds_preds = enumerate(zip(golds, preds))
        with tqdm(iter_golds_preds, total=len(golds), leave=False, desc="Evaluating") as pbar:
            for i, (g, p) in pbar:
                if p == EMPTY_ANSWER:
                    results.append("(incorrect)")
                elif p == g:
                    results.append(1)
                else:
                    num_llm_evals += 1
                    # TODO: Further can be improved with asynchronous requests
                    response = self.oai_client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": PROMPT.format(gold=g, answer=p)}],
                        temperature=0.0,
                        n=1,
                    )

                    response_msg = response.choices[0].message.content
                    try:
                        results.append(response_msg)
                    except KeyError:
                        print(f"Unexpected response: {response_msg}")
                        results.append(-1)
                pbar.set_postfix({"llm_calls": f"{num_llm_evals}/{i + 1}"})

        results_summary = dict.fromkeys(allowed_answers, 0)
        for judgement in results:
            try:
                results_summary[judgement] += 1
            except KeyError:
                results_summary[INVALID_JUDGMENT] += 1

        results_summary = {name: val / len(results) for name, val in results_summary.items()}

        return results_summary


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
