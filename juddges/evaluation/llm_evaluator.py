import yaml
from openai import OpenAI
from tqdm.auto import tqdm

from juddges.evaluation.parse import EMPTY_ANSWER, parse_results

PROMPT = """
You are comparing a submitted Asnwer to a Reference answer. Answer contains information extracted in Polish.
[BEGIN DATA]
************
[Reference]: {gold}
************
[Answer]: {answer}
************
[END DATA]

Compare the submitted Answer with the Reference answer, and state its correctness.
Ignore any minor differences in style, grammar, or punctuation, accept abbreviations.
If Answer contains any extra information that is not present in the Reference, mark it as incorrect.
If Answer is partial, misses information present in the Reference, mark it as incorrect.
Conversely, when comparing dates or numbers, be very precise.
Format your answer as only a single word in parentheses "(correct)" or "(incorrect)".
"""

answer_2_num = {
    "(correct)": 1,
    "(incorrect)": 0,
}


class StructuredLLMJudgeEvaluator:
    """Evaluation pipeline which, whenver justified, queries LLM for a judgement.
    The pipeline works as follows for each extracted field:
        1. Parsing yaml responses if possible, if not, treating them as incorrect.
        2. Comparing the gold and predicted answers:
            - If the predicted answer is the same as the gold, it is treated as correct.
            - Otherwise, the LLM is queried for a judgement.
        3. Aggregating the results and computing accuracy.
    """

    def __init__(self, oai_client: OpenAI, model_name: str):
        self.oai_client = oai_client
        self.model_name = model_name

    def evaluate(self, results: list[dict[str, str]]) -> dict[str, dict[str, float]]:
        """Evaluates information extraction by computing metrics per each field."""
        res_gold, res_pred = parse_results(results)
        return {
            field: self.compute_metrics(res_gold[field], res_pred[field])
            for field in tqdm(res_gold, desc="Fields")
        }

    def compute_metrics(self, golds: list[str], preds: list[str]) -> dict[str, float]:
        assert len(golds) == len(preds)
        results = []
        num_llm_evals = 0
        iterable = enumerate(zip(golds, preds))
        with tqdm(iterable, total=len(golds), leave=False, desc="Evaluating") as pbar:
            for i, (g, p) in pbar:
                if p == EMPTY_ANSWER:
                    results.append("(incorrect)")
                elif p == g:
                    results.append(1)
                else:
                    num_llm_evals += 1
                    response = self.oai_client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": PROMPT.format(gold=g, answer=p)}],
                        temperature=0.0,
                        n=1,
                    )

                    response_msg = response.choices[0].message.content
                    try:
                        results.append(answer_2_num[response_msg])
                    except KeyError:
                        print(f"Unexpected response: {response_msg}")
                        results.append(-1)
                pbar.set_postfix({"llm_calls": f"{num_llm_evals}/{i+1}"})

        return {"accuracy": sum(results) / len(results)}


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
