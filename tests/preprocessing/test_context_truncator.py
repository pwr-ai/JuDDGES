import unittest

from transformers import AutoTokenizer

from juddges.preprocessing.context_truncator import ContextTruncator, ContextTruncatorTiktoken

PROMPT_TEMPLATE = "How many helicopters can a human eat in one sitting? {context}"


class TestContextTruncator(unittest.TestCase):
    def test_mistral(self):
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        self._check(model_id, 120)

    def test_llama3(self):
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        self._check(model_id, 120)

    def test_mistral_warn(self):
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        max_length = 3
        context = " ".join([str(i) for i in range(max_length * 2)])
        output = "None."
        with self.assertWarns(Warning):
            ContextTruncator(
                prompt_without_context=PROMPT_TEMPLATE.format(context=""),
                tokenizer=tokenizer,
                max_length=max_length,
            )(context, output)

    def _check(self, model_id: str, max_length: int):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        context = " ".join([str(i) for i in range(max_length * 2)])
        output = "None."
        first_message = PROMPT_TEMPLATE.format(context=context)

        messages = [
            {"role": "user", "content": first_message},
            {"role": "assistant", "content": output},
        ]

        original_tokenized = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True
        )
        original_length = len(original_tokenized.data["input_ids"])

        truncated_context = ContextTruncator(
            prompt_without_context=PROMPT_TEMPLATE.format(context=""),
            tokenizer=tokenizer,
            max_length=max_length,
        )(context, output)["context"]

        self.assertGreaterEqual(len(context), len(truncated_context))

        first_message = PROMPT_TEMPLATE.format(context=truncated_context)
        messages = [
            {"role": "user", "content": first_message},
            {"role": "assistant", "content": output},
        ]
        truncated_tokenized = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_length=True
        )
        truncated_length = len(truncated_tokenized.data["input_ids"])

        self.assertLess(truncated_length, original_length)
        self.assertLessEqual(truncated_length, max_length)

        self.assertListEqual(
            original_tokenized["input_ids"][-5:], truncated_tokenized["input_ids"][-5:]
        )


class TestContextTruncatorTiktoken(unittest.TestCase):
    def test_gpt_4o(self):
        max_length = 120
        context = " ".join([str(i) for i in range(max_length * 2)])
        output = "None."

        for model in ["gpt-4o", "gpt-4o-mini"]:
            with self.subTest(model=model):
                truncator = ContextTruncatorTiktoken(
                    prompt_without_context=PROMPT_TEMPLATE.format(context=""),
                    model=model,
                    max_length=max_length,
                )
                truncated_context = truncator(context, output)["context"]

                self.assertLess(len(truncated_context), len(context))
                self.assertGreater(len(truncated_context), 0)
                self.assertIn(truncated_context, context)

    def test_truncator_raises_on_too_long_prompt_and_output(self):
        max_length = 10
        context = " ".join([str(i) for i in range(max_length * 2)])
        output = "None."

        for model in ["gpt-4o", "gpt-4o-mini"]:
            with self.subTest(model=model):
                truncator = ContextTruncatorTiktoken(
                    prompt_without_context=PROMPT_TEMPLATE.format(context=""),
                    model=model,
                    max_length=max_length,
                )
                self.assertRaises(ValueError, lambda: truncator(context, output))
