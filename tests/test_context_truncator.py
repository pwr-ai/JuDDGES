import unittest

from transformers import AutoTokenizer

from juddges.preprocessing.context_truncator import ContextTruncator, ContextTruncatorTiktoken


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
        prompt, context, output = (
            "How many helicopters can a human eat in one sitting? {context}",
            " ".join([str(i) for i in range(max_length * 2)]),
            "None.",
        )
        with self.assertWarns(Warning):
            ContextTruncator(tokenizer, max_length)(prompt, context, output)

    def _check(self, model_id: str, max_length: int):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        prompt, context, output = (
            "How many helicopters can a human eat in one sitting? {context}",
            " ".join([str(i) for i in range(max_length * 2)]),
            "None.",
        )
        first_message = prompt.format(context=context)

        messages = [
            {"role": "user", "content": first_message},
            {"role": "assistant", "content": output},
        ]

        original_tokenized = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True
        )
        original_length = len(original_tokenized.data["input_ids"])

        truncated_context = ContextTruncator(tokenizer, max_length)(prompt, context, output)

        self.assertGreaterEqual(len(context), len(truncated_context))

        first_message = prompt.format(context=truncated_context)
        messages = [
            {"role": "user", "content": first_message},
            {"role": "assistant", "content": output},
        ]
        truncated_tokenized = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_length=True
        )
        truncated_length = len(truncated_tokenized.data["input_ids"])

        self.assertLess(truncated_length, original_length)
        self.assertLess(truncated_length, max_length)

        self.assertListEqual(
            original_tokenized["input_ids"][-5:], truncated_tokenized["input_ids"][-5:]
        )


class TestContextTruncatorTiktoken(unittest.TestCase):
    def test_gpt_4o(self):
        max_length = 120
        prompt, context, output = (
            "How many helicopters can a human eat in one sitting? {context}",
            " ".join([str(i) for i in range(max_length * 2)]),
            "None.",
        )

        for model in ["gpt-4o", "gpt-4o-mini"]:
            with self.subTest(model=model):
                truncator = ContextTruncatorTiktoken(model=model, max_length=max_length)
                truncated_context = truncator(prompt, context, output)

                self.assertLess(len(truncated_context), len(context))
                self.assertGreater(len(truncated_context), 0)
                self.assertIn(truncated_context, context)

    def test_truncator_raises_on_too_long_prompt_and_output(self):
        max_length = 10
        prompt, context, output = (
            "How many helicopters can a human eat in one sitting? {context}",
            " ".join([str(i) for i in range(max_length * 2)]),
            "None.",
        )

        for model in ["gpt-4o", "gpt-4o-mini"]:
            with self.subTest(model=model):
                truncator = ContextTruncatorTiktoken(model=model, max_length=max_length)
                self.assertRaises(ValueError, lambda: truncator(prompt, context, output))
