"""
QA - Question Answer
COT - Chain of Thought
FEW - Few-Shot
ZERO - Zero-Shot
CHN - Prompt Chaining
"""


GEN_QA_COT_PROMPT = """\
You are a question-answer generator. Your goal is to generate question-answer pairs given the Context `document`.
Do not tranlate the Context, generate questions and answers in original language - {language}.

Context:
```
{context}
```

Step 1: Identify spans that are likely to be answers to questions, identify as many as possible.
Step 2: For each identified span, generate a question.
Step 3: Respond to the question in only a few tokens concisely.

Ensure that you distinctly label and delineate Steps 1, 2 and 3.

{format_instructions}

Output:
```{format_md_ext}
"""

GEN_QA_BASELINE_PROMPT = """\
You are a question-answer generator. Your goal is to generate question-answer pairs given the Context `document`.
Do not tranlate the Context, generate questions and answers in original language - {language}.

Context:
```
{context}
```

{format_instructions}

Output:
```{format_md_ext}
"""
