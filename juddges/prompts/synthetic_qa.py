"""
QA - Question Answer
COT - Chain of Thought
FEW - Few-Shot
ZERO - Zero-Shot
CHN - Prompt Chaining
"""

QA_GENERATOR_ROLE_INCRUCT = """\
You are a question-answer generator.
Your goal is to generate question-answer pairs given the Context and Metadata of the document.
Do not tranlate the Context, generate questions and answers in original language - {lang_code} (ISO 639-1).
Do not provide information that are not explicitly mentioned in Context document.
Ensure that the answers are concise and relevant to the questions.
"""

GEN_QA_COT_PROMPT = (
    QA_GENERATOR_ROLE_INCRUCT
    + """\

Metadata below follows the schema:
```
{context_metadata_schema}
```

Metadata:
```
{context_metadata}
```

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
)

GEN_QA_BASELINE_PROMPT = (
    QA_GENERATOR_ROLE_INCRUCT
    + """\

Metadata below follows the schema:
```
{context_metadata_schema}
```

Metadata:
```
{context_metadata}
```

Context:
```
{context}
```

{format_instructions}

Output:
```{format_md_ext}
"""
)


GEN_QA_LLM_AS_JUDGE_PROMPT = """\
You are a question-answer generator's supervisor.
Your goal is to evaluate the quality of the generated question-answer pairs.

Generated Data Schema:
```
{generated_data_schema}
```

Generated Data:
```
{generated_data_dict}
```

{format_instructions}

Output:
```{format_md_ext}
"""
