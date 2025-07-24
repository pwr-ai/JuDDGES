# Iterative Refinement of Prompt and Schema for Automated Information Extraction from Legal Judgments with LLMs

![Version](https://img.shields.io/badge/version-20.06.2025-blue)

This guide outlines a practical, iterative process for refining prompts and schema descriptions to maximize the quality of automated information extraction from legal judgments using large language models (LLMs).

## 1. Preparation of Prompt and Schema Description

- **Draft the Prompt:** Prepare a clear, instructional prompt for the LLM. The prompt should explain the extraction task, specify language requirements, and provide formatting instructions. See [`info_extraction_annotated_json.yaml`](../configs/prompt/info_extraction_json.yaml) for an example - this prompt were utilized in our paper experiment and should work quite well in most cases.
- **Design the Schema:** Create a detailed, unambiguous schema description for the information to be extracted. Each field should have a clear type, eventually allowed values (enums), and precise descriptions to minimize LLM misinterpretation. See [`swiss_franc_loans_refined.yaml`](../configs/ie_schema/swiss_franc_loans_refined.yaml) for an example.
- Finally, prompt template is filled with the schema and the judgment text.

## 2. Model Selection and Inference

### 2.1 Choose right LLM

Select a language model with a context window large enough to accommodate lengthy legal texts and the full prompt (for Polish judgments, up to 60k tokens was required).

### 2.2 Running Inference

For maximal efficiency, use the `scripts/sft/predict_vllm.py` script, which is optimized for high-throughput, batched inference with large language models (LLMs) using the vLLM engine.

  **Example command:**

  ```bash
  CUDA_VISIBLE_DEVICES=<your_gpu_id_if_necessary> python scripts/sft/predict_vllm.py \
      llm=llama_3.3_70b_instruct \
      llm.batch_size=1 \
      max_model_len=65536 \
      dataset=pl_court_swiss_franc_loans \
      generate_kwargs.max_tokens=1000 \
      prompt=info_extraction_annotated_json_refined \
      ie_schema=swiss_franc_loans_refined \
      split=annotated \
      random_seed=42
  ```

- **llm**: Name of the LLM config (one from [`configs/llm/`](../configs/llm/))
- **llm.batch_size**: Number of samples per batch (increase for faster inference if memory allows)
- **max_model_len**: Maximum context length (`total = input + generated` tokens)
- **dataset**: Dataset config name (one from [`configs/dataset/`](../configs/dataset/))
- **generate_kwargs.max_tokens**: Maximum new tokens to generate per sample
- **prompt**: Prompt template config
- **ie_schema**: Schema config for extraction
- **split**: Dataset split to use
- **random_seed**: For reproducibility

### 2.3 How it works

- **Batch Processing:** Adjust `llm.batch_size` according to your available GPU memory for optimal throughput. Larger batch sizes speed up processing but require more memory.
- **Hydra-Based Configuration:** The script uses [Hydra](https://hydra.cc/) for flexible configuration management. You can compose and override config files and parameters directly from the command line. The full, resolved config is printed at the start of each run for transparency.
- **No DVC Required:** Since it is instruction for fast iterating over prompt and schema, DVC is not needed to run this script—configuration and data paths are managed via Hydra.
- **Adding New Models or Datasets:**
  - To add a new LLM, create a new config file under `configs/llm/` (see existing configs).
  - To add a new dataset, add a new config file under `configs/dataset/` (see existing configs).
- **Schema and Prompt Configs Used in the Paper:**
  - For the English dataset, the schema and prompt configs used were:
    - Schema: `configs/ie_schema/en_appealcourt.yaml`
    - Prompt: `configs/prompt/info_extraction_json.yaml`
- **Resource Requirements Example:**
  - For the Polish dataset (context size ≈ 40k tokens, batch size = 8), running [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) required ~90GB of GPU memory (tested on a single NVIDIA H100).
  - **Note:** vLLM pre-allocates the key-value (KV) cache for the configured context size. If insufficient memory is available, the script will fail immediately at initialization.

### 2.4 Tips

- Always check the printed config at the start of the run to confirm all settings.
- If you encounter out-of-memory errors, reduce `llm.batch_size` or `max_model_len`.
- For new models or datasets, copy and adapt existing config files as templates. For example, if you want to add a new LLM, you can copy the config file from `configs/llm/` and modify it to your needs.
- Hydra allows config values to be overridden from multiple sources (e.g., command line, environment variables, defaults in config files). **Always carefully review the full config printed at the start of the script to ensure all parameters are set as intended.** If results are surprising, double-check for any hydra overrides.

## 3. Result Analysis

### 3.1 Qualitative Analysis

Once you have obtained the results, you can analyze them to verify whether the LLM output is valid JSON and conforms to the expected schema. The recommended approach is to use a notebook, such as [dev_notebooks/analyse_llm_as_judge.ipynb](../dev_notebooks/analyse_llm_as_judge.ipynb). **Note:** This notebook is a working version intended as a starting point—it should be adjusted and modified to fit your particular needs and analysis goals. During your analysis, you can check the following aspects:

- **Format Failures:** Check if the LLM output is valid JSON and matches the expected schema. Track and analyze any parsing errors.
- **Context Length Issues:** Monitor for incomplete generations due to context or output token limits. If the output is truncated, consider using a model with a larger context window or further truncating the input context.
- **Response Quality:** Manually review a sample of outputs, comparing extracted fields to gold annotations. Look for:
  - Misinterpretations due to vague schema descriptions
  - Prompt instructions not being followed
  - Systematic errors or omissions

### 3.2 Quantitative Analysis

- Quantitative analysis depends on the concrete schema and we have not fully developed it for the Polish dataset yet.
- Up until now, we have been using llm-as-judge for evaluation, but we need to further refine the prompt as we have spotted some minor flaws in the current prompts.
- At this stage, qualitative analysis is considered way more important than quantitative analysis.

However, ultimately, quantitative analysis is needed to track the progress and compare the effectiveness of different prompt/schema/model configurations over time.

## 4. Refinement

- **Prompt/Schema Updates:** Based on qualitative and quantitative findings, iteratively refine the prompt and schema. Make descriptions more specific, clarify ambiguous fields, or adjust instructions as needed.
- **Model Scaling:** If instructions and schema are sufficiently clear but the model still underperforms, consider switching to a larger or more capable LLM.

## 5. Repeat

- **Iterate:** Repeat steps 2–5 until extraction quality is satisfactory for your use case.

---

**Tip:** Document each iteration's changes and results to build a knowledge base for future schema/prompt engineering efforts.

## Future Work

We are planning the following directions for further improvement and experimentation:

- [ ] Further refine schema descriptions and remove ambiguous fields (make schema more precise and unambiguous, eliminate unclear fields, we are at 60% done with this)
- [ ] Test larger LLMs (experiment with even larger and more capable language models, as current models may have reached their ceiling, we started experimenting with 32B and 70B models)
- [ ] Test constrained decoding (enforce output validity and improve extraction accuracy, especially for Polish names and formatting)
