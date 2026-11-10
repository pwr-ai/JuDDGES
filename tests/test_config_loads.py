import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig

from juddges.config import FineTuningConfig, PredictInfoExtractionConfig
from juddges.utils.config import resolve_config


@pytest.mark.parametrize(
    "llm",
    [
        "mistral_nemo_instruct_2407",
        "llama_3.1_8b_instruct",
        "llama_3.2_3b_instruct",
        "pllum_12b_instruct",
        "bielik_11b_v23_instruct",
        "phi_4",
        "phi_4_mini_instruct",
    ],
)
@pytest.mark.parametrize(
    "dataset",
    [
        "pl_court_swiss_franc_loans",
    ],
)
@pytest.mark.parametrize(
    "prompt",
    [
        "info_extraction_json",
    ],
)
@pytest.mark.parametrize(
    "ie_schema",
    [
        "swiss_franc_loans",
    ],
)
def test_load_config_peft_fine_tuning(llm: str, dataset: str, prompt: str, ie_schema: str):
    with initialize(version_base=None, config_path="../configs", job_name="test"):
        raw_cfg = compose(
            config_name="peft_fine_tuning",
            overrides=[
                f"llm={llm}",
                f"dataset={dataset}",
                f"prompt={prompt}",
                f"ie_schema={ie_schema}",
                # --- hydra needs it ---
                "hydra.run.dir=.",
                "hydra.job.num=1",
            ],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(raw_cfg)
    cfg_dict = resolve_config(raw_cfg)
    del cfg_dict["hydra"]
    config = FineTuningConfig(**cfg_dict)

    assert config.use_peft


@pytest.mark.parametrize(
    "llm",
    [
        "mistral_nemo_instruct_2407",
        "llama_3.1_8b_instruct",
        "llama_3.2_3b_instruct",
        "pllum_12b_instruct",
        "phi_4",
        "phi_4_mini_instruct",
    ],
)
@pytest.mark.parametrize(
    "dataset",
    [
        "pl_court_swiss_franc_loans",
    ],
)
@pytest.mark.parametrize(
    "prompt",
    [
        "info_extraction_json",
        "info_extraction_annotated_json",
    ],
)
@pytest.mark.parametrize(
    "ie_schema",
    [
        "swiss_franc_loans",
    ],
)
def test_load_config_predict(llm: str, dataset: str, prompt: str, ie_schema: str):
    with initialize(version_base=None, config_path="../configs", job_name="test"):
        raw_cfg = compose(
            config_name="predict_vllm",
            overrides=[
                f"llm={llm}",
                f"dataset={dataset}",
                f"prompt={prompt}",
                f"ie_schema={ie_schema}",
                f"random_seed={42}",
                # --- hydra needs it ---
                "hydra.run.dir=.",
                "hydra.job.num=1",
            ],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(raw_cfg)
    cfg_dict = resolve_config(raw_cfg)
    del cfg_dict["hydra"]
    PredictInfoExtractionConfig(**cfg_dict)


@pytest.mark.parametrize("llm", ["bielik_11b_v23_instruct", "llama_3.1_8b_instruct"])
def test_load_config_sft_24gb_profile(llm: str):
    with initialize(version_base=None, config_path="../configs", job_name="test"):
        raw_cfg = compose(
            config_name="peft_fine_tuning",
            overrides=[
                f"llm={llm}",
                "dataset=pl_court_swiss_franc_loans",
                "prompt=info_extraction_annotated_json",
                "ie_schema=swiss_franc_loans",
                f"+profile={llm}_sft_24gb",
                "training_args.seed=997",
                "hydra.run.dir=.",
                "hydra.job.num=1",
            ],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(raw_cfg)
    cfg_dict = resolve_config(raw_cfg)
    del cfg_dict["hydra"]
    config = FineTuningConfig(**cfg_dict)

    assert config.llm.use_4bit
    assert config.peft_args["r"] == 16 and config.peft_args["lora_alpha"] == 32
    assert config.training_args["packing"] is False and config.truncate_context
    assert config.training_args["seed"] == 997
    assert config.output_dir.name == "seed_997"
    assert "seed_997" in config.run_name


@pytest.mark.parametrize("llm", ["bielik_11b_v23_instruct", "llama_3.1_8b_instruct"])
@pytest.mark.parametrize("profile", [True, False])
def test_load_config_predict_24gb_profile(llm: str, profile: bool):
    overrides = [f"+profile={llm}_predict_24gb"] if profile else []
    with initialize(version_base=None, config_path="../configs", job_name="test"):
        raw_cfg = compose(
            config_name="predict_vllm",
            overrides=[
                f"llm={llm}",
                "dataset=pl_court_swiss_franc_loans",
                "prompt=info_extraction_annotated_json",
                "ie_schema=swiss_franc_loans",
                "random_seed=42",
                "split=annotated",
                "guided_json=true",
                *overrides,
                "hydra.run.dir=.",
                "hydra.job.num=1",
            ],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(raw_cfg)
    cfg_dict = resolve_config(raw_cfg)
    del cfg_dict["hydra"]
    config = PredictInfoExtractionConfig(**cfg_dict)

    assert config.guided_json
    # without the profile the old behaviour is kept
    assert (config.quantization == "bitsandbytes") is profile
    assert (config.max_model_len == 8192) is profile
