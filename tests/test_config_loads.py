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
            config_name="predict",
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
