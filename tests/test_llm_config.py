import pytest

from juddges.config import LLMConfig


def _llm_config(adapter_path=None, **kwargs) -> LLMConfig:
    return LLMConfig(
        name="m/x",
        tokenizer_name="m/x",
        adapter_path=adapter_path,
        padding="longest",
        batch_size=1,
        use_4bit=True,
        **kwargs,
    )


def test_bnb_quant_type_is_opt_in():
    assert _llm_config().bnb_4bit_kwargs == {"load_in_4bit": True}  # old behavior
    assert _llm_config(bnb_4bit_quant_type="nf4").bnb_4bit_kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
    }


def test_24gb_sft_profiles_enable_nf4():
    import yaml

    for model in ["bielik_11b_v23_instruct", "llama_3.1_8b_instruct"]:
        profile = yaml.safe_load(open(f"configs/profile/{model}_sft_24gb.yaml"))
        assert profile["llm"]["bnb_4bit_quant_type"] == "nf4"


def test_adapter_path_prefers_top_level_adapter(tmp_path):
    (tmp_path / "adapter_model.safetensors").touch()
    (tmp_path / "checkpoint-10").mkdir()

    assert _llm_config(tmp_path).adapter_path_or_first_ckpt_path == tmp_path


def test_adapter_path_takes_last_checkpoint_numerically(tmp_path):
    for step in [9, 10, 100, 2]:
        (tmp_path / f"checkpoint-{step}").mkdir()

    path = _llm_config(tmp_path).adapter_path_or_first_ckpt_path

    assert path == tmp_path / "checkpoint-100"  # lexicographic order would give checkpoint-9


def test_adapter_path_fails_without_adapter(tmp_path):
    (tmp_path / "checkpoint-final-notes").mkdir()

    with pytest.raises(ValueError, match="No adapter_model.safetensors"):
        _llm_config(tmp_path).adapter_path_or_first_ckpt_path
