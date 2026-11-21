import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from SneppX_ALG.interface_bindings.model_zoo import (
    ModelConfig, LlamaConfig, MistralConfig, Qwen2Config, DeepSeekV2Config,
    get_model_config, get_model_config_obj, from_pretrained, build_model_from_config,
    ModelHub, _HAS_C_BACKEND
)

tests_passed = 0
tests_total = 0


def test(name):
    global tests_total
    tests_total += 1
    print(f"test_{name}... ", end="", flush=True)


def pass_():
    global tests_passed
    tests_passed += 1
    print("PASS")


# ---------------------------------------------------------------------------
# 1. ModelConfig → JSON → C config → ModelConfig roundtrip
# ---------------------------------------------------------------------------

def test_python_to_c_roundtrip():
    """Create Python ModelConfig, convert to C via registry, verify fields survive."""
    mc = ModelConfig(
        name="roundtrip-test",
        version="1.0",
        architecture="transformer",
        model_type="causal-lm",
        framework="sneppx",
        num_layers=24,
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=4,
        vocab_size=32000,
        max_seq_len=8192,
        hidden_act="silu",
        gated_ffn=True,
        use_rms_norm=True,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        use_flash_attention=True,
        learning_rate=3e-4,
    )

    py_json = mc.to_json()
    assert "1024" in py_json
    assert "silu" in py_json

    if _HAS_C_BACKEND:
        c_cfg = get_model_config("llama2", "7B")
        assert c_cfg is not None
        assert c_cfg.get("hidden_size") == 4096


def test_all_presets_accessible():
    """Verify all model presets can be loaded and produce valid configs."""
    presets = [
        ("llama2", "7B", 4096, 32, 32),
        ("llama2", "13B", 5120, 40, 40),
        ("llama3", "8B", 4096, 32, 8),
        ("mistral", "7B", 4096, 32, 8),
        ("qwen2", "7B", 3584, 28, 4),
    ]
    for family, size, hs, layers, kv in presets:
        cfg = get_model_config(family, size)
        assert cfg["hidden_size"] == hs, f"{family}/{size}: expected hs={hs}, got {cfg['hidden_size']}"
        assert cfg["num_hidden_layers"] == layers, f"{family}/{size}: expected layers={layers}, got {cfg['num_hidden_layers']}"


def test_model_config_dataclass_conversion():
    """Test LlamaConfig → ModelConfig → LlamaConfig roundtrip."""
    orig = LlamaConfig()
    mc = ModelConfig(
        name="test",
        version="1.0",
        architecture="transformer",
        model_type="causal-lm",
        framework="sneppx",
        num_layers=orig.num_hidden_layers,
        hidden_size=orig.hidden_size,
        num_attention_heads=orig.num_attention_heads,
        num_key_value_heads=orig.num_key_value_heads,
        vocab_size=orig.vocab_size,
        max_seq_len=orig.max_position_embeddings,
        hidden_act="silu",
        gated_ffn=True,
        use_rms_norm=True,
        rms_norm_eps=orig.rms_norm_eps,
        rope_theta=orig.rope_theta,
        use_flash_attention=True,
        learning_rate=2e-4,
    )
    restored = LlamaConfig.from_model_config(mc)
    assert restored.hidden_size == orig.hidden_size
    assert restored.num_hidden_layers == orig.num_hidden_layers
    assert restored.num_attention_heads == orig.num_attention_heads


# ---------------------------------------------------------------------------
# 2. ModelHub from_pretrained / save_pretrained full pipeline
# ---------------------------------------------------------------------------

def test_hub_save_load_pipeline():
    """Create a model via ModelHub, save to temp dir, verify files."""
    hub = ModelHub()
    model = hub.from_pretrained("llama-2-7b")
    assert model["model_id"] == "llama-2-7b"
    assert model["family"] == "llama2"
    assert model["config"]["hidden_size"] == 4096

    with tempfile.TemporaryDirectory() as tmpdir:
        hub.save_pretrained(model, tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "config.json"))
        assert os.path.exists(os.path.join(tmpdir, "README.md"))

        with open(os.path.join(tmpdir, "config.json")) as f:
            saved = json.load(f)
        assert saved["hidden_size"] == 4096


def test_hub_cache_dir_creation():
    """Verify ModelHub creates its cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = os.path.join(tmpdir, "sneppx-cache")
        hub = ModelHub(cache_dir=cache)
        assert os.path.exists(cache)


def test_hub_model_card_generation():
    """Verify ModelHub generates a model card with correct metadata."""
    hub = ModelHub()
    model = hub.from_pretrained("mistral-7b")
    card = hub._generate_model_card(model)
    assert "mistral" in card
    assert "7B" in card
    assert "SneppX" in card


# ---------------------------------------------------------------------------
# 3. from_pretrained standalone function
# ---------------------------------------------------------------------------

def test_from_pretrained_standalone():
    """Verify the standalone from_pretrained function works."""
    result = from_pretrained("qwen2-7b", verbose=False)
    assert result["model_id"] == "qwen2-7b"
    assert result["config"]["hidden_size"] in (3584, 4096)


def test_from_pretrained_unknown_model():
    """Verify from_pretrained raises on unknown model_id."""
    try:
        from_pretrained("nonexistent-model-9999b")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 4. FrameworkConfig → ModelConfig conversions (all 4 framework types)
# ---------------------------------------------------------------------------

def test_mistral_config_conversion():
    """Test MistralConfig ↔ ModelConfig conversion."""
    mc = ModelConfig(
        hidden_size=5120, intermediate_size=13824,
        num_layers=40, num_attention_heads=40,
        num_key_value_heads=8, vocab_size=32000,
        max_seq_len=4096, rope_theta=10000.0,
        sliding_window=4096, learning_rate=2e-4,
        name="mistral-test", version="1.0",
        architecture="transformer", model_type="causal-lm",
        framework="sneppx", hidden_act="silu",
        gated_ffn=True, use_rms_norm=True,
    )
    mist = MistralConfig.from_model_config(mc)
    assert mist.hidden_size == 5120
    assert mist.num_key_value_heads == 8
    assert mist.sliding_window == 4096


def test_qwen2_config_conversion():
    """Test Qwen2Config ↔ ModelConfig conversion."""
    mc = ModelConfig(
        hidden_size=3584, intermediate_size=18944,
        num_layers=28, num_attention_heads=28,
        num_key_value_heads=4, vocab_size=152064,
        max_seq_len=8192, rope_theta=1000000.0,
        learning_rate=2e-4, name="qwen-test", version="1.0",
        architecture="transformer", model_type="causal-lm",
        framework="sneppx", hidden_act="silu",
        gated_ffn=True, use_rms_norm=True,
    )
    qwen = Qwen2Config.from_model_config(mc)
    assert qwen.hidden_size == 3584
    assert qwen.num_hidden_layers == 28


def test_deepseek_config_conversion():
    """Test DeepSeekV2Config ↔ ModelConfig conversion."""
    mc = ModelConfig(
        hidden_size=4096, intermediate_size=11008,
        num_layers=32, num_attention_heads=32,
        num_key_value_heads=32, vocab_size=102400,
        max_seq_len=4096, rope_theta=10000.0,
        learning_rate=2e-4, name="deepseek-test", version="1.0",
        architecture="transformer", model_type="causal-lm",
        framework="sneppx", hidden_act="silu",
        gated_ffn=True, use_rms_norm=True,
    )
    ds = DeepSeekV2Config.from_model_config(mc)
    assert ds.hidden_size == 4096
    assert ds.num_hidden_layers == 32


# ---------------------------------------------------------------------------
# 5. build_model_from_config
# ---------------------------------------------------------------------------

def test_build_model_from_config():
    """Verify build_model_from_config returns expected structure."""
    config = get_model_config("llama3", "8B")
    model = build_model_from_config(config)
    assert "config" in model
    assert "total_params" in model
    assert model["hidden_size"] == 4096
    assert model["layers"] == 32


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Running Phase 1.8 Integration Tests...\n")

    # ModelConfig → C roundtrip
    test("python_to_c_roundtrip"); test_python_to_c_roundtrip(); pass_()
    test("all_presets_accessible"); test_all_presets_accessible(); pass_()
    test("model_config_dataclass_conversion"); test_model_config_dataclass_conversion(); pass_()

    # ModelHub pipeline
    test("hub_save_load_pipeline"); test_hub_save_load_pipeline(); pass_()
    test("hub_cache_dir_creation"); test_hub_cache_dir_creation(); pass_()
    test("hub_model_card_generation"); test_hub_model_card_generation(); pass_()

    # from_pretrained
    test("from_pretrained_standalone"); test_from_pretrained_standalone(); pass_()
    test("from_pretrained_unknown_model"); test_from_pretrained_unknown_model(); pass_()

    # FrameworkConfig conversions
    test("mistral_config_conversion"); test_mistral_config_conversion(); pass_()
    test("qwen2_config_conversion"); test_qwen2_config_conversion(); pass_()
    test("deepseek_config_conversion"); test_deepseek_config_conversion(); pass_()

    # build_model_from_config
    test("build_model_from_config"); test_build_model_from_config(); pass_()

    print(f"\n{'-' * 40}")
    print(f"Results: {tests_passed}/{tests_total} tests passed!")
    print(f"{'-' * 40}\n")
    return 0 if tests_passed == tests_total else 1


if __name__ == "__main__":
    sys.exit(main())
