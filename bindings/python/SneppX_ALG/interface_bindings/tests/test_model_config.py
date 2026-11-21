"""Test for ModelConfig dataclass."""
import json
from SneppX_ALG.interface_bindings.model_zoo import ModelConfig, LlamaConfig, MistralConfig, get_model_config_obj, ModelHub


def test_model_config_creation():
    """Test basic ModelConfig creation."""
    config = ModelConfig(
        name="test-model",
        version="1.0.0",
        description="A test model",
        author="Test Author",
        license="MIT",
        architecture="transformer",
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=32000,
        max_seq_len=4096,
        learning_rate=2e-4,
    )
    
    assert config.name == "test-model"
    assert config.version == "1.0.0"
    assert config.hidden_size == 4096
    assert config.num_layers == 32
    assert config.num_attention_heads == 32
    assert config.num_key_value_heads == 8
    print("test_model_config_creation: PASS")


def test_model_config_json():
    """Test JSON serialization/deserialization."""
    config = ModelConfig(
        name="test-model",
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        vocab_size=32000,
    )
    
    json_str = config.to_json()
    loaded = ModelConfig.from_json(json_str)
    
    assert loaded.name == "test-model"
    assert loaded.hidden_size == 4096
    assert loaded.num_layers == 32
    assert loaded.num_attention_heads == 32
    assert loaded.vocab_size == 32000
    print("test_model_config_json: PASS")


def test_model_config_validation():
    """Test ModelConfig validation."""
    config = ModelConfig(name="test")
    errors = config.validate()
    assert len(errors) > 0  # Should have errors for missing required fields
    
    config = ModelConfig(
        name="test",
        version="1.0",
        architecture="transformer",
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        vocab_size=32000,
        learning_rate=2e-4,
    )
    errors = config.validate()
    assert len(errors) == 0  # Should be valid
    print("test_model_config_validation: PASS")


def test_llama_config_to_model_config():
    """Test LlamaConfig to ModelConfig conversion."""
    llama_config = LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=32000,
        max_position_embeddings=4096,
        name="llama2-7b",
        version="1.0",
    )
    
    model_config = llama_config.to_model_config()
    
    assert model_config.name == "llama2-7b"
    assert model_config.hidden_size == 4096
    assert model_config.num_layers == 32
    assert model_config.num_attention_heads == 32
    assert model_config.num_key_value_heads == 8
    assert model_config.vocab_size == 32000
    assert model_config.max_seq_len == 4096
    print("test_llama_config_to_model_config: PASS")


def test_mistral_config_to_model_config():
    """Test MistralConfig to ModelConfig conversion."""
    mistral_config = MistralConfig(
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=32000,
        max_position_embeddings=32768,
        sliding_window=4096,
        name="mistral-7b",
        version="0.1",
    )
    
    model_config = mistral_config.to_model_config()
    
    assert model_config.name == "mistral-7b"
    assert model_config.hidden_size == 4096
    assert model_config.num_layers == 32
    assert model_config.num_attention_heads == 32
    assert model_config.num_key_value_heads == 8
    assert model_config.vocab_size == 32000
    assert model_config.max_seq_len == 32768
    assert model_config.sliding_window == 4096
    print("test_mistral_config_to_model_config: PASS")


def test_model_config_from_registry():
    """Test getting ModelConfig from registry."""
    config = get_model_config_obj("llama2", "7B")
    
    assert config.name == "llama2-7B"
    assert config.hidden_size == 4096
    assert config.num_layers == 32
    assert config.num_attention_heads == 32
    assert config.vocab_size == 32000
    assert config.max_seq_len == 4096
    print("test_model_config_from_registry: PASS")


def test_model_config_file_io():
    """Test saving and loading ModelConfig from file."""
    import tempfile
    import os
    
    config = ModelConfig(
        name="file-test",
        hidden_size=2048,
        num_layers=16,
        num_attention_heads=16,
        vocab_size=16000,
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        config.save(temp_path)
        loaded = ModelConfig.load(temp_path)
        
        assert loaded.name == "file-test"
        assert loaded.hidden_size == 2048
        assert loaded.num_layers == 16
        assert loaded.num_attention_heads == 16
        assert loaded.vocab_size == 16000
        print("test_model_config_file_io: PASS")
    finally:
        os.unlink(temp_path)


def test_llama_config_from_model_config():
    """Test LlamaConfig from ModelConfig conversion."""
    model_config = ModelConfig(
        hidden_size=5120,
        intermediate_size=13824,
        num_layers=40,
        num_attention_heads=40,
        num_key_value_heads=40,
        vocab_size=32000,
        max_seq_len=4096,
        rope_theta=10000.0,
        learning_rate=2e-4,
    )
    
    llama_config = LlamaConfig.from_model_config(model_config)
    
    assert llama_config.hidden_size == 5120
    assert llama_config.intermediate_size == 13824
    assert llama_config.num_hidden_layers == 40
    assert llama_config.num_attention_heads == 40
    assert llama_config.num_key_value_heads == 40
    assert llama_config.vocab_size == 32000
    assert llama_config.max_position_embeddings == 4096
    assert llama_config.rope_theta == 10000.0
    print("test_llama_config_from_model_config: PASS")


def test_mistral_config_from_model_config():
    """Test MistralConfig from ModelConfig conversion."""
    model_config = ModelConfig(
        hidden_size=5120,
        intermediate_size=13824,
        num_layers=40,
        num_attention_heads=40,
        num_key_value_heads=40,
        vocab_size=32000,
        max_seq_len=4096,
        rope_theta=10000.0,
        sliding_window=4096,
        learning_rate=2e-4,
    )
    
    mistral_config = MistralConfig.from_model_config(model_config)
    
    assert mistral_config.hidden_size == 5120
    assert mistral_config.intermediate_size == 13824
    assert mistral_config.num_hidden_layers == 40
    assert mistral_config.num_attention_heads == 40
    assert mistral_config.num_key_value_heads == 40
    assert mistral_config.vocab_size == 32000
    assert mistral_config.max_position_embeddings == 4096
    assert mistral_config.rope_theta == 10000.0
    assert mistral_config.sliding_window == 4096
    print("test_mistral_config_from_model_config: PASS")


def test_mistral_config_from_model_config():
    """Test MistralConfig from ModelConfig conversion."""
    model_config = ModelConfig(
        hidden_size=5120,
        intermediate_size=13824,
        num_layers=40,
        num_attention_heads=40,
        num_key_value_heads=40,
        vocab_size=32000,
        max_seq_len=4096,
        rope_theta=10000.0,
        sliding_window=4096,
        learning_rate=2e-4,
    )
    
    mistral_config = MistralConfig.from_model_config(model_config)
    
    assert mistral_config.hidden_size == 5120
    assert mistral_config.intermediate_size == 13824
    assert mistral_config.num_hidden_layers == 40
    assert mistral_config.num_attention_heads == 40
    assert mistral_config.num_key_value_heads == 40
    assert mistral_config.vocab_size == 32000
    assert mistral_config.max_position_embeddings == 4096
    assert mistral_config.rope_theta == 10000.0
    assert mistral_config.sliding_window == 4096
    print("test_mistral_config_from_model_config: PASS")


def test_model_hub():
    """Test ModelHub class."""
    hub = ModelHub()
    
    # Test initialization
    assert hub.cache_dir.exists()
    
    # Test from_pretrained (stub)
    model = hub.from_pretrained("llama-2-7b")
    assert "model_id" in model
    assert model["model_id"] == "llama-2-7b"
    
    # Test save_pretrained
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        model = {"model_id": "test-model", "config": {"hidden_size": 512}}
        hub.save_pretrained(model, tmpdir)
        import os
        assert os.path.exists(os.path.join(tmpdir, "config.json"))
        assert os.path.exists(os.path.join(tmpdir, "README.md"))
    
    print("test_model_hub: PASS")


def main():
    """Run all tests."""
    print("Running ModelConfig tests...\n")
    
    test_model_config_creation()
    test_model_config_json()
    test_model_config_validation()
    test_llama_config_to_model_config()
    test_mistral_config_to_model_config()
    test_model_config_from_registry()
    test_model_config_file_io()
    test_llama_config_from_model_config()
    test_mistral_config_from_model_config()
    test_model_hub()
    
    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    main()