import pytest

from backend.model_config import BaseConfig


def test_model_config_round_trip(tmp_path):
  config = BaseConfig(output_attentions=True)
  config.custom_value = 42

  config.save_model_config(tmp_path)

  restored = BaseConfig()
  restored.load_model_config(tmp_path)
  assert restored.to_dict() == config.to_dict()


def test_missing_model_config_is_explicit(tmp_path):
  with pytest.raises(FileNotFoundError, match="Model config does not exist"):
    BaseConfig().load_model_config(tmp_path)
