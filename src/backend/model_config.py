"""Lightweight model configuration primitives."""

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Union


class BaseConfig:
  """Serializable base class for model configurations."""

  def __init__(
    self,
    output_hidden_states: bool = False,
    output_attentions: bool = False,
    return_dict: bool = False,
  ) -> None:
    self.output_hidden_states = output_hidden_states
    self.output_attentions = output_attentions
    self.return_dict = return_dict

  def to_dict(self) -> Dict[str, Any]:
    return copy.deepcopy(self.__dict__)

  def load_from_dict(self, config_dict: Dict[str, Any]) -> None:
    for key, value in config_dict.items():
      setattr(self, key, value)

  def save_model_config(self, output_dir: Union[str, os.PathLike]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "model_args.json").open("w", encoding="utf-8") as handle:
      json.dump(self.to_dict(), handle, indent=2)

  def load_model_config(self, input_dir: Union[str, os.PathLike]) -> None:
    model_args_file = Path(input_dir) / "model_args.json"
    if not model_args_file.is_file():
      raise FileNotFoundError(f"Model config does not exist: {model_args_file}")
    with model_args_file.open(encoding="utf-8") as handle:
      self.load_from_dict(json.load(handle))
