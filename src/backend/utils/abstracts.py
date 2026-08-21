import copy
import importlib.util
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from lightning import LightningDataModule
from torch import FloatTensor, LongTensor, Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

is_lightning_available = importlib.util.find_spec("pytorch_lightning") is not None
if is_lightning_available:
  logger.info("pytorch-lightning is available")
else:
  logger.info("pytorch-lightning is not available")

# For trainer
CONFIG_NAME = "config.json"
ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"
WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"

# For models
ALL_LAYERNORM_LAYERS = [nn.LayerNorm]


@dataclass
class BaseTrainingArguments:
  epochs: Optional[int] = field(default=None)
  batch_size: Optional[int] = field(default=None)
  eval_batch_size: Optional[int] = field(default=None)
  seq_len: Optional[int] = field(default=None)
  # TODO


@dataclass
class BaseModelInputs:
  input_ids: torch.LongTensor
  attention_mask: Optional[torch.Tensor] = None
  input_embeds: Optional[torch.FloatTensor] = None
  labels: torch.LongTensor = None
  use_cache: bool = False
  output_attentions: Optional[bool] = None
  output_hidden_states: Optional[bool] = None
  cache_position: Optional[torch.LongTensor] = None
  num_logits_to_keep: int = 0
  # TODO


@dataclass
class BaseModelOutput:
  last_hidden_state: Optional[torch.FloatTensor] = field(default=None)
  hidden_states: Optional[Tuple[torch.FloatTensor]] = field(default=None)
  attentions: Optional[Tuple[torch.FloatTensor]] = field(default=None)


@dataclass
class BaseLayerOutput:
  hidden_states: Optional[Tuple[torch.FloatTensor]] = field(default=None)
  attentions: Optional[Tuple[torch.FloatTensor]] = field(default=None)


# pyTorch Lightning
@dataclass
class LightningStatus:
  is_available = is_lightning_available


@dataclass
class LightningTrainingArguments(BaseTrainingArguments):
  seed: Optional[int] = field(default=None)
  model_name: str = field(default="default_model")
  accelerator: Optional[str] = field(default="auto")
  devices: Optional[Union[List[int], str, int]] = field(default="auto")
  num_nodes: Optional[int] = field(default=1)
  min_epochs: Optional[int] = field(
    default=None,
    metadata="Force training for at least these many epochs. Disabled by default (None).",
  )
  max_epochs: Optional[int] = field(
    default=None,
    metadata=(
      "Stop training once this number of epochs is reached. Disabled by default (None)."
      " If both max_epochs and max_steps are not specified, defaults to max_epochs = 1000. To enable infinite training, set max_epochs = -1."
    ),
  )
  min_steps: Optional[int] = field(
    default=None,
    metadata="Force training for at least these number of steps. Disabled by default (None).",
  )
  max_steps: Optional[int] = field(
    default=-1,
    metadata=(
      "Stop training after this number of steps. Disabled by default (-1). If `max_steps` = -1 and `max_epochs` = None,"
      " will default to `max_epochs` = 1000. To enable infinite training, set `max_epochs` to -1."
    ),
  )
  logger_type: Optional[str] = field(
    default="tensorboard",
    metadata={
      "note": "Set up the logger for the trainer, currently possible choices: 'csv', 'tensorboard', 'wandb'. Default: 'tensorboard'."
    },
  )
  log_every_n_steps: Optional[int] = field(
    default=50, metadata="How often to log within steps. Default: 50."
  )

  checkpoint_dir: Optional[Union[str, os.PathLike]] = field(
    default=".", metadata={"note": ""}
  )  # FIXME
  test_checkpoint_path: Optional[Union[str, os.PathLike]] = field(
    default=None, metadata={"note": ""}
  )  # FIXME
  resume_from_checkpoint: Optional[str] = field(
    default=None, metadata={"note": 'Path to a .ckpt file or "latest"'}
  )

  num_workers: Optional[int] = field(default=0)
  train_dataloaders: Optional[Union[Any, DataLoader, LightningDataModule]] = field(
    default=None
  )
  val_dataloaders: Optional[Union[Any, DataLoader, LightningDataModule]] = field(
    default=None
  )
  test_dataloaders: Optional[Union[Any, DataLoader, LightningDataModule]] = field(
    default=None
  )

  # (Other fields you might have)
  # ...

  def __post_init__(self):
    if self.checkpoint_dir:
      self.checkpoint_dir = str(Path(self.checkpoint_dir).absolute())

    if self.eval_batch_size is None:
      self.eval_batch_size = self.batch_size

    # Ensure dataloaders are attributes even if None initially
    if not hasattr(self, "train_dataloaders"):
      self.train_dataloaders = None
    if not hasattr(self, "val_dataloaders"):
      self.val_dataloaders = None
    if not hasattr(self, "test_dataloaders"):
      self.test_dataloaders = None

    self.collate()

  def to_dict(self) -> Dict[str, Any]:
    """Serializes this instance to a Python Dictionary"""
    return copy.deepcopy(self.__dict__)

  def collate(self) -> None:
    """Collation Abstract"""
    raise NotImplementedError(
      "LightningTrainingArguments: Make sure to implement `collate` in a subclass."
    )


@dataclass
class LightningModelInputs(BaseModelInputs):
  input_ids: Optional[Union[LongTensor, FloatTensor]] = field(
    default=None,
    metadata={
      "note": (
        "For embedded logits (e.g. Language Model), `input_ids` should be specified. "
        "For 2D or image-like inputs: `pixel_values` should be specified."
      )
    },
  )
  pixel_values: Optional[Union[LongTensor, FloatTensor]] = field(
    default=None,
    metadata={
      "note": (
        "For embedded logits (e.g. Language Model), `input_ids` should be specified. "
        "For 2D or image-like inputs: `pixel_values` should be specified."
      )
    },
  )

  attention_mask: Optional[Tensor] = field(default=None)
  position_ids: Optional[LongTensor] = field(default=None)
  cache_position: Optional[LongTensor] = field(default=None)

  inputs_embeds: Optional[FloatTensor] = field(default=None)

  labels: Optional[LongTensor] = field(
    default=None,
    metadata={
      "note": (
        "For normal cases, `labels` is not supposed to be within the `TrainingInputs`."
        " During the training loop, the `labels` should passed by `Dataset` separately."
        "However, `labels` should be specified within the `LightningTrainingInputs` while"
        " training model via `pytorch-lightning`."
      )
    },
  )

  num_logits_to_keep: int = field(default=0)

  use_cache: Optional[bool] = field(default=None)
  output_attentions: Optional[bool] = field(default=None)
  output_hidden_states: Optional[bool] = field(default=None)
