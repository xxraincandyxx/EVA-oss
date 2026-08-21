# configuration_focus.py

"""Focus on You"""

import logging
import warnings
from math import sqrt
from typing import Optional

from ....model_config import BaseConfig

logger = logging.getLogger(__name__)


class FocusConfig(BaseConfig):
  def __init__(
    self,
    model_name: str = "focus_icm",  # stands for 'focus image classification model', not elegant thx
    num_channels: Optional[int] = 3,
    manifest_patch_size: Optional[int] = 32,
    manifest_patch_stride: Optional[int] = None,
    latent_patch_size: Optional[int] = 4,
    latent_patch_stride: Optional[int] = None,
    seq_len: Optional[int] = 144,  # = `shared_len` + `latent_len`
    shared_len: Optional[int] = 80,
    latent_len: Optional[int] = 64,
    hidden_size: Optional[int] = 768,
    intermediate_size: Optional[int] = 1536,
    attn_num_heads: Optional[int] = 12,
    attn_head_dim: Optional[int] = None,
    num_classes: Optional[int] = None,
    num_manifest_layers: Optional[int] = 2,
    num_latent_layers: Optional[int] = 6,
    max_rel_dist: Optional[int] = None,
    rms_epsilon: Optional[float] = 1e-12,
    sdpa_enable: Optional[bool] = True,
    _focus_loc_len: Optional[int] = None,
    # for the following argument: currently only two available choices: 'nomad' for a normal model and number of params; 'debug' for light-weight model and params
    operating_mode: Optional[str] = "nomad",
  ):
    self.model_name = model_name
    self.num_channels = num_channels
    self.latent_patch_size = latent_patch_size
    self.manifest_patch_size = manifest_patch_size
    self.latent_patch_stride = (
      latent_patch_stride if latent_patch_stride is not None else latent_patch_size
    )
    self.manifest_patch_stride = (
      manifest_patch_stride
      if manifest_patch_stride is not None
      else manifest_patch_size
    )

    self.seq_len = seq_len if seq_len is not None else latent_len + shared_len
    if int(sqrt(seq_len)) ** 2 != seq_len:
      raise ValueError(f"`seq_len` must be a perfect square, got {seq_len}")
    self.latent_len = latent_len
    self.shared_len = shared_len

    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.attn_num_heads = attn_num_heads
    self.attn_head_dim = (
      attn_head_dim if attn_head_dim is not None else hidden_size // attn_num_heads
    )

    self.num_classes = num_classes
    self.num_manifest_layers = num_manifest_layers
    self.num_latent_layers = num_latent_layers

    self.max_rel_dist = max_rel_dist
    self.rms_epsilon = rms_epsilon
    self.sdpa_enable = sdpa_enable

    # TODO (emer): try to remove the pre-settings
    if _focus_loc_len is not None and _focus_loc_len != 49:
      # replace with `warning_once`, which thx is equivalent...
      warnings.warn("You should not modify inner variable `_focus_loc_len`.")
    _focus_loc_len = 49
    self._focus_loc_len = _focus_loc_len

    self.operating_mode = operating_mode

    # __post_init__
    if self.operating_mode == "debug":
      self.hidden_size = 66
      self.intermediate_size = 79
      self.attn_num_heads = 3
      # pre-calculate the `attn_head_dim` to avoid the modification within the `__init__`
      self.attn_head_dim = self.hidden_size // self.attn_num_heads

      # dummy
      self.num_manifest_layers = 1
      self.num_latent_layers = 2

    elif self.operating_mode == "light":
      self.hidden_size = 128
      self.intermediate_size = 256
      self.attn_num_heads = 4
      self.attn_head_dim = self.hidden_size // self.attn_num_heads  # 64

      self.num_manifest_layers = 3
      self.num_latent_layers = 6

    if self.num_classes is None:
      raise ValueError("FocusConfig: `num_classes` must be set.")

    # Self-Collation
    logger.info(f"Focus Configuration: {self.to_dict()}")


# configuration_focus.py ends here
