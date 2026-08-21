# modeling_focus.py

"""Focus on You"""

import logging
import math
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor, Tensor
from torch.nn import CrossEntropyLoss

from .collation_focus import FocusCollator
from .configuration_focus import FocusConfig

logger = logging.getLogger(__name__)


class FocusRMSNorm(nn.Module):
  """FocusRMSNorm is same to T5RMSNorm & LlamaRMSNorm"""

  def __init__(self, hidden_size: int, epsilon=1e-12):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = epsilon

  def forward(self, hidden_states: FloatTensor) -> FloatTensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


class FocusPatchEmbedding(nn.Module):
  def __init__(
    self,
    config: FocusConfig,
    collator: Optional[FocusCollator] = None,
    patch_size: Optional[int] = None,
    patch_stride: Optional[int] = None,
    **kwargs,
  ):
    super().__init__()
    self.config = config
    self.collator = collator
    patch_size = patch_size if patch_size is not None else config.latent_patch_size
    patch_stride = (
      patch_stride if patch_stride is not None else config.latent_patch_stride
    )
    self.conv = nn.Conv2d(
      config.num_channels,
      config.hidden_size,
      kernel_size=patch_size,
      stride=patch_stride,
    )

  def forward(self, pixel_values: LongTensor) -> FloatTensor:
    # [batch_size, num_channels, height, width] -> [batch_size, num_patches, hidden_size]
    return self.conv(pixel_values).flatten(-2).transpose(-2, -1)


class MultiLayerPerception(nn.Module):
  def __init__(
    self,
    in_hidden_size: int,
    intermediate_size: int,
    out_hidden_size: int,
    is_bias: Optional[bool] = False,
    **kwargs,
  ):
    super().__init__()
    is_bias = is_bias if is_bias is not None else False

    self.gate_proj = nn.Linear(in_hidden_size, intermediate_size, bias=is_bias)
    self.up_proj = nn.Linear(in_hidden_size, intermediate_size, bias=is_bias)
    self.down_proj = nn.Linear(intermediate_size, out_hidden_size, bias=is_bias)
    self.act_fn = nn.SiLU()  # default: nn.GELU()

  def forward(self, hidden_states: FloatTensor) -> FloatTensor:
    # Assuming input shape: [batch_size, seq_len, in_hidden_size]
    #    Then output shape: [batch_size, seq_len, out_hidden_size]
    mlp_inputs = self.act_fn(self.gate_proj(hidden_states))
    mlp_weights = self.up_proj(hidden_states)
    mlp_outputs = self.down_proj(mlp_weights * mlp_inputs)
    return mlp_outputs


class FocusRotaryEmbedding(nn.Module):
  def __init__(
    self,
    dim: Optional[int] = None,
    base: Optional[float] = 10000.0,
    device: Optional[Union["torch.device", Any]] = None,
    scaling_factor: Optional[float] = 1.0,  # unused
    config: Optional[FocusConfig] = None,
    collator: Optional[FocusCollator] = None,
  ):
    # FIXME: complete this docstring
    """_summary_

    Args:
        dim (Optional[int], optional): _description_. Defaults to None.
        max_pos_emb (Optional[int], optional): _description_. Defaults to 256.
        base (Optional[float], optional): _description_. Defaults to 1000.0.
        device (Optional[Union[&quot;torch.device&quot;, Any]], optional): _description_. Defaults to None, we will move the tensor to `cpu` in this case.
        scaling_factor (Optional[float], optional): _description_. Defaults to 1.0.
        config (Optional[FocusConfig], optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_
    """
    super().__init__()
    # TODO: apply rope init func
    self.config = config
    self.collator = collator
    if dim is None and config is None:
      raise ValueError("You should specify either `dim` or `config`.")
    dim = dim if dim is not None else config.attn_head_dim
    device = device if device is not None else torch.device("cpu")

    # `inv_freq` shape: [dim // 2,]
    inv_freq = 1.0 / (
      base
      ** (
        torch.arange(start=0, end=dim, step=2, dtype=torch.float32, device=device) / dim
      )
    )
    self.register_buffer("inv_freq", inv_freq, persistent=False)
    logger.info(f"`inv_freq` device: {self.inv_freq.device}")

  @torch.no_grad()
  def forward(
    self, states: FloatTensor, position_ids: LongTensor
  ) -> Tuple[FloatTensor, FloatTensor]:
    if self.inv_freq.device != torch.device("cpu"):
      # avoid auto casting (e.g. pytorch lightning) to move the tensor to cpu
      self.inv_freq = self.inv_freq.to(torch.device("cpu"))

    # supposed shape of `inv_freq_extended`: [batch_size, dim // 2, 1]; `position_ids_extended`: [batch_size, 1, seq_len]
    device = states.device  # whatever, to apply the RoPE, the output `cos` and `sin` should be on the same device with `states`
    inv_freq_extended = (
      self.inv_freq[None, :, None]
      .float()
      .expand(position_ids.shape[0], -1, 1)
      .to(device)
    )
    position_ids_extended = position_ids[:, None, :].float().to(device)

    logger.debug(
      f"FocusRotaryEmbedding.forward() - `batch_size` = {position_ids.shape[0]}; `seq_len` = {position_ids.shape[1]}"
    )

    logger.debug(
      "FocusRotaryEmbedding.forward() - "
      "Supposed shape of `inv_freq_extended`: [batch_size, dim // 2, 1]; `position_ids_extended`: [batch_size, 1, seq_len] - "
      f"`inv_freq_extended` device: {inv_freq_extended.device}, shape: {inv_freq_extended.shape}; "
      f"`position_ids_extended` device: {position_ids_extended.device}, shape: {position_ids_extended.shape}"
    )

    # force float32
    device_type = states.device.type
    device_type = (
      device_type if isinstance(device_type, str) and device_type != "mpu" else "cpu"
    )
    with torch.autocast(device_type=device_type, enabled=False):
      # @.T -> [batch_size, seq_len, dim // 2]
      freqs = (torch.matmul(inv_freq_extended, position_ids_extended)).transpose(1, 2)
      emb = torch.cat([freqs, freqs], dim=-1)
      cos = emb.cos()
      sin = emb.sin()

    # Output device & shape re-check
    logger.debug(
      "FocusRotaryEmbedding.forward() - "
      "Supposed shape of `cos` and `sin`: [batch_size, seq_len, dim] - "
      f"`cos` device: {cos.device}, shape: {cos.shape}; `sin` device: {sin.device}, shape: {sin.shape}"
    )

    # `attention_scaling` should've been implemented here, yet unused in this case
    return cos.to(dtype=states.dtype), sin.to(dtype=states.dtype)


def _rotate_half(states: FloatTensor) -> FloatTensor:
  links = states[..., : states.shape[-1] >> 1]  # left
  rechts = states[..., states.shape[-1] >> 1 :]  # right
  return torch.cat([-rechts, links], dim=-1)


def _apply_rotary_pos_emb(
  query: FloatTensor,
  key: FloatTensor,
  cos: FloatTensor,
  sin: FloatTensor,
  unsqueeze_dim: Optional[int] = 1,
) -> Tuple[FloatTensor, FloatTensor]:
  cos = cos.unsqueeze(unsqueeze_dim)
  sin = sin.unsqueeze(unsqueeze_dim)
  query_embed = (query * cos) + (_rotate_half(query) * sin)
  key_embed = (key * cos) + (_rotate_half(key) * sin)
  return query_embed, key_embed


class MultiHeadRelativeEmbedding(nn.Module):
  def __init__(
    self,
    seq_len: int,
    shared_len: int,
    num_heads: int,
    head_dim: int,
    max_rel_dist: Optional[int] = None,
    collator: Optional[Union[FocusCollator, Any]] = None,
  ):
    super().__init__()

    rel_size = int(math.sqrt(seq_len - shared_len))
    self.collator = collator
    self.shared_len = shared_len
    # the default value of `max_rel_dist` is set to max available size
    self.max_rel_dist = max_rel_dist if max_rel_dist is not None else rel_size - 1
    if seq_len != rel_size**2 + shared_len:
      raise ValueError(
        f"`seq_len`({seq_len}) - `shared_len`({shared_len}) is supposed to be a perfect square."
      )

    self.h_rel_table = nn.Parameter(
      torch.randn(num_heads, (self.max_rel_dist << 1) + 1 + shared_len, head_dim >> 1)
    )
    self.w_rel_table = nn.Parameter(
      torch.randn(num_heads, (self.max_rel_dist << 1) + 1 + shared_len, head_dim >> 1)
    )

    h_rel_pos, w_rel_pos = torch.meshgrid(
      torch.arange(rel_size), torch.arange(rel_size), indexing="ij"
    )
    h_rel_pos, w_rel_pos = h_rel_pos.reshape(-1), w_rel_pos.reshape(-1)
    h_dist, w_dist = (
      self._get_rel_dist(h_rel_pos),
      self._get_rel_dist(w_rel_pos),
    )

    self.register_buffer("h_dist", h_dist)
    self.register_buffer("w_dist", w_dist)

  def _get_rel_dist(self, rel_pos: FloatTensor) -> FloatTensor:
    dist = rel_pos.unsqueeze(1) - rel_pos.unsqueeze(0)
    dist = torch.clamp(dist, -self.max_rel_dist, self.max_rel_dist)
    dist = dist + self.max_rel_dist + self.shared_len
    # pad last dim by (shared_len, 0) and 2nd to last by (shared_len, 0)
    dist = nn.functional.pad(
      dist, (self.shared_len, 0, self.shared_len, 0), "constant", 0
    )
    return dist

  def forward(self) -> FloatTensor:
    h_pos_emb = self.h_rel_table[:, self.h_dist]
    w_pos_emb = self.w_rel_table[:, self.w_dist]
    rel_pos_emb = torch.cat([h_pos_emb, w_pos_emb], dim=-1)
    return rel_pos_emb


class FocusAttention(nn.Module):
  """Multi-headed attention from 'Attention Is All You Need' paper"""

  def __init__(
    self,
    hidden_size: int,
    num_heads: int,
    head_dim: Optional[int] = None,
    collator: Optional[Union[FocusCollator, Any]] = None,
    layer_idx: Optional[int] = None,
    **kwargs,
  ):
    super().__init__()
    self.collator = collator
    self.layer_idx = layer_idx
    self.num_heads = num_heads
    self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
    if layer_idx is None:
      raise ValueError  # TODO: Implement and Use `warning_once`

    logger.debug(
      f"FocusAttention -- `num_heads` = {self.num_heads}, `head_dim` = {self.head_dim}"
    )

    self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
    self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
    self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

  def forward(
    self,
    hidden_states: FloatTensor,
    position_embeddings: Optional[Tuple[FloatTensor, FloatTensor]] = None,
    **kwargs,
  ) -> FloatTensor:
    """To implement RoPE, You should specify `position_embeddings` = `cos`, `sin`"""
    # seq_len = query_len = key_len = value_len
    batch_size, seq_len, _ = hidden_states.size()
    logger.debug(
      f"FocusAttention.forward -- `hidden_states` shape: {hidden_states.shape}"
    )

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
      batch_size, seq_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
      batch_size, seq_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
      batch_size, seq_len, self.num_heads, self.head_dim
    ).transpose(1, 2)

    logger.debug(
      f"FocusAttention.forward -- `query_states` shape: {query_states.shape}"
    )

    if position_embeddings is not None:
      cos, sin = position_embeddings
      query_states, key_states = _apply_rotary_pos_emb(
        query_states, key_states, cos, sin
      )
      logger.info("FocusAttention: Applies `RoPE`")

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
      self.head_dim
    )
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
      query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (
      batch_size,
      self.num_heads,
      seq_len,
      self.head_dim,
    ):
      raise ValueError(
        f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
        f" {attn_output.size()}"
      )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output


class FocusRelativeAttention(nn.Module):
  def __init__(
    self,
    seq_len: int,
    shared_len: int,
    hidden_size: int,
    num_heads: int,
    head_dim: Optional[int] = None,
    max_rel_dist: Optional[int] = None,
    collator: Optional[Union[FocusCollator, Any]] = None,
    layer_idx: Optional[int] = None,
    **kwargs,
  ):
    super().__init__()
    self.collator = collator
    self.layer_idx = layer_idx
    self.seq_len = seq_len
    self.num_heads = num_heads
    head_dim = head_dim if head_dim is not None else hidden_size // num_heads
    self.head_dim = head_dim

    self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
    self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
    self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
    self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    self.q_pos_emb = MultiHeadRelativeEmbedding(
      seq_len, shared_len, num_heads, head_dim, max_rel_dist, collator
    )
    self.v_pos_emb = MultiHeadRelativeEmbedding(
      seq_len, shared_len, num_heads, head_dim, max_rel_dist, collator
    )

  def forward(self, hidden_states: FloatTensor, **kwargs) -> FloatTensor:
    # seq_len = query_len = key_len = value_len
    batch_size = hidden_states.shape[0]
    logger.debug(
      f"FocusRelativeAttention.forward -- `hidden_states` shape: {hidden_states.shape}"
    )

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
      batch_size, self.seq_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
      batch_size, self.seq_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
      batch_size, self.seq_len, self.num_heads, self.head_dim
    ).transpose(1, 2)

    # [batch_size, num_heads, query_len, head_dim] @ [batch_size, head_dim, num_heads, key_len] -> [batch_size, num_heads, query_len, key_len]
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

    # q_pos: [num_heads, query_len, key_len, head_dim] -> [num_heads * query_len, head_dim, key_len]
    q_pos = self.q_pos_emb().transpose(2, 3)
    q_pos = q_pos.reshape(-1, self.head_dim, self.seq_len)

    # query_states: [batch_size, num_heads, seq_len, head_dim] -> [num_heads * query_len, batch_size, head_dim]
    query_states = query_states.permute(1, 2, 0, 3)
    query_states = query_states.reshape(-1, batch_size, self.head_dim)

    # rel_q: @ -> [num_heads * query_len, batch_size, key_len] -> [batch_size, num_heads, query_len, key_len]
    rel_q = torch.matmul(query_states, q_pos)
    rel_q = rel_q.reshape(
      self.num_heads, self.seq_len, batch_size, self.seq_len
    ).permute(2, 0, 1, 3)

    attn_weights = (attn_weights + rel_q) / math.sqrt(self.head_dim)
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

    # attn_values: @ -> [batch_size, num_heads, query_len, head_dim]
    attn_values = torch.matmul(attn_weights, value_states)

    # v_pos: [num_heads, query_len, key_len, head_dim] -> [num_heads * query_len, key_len, head_dim]
    v_pos = self.v_pos_emb()
    v_pos = v_pos.reshape(-1, self.seq_len, self.head_dim)

    # attn_weights: [batch_size, num_heads, query_len, key_len] -> [num_heads * query_len, batch_size, key_len]
    attn_weights = attn_weights.permute(1, 2, 0, 3).reshape(
      -1, batch_size, self.seq_len
    )

    # rel_v: [num_heads * query_len, batch_size, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
    rel_v = torch.matmul(attn_weights, v_pos)
    rel_v = rel_v.reshape(
      self.num_heads, self.seq_len, batch_size, self.head_dim
    ).permute(2, 0, 1, 3)

    # attn_values: [batch_size, num_heads, seq_len, head_dim]
    attn_values = attn_values + rel_v

    # reform for return -> [batch_size, seq_len, num_heads * head_dim]
    attn_values = attn_values.transpose(1, 2).contiguous()
    attn_values = attn_values.reshape(batch_size, self.seq_len, -1)

    attn_output = self.o_proj(attn_values)
    return attn_output


class FocusSdpaAttention(FocusAttention):
  """
  Focus attention module using torch.nn.functional.scaled_dot_product_attention.
  This module inherits from `FocusAttention` as the weights of the module stays untouched.
  The only changes are on the forward pass to adapt to the SDPA API.
  This is copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L407
  """

  # Inherited from FocusAttention.forward()
  def forward(
    self,
    hidden_states: FloatTensor,
    positional_embeddings: Optional[Tuple[FloatTensor, FloatTensor]] = None,
    **kwargs,
  ):
    """Equivalent to FocusAttention, To implement RoPE, You should specify `positional_embeddings` = `cos`, `sin`"""
    # seq_len = query_len = key_len = value_len
    batch_size, seq_len, _ = hidden_states.size()
    logger.debug(
      f"FocusSdpaAttention.forward() - `hidden_states` shape: {hidden_states.shape}"
    )

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
      batch_size, seq_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
      batch_size, seq_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
      batch_size, seq_len, self.num_heads, self.head_dim
    ).transpose(1, 2)

    logger.debug(
      f"FocusSdpaAttention.forward() - `query_states` shape: {query_states.shape}"
    )

    if positional_embeddings is not None:
      cos, sin = positional_embeddings
      query_states, key_states = _apply_rotary_pos_emb(
        query_states, key_states, cos, sin
      )
      logger.info("FocusSdpaAttention.forward() - applies `RoPE`")

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda":
      query_states = query_states.contiguous()
      key_states = key_states.contiguous()
      value_states = value_states.contiguous()

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
      self.head_dim
    )
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
      query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
      query=query_states,
      key=key_states,
      value=value_states,
      attn_mask=None,
      dropout_p=0.0,
      is_causal=False,
      scale=None,
    )

    # the following code is not accepted by torch.jit for python boolean values, deprecating...
    # if attn_output.size() != torch.Size((batch_size, self.num_heads, seq_len, self.head_dim)):
    #   raise ValueError(
    #     f"Attention output shape mismatch. Expected {torch.Size((batch_size, self.num_heads, seq_len, self.head_dim))}, "
    #     f"got {attn_output.size()}"
    #   )
    attn_output.view(batch_size, self.num_heads, seq_len, self.head_dim)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output


FOCUS_ATTENTION_CLASSES = {
  "eager": FocusAttention,
  "relative": FocusRelativeAttention,
  "sdpa": FocusSdpaAttention,
}


# TODO: try to use rel emb as the first layer and sdpa as others
class FocusEncoder(nn.Module):
  def __init__(
    self,
    config: FocusConfig,
    collator: Optional[Union[FocusCollator, Any]] = None,
    attn_implementation: Optional[str] = "sdpa",
    layer_idx: Optional[int] = None,
  ):
    super().__init__()
    self.config = config
    self.collator = collator

    self.layer_idx = layer_idx

    self.input_layernorm = FocusRMSNorm(config.hidden_size, epsilon=config.rms_epsilon)
    self.self_attn = FOCUS_ATTENTION_CLASSES[attn_implementation](
      seq_len=config.seq_len,
      shared_len=config.shared_len,
      hidden_size=config.hidden_size,
      num_heads=config.attn_num_heads,
      head_dim=config.attn_head_dim,
      max_rel_dist=config.max_rel_dist,
      layer_idx=layer_idx,
      collator=collator,
    )
    self.post_attn_layernorm = FocusRMSNorm(
      config.hidden_size, epsilon=config.rms_epsilon
    )
    self.mlp = MultiLayerPerception(
      in_hidden_size=config.hidden_size,
      intermediate_size=config.intermediate_size,
      out_hidden_size=config.hidden_size,
      is_bias=False,
    )

  def forward(
    self,
    hidden_states: FloatTensor,
    position_embeddings: Optional[Tuple[FloatTensor, FloatTensor]] = None,
  ) -> FloatTensor:
    logger.debug(
      f"FocusEncoder({self.layer_idx}).forward -- `hidden_states` shape: {hidden_states.shape}"
    )

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(
      hidden_states, position_embeddings=position_embeddings
    )  # if using Relative Attention, the latter argument will be passed to **kwargs
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attn_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


class FocusRouter(nn.Module):
  """Locate the Position to Focus on"""

  def __init__(self, config: FocusConfig, collator: Optional[FocusCollator] = None):
    super().__init__()
    self.config = config
    sdpa_enable = getattr(config, "sdpa_enable", True)
    attn_implementation = "sdpa" if sdpa_enable else "eager"

    self.manifest_emb = FocusPatchEmbedding(
      config,
      collator,
      patch_size=config.manifest_patch_size,
      patch_stride=config.manifest_patch_stride,
    )
    self.rotary_emb = FocusRotaryEmbedding(
      dim=config.attn_head_dim, base=1000.0, config=config, collator=collator
    )
    self.manifest_layers = nn.ModuleList(
      [
        FocusEncoder(config, collator, attn_implementation, layer_idx)
        for layer_idx in range(config.num_manifest_layers)
      ]
    )  # we will pass `position_embeddings` within the forward pass, to apply `RoPE`
    self.norm = FocusRMSNorm(config.hidden_size, epsilon=config.rms_epsilon)

    # TODO: implement func for weights init & final processing

  def forward(
    self,
    pixel_values: FloatTensor,
    shared_states: FloatTensor,
    position_ids: Optional[LongTensor] = None,
    previous_manifest_states: Optional[FloatTensor] = None,
  ) -> Tuple[FloatTensor, FloatTensor]:
    if position_ids is None:
      # TODO: warning once + replace rope with relative if `position_ids` is not provided, or calculate it within this scope
      raise ValueError(
        "You should specify `position_ids` for RoPE of the first manifest layer."
      )

    if previous_manifest_states is None:
      logger.info(
        "FocusRouter.forward() - `previous_manifest_states` is not provided, patchifying the `pixel_values`..."
      )
      manifest_states = self.manifest_emb(pixel_values)  # `seq_len` is variable

    else:
      logger.info(
        "FocusRouter.forward() - `previous_manifest_states` is provided, using it..."
      )
      manifest_states = previous_manifest_states

    logger.debug(
      f"FocusRouter.forward() - `manifest_states` shape: {manifest_states.shape}"
    )

    # hidden_states shape: [batch_size, seq_len' = shared_len + manifest_len (UNKNOWN), hidden_size]
    hidden_states = torch.cat([shared_states, manifest_states], dim=-2)
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for layer_idx, manifest_layer in enumerate(self.manifest_layers):
      hidden_states = (
        manifest_layer(hidden_states, position_embeddings)  # sdpa with rope
        if layer_idx <= 0
        else manifest_layer(hidden_states, None)  # sdpa without positional embedding
      )
    hidden_states = self.norm(hidden_states)

    shared_states = hidden_states[:, : self.config.shared_len, :]
    manifest_states = hidden_states[:, self.config.shared_len :, :]
    return shared_states, manifest_states


class FocusDecoder(nn.Module):
  def __init__(self, config: FocusConfig, collator: FocusCollator):
    super().__init__()
    self.config = config
    self.collator = collator

    _attn_implementation = "relative"  # IMMUTABLE

    self.latent_embed = FocusPatchEmbedding(
      config,
      collator,
      patch_size=config.latent_patch_size,
      patch_stride=config.latent_patch_stride,
    )
    self.latent_layers = nn.ModuleList(
      [
        FocusEncoder(config, collator, _attn_implementation, layer_idx)
        for layer_idx in range(config.num_latent_layers)
      ]
    )
    self.norm = FocusRMSNorm(config.hidden_size, epsilon=config.rms_epsilon)

    # TODO: implement func for weights init & final processing

  def forward(
    self, pixel_values: FloatTensor, shared_states: FloatTensor
  ) -> Tuple[FloatTensor, FloatTensor]:
    # `states` shape: [batch_size, seq_len = shared_len + latent_len, hidden_size]
    logger.debug(f"FocusDecoder.forward() - `pixel_values` shape: {pixel_values.shape}")

    # `latent_states` shape: [batch_size, latent_len, hidden_size]
    latent_states = self.latent_embed(
      pixel_values
    )  # `seq_len` is constant within this layer
    logger.debug(
      f"FocusDecoder.forward() - `latent_states` shape: {latent_states.shape}"
    )
    logger.debug(
      f"FocusDecoder.forward() - `shared_states` shape: {shared_states.shape}"
    )

    # `shared_states` shape: [batch_size, shared_len, hidden_size]
    hidden_states = torch.cat([shared_states, latent_states], dim=1)
    logger.debug(
      f"FocusDecoder.forward() - `hidden_states` shape: {hidden_states.shape}"
    )

    for latent_layer in self.latent_layers:
      # we apply the skip connections within the `latent_layer`
      hidden_states = latent_layer(hidden_states)
    hidden_states = self.norm(hidden_states)

    shared_states = hidden_states[:, : self.config.shared_len, :]
    latent_states = hidden_states[:, self.config.shared_len :, :]
    return shared_states, latent_states


class FocusModel(nn.Module):
  def __init__(self, config: FocusConfig, collator: FocusCollator):
    super().__init__()
    self.config = config
    self.collator = collator

    # univariate state of `shared_states` trainable/learnable
    self.shared_state = nn.Parameter(
      torch.randn(1, config.shared_len, config.hidden_size)
    )

    self.route = FocusRouter(config, collator)
    self.decode = FocusDecoder(config, collator)

    self.focus_expert = nn.Linear(
      config.hidden_size, config._focus_loc_len
    )  # inner function, this defines the logic of `self._focus_on()`: [1] index for row-majored in-order patches (TEMPORARY SOLUTION);
    self.cls_expert = nn.Linear(
      config.hidden_size, config.num_classes + 2
    )  # [1] classes (one-hot vector of size `config.num_classes`); [2] continue or break (confidence as a one-hot vector of size 2);

    # init attributes
    self.logged_focus_logits = []

    # TODO: implement func for weights init & final processing

  def forward(
    self, pixel_values: FloatTensor
  ) -> Tuple[FloatTensor, Union[List[FloatTensor], None]]:
    logger.debug(
      f"FocusModel - forward() - input pixel_values shape: {pixel_values.size()}"
    )
    # returns `cls_logits`, `focus_logits` (if configured)
    batch_size = pixel_values.shape[0]
    # -> `manifest_position_ids` shape: [batch_size, manifest_num_patches_height * manifest_num_patches_width]
    manifest_position_ids = self._get_manifest_position_ids(pixel_values)
    focus_window_lookup = manifest_position_ids.size()
    shared_states = self.shared_state.expand(batch_size, -1, -1)

    logger.debug(
      f"FocusModel.forward() - `manifest_position_ids` shape & preview: {manifest_position_ids.shape}"
      f"\n - first:\n{manifest_position_ids[0]}\n - last:\n{manifest_position_ids[-1]}"
    )

    # A loop iterating to decide where to break the focusing progress
    # 02-03-2025: we will first test this model with only one router and one decoder, then build func & loop based on this
    # 02-04-2025: our current preference for construction of `shared_states` - [0] classes & confidence; [1] focus location;
    # 06-06-2025: initialize construction...
    # I suggest that we need a stronger policy here, I don't expect that the model would find that more iterations will
    # make it more clear about the object it supposed to clarify.
    manifest_states = None
    max_focus_iters = getattr(self.config, "max_focus_iters", 4)

    focus_logits_cache = []
    # temporal version, we just ignore the confidence of the model and simply do 4 iterations
    for iter in range(max_focus_iters):
      logger.info(f"==== Focusing Iteration Index [{iter}] ====")

      # during the first iter, `manifest_states` is supposed to be passed as a NoneType value, which would be ignored by the route layer and it will generate a new one
      shared_states, manifest_states = self.route(
        pixel_values,
        shared_states,
        manifest_position_ids,
        previous_manifest_states=manifest_states,
      )
      logger.debug(
        f"FocusModel.forward() - iter[{iter}] - post router - `shared_states` shape: {shared_states.shape}"
      )

      focus_logits = self.focus_expert(
        shared_states[:, 1, :]
      )  # extract the latent focus location
      # crop wanted window from original `pixel_values`
      # [batch_size, num_channels, focus_window_size, focus_window_size] (`focus_window_size` = `manifest_patch_size`)
      focus_windows = self._focus_on(pixel_values, focus_logits, focus_window_lookup)
      logger.debug(
        f"FocusModel.forward() - iter[{iter}] - `focus_windows[0]` shape: {focus_windows[0].shape}"
      )

      shared_states, _ = self.decode(
        focus_windows, shared_states
      )  # omit `latent_states`
      logger.debug(
        f"FocusModel.forward() - iter[{iter}] - post decoder - `shared_states` shape: {shared_states.shape}"
      )

      cls_logits = self.cls_expert(
        shared_states[:, 0, :]
      )  # extract the latent classes & confidence

      # cache focus logits
      focus_logits_cache.append(focus_logits)

    if not getattr(self.config, "return_focus_logits", True):
      focus_logits_cache = []  # clear cache
    return cls_logits, focus_logits_cache

  @torch.no_grad()
  def _focus_on(
    self,
    pixel_values: FloatTensor,
    focus_logits: FloatTensor,
    focus_window_lookup: torch.Size,
  ) -> FloatTensor:
    batch_size, num_channels, pixel_height, pixel_width = pixel_values.size()
    focus_window_size = self.config.manifest_patch_size  # (32)
    # NOTE: currently based on the following config settings:
    # CONFIG (TEMPORARY):
    # input_image_size: 224x224
    # manifest_patch_size: 32x32; manifest_patch_stride: 32;
    # -> num_manifest_patches: 7x7 = 49 (num_windows)
    # ---- based on the above config, we set the following values for 'focus' ---- #
    # _focus_loc_len = 49  <-- preparing configuration for where to focus on
    # latent_patch_size: 4x4; latent_patch_stride: 4;
    # -> num_latent_patches: 8x8 = 64

    # NONSENSE (06-05-25): What would happen if the image size isn't set to (224x224)?
    #
    # Preliminary:
    #  This is a experimental note for that I accidentally set the image size to (256x256) instead of (224x224),
    # but trained surprisingly fluently without any errors! THIS IS WEIRD, MAKE ME CURIOUS.
    #
    # Note:
    #  Fine, After a thorough diagnose, this is nothing interesting... The `focus_expert` extract the latent focus location
    # in size of 49 whatever the original image size and the within the inner function __idx_to_window_ids(), the calculation
    # just simply views image size as (224x224) and the returned result limited this range, as well. Therefore the pixels
    # exceeding the original image size's range are just ignored. This is bugged and supposed to be fixed, pity though...
    #
    # Re-Note (06-05-2025):
    #  I experiment with the config-fixed version, but the result comes worse compared to the one above. I don't know why,
    # maybe caused by the batch_size or some other influential factors? weird...

    # FIXME (emer): try to build this method independent of the config settings above

    logger.debug(
      f"FocusModel._focus_on() - `focus_window_lookup`: {focus_window_lookup}"
    )

    # `pixel_values` is split to 49 patches and we will decide which one to 'focus' on based on the linear output `focus_logits`
    focus_window_idx = torch.argmax(
      nn.functional.softmax(focus_logits, dim=-1), dim=-1, keepdim=False
    )  # [batch_size,]
    logger.debug(f"FocusModel._focus_on() - `focus_window_idx`:\n{focus_window_idx}")

    def __idx_to_window_ids(window_idx: LongTensor) -> LongTensor:
      # FIXME: this method is supposed to use the passed info of manifest patches, currently we assign it with a default value,
      #  try to improve it in the future
      manifest_num_patches_width = 7
      manifest_patch_size = 32
      window_height_ids = window_idx // manifest_num_patches_width * manifest_patch_size
      window_width_ids = window_idx % manifest_num_patches_width * manifest_patch_size

      # initial code
      # ```python
      # window_height_ids = torch.stack(
      #   [torch.arange(start=height_id, end=height_id + manifest_patch_size) for height_id in window_height_ids]
      # )
      # window_width_ids = torch.stack(
      #   [torch.arange(start=width_id, end=width_id + manifest_patch_size) for width_id in window_width_ids]
      # )
      # ```

      # improved code
      # Create a range tensor for the patch offsets [0, 1, 2, ..., manifest_patch_size-1]
      offsets = torch.arange(manifest_patch_size, device=window_height_ids.device)

      # Generate window indices using broadcasting
      window_height_ids = window_height_ids.unsqueeze(1) + offsets
      window_width_ids = window_width_ids.unsqueeze(1) + offsets

      # TODO: improve the above code, seeming redundant and non-efficient
      # NOTE (06-09-25): Preliminarily improve the code, to avoid iterating over the tensors, which is not accepted by torch.jit
      # And the above improved code has been tested, see `{projectRoot}/utests/python_utests/models/perf/perf-__idx_to_window_ids.py`

      return (window_height_ids, window_width_ids)

    window_height_ids, window_width_ids = __idx_to_window_ids(focus_window_idx)
    logger.debug(f"FocusModel._focus_on() - `window_height_ids`:\n{window_height_ids}")
    logger.debug(f"FocusModel._focus_on() - `window_width_ids`:\n{window_width_ids}")

    focus_windows = pixel_values[
      torch.arange(batch_size, dtype=torch.int32, requires_grad=False)[
        :, None, None, None
      ],  # [batch_size, 1, 1, 1],
      torch.arange(num_channels, dtype=torch.int32, requires_grad=False)[
        None, :, None, None
      ],  # [1, num_channels, 1, 1],
      window_height_ids[
        :, None, :, None
      ],  # [batch_size, 1, manifest_num_patches_height, 1],
      window_width_ids[
        :, None, None, :
      ],  # [batch_size, 1, 1, manifest_num_patches_width],
    ].contiguous()  # FIXME: any way to improve this code?
    focus_windows = focus_windows.view(
      batch_size, num_channels, focus_window_size, focus_window_size
    )
    return focus_windows

  @torch.no_grad()
  def _get_manifest_position_ids(
    self, pixel_values: FloatTensor
  ) -> Tuple[LongTensor, torch.Size]:
    # TODO: docstring?
    # [batch_size, num_channels, height, width] -> [batch_size, height, width]
    pixel_shapes = list(
      [pixel_values.shape[0], pixel_values.shape[2], pixel_values.shape[2]]
    )  # avoid using tuple or torch.Size (subclass of tuple) for assignment issue

    # FIXME: the following operations have problems (e.g. `height` or `width` is not dividable by `patch_stride` after subtraction)
    # and we may improve the following codes in the future

    # assuming `height` = `width` = 224, `manifest_patch_size` = `manifest_patch_stride` = 32, then the new `pixel_shapes` = [batch_size, 7, 7]
    pixel_shapes[1] = (
      pixel_shapes[1] - self.config.manifest_patch_size
    ) // self.config.manifest_patch_stride + 1
    pixel_shapes[2] = (
      pixel_shapes[2] - self.config.manifest_patch_size
    ) // self.config.manifest_patch_stride + 1
    # -> [batch_size, manifest_len (UNKNOWN) = num_patches_height * num_patches_width]
    position_ids_shape = torch.Size(
      [pixel_shapes[0], pixel_shapes[1] * pixel_shapes[2]]
    )  # [batch_size, 49]
    logger.debug(
      f"FocusModel._get_manifest_position_ids() - `position_ids_shape`: {position_ids_shape}"
    )

    # -> [batch_size, seq_len' = shared_len + manifest_len (UNKNOWN)], with `shared_len` padded with 0
    position_ids = torch.arange(
      start=1,
      end=position_ids_shape[1] + 1,
      dtype=torch.int32,
      requires_grad=False,
    )  # `0` is for `shared_states`
    position_ids = torch.cat(
      [
        torch.zeros(self.config.shared_len, dtype=torch.int32, requires_grad=False),
        position_ids,
      ],
      dim=-1,
    )
    position_ids = position_ids[None, :].expand(pixel_shapes[0], -1)
    return position_ids

  def _log_focus_logits(self, focus_logits_cache: List[torch.Tensor]):
    self.logged_focus_logits.append(focus_logits_cache)

  def get_focus_logits(self) -> List[List[torch.Tensor]]:
    return self.logged_focus_logits


class FocusForClassification(nn.Module):
  def __init__(self, config: FocusConfig, collator: Optional[FocusCollator] = None):
    super().__init__()
    self.config = config
    self.collator = collator
    self.model = FocusModel(config, collator)

    self.cls_loss_fn = CrossEntropyLoss()
    self.con_loss_fn = CrossEntropyLoss()

    self.alpha = getattr(config, "alpha", 0.5)
    # TODO: init weights & final processing

  def forward(
    self,
    pixel_values: FloatTensor,
    cls_labels: Optional[LongTensor] = None,
    **kwargs,
  ) -> Tuple[
    Union[FloatTensor, None],
    Tuple[Union[FloatTensor, None], Union[FloatTensor, None]],
    Tuple[FloatTensor, FloatTensor],
  ]:
    batch_size = pixel_values.shape[0]
    cls_logits, focus_logits_cache = self.model(pixel_values)
    # `focus_logits` would be NoneType if `return_focus_logits` is configured False

    # declare and initiate before moving into next session
    loss = None
    cls_loss = None
    con_loss = None
    con_logits = None

    if cls_labels is not None:
      # ---- loss ---- #
      # Upcast to float to avoid precision issues
      cls_logits = cls_logits.float()
      cls_labels = cls_labels.long().to(cls_logits.device)
      logger.debug(
        f"FocusForClassification.forward() -- input `cls_logits` shape: {cls_logits.shape}"
      )

      # Slice to match the construction, contiguous() to enable view()
      # 03-04-25: this is not the final version thus we apply the following code to maintain its mutability
      con_logits = cls_logits[:, self.config.num_classes :].contiguous()
      cls_logits = cls_logits[:, : self.config.num_classes].contiguous()
      cls_labels = cls_labels.contiguous()

      # Collate Guide (03-04-25):
      # The final output of `cls_logits` includes classes (of size `config.num_classes`) and confidence (one-hot vector of size 2, deciding whether to continue focusing),
      # with the sum `config.num_classes` + 2 as the judging vector's size.
      # For which we gotta extend the labels for it before further processing.
      cls_results = torch.argmax(
        nn.functional.softmax(cls_logits, dim=-1), dim=-1, keepdim=False
      )  # [batch_size,]
      con_labels = (cls_results == cls_labels).long().to(cls_logits.device)
      logger.debug(
        f"FocusForClassification.forward() - `cls_logits` shape: {cls_logits.shape}"
      )
      logger.debug(
        f"FocusForClassification.forward() - `cls_logits[: 16]`:\n{cls_logits[:16]}"
      )
      logger.debug(
        f"FocusForClassification.forward() - `con_logits` shape: {con_logits.shape}"
      )
      logger.debug(
        f"FocusForClassification.forward() - `con_logits[: 16]`:\n{con_logits[:16]}"
      )
      logger.debug(
        f"FocusForClassification.forward() - `cls_labels` shape: {cls_labels.shape}"
      )
      logger.debug(
        f"FocusForClassification.forward() - `cls_labels[: 16]`:\n{cls_labels[:16]}"
      )
      logger.debug(
        f"FocusForClassification.forward() - `cls_results` shape: {cls_results.shape}"
      )
      logger.debug(
        f"FocusForClassification.forward() - `cls_results[: 16]`: {cls_results[:16]}"
      )
      logger.debug(
        f"FocusForClassification.forward() - `con_labels` shape: {con_labels.shape}"
      )
      logger.debug(
        f"FocusForClassification.forward() - `con_labels[: 16]`: {con_labels[:16]}"
      )

      # Recheck shapes
      cls_logits = cls_logits.view(batch_size, self.config.num_classes)
      con_logits = con_logits.view(batch_size, 2)
      cls_labels = cls_labels.view(batch_size)
      con_labels = con_labels.view(batch_size)

      # Calculate loss
      cls_loss = self.cls_loss_fn(cls_logits, cls_labels)
      con_loss = self.con_loss_fn(con_logits, con_labels)

      # alpha is a hyperparameter
      loss = self.alpha * cls_loss + (1 - self.alpha) * con_loss

    return loss, (cls_loss, con_loss), (cls_logits, con_logits)

  @torch.no_grad()
  def classify(self, pixel_values: FloatTensor) -> Tuple[Tensor, Tensor]:
    if (
      len(pixel_values.size()) < 4
    ):  # should be [batch_size, num_channels, height, width]
      # the following line is equivalent to set `batch_size` to 1
      pixel_values = pixel_values.unsqueeze(0)
      logger.debug(
        f"FocusForClassification.classify() - `pixel_values` shape: {pixel_values.shape}"
      )

    cls_logits, focus_logits_cache = self.model(pixel_values)
    cls_logits = cls_logits[:, : self.config.num_classes].contiguous()
    # Review for `cls_logits` and `focus_logits`, this is in reference to the comments within the `__init__` of FocusModel
    # cls_logits -- [1] classes (one-hot vector of size `config.num_classes`); [2] continue or break (confidence as a one-hot vector of size 2);
    # focus_logits -- inner variate, this defines the logic of `self._focus_on`: [1] index for in-order patches (TEMPORARY SOLUTION - NOTE: Currently set to 49);

    cls_results = torch.argmax(
      nn.functional.softmax(cls_logits, dim=-1), dim=-1, keepdim=False
    )  # [batch_size,]
    focus_window_idx_lst = [
      torch.argmax(
        nn.functional.softmax(singular_focus_logits, dim=-1),
        dim=-1,
        keepdim=False,
      )
      for singular_focus_logits in focus_logits_cache
    ]  # List[tensors of shape [batch_size,]], length is the iterated times

    return cls_results, focus_window_idx_lst


# modeling_focus.py ends here
