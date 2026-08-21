# tools.py

import math
import warnings
from typing import List, Tuple, Union

import numpy as np
import torch

# --- Helper Functions for Graceful Plotting --- #


# this method doesn't support non-squared images currently
def window_to_xyxy_box(
  window_idx: int,
  image_size: Tuple[int, int] = (224, 224),
  num_windows: int = 49,
) -> Union[Tuple[int], None]:
  side_num = int(math.sqrt(num_windows))
  if side_num**2 != num_windows:
    warnings.warn(f"`num_windows` ({num_windows}) must be a perfect square.")
    return None

  side_len = image_size[0] // side_num

  x1 = (window_idx // side_num) * side_len
  y1 = (window_idx % side_num) * side_len
  x2 = x1 + side_len
  y2 = y1 + side_len

  return (x1, y1, x2, y2)


def darken_color(color, factor=0.2):
  """
  Returns a darker version of the input color.

  Parameters:
  - color: Input color as hex string (e.g., "#RRGGBB") or RGB tuple (e.g., (R, G, B))
  - factor: Darkening factor (0-1), where 0 is no change and 1 is completely black

  Returns:
  - Darker color in the same format as input
  """

  # Handle hex color input
  if isinstance(color, str) and color.startswith("#"):
    # Convert hex to RGB
    hex_color = color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    # Darken each channel
    r = max(0, int(r * (1 - factor)))
    g = max(0, int(g * (1 - factor)))
    b = max(0, int(b * (1 - factor)))

    # Convert back to hex
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

  # Handle RGB tuple input
  elif isinstance(color, (tuple, list)) and len(color) in (3, 4):
    # Darken each RGB channel (ignore alpha if present)
    r = max(0, int(color[0] * (1 - factor)))
    g = max(0, int(color[1] * (1 - factor)))
    b = max(0, int(color[2] * (1 - factor)))

    # Return RGB or RGBA depending on input
    if len(color) == 4:
      return (r, g, b, color[3])
    return (r, g, b)

  else:
    raise ValueError("Input color must be hex string or RGB tuple")


def lighten_color(color, factor=0.2):
  """
  Returns a lighter version of the input color.

  Parameters:
  - color: Input color as hex string (e.g., "#RRGGBB") or RGB tuple (e.g., (R, G, B))
  - factor: Lightening factor (0-1), where 0 is no change and 1 is completely white

  Returns:
  - Lighter color in the same format as input
  """

  # Handle hex color input
  if isinstance(color, str) and color.startswith("#"):
    # Convert hex to RGB
    hex_color = color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    # Lighten each channel
    r = min(255, int(r + (255 - r) * factor))
    g = min(255, int(g + (255 - g) * factor))
    b = min(255, int(b + (255 - b) * factor))

    # Convert back to hex
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

  # Handle RGB tuple input
  elif isinstance(color, (tuple, list)) and len(color) in (3, 4):
    # Lighten each RGB channel (ignore alpha if present)
    r = min(255, int(color[0] + (255 - color[0]) * factor))
    g = min(255, int(color[1] + (255 - color[1]) * factor))
    b = min(255, int(color[2] + (255 - color[2]) * factor))

    # Return RGB or RGBA depending on input
    if len(color) == 4:
      return (r, g, b, color[3])
    return (r, g, b)

  else:
    raise ValueError("Input color must be hex string or RGB tuple")


def denormalize_static(
  normalized_image: Union[torch.Tensor, np.ndarray],
  mean=(0.485, 0.456, 0.406),
  std=(0.229, 0.224, 0.225),
  max_pixel_values=255.0,
) -> np.ndarray:
  """Reverts the normalization on an image

  This is only a temporal solution, given the arguments/attributes
  initialized within the class cannot affect this static method.

  TODO: fix the above problem
  """

  if isinstance(normalized_image, torch.Tensor):
    normalized_image = normalized_image.numpy()
  elif isinstance(normalized_image, List):
    normalized_image = np.array(normalized_image)

  mean = np.array(mean)
  std = np.array(std)
  # Multiply by std and add mean (then scale back to 0-255)
  denormalized = (normalized_image * std + mean) * max_pixel_values
  # Clip to ensure valid pixel range
  denormalized = np.clip(denormalized, 0, max_pixel_values)
  return denormalized.astype(np.uint8)


# tools.py ends here
