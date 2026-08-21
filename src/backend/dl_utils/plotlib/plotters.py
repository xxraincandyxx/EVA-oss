# plotters.py

import io
import os
import warnings
from typing import List, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont

from .tools import denormalize_static

try:
  from backend.utils import get_logger
except Exception as e:
  warnings.warn(
    f"Failed to recognize eva as module, try relative import. Message - {e}"
  )
  from ...utils import get_logger

# The `name` is no needed 'cause the function will return
# the Eva Logger whatever the argument passed is
logger = get_logger()


# Helper function to make numpy array input valid
def _handle_ndarray_conversion(image_path_or_array):
  logger.debug_once(
    f"Input image_path_or_array - min: {image_path_or_array.min()}, max: {image_path_or_array.max()}"
  )
  if image_path_or_array.dtype == np.float32 or image_path_or_array.dtype == np.float64:
    if image_path_or_array.max() <= 1.0 and image_path_or_array.min() >= 0.0:
      image_path_or_array = (image_path_or_array * 255).astype(np.uint8)
    else:
      image_path_or_array = image_path_or_array.astype(np.uint8)
  elif image_path_or_array.dtype != np.uint8:
    image_path_or_array = image_path_or_array.astype(np.uint8)

  if image_path_or_array.ndim == 2:
    img = Image.fromarray(image_path_or_array).convert("RGB")
  elif image_path_or_array.ndim == 3 and image_path_or_array.shape[2] == 1:
    img = Image.fromarray(image_path_or_array.squeeze(), mode="L").convert("RGB")
  elif image_path_or_array.ndim == 3 and image_path_or_array.shape[2] == 4:
    img = Image.fromarray(image_path_or_array, mode="RGBA").convert("RGB")
  elif image_path_or_array.ndim == 3 and image_path_or_array.shape[2] == 3:
    img = Image.fromarray(image_path_or_array, mode="RGB")

  else:
    print(
      f"Error: Unsupported NumPy array shape or type: {image_path_or_array.shape}, dtype: {image_path_or_array.dtype}"
    )
    return None
  return img


# Helper function to convert various color inputs to a 3-element RGB int tuple
def _standardize_color_to_rgb(color_input, default_color_tuple=(255, 0, 0)):
  """Converts a color input to an (R, G, B) tuple of integers."""
  if isinstance(color_input, tuple):
    if len(color_input) == 3:  # (R, G, B)
      return tuple(int(c) for c in color_input)
    elif len(color_input) == 4:  # (R, G, B, A)
      return tuple(int(c) for c in color_input[:3])
    else:
      print(f"Warning: Invalid color tuple '{color_input}'. Using default.")
      return default_color_tuple
  elif isinstance(color_input, str):
    try:
      return ImageColor.getrgb(color_input)  # Parses names, hex, etc.
    except ValueError:
      print(f"Warning: Invalid color string '{color_input}'. Using default.")
      return default_color_tuple
  else:
    print(
      f"Warning: Invalid color type '{type(color_input)}' for '{color_input}'. Using default."
    )
    return default_color_tuple


def plot_image_with_boxes(
  image_path_or_array,
  boxes: List[Union[List, Tuple, torch.Tensor]],
  labels: Optional[Union[List[str], torch.Tensor]] = None,
  scores: Optional[Union[List[float], torch.FloatTensor]] = None,
  colors: Optional[Union[List[Union[str, Tuple]], str, Tuple]] = None,
  line_thickness: Optional[int] = 2,
  font_size: Optional[int] = 16,
  show_labels: Optional[bool] = True,
  show_scores: Optional[bool] = True,
  score_threshold: Optional[float] = 0.0,  # Minimum score to display a box
  figsize: Optional[Tuple] = (12, 12),
  title: Optional[str] = None,
  output_path: Optional[Union[str, os.PathLike]] = None,
  return_obj: Optional[bool] = False,
  return_obj_type: Optional[str] = "plt",  # or 'cv2'
) -> Union[None, Image.Image]:
  """
  Plots an image with bounding boxes, labels, and scores.

  Args:
      image_path_or_array (str or np.ndarray or PIL.Image.Image):
          Path to the image file, a NumPy array (H, W, C), or a PIL Image.
      boxes (list of lists/tuples):
          A list of bounding boxes, where each box is [x_min, y_min, x_max, y_max].
      labels (list of str, optional):
          A list of labels corresponding to each box. Defaults to None.
      scores (list of float, optional):
          A list of confidence scores (0-1) for each box. Defaults to None.
      colors (list of str or tuple, or str or tuple, optional):
          - A list of colors for each box (e.g., ['red', 'blue']).
          - A single color for all boxes (e.g., 'red' or (255,0,0)).
          - If None, unique colors will be generated for each class label if labels are provided,
            otherwise defaults to 'red'.
      line_thickness (int, optional):
          Thickness of the bounding box lines. Defaults to 2.
      font_size (int, optional):
          Font size for labels and scores. Defaults to 16.
      show_labels (bool, optional):
          Whether to display labels. Defaults to True.
      show_scores (bool, optional):
          Whether to display scores. Defaults to True.
      score_threshold (float, optional):
          Minimum score for a box to be displayed. Defaults to 0.0 (display all).
      figsize (tuple, optional):
          Figure size for matplotlib. Defaults to (12, 12).
      title (str, optional):
          Title for the plot. Defaults to None.
      output_path (str, optional):
          If provided, saves the image to this path instead of/in addition to showing it.
          The format is inferred from the extension (e.g., 'output.png').
      return_obj (bool, optional):
          If true, doesn't show the image but return the image object instead; Otherwise, show
          the plot directly.
      return_obj_type (str, optional):
          The type of the returned image object, currently support ['plt', 'cv2']; will not work
          if `return_obj` is not set or set to False; Defaults to 'plt'.
  """

  logger.info_once("--- Plot Image With Boxes ---")

  # Load Image and Preprocess Labels & Boxes
  image_size = None
  if isinstance(image_path_or_array, str):
    try:
      img = Image.open(image_path_or_array).convert("RGB")
    except FileNotFoundError:
      logger(f"Error: Image file not found at {image_path_or_array}")
      return
  elif isinstance(image_path_or_array, np.ndarray):
    img = _handle_ndarray_conversion()
  elif isinstance(image_path_or_array, torch.Tensor):
    if image_path_or_array.shape[0] == 3:
      img = image_path_or_array.permute(1, 2, 0)
    else:
      img = image_path_or_array

    image_size = tuple(img.size())  # [height, width, n_channels]
    img = denormalize_static(img)  # -> converted to np.ndarray
    logger.info(f"Plotter gets torch.Tensor type image with size: {image_size}")
    img = _handle_ndarray_conversion(img)
  elif isinstance(image_path_or_array, Image.Image):
    img = image_path_or_array.convert("RGB")
  else:
    logger.error("plot_image_with_boxes - Error: Invalid image_path_or_array type.")
    return

  if img is None:
    logger.error("plot_image_with_boxes - Error: Invalid image_path_or_array type.")
    return

  if isinstance(labels, torch.Tensor):
    labels = labels.tolist()

  if isinstance(boxes, torch.Tensor):
    logger.info_once(f"Plotter gets torch.Tensor type boxes with shape: {boxes.size()}")
    logger.info_once(f"Plotter gets torch.Tensor type boxes with Pre-peek:\n{boxes[0]}")
    boxes = boxes.numpy()
    if boxes.min() >= 0.0 and boxes.max() <= 1.0:
      for idx in range(boxes.shape[0]):
        boxes[idx, 0] = boxes[idx, 0] * image_size[1]  # <- width
        boxes[idx, 2] = boxes[idx, 2] * image_size[1]  # <- width
        boxes[idx, 1] = boxes[idx, 1] * image_size[0]  # <- height
        boxes[idx, 3] = boxes[idx, 3] * image_size[0]  # <- height
    logger.info_once(
      f"Plotter gets torch.Tensor type boxes with Post-peek:\n{boxes[0]}"
    )

    boxes = boxes.tolist()

  draw = ImageDraw.Draw(img)

  # Font (same as before)
  try:
    font_path = "arial.ttf"
    if not os.path.exists(font_path):
      possible_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "LiberationSans-Regular.ttf",
      ]
      for pf in possible_fonts:
        if os.path.exists(pf):
          font_path = pf
          break
      else:
        font_path = None
    font = (
      ImageFont.truetype(font_path, font_size)
      if font_path
      else ImageFont.load_default()
    )
  except IOError:
    print("Warning: Font file not found/unreadable. Using default PIL font.")
    font = ImageFont.load_default()

  # Colors - Pre-processing to RGB tuples
  default_rgb_tuple = _standardize_color_to_rgb("red")  # Default to red
  label_to_rgb_map = {}
  processed_color_list = None  # For when 'colors' is a list

  if colors is None:
    if labels is not None:
      unique_labels = sorted(list(set(labels)))
      num_unique_labels = len(unique_labels)
      palette_name = "tab20" if num_unique_labels <= 20 else "viridis"
      try:
        palette = cm.get_cmap(palette_name, num_unique_labels)
      except ValueError:
        print(f"Warning: Colormap '{palette_name}' not found. Using 'viridis'.")
        palette = cm.get_cmap("viridis", num_unique_labels)

      for i, label in enumerate(unique_labels):
        # cmap returns (R,G,B,A) floats in [0,1]
        color_rgba_float = palette(i)
        # Convert to (R,G,B) int tuple
        label_to_rgb_map[label] = tuple(int(c * 255) for c in color_rgba_float[:3])
    # If no labels, default_rgb_tuple will be used for all
  elif isinstance(colors, (str, tuple)):  # Single color for all boxes
    default_rgb_tuple = _standardize_color_to_rgb(colors, default_rgb_tuple)
  elif isinstance(colors, list):
    processed_color_list = [
      _standardize_color_to_rgb(c, default_rgb_tuple) for c in colors
    ]
  else:
    print(
      f"Warning: Invalid 'colors' argument type: {type(colors)}. Using default red."
    )
    # default_rgb_tuple is already set

  # Draw Boxes and Text
  for i, box in enumerate(boxes):
    current_score = scores[i] if scores and i < len(scores) else 1.0
    if current_score < score_threshold:
      continue

    x_min, y_min, x_max, y_max = map(int, box)

    # Determine the RGB tuple for this box
    box_rgb_color = default_rgb_tuple  # Start with the fallback/single color

    if colors is None:  # Auto-generated colors based on label
      if labels and i < len(labels) and labels[i] in label_to_rgb_map:
        box_rgb_color = label_to_rgb_map[labels[i]]
    elif processed_color_list:  # A list of colors was provided and processed
      box_rgb_color = processed_color_list[i % len(processed_color_list)]
    # If 'colors' was a single color string/tuple, box_rgb_color is already default_rgb_tuple (which was updated)

    # Now, box_rgb_color is guaranteed to be an (R,G,B) int tuple
    draw.rectangle(
      [x_min, y_min, x_max, y_max], outline=box_rgb_color, width=line_thickness
    )

    text_to_display = ""
    if show_labels and labels and i < len(labels):
      text_to_display += str(labels[i])
    if show_scores and scores and i < len(scores):
      score_str = f"{scores[i]:.2f}"
      if text_to_display:
        text_to_display += f" {score_str}"
      else:
        text_to_display += score_str

    if text_to_display:
      try:
        if hasattr(draw, "textbbox"):
          text_bbox = draw.textbbox((x_min, y_min), text_to_display, font=font)
          text_width, text_height = (
            text_bbox[2] - text_bbox[0],
            text_bbox[3] - text_bbox[1],
          )
        else:
          text_width, text_height = draw.textsize(text_to_display, font=font)
      except Exception:
        text_width, text_height = draw.textsize(text_to_display, font=font)

      text_bg_y_min = y_min - text_height - line_thickness
      text_bg_y_max = y_min - line_thickness
      if text_bg_y_min < 0:
        text_bg_y_min = y_max + line_thickness
        text_bg_y_max = y_max + text_height + line_thickness

      draw.rectangle(
        [
          x_min,
          text_bg_y_min,
          x_min + text_width + (line_thickness // 2),
          text_bg_y_max,
        ],
        fill=box_rgb_color,  # Use the RGB tuple directly
      )

      # Determine text color based on background (box_rgb_color) brightness
      # box_rgb_color is an (R,G,B) tuple here
      brightness = sum(box_rgb_color[:3])  # Sum R,G,B components
      # Threshold for brightness (midpoint of 0-765)
      if brightness > (255 * 3 / 2):
        text_fill_color_name = "black"  # Dark text on light background
      else:
        text_fill_color_name = "white"  # Light text on dark background

      # PIL's draw.text accepts color names or RGB tuples for 'fill'
      draw.text(
        (x_min + (line_thickness // 2), text_bg_y_min),
        text_to_display,
        fill=text_fill_color_name,  # "black" or "white"
        font=font,
      )

  # Display/Save Image (same as before)
  plt.figure(figsize=figsize)
  if title:
    plt.title(title)
  plt.imshow(np.array(img))
  plt.axis("off")
  if output_path:
    try:
      output_dir = os.path.dirname(output_path)
      if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
      plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
      print(f"Image saved to {output_path}")
    except Exception as e:
      logger.error(f"Error saving image to {output_path}: {e}")
      warnings.warn(f"Error saving image to {output_path}: {e}")

  # Handle display vs return object
  if not return_obj:
    plt.show()
    return None
  else:
    if return_obj_type == "plt":
      # Save the plot to a buffer and return as PIL Image
      buf = io.BytesIO()
      plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.0)
      buf.seek(0)
      pil_img = Image.open(buf)
      plt.close()  # Close the figure to free memory
      return pil_img
    elif return_obj_type == "cv2":
      # Save to buffer and convert to OpenCV format (BGR numpy array)
      buf = io.BytesIO()
      plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.0)
      buf.seek(0)
      pil_img = Image.open(buf)

      # Convert to numpy array and handle alpha channel
      if pil_img.mode == "RGBA":
        pil_img = pil_img.convert("RGB")
      np_img = np.array(pil_img)
      plt.close()  # Close the figure to free memory

      # Convert RGB to BGR for OpenCV
      return np_img[:, :, ::-1].copy()
    else:
      logger.error(
        f"Currently `return_obj_type` supports only 'plt' or 'cv2', but got ({return_obj_type})."
      )
      warnings.warn(
        f"Currently `return_obj_type` supports only 'plt' or 'cv2', but got ({return_obj_type})."
      )
      plt.close()
      return None


# ----------------------------- #
# --- Guide / Example Usage --- #
# ----------------------------- #


def plot_demo():
  # Create a dummy image for testing
  dummy_img_array = np.zeros((300, 400, 3), dtype=np.uint8)
  dummy_img_array[50:150, 100:200, 0] = 200  # Red patch
  dummy_img_array[180:280, 250:350, 1] = 200  # Green patch
  dummy_img_array[:, :, 2] = 50  # Blueish background

  # Example 1: Simple boxes
  print("\n--- Example 1: Simple ---")
  boxes1 = [
    [50, 50, 150, 150],  # xyxy format
    [200, 80, 300, 180],
  ]
  plot_image_with_boxes(dummy_img_array, boxes1, title="Example 1: Simple Boxes")

  # Example 2: With labels and scores
  print("\n--- Example 2: Labels & Scores ---")
  boxes2 = [[50, 50, 150, 150], [200, 80, 300, 180], [10, 10, 80, 80]]
  labels2 = ["cat", "dog", "cat"]
  scores2 = [0.95, 0.88, 0.72]
  plot_image_with_boxes(
    dummy_img_array,
    boxes2,
    labels=labels2,
    scores=scores2,
    title="Example 2: With Labels and Scores",
    font_size=12,
  )

  # Example 3: Custom colors and score threshold
  print("\n--- Example 3: Custom Colors & Threshold ---")
  boxes3 = [
    [50, 50, 150, 150],
    [200, 80, 300, 180],
    [180, 250, 280, 290],  # This one is on the green patch
  ]
  labels3 = ["person", "car", "person"]
  scores3 = [0.99, 0.65, 0.45]  # Last box should be filtered out
  colors3 = ["magenta", "cyan", "yellow"]  # Explicit list of colors
  plot_image_with_boxes(
    dummy_img_array,
    boxes3,
    labels=labels3,
    scores=scores3,
    colors=colors3,
    score_threshold=0.5,
    line_thickness=3,
    title="Example 3: Custom Colors & Score Threshold (0.5)",
  )

  # Example 4: Single custom color for all boxes
  print("\n--- Example 4: Single Custom Color ---")
  plot_image_with_boxes(
    dummy_img_array,
    boxes1,
    colors="lime",
    title="Example 4: Single Custom Color (Lime)",
    labels=labels2[:2],
  )

  # Example 5: Auto-colors based on labels, saving to file
  print("\n--- Example 5: Auto-colors & Save ---")
  # Using Pillow image directly
  pil_image = Image.fromarray(dummy_img_array)
  plot_image_with_boxes(
    pil_image,
    boxes2,
    labels=labels2,
    scores=scores2,
    title="Example 5: Auto-colors & Save",
    output_path="output_detections.png",
  )

  # Example 6: Test with a real image (replace 'path/to/your/image.jpg' with an actual image)
  # This will likely fail if you don't provide a real image path
  print("\n--- Example 6: Real Image (if path provided) ---")
  real_image_path = "test_image.jpg"  # <<-- REPLACE WITH YOUR IMAGE PATH
  # Create a dummy file if it doesn't exist for the example to run without error
  if not os.path.exists(real_image_path):
    try:
      Image.fromarray(dummy_img_array).save(real_image_path)
      print(
        f"Created dummy '{real_image_path}' for testing. Replace it with your own image."
      )
    except Exception as e:
      print(
        f"Could not create dummy image '{real_image_path}': {e}. Skipping Example 6."
      )
      real_image_path = None

  if real_image_path and os.path.exists(real_image_path):
    real_boxes = [
      [100, 100, 200, 250],  # Adjust these to your image content
      [300, 50, 450, 180],
    ]
    real_labels = ["object1", "object2"]
    real_scores = [0.92, 0.78]
    plot_image_with_boxes(
      real_image_path,
      real_boxes,
      labels=real_labels,
      scores=real_scores,
      title="Example 6: Detections on Real Image",
      output_path="real_image_detections.png",
    )
  else:
    print(
      f"Skipping Example 6 as '{real_image_path}' was not found or couldn't be created."
    )


if __name__ == "__main__":
  # dataclass_demo()
  plot_demo()


# plotters.py ends here
