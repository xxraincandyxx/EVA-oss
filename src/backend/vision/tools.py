# tools.py

import cv2
import numpy as np

# ------------------------ #
# --- Helper Functions --- #
# ------------------------ #


def detect_cover(
  frame: np.ndarray,
  skin_ratio_thresh: float = 0.20,
  dark_ratio_thresh: float = 0.25,
  dark_value_thresh: int = 40,
) -> bool:
  """
  Decide whether the image is largely covered by:
    1) skin-colored objects (e.g. a hand over the lens), OR
    2) very dark/black regions (e.g. occlusion, lens cap, heavy shadows).

  Args:
      frame: BGR image from cv2.
      skin_ratio_thresh: fraction of image area; above this returns True for skin.
      dark_ratio_thresh: fraction of image area; above this returns True for dark.
      dark_value_thresh: grayscale threshold below which a pixel counts as "dark".

  Returns:
      True if either skin-colored pixels > skin_ratio_thresh
      OR dark pixels > dark_ratio_thresh.
  """
  h, w = frame.shape[:2]
  total = h * w

  # --- 1) Skin detection (HSV) ---
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  # a bit wider range for more sensitivity
  lower = np.array([0, 20, 40], dtype=np.uint8)
  upper = np.array([25, 200, 255], dtype=np.uint8)
  skin_mask = cv2.inRange(hsv, lower, upper)
  # clean up noise
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
  skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
  skin_ratio = cv2.countNonZero(skin_mask) / total

  if skin_ratio > skin_ratio_thresh:
    # too much skin-colored area!
    return True

  # --- 2) Dark/black area detection ---
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # mask of dark pixels
  dark_mask = cv2.threshold(gray, dark_value_thresh, 255, cv2.THRESH_BINARY_INV)[1]
  # optional: clean small specks
  dark_mask = cv2.morphologyEx(
    dark_mask,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
    iterations=1,
  )
  dark_ratio = cv2.countNonZero(dark_mask) / total

  if dark_ratio > dark_ratio_thresh:
    # too much darkness!
    return True

  # Otherwise—no big cover detected
  return False


def draw_boxes_and_resize(
  frame: np.ndarray, boxes: np.ndarray, output_size: tuple[int, int]
) -> np.ndarray:
  """
  Draw axis-aligned bounding boxes on the image and resize to a specified output size.

  Args:
      frame: BGR image from cv2.
      boxes: array-like of shape (N,4), each box as [x_min, y_min, x_max, y_max].
      output_size: (width, height) tuple for the resized image.

  Returns:
      new_frame: the image with rectangles drawn, then resized.
  """
  img = frame.copy()

  # Draw each box
  for x1, y1, x2, y2 in boxes:
    pt1 = (int(x1), int(y1))
    pt2 = (int(x2), int(y2))
    cv2.rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=2)

  # Resize (this may change aspect ratio; if you want to keep aspect use cv2.resize with interpolation and compute padding)
  new_frame = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
  return new_frame


def add_text_and_resize(
  frame: np.ndarray,
  text: str,
  font_scale: float,
  thickness: int,
  output_size: tuple[int, int],
  text_color: tuple[int, int, int] = (255, 255, 255),
  bg_color: tuple[int, int, int] = (0, 0, 0),
  padding: int = 10,
) -> np.ndarray:
  """
  Draws a text banner at the top of the image and resizes the result.

  Args:
      frame: BGR image from cv2.
      text: The text string to overlay.
      font_scale: Scale factor for text size.
      thickness: Thickness of the text strokes.
      output_size: (width, height) tuple for the final resized image.
      text_color: BGR color for the text (default white).
      bg_color: BGR color for the text background strip (default black).
      padding: Vertical padding in pixels above & below the text.

  Returns:
      new_frame: The image with text banner, then resized.
  """
  img = frame.copy()
  h, w = img.shape[:2]

  # Choose font
  font = cv2.FONT_HERSHEY_SIMPLEX

  # Get text size
  ((text_w, text_h), baseline) = cv2.getTextSize(text, font, font_scale, thickness)
  total_h = text_h + baseline + padding * 2

  # Draw background rectangle at top
  cv2.rectangle(img, (0, 0), (w, total_h), color=bg_color, thickness=-1)

  # Position text centered horizontally, with padding from top
  org = (int((w - text_w) / 2), padding + text_h)
  cv2.putText(
    img,
    text,
    org,
    font,
    font_scale,
    text_color,
    thickness,
    lineType=cv2.LINE_AA,
  )

  # Resize to desired output size
  new_frame = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
  return new_frame


def demo_foo():
  frame = None  # Gets cv2 image

  # 1. Check coverage
  if detect_cover(frame, skin_ratio_thresh=0.3):
    print("Frame is likely covered by hand!")
  else:
    print("Frame is OK.")

  # 2. Draw boxes and get resized output
  # Suppose you detected two regions:
  boxes = np.array(
    [
      [50, 60, 200, 300],
      [400, 100, 550, 350],
    ]
  )
  out_w, out_h = 640, 480
  boxed_resized = draw_boxes_and_resize(frame, boxes, (out_w, out_h))

  # Show result
  cv2.imshow("Boxes", boxed_resized)
  cv2.waitKey(1)


def demo_bar():
  ret, frame = None  # Gets cv2 image

  if not ret:
    # handle read error...
    pass

  # Add a sweet banner and resize to 800×600
  texted = add_text_and_resize(
    frame,
    text="Hello",
    font_scale=1.0,
    thickness=2,
    output_size=(800, 600),
    text_color=(255, 255, 255),  # white text
    bg_color=(0, 0, 0),  # black strip
    padding=15,
  )

  cv2.imshow("With Text", texted)
  cv2.waitKey(1)


# tools.py ends here
