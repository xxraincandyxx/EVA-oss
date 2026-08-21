# fatuus._yolo

import glob
import os
import random
import warnings
from typing import List, Optional, Union

import albumentations as A
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import CACHE_DIR

try:
  from backend.utils import get_logger
except Exception as e:
  warnings.warn(
    f"Failed to recognize eva as module, try relative import. Message - {e}"
  )
  from ..eva.utils import get_logger

# The `name` is no needed 'cause the function will return
# the Eva Logger whatever the argument passed is
logger = get_logger()


class ObjectTransforms:
  def __init__(
    self,
    image_size=(640, 640),
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    max_pixel_values=255.0,
  ):
    self.image_size = image_size
    self.mean = mean
    self.std = std
    self.max_pixel_values = max_pixel_values

  def get_object_detection_transforms(self, is_train):
    """
    Defines transformations for object detection.
    Args:
        image_size (tuple): Target image size (height, width).
        is_train (bool): If True, applies training augmentations.
        mean (tuple): Normalization mean.
        std (tuple): Normalization standard deviation.
    Returns:
        A.Compose: Albumentations composition of transforms.
    """
    bbox_params = A.BboxParams(
      format="yolo",  # Our input labels are in YOLO format [x_c, y_c, w, h] normalized
      label_fields=["category_ids"],
      min_visibility=0.2,  # Fraction of bbox area that must be visible after transform
      min_area=100,  # Minimum bbox area in pixels (after resizing) to keep the box
      # This depends on image_size, adjust if needed
    )

    if is_train:
      transforms = [
        # Geometric transforms
        A.RandomSizedBBoxSafeCrop(
          height=self.image_size[0],
          width=self.image_size[1],
          erosion_rate=0.2,
          p=0.5,
        ),
        A.Resize(
          height=self.image_size[0], width=self.image_size[1]
        ),  # Ensure final size
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
          shift_limit=0.05,
          scale_limit=0.1,
          rotate_limit=15,
          p=0.5,
          border_mode=0,
          # value=self.mean,
        ),  # Use mean for padding
        # Color/Pixel-level transforms
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(
          hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3
        ),
        A.ISONoise(p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        A.Normalize(
          mean=self.mean, std=self.std, max_pixel_value=self.max_pixel_values
        ),
        ToTensorV2(),
      ]
    else:  # Validation/Test
      transforms = [
        A.Resize(height=self.image_size[0], width=self.image_size[1]),
        A.Normalize(
          mean=self.mean, std=self.std, max_pixel_value=self.max_pixel_values
        ),
        ToTensorV2(),
      ]

    # NOTE: we do not like the transforms above, thus a override here
    transforms = [
      A.Resize(height=self.image_size[0], width=self.image_size[1]),
      # A.HorizontalFlip(p=0.5),  <- if this is enabled, then the labels would not be correct
      A.Normalize(mean=self.mean, std=self.std, max_pixel_value=self.max_pixel_values),
      ToTensorV2(),
    ]
    return A.Compose(transforms, bbox_params=bbox_params)

  def get_object_detection_denormalize(
    self, normalized_image: Union[torch.Tensor, np.ndarray]
  ) -> np.ndarray:
    """Reverts the normalization on an image"""
    if isinstance(normalized_image, torch.Tensor):
      # If tensor is on GPU, move to CPU. Permute from (C, H, W) to (H, W, C).
      normalized_image = normalized_image.cpu().numpy().transpose(1, 2, 0)

    if isinstance(normalized_image, torch.Tensor):
      normalized_image = normalized_image.numpy()
    elif isinstance(normalized_image, List):
      normalized_image = np.array(normalized_image)

    mean = np.array(self.mean)
    std = np.array(self.std)
    # Multiply by std and add mean (then scale back to 0-255)
    denormalized = (normalized_image * std + mean) * self.max_pixel_values
    # Clip to ensure valid pixel range
    denormalized = np.clip(denormalized, 0, self.max_pixel_values)
    return denormalized.astype(np.uint8)


def object_detection_collate_fn(batch):
  """
  Custom collate function for object detection.
  Filters out None samples (from loading errors) and handles varying numbers of boxes.
  Args:
      batch: list of (image, target) tuples.
  Returns:
      images: torch.Tensor of stacked images.
      targets: list of target dictionaries.
  """
  # Filter out None samples
  batch = [sample for sample in batch if sample[0] is not None]
  if not batch:  # If all samples in batch were None
    return None, None

  images = torch.stack([item[0] for item in batch], 0)
  targets = [item[1] for item in batch]
  return images, targets


class CustomObjectDetectionDataset(Dataset):
  def __init__(
    self,
    image_dir,
    label_dir,
    image_filenames,
    transform=None,
    image_load_mode="RGB",
  ):
    """
    Args:
        image_dir (str): Directory with all images.
        label_dir (str): Directory with all label files.
        image_filenames (list): List of image filenames (without extension) to include in this dataset split.
        transform (callable, optional): Optional transform to be applied on a sample.
        image_load_mode (str): 'RGB', 'L' (grayscale), etc. for PIL.
    """
    self.image_dir = image_dir
    self.label_dir = label_dir
    self.image_filenames = image_filenames  # e.g., ['img1', 'img2']
    self.transform = transform
    self.image_load_mode = image_load_mode

  def __len__(self):
    return len(self.image_filenames)

  def __getitem__(self, idx):
    img_name_no_ext = self.image_filenames[idx]
    img_path = os.path.join(self.image_dir, img_name_no_ext + ".jpg")
    label_path = os.path.join(self.label_dir, img_name_no_ext + ".txt")

    try:
      # --- Image Loading ---
      image = Image.open(img_path).convert(self.image_load_mode)
      image_np = np.array(image)  # Albumentations prefers numpy arrays
    except Exception as e:
      print(f"Error loading image {img_path}: {e}")
      # Return None to be filtered by collate_fn or handle error differently
      return None, None

    # --- Label Loading & Processing ---
    boxes = []  # Store [class_id, x_center, y_center, width, height]

    if os.path.exists(label_path):
      with open(label_path, "r") as f:
        for line in f:
          line = line.strip()
          if not line:
            continue
          parts = line.split()
          try:
            class_id = int(parts[0])
            # YOLO format: x_center, y_center, width, height (all normalized 0-1)
            x_c, y_c, w, h = map(float, parts[1:5])
            boxes.append(
              [x_c, y_c, w, h, class_id]
            )  # Keep class_id at the end for Albumentations
          except ValueError:
            logger.warning(f"Warning: Malformed line in {label_path}: '{line}'")
            continue
    # else:
    # print(f"Warning: Label file not found {label_path}, assuming no objects.")
    # No objects in this image if file doesn't exist or is empty

    # Convert to numpy array for albumentations
    # Albumentations expects bboxes: list of [x_min, y_min, x_max, y_max] or [x_c, y_c, w, h]
    # And category_ids: list of [category_id]
    if boxes:
      boxes_np = np.array(boxes, dtype=np.float32)
      # Separate bboxes (coordinates) and category_ids for Albumentations
      bboxes_coords = boxes_np[:, :4].tolist()  # List of [x_c, y_c, w, h]
      category_ids = boxes_np[:, 4].astype(int).tolist()  # List of [class_id]
    else:
      bboxes_coords = []
      category_ids = []

    # --- Apply Transformations ---
    target = {}  # Standard PyTorch target format for detection
    if self.transform:
      try:
        transformed = self.transform(
          image=image_np, bboxes=bboxes_coords, category_ids=category_ids
        )
        image_tensor = transformed["image"]

        # Convert transformed bboxes (still YOLO) to [xmin, ymin, xmax, ymax]
        # This is a common format expected by models like Faster R-CNN, DETR
        # If your model expects YOLO format directly, you can skip this conversion.
        transformed_bboxes_yolo = transformed["bboxes"]
        transformed_labels = transformed["category_ids"]

        # If no boxes after transform (e.g., all cropped out)
        if not transformed_bboxes_yolo:
          target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
          target["labels"] = torch.zeros(0, dtype=torch.int64)
        else:
          # Convert YOLO [xc, yc, w, h] to [xmin, ymin, xmax, ymax]
          final_boxes_xyxy = []
          for box_yolo in transformed_bboxes_yolo:
            xc, yc, w, h_box = box_yolo
            x1 = xc - w / 2  # * w_img # If you need pixel coords
            y1 = yc - h_box / 2  # * h_img
            x2 = xc + w / 2  # * w_img
            y2 = yc + h_box / 2  # * h_img
            final_boxes_xyxy.append([x1, y1, x2, y2])

          target["boxes"] = torch.as_tensor(final_boxes_xyxy, dtype=torch.float32)
          target["labels"] = torch.as_tensor(transformed_labels, dtype=torch.int64)

      except Exception as e:
        print(f"Error applying transform to {img_path}: {e}")
        # Fallback or error handling
        return None, None  # Filtered by collate_fn
    else:
      # If no transform, just convert image to tensor and format labels
      # This path is less common for training deep models
      image_tensor = ToTensorV2()(image=image_np)["image"]  # Basic conversion
      if not bboxes_coords:  # No objects
        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        target["labels"] = torch.zeros(0, dtype=torch.int64)
      else:
        # Convert YOLO to xyxy
        final_boxes_xyxy = []
        for box_yolo in bboxes_coords:
          xc, yc, w, h_box = box_yolo
          x1, y1 = xc - w / 2, yc - h_box / 2
          x2, y2 = xc + w / 2, yc + h_box / 2
          final_boxes_xyxy.append([x1, y1, x2, y2])
        target["boxes"] = torch.as_tensor(final_boxes_xyxy, dtype=torch.float32)
        target["labels"] = torch.as_tensor(category_ids, dtype=torch.int64)

    return image_tensor, target


class ObjectDetectionDataModule(pl.LightningDataModule):
  def __init__(
    self,
    image_dir: str,
    label_dir: str,
    image_size: tuple = (640, 640),
    train_val_test_split_ratio: tuple = (0.7, 0.15, 0.15),
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    random_seed: int = 42,
  ):
    super().__init__()
    self.save_hyperparameters()

    self.image_dir = image_dir
    self.label_dir = label_dir
    self.image_size = image_size
    self.train_val_test_split_ratio = train_val_test_split_ratio
    self.batch_size = batch_size
    if num_workers is not None:
      self.num_workers = num_workers
    else:
      self.num_workers = os.cpu_count() if os.cpu_count() is not None else 0
    self.pin_memory = pin_memory
    self.persistent_workers = persistent_workers if self.num_workers > 0 else False
    self.random_seed = random_seed

    self.object_transforms = ObjectTransforms(image_size=self.image_size)

    self.train_transforms = self.object_transforms.get_object_detection_transforms(
      is_train=True
    )
    self.val_test_transforms = self.object_transforms.get_object_detection_transforms(
      is_train=False
    )

    self.dataset_train = None
    self.dataset_val = None
    self.dataset_test = None
    self.all_image_filenames_no_ext = []

  def prepare_data(self):
    """Checks if data directories exist."""
    if not os.path.isdir(self.image_dir):
      raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
    if not os.path.isdir(self.label_dir):
      raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

  def setup(self, stage: Optional[str] = None):
    """
    Load data paths and create train/val/test splits.
    This method is called by Lightning with populat_available_datasets=True,
    so it's safe to assign datasets here.
    """
    # Find all .jpg images and derive corresponding .txt label files
    # Only consider images that have a corresponding label file (or vice-versa, user choice)

    # Get all image basenames (filename without extension)
    image_paths = glob.glob(os.path.join(self.image_dir, "*.jpg"))
    if not image_paths:
      raise ValueError(f"No .jpg images found in {self.image_dir}")

    all_image_filenames_no_ext = sorted(
      [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
    )

    # Filter: Keep only images for which a label file also exists
    # This ensures consistency.
    self.all_image_filenames_no_ext = [
      fname
      for fname in all_image_filenames_no_ext
      if os.path.exists(os.path.join(self.label_dir, fname + ".txt"))
    ]

    if not self.all_image_filenames_no_ext:
      raise ValueError(
        f"No image/label pairs found. Checked {len(all_image_filenames_no_ext)} potential images."
      )

    # Shuffle once for reproducible splits
    random.Random(self.random_seed).shuffle(self.all_image_filenames_no_ext)

    # Split filenames
    n_total = len(self.all_image_filenames_no_ext)
    n_train = int(n_total * self.train_val_test_split_ratio[0])
    n_val = int(n_total * self.train_val_test_split_ratio[1])

    train_files = self.all_image_filenames_no_ext[:n_train]
    val_files = self.all_image_filenames_no_ext[n_train : n_train + n_val]
    test_files = self.all_image_filenames_no_ext[n_train + n_val :]

    if stage == "fit" or stage is None:
      self.dataset_train = CustomObjectDetectionDataset(
        image_dir=self.image_dir,
        label_dir=self.label_dir,
        image_filenames=train_files,
        transform=self.train_transforms,
      )
      self.dataset_val = CustomObjectDetectionDataset(
        image_dir=self.image_dir,
        label_dir=self.label_dir,
        image_filenames=val_files,
        transform=self.val_test_transforms,
      )
      logger.info(f"Train dataset size: {len(self.dataset_train)}")
      logger.info(f"Validation dataset size: {len(self.dataset_val)}")

    if stage == "test" or stage is None:
      self.dataset_test = CustomObjectDetectionDataset(
        image_dir=self.image_dir,
        label_dir=self.label_dir,
        image_filenames=test_files,
        transform=self.val_test_transforms,
      )
      logger.info(
        f"Test dataset size: {len(self.dataset_test) if self.dataset_test else 0}"
      )

    if not self.dataset_train and not self.dataset_val and not self.dataset_test:
      warnings.warn("No datasets were setup. Check stage or data.")

  def train_dataloader(self):
    if not self.dataset_train:
      self.setup("fit")
    return DataLoader(
      self.dataset_train,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=self.num_workers,
      pin_memory=self.pin_memory,
      persistent_workers=self.persistent_workers,
      collate_fn=object_detection_collate_fn,  # IMPORTANT
      drop_last=True,  # Can be useful for stability
    )

  def val_dataloader(self):
    if not self.dataset_val:
      self.setup("fit")
    return DataLoader(
      self.dataset_val,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=self.num_workers,
      pin_memory=self.pin_memory,
      persistent_workers=self.persistent_workers,
      collate_fn=object_detection_collate_fn,  # IMPORTANT
    )

  def test_dataloader(self):
    if not self.dataset_test:
      self.setup("test")
    return DataLoader(
      self.dataset_test,
      batch_size=self.batch_size,
      shuffle=False,
      num_workers=self.num_workers,
      pin_memory=self.pin_memory,
      persistent_workers=self.persistent_workers,
      collate_fn=object_detection_collate_fn,  # IMPORTANT
    )


# --- Main training script / test --- #
def dataclass_demo():
  # Create dummy data for testing
  IMAGE_DIR = os.path.join(CACHE_DIR, "dummy_od_images")
  LABEL_DIR = os.path.join(CACHE_DIR, "dummy_od_labels")
  os.makedirs(IMAGE_DIR, exist_ok=True)
  os.makedirs(LABEL_DIR, exist_ok=True)

  # Create 5 dummy images and labels
  for i in range(5):
    img_name = f"image_{i + 1}"
    # Create a dummy JPG image
    try:
      dummy_img = Image.new(
        "RGB",
        (np.random.randint(200, 800), np.random.randint(200, 800)),
        color=(
          np.random.randint(0, 255),
          np.random.randint(0, 255),
          np.random.randint(0, 255),
        ),
      )
      dummy_img.save(os.path.join(IMAGE_DIR, img_name + ".jpg"))
    except Exception as e:
      print(f"Error creating dummy image {img_name}.jpg: {e}")
      continue

    # Create a dummy TXT label file
    with open(os.path.join(LABEL_DIR, img_name + ".txt"), "w") as f:
      num_objects = np.random.randint(0, 4)  # 0 to 3 objects
      for _ in range(num_objects):
        class_id = np.random.randint(0, 3)  # 3 classes (0, 1, 2)
        x_c = np.random.uniform(0.1, 0.9)
        y_c = np.random.uniform(0.1, 0.9)
        w = np.random.uniform(0.05, 0.5)  # width relative to image width
        h = np.random.uniform(0.05, 0.5)  # height relative to image height
        # Ensure box is within image (roughly)
        x_c = np.clip(x_c, w / 2 + 0.01, 1 - w / 2 - 0.01)
        y_c = np.clip(y_c, h / 2 + 0.01, 1 - h / 2 - 0.01)
        f.write(f"{class_id} {x_c:.4f} {y_c:.4f} {w:.4f} {h:.4f}\n")
    print(f"Created dummy data for {img_name}")

  print(f"Contents of {IMAGE_DIR}: {os.listdir(IMAGE_DIR)}")
  print(f"Contents of {LABEL_DIR}: {os.listdir(LABEL_DIR)}")

  # Configure and test the DataModule
  data_module = ObjectDetectionDataModule(
    image_dir=IMAGE_DIR,
    label_dir=LABEL_DIR,
    image_size=(320, 320),  # Smaller for faster test
    train_val_test_split_ratio=(
      0.6,
      0.2,
      0.2,
    ),  # For 5 images: 3 train, 1 val, 1 test
    batch_size=2,
    num_workers=0,  # Set to 0 for easier debugging, >0 for performance
    random_seed=123,
  )

  data_module.prepare_data()
  data_module.setup(stage="fit")  # Setup for training and validation

  print("\n--- Testing Train Dataloader ---")
  train_loader = data_module.train_dataloader()
  if train_loader and len(train_loader) > 0:
    for i, batch in enumerate(train_loader):
      if batch[0] is None:  # Skipped batch due to all samples failing
        print(f"Batch {i + 1} was skipped (all samples failed to load/transform).")
        continue
      images, targets = batch
      print(f"\nTrain Batch {i + 1}:")
      print(
        "Images shape:", images.shape
      )  # e.g., torch.Size([batch_size, 3, 320, 320])
      print("Targets (list of dicts, length):", len(targets))
      for t_idx, T_item in enumerate(targets):
        print(
          f"  Target {t_idx}: boxes shape: {T_item['boxes'].shape}, labels shape: {T_item['labels'].shape}"
        )
        print(f"    Boxes (xyxy): {T_item['boxes']}")
        print(f"    Labels: {T_item['labels']}")
      if i == 0:
        break  # Just show first batch
  else:
    print("Train Dataloader is empty or None.")

  print("\n--- Testing Validation Dataloader ---")
  val_loader = data_module.val_dataloader()
  if val_loader and len(val_loader) > 0:
    for i, batch in enumerate(val_loader):
      if batch[0] is None:
        print(f"Batch {i + 1} was skipped.")
        continue
      images, targets = batch
      print(f"\nVal Batch {i + 1}:")
      print("Images shape:", images.shape)
      print("Targets (list of dicts, length):", len(targets))
      if i == 0:
        break
  else:
    print("Validation Dataloader is empty or None.")

  # To cleanup dummy data
  # import shutil
  # shutil.rmtree(IMAGE_DIR)
  # shutil.rmtree(LABEL_DIR)


# fatuus._yolo ends here
