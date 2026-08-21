# fatuus._imagenet

import glob
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

is_standalone = True  # is running itself independently
try:
  from . import CACHE_DIR

  try:
    from backend.utils import get_logger
    from backend.utils.abstracts import LightningModelInputs
  except Exception as e:
    warnings.warn(
      f"Failed to recognize eva as module, try relative import. Message - {e}"
    )
    from ..eva.utils import get_logger
    from ..eva.utils.abstracts import LightningModelInputs

  logger = get_logger()
  is_standalone = False  # not running independently

except Exception as e:
  DATA_DIR = str(
    Path(
      os.getenv(
        "EVA_TRAINING_DATA_DIR", Path.home() / ".local" / "share" / "eva" / "training"
      )
    ).expanduser()
  )
  CACHE_DIR = str(
    Path(
      os.getenv(
        "EVA_TRAINING_CACHE_DIR",
        Path.home() / ".cache" / "eva" / "training",
      )
    ).expanduser()
  )

  import logging
  import sys

  # --- Create a local logger for debugging ---
  logger = logging.getLogger("_imagenet")
  logger.setLevel(logging.INFO)
  stream_handler = logging.StreamHandler(sys.stderr)
  formatter = logging.Formatter(
    "%(asctime)s %(levelname)-8s %(threadName)s:%(process)d [%(filename)s:%(funcName)s():%(lineno)d] %(message)s"
  )
  stream_handler.setFormatter(formatter)
  logger.addHandler(stream_handler)
  logger.info(
    f"Detected running this as a DEMO, deprecate relative import. Message - {e}"
  )


# --- Image Transformation Class ---
# This class defines the image preprocessing and augmentation pipelines.


class ImageTransforms:
  """
  Manages and provides image transformations for classification tasks.
  """

  def __init__(
    self,
    image_size: tuple = (224, 224),
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
    max_pixel_value: float = 255.0,
  ):
    self.image_size = image_size
    self.mean = mean
    self.std = std
    self.max_pixel_value = max_pixel_value

  def get_classification_transforms(self, is_train: bool) -> A.Compose:
    """
    Defines and returns transformations for image classification.

    Args:
        is_train (bool): If True, applies training augmentations.

    Returns:
        A.Compose: Albumentations composition of transforms.
    """
    if is_train:
      # Augmentations for the training set
      transforms = [
        # Resize the image while maintaining aspect ratio, then crop to the target size.
        # This is often better than a direct, distorting resize.
        A.LongestMaxSize(max_size=max(self.image_size), p=1.0),
        A.PadIfNeeded(
          min_height=self.image_size[0],
          min_width=self.image_size[1],
          border_mode=0,  # cv2.BORDER_CONSTANT
          value=0,  # Pad with black
        ),
        A.RandomCrop(height=self.image_size[0], width=self.image_size[1], p=1.0),
        # Geometric and color augmentations
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
          shift_limit=0.05,
          scale_limit=0.1,
          rotate_limit=15,
          p=0.3,
          border_mode=0,
          value=0,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(
          hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3
        ),
        # Normalization and conversion to tensor
        A.Normalize(mean=self.mean, std=self.std, max_pixel_value=self.max_pixel_value),
        ToTensorV2(),
      ]
    else:  # Validation/Test
      # For validation and testing, we only resize and normalize.
      transforms = [
        A.Resize(height=self.image_size[0], width=self.image_size[1]),
        A.Normalize(mean=self.mean, std=self.std, max_pixel_value=self.max_pixel_value),
        ToTensorV2(),
      ]

    # NOTE: we do not like the transforms above, thus a override here
    transforms = [
      A.Resize(height=self.image_size[0], width=self.image_size[1]),
      # A.HorizontalFlip(p=0.5),  <- if this is enabled, then the labels would not be correct
      A.Normalize(mean=self.mean, std=self.std, max_pixel_value=self.max_pixel_value),
      ToTensorV2(),
    ]
    return A.Compose(transforms)

  def denormalize(
    self, normalized_image: Union[torch.Tensor, np.ndarray]
  ) -> np.ndarray:
    """Reverts the normalization on an image tensor/array for visualization."""
    if isinstance(normalized_image, torch.Tensor):
      # If tensor is on GPU, move to CPU. Permute from (C, H, W) to (H, W, C).
      normalized_image = normalized_image.cpu().numpy().transpose(1, 2, 0)

    mean = np.array(self.mean)
    std = np.array(self.std)

    # Denormalize: (tensor * std) + mean
    denormalized = (normalized_image * std) + mean
    # Scale back to 0-255 pixel range
    denormalized = denormalized * self.max_pixel_value
    # Clip values to be in the valid [0, 255] range
    denormalized = np.clip(denormalized, 0, 255)

    return denormalized.astype(np.uint8)


# --- Custom Collate Function ---
# This function formats the output of the DataLoader into the desired dictionary.


def classification_collate_fn(
  batch: List[Tuple[torch.Tensor, int]],
) -> Union[Dict[str, torch.Tensor], "LightningModelInputs"]:
  """
  Custom collate function for image classification.
  Filters out None samples and batches data into a dictionary.

  Args:
      batch: A list of (image_tensor, label_id) tuples.

  Returns:
      A dictionary with 'pixel_values' and 'labels' tensors.
  """
  # Filter out samples that failed to load (e.g., corrupted images)
  batch = [sample for sample in batch if sample is not None and sample[0] is not None]
  if not batch:  # fallback
    # Return an empty dict or handle as needed if the whole batch is invalid
    if is_standalone:
      return {"pixel_values": torch.empty(0), "labels": torch.empty(0)}

    # Default
    return LightningModelInputs(pixel_values=torch.empty(0), labels=torch.empty(0))

  images = torch.stack([item[0] for item in batch], dim=0)
  labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

  # A fallback choice, return dict - almost equivalent to the default one
  if is_standalone:
    return {"pixel_values": images, "labels": labels}

  # Default
  return LightningModelInputs(pixel_values=images, labels=labels)


# --- Custom Dataset Class ---
# A clean, simple Dataset class that receives prepared data.


class CustomImageClassificationDataset(Dataset):
  """
  A generic image classification dataset.
  It expects a list of image paths and a corresponding list of integer labels.
  The heavy lifting of discovering files and mapping labels is done by the DataModule.
  """

  def __init__(
    self,
    image_paths: List[str],
    labels: List[int],
    transform: Optional[A.Compose] = None,
    image_load_mode: str = "RGB",
  ):
    self.image_paths = image_paths
    self.labels = labels
    self.transform = transform
    self.image_load_mode = image_load_mode

  def __len__(self) -> int:
    return len(self.image_paths)

  def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, int]]:
    img_path = self.image_paths[idx]
    label = self.labels[idx]

    try:
      # Load image using PIL
      image = Image.open(img_path).convert(self.image_load_mode)
      image_np = np.array(image)
    except Exception as e:
      logger.warning(f"Could not load image {img_path}: {e}")
      return None  # The collate_fn will filter this out

    # Apply transformations if they exist
    if self.transform:
      try:
        transformed = self.transform(image=image_np)
        image_tensor = transformed["image"]
      except Exception as e:
        logger.warning(f"Error applying transform to {img_path}: {e}")
        return None
    else:
      # Basic conversion if no transforms are provided
      image_tensor = ToTensorV2()(image=image_np)["image"]

    return image_tensor, label


# --- Lightning DataModule ---
# The main orchestrator for data handling.


class ImageClassificationDataModule(pl.LightningDataModule):
  """
  PyTorch Lightning DataModule for handling image classification datasets.
  It supports two types of dataset structures:
  1. 'ilsvrc_subset': ImageNet-style with class subfolders (e.g., train.X1/n01440764/...).
  2. 'mini_imagenet': Flat image directory with CSV files for splits.
  """

  def __init__(
    self,
    data_dir: str,
    dataset_type: str,  # 'ilsvrc_subset' or 'mini_imagenet'
    image_size: Tuple[int, int] = (224, 224),
    train_val_test_split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    random_seed: int = 42,
  ):
    super().__init__()
    # Validate dataset_type
    if dataset_type not in ["ilsvrc_subset", "mini_imagenet"]:
      raise ValueError(
        f"dataset_type must be 'ilsvrc_subset' or 'mini_imagenet', but got {dataset_type}"
      )

    # Here's a little trick, the arguments passed to this class seeming not been used at all.
    # Yet in fact the following line of code has saved all these hyperparameters to attribute
    # `self.hparams` and for which we can extract hparam like, e.g., `self.hparams.datadir`.
    self.save_hyperparameters()

    self.image_transforms = ImageTransforms(image_size=self.hparams.image_size)
    self.train_transforms = self.image_transforms.get_classification_transforms(
      is_train=True
    )
    self.val_test_transforms = self.image_transforms.get_classification_transforms(
      is_train=False
    )

    # Placeholders for datasets and metadata
    self.dataset_train = None
    self.dataset_val = None
    self.dataset_test = None
    self.class_to_idx: Dict[str, int] = {}
    self.idx_to_class: Dict[int, str] = {}
    self.num_classes: int = 0

  def prepare_data(self):
    """Check if the data directory exists. No downloads or modifications here."""
    if not os.path.isdir(self.hparams.data_dir):
      raise FileNotFoundError(f"Data directory not found: {self.hparams.data_dir}")

  def setup(self, stage: Optional[str] = None):
    """
    This is the core logic for loading, splitting, and preparing datasets.
    It's called on every GPU in DDP training.
    """
    # This setup logic should only run once
    if self.dataset_train and self.dataset_val and self.dataset_test:
      return

    logger.info(
      f"Setting up data for stage '{stage}' using '{self.hparams.dataset_type}' structure."
    )

    # --- Data discovery and splitting based on dataset type ---
    if self.hparams.dataset_type == "ilsvrc_subset":
      all_data, self.class_to_idx = self._discover_ilsvrc_data()
      train_data, val_data, test_data = self._split_data(all_data)
    elif self.hparams.dataset_type == "mini_imagenet":
      train_data, val_data, test_data, self.class_to_idx = (
        self._discover_mini_imagenet_data()
      )
    else:
      # This case is already handled in __init__, but as a safeguard:
      raise ValueError("Invalid dataset_type")

    self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
    self.num_classes = len(self.class_to_idx)
    logger.info(f"Discovered {self.num_classes} classes.")

    # --- Instantiate Datasets ---
    if stage == "fit" or stage is None:
      train_paths, train_labels = self._prepare_dataset_args(train_data)
      self.dataset_train = CustomImageClassificationDataset(
        image_paths=train_paths,
        labels=train_labels,
        transform=self.train_transforms,
      )

      val_paths, val_labels = self._prepare_dataset_args(val_data)
      self.dataset_val = CustomImageClassificationDataset(
        image_paths=val_paths,
        labels=val_labels,
        transform=self.val_test_transforms,
      )
      logger.info(f"Train dataset size: {len(self.dataset_train)}")
      logger.info(f"Validation dataset size: {len(self.dataset_val)}")

    if stage == "test" or stage is None:
      test_paths, test_labels = self._prepare_dataset_args(test_data)
      self.dataset_test = CustomImageClassificationDataset(
        image_paths=test_paths,
        labels=test_labels,
        transform=self.val_test_transforms,
      )
      logger.info(
        f"Test dataset size: {len(self.dataset_test) if self.dataset_test else 0}"
      )

  def _prepare_dataset_args(
    self, data_list: List[Tuple[str, str]]
  ) -> Tuple[List[str], List[int]]:
    """Converts a list of (path, class_name) tuples to (path_list, label_idx_list)"""
    if not data_list:
      return [], []
    paths = [item[0] for item in data_list]
    # Map class name string to integer index
    labels = [self.class_to_idx[item[1]] for item in data_list]
    return paths, labels

  def _discover_ilsvrc_data(
    self,
  ) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """Finds all images in the ILSVRC-style subfolder structure."""
    logger.info("Discovering data from ILSVRC-style directories...")
    search_patterns = [
      os.path.join(self.hparams.data_dir, "train.X*", "*", "*.JPEG"),
      os.path.join(self.hparams.data_dir, "val.X", "*", "*.JPEG"),
    ]
    all_image_paths = []
    for pattern in search_patterns:
      all_image_paths.extend(glob.glob(pattern))

    if not all_image_paths:
      raise FileNotFoundError(
        f"No .JPEG images found in {self.hparams.data_dir} with ILSVRC structure."
      )

    all_data = [
      (path, os.path.basename(os.path.dirname(path))) for path in all_image_paths
    ]

    # Create a consistent class-to-index mapping
    class_names = sorted(list(set(item[1] for item in all_data)))
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    return all_data, class_to_idx

  def _split_data(self, all_data: List[Tuple[str, str]]) -> Tuple[List, List, List]:
    """Splits data while maintaining class distribution (stratification)."""
    train_ratio, val_ratio, test_ratio = self.hparams.train_val_test_split_ratio

    paths = [item[0] for item in all_data]
    labels = [item[1] for item in all_data]

    # Split into train and temp (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
      paths,
      labels,
      test_size=(val_ratio + test_ratio),
      random_state=self.hparams.random_seed,
      stratify=labels,
    )

    # Split temp into val and test
    # Adjust test_size to be proportional to the temp set
    relative_test_size = test_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
      temp_paths,
      temp_labels,
      test_size=relative_test_size,
      random_state=self.hparams.random_seed,
      stratify=temp_labels,
    )

    # Re-combine paths and labels into the original tuple format
    train_data = list(zip(train_paths, train_labels))
    val_data = list(zip(val_paths, val_labels))
    test_data = list(zip(test_paths, test_labels))

    return train_data, val_data, test_data

  def _discover_mini_imagenet_data(
    self,
  ) -> Tuple[List, List, List, Dict[str, int]]:
    """Loads data splits from Mini-ImageNet CSV files."""
    logger.info("Discovering data from Mini-ImageNet CSV files...")
    image_base_dir = os.path.join(self.hparams.data_dir, "images")
    if not os.path.isdir(image_base_dir):
      raise FileNotFoundError(
        f"Image directory 'images' not found inside {self.hparams.data_dir}"
      )

    try:
      train_df = pd.read_csv(os.path.join(self.hparams.data_dir, "train.csv"))
      val_df = pd.read_csv(os.path.join(self.hparams.data_dir, "val.csv"))
      test_df = pd.read_csv(os.path.join(self.hparams.data_dir, "test.csv"))
    except FileNotFoundError as e:
      raise FileNotFoundError(
        f"Could not find train/val/test CSV files in {self.hparams.data_dir}. Error: {e}"
      )

    # Create a unified, sorted class-to-index mapping from all splits
    all_labels = pd.concat(
      [train_df["label"], val_df["label"], test_df["label"]]
    ).unique()
    class_to_idx = {name: i for i, name in enumerate(sorted(all_labels))}

    # Function to convert DataFrame to list of (path, label) tuples
    def df_to_data_list(df: pd.DataFrame) -> List[Tuple[str, str]]:
      data_list = []
      for _, row in df.iterrows():
        path = os.path.join(image_base_dir, row["filename"])
        data_list.append((path, row["label"]))
      return data_list

    train_data = df_to_data_list(train_df)
    val_data = df_to_data_list(val_df)
    test_data = df_to_data_list(test_df)

    return train_data, val_data, test_data, class_to_idx

  # --- Dataloader Methods ---
  def train_dataloader(self) -> DataLoader:
    if not self.dataset_train:
      self.setup("fit")
    return DataLoader(
      self.dataset_train,
      batch_size=self.hparams.batch_size,
      shuffle=True,
      num_workers=self.hparams.num_workers,
      pin_memory=self.hparams.pin_memory,
      persistent_workers=self.hparams.persistent_workers
      if self.hparams.num_workers > 0
      else False,
      collate_fn=classification_collate_fn,
      drop_last=True,
    )

  def val_dataloader(self) -> DataLoader:
    if not self.dataset_val:
      self.setup("fit")
    return DataLoader(
      self.dataset_val,
      batch_size=self.hparams.batch_size,
      shuffle=False,
      num_workers=self.hparams.num_workers,
      pin_memory=self.hparams.pin_memory,
      persistent_workers=self.hparams.persistent_workers
      if self.hparams.num_workers > 0
      else False,
      collate_fn=classification_collate_fn,
    )

  def test_dataloader(self) -> DataLoader:
    if not self.dataset_test:
      self.setup("test")
    return DataLoader(
      self.dataset_test,
      batch_size=self.hparams.batch_size,
      shuffle=False,
      num_workers=self.hparams.num_workers,
      pin_memory=self.hparams.pin_memory,
      persistent_workers=self.hparams.persistent_workers
      if self.hparams.num_workers > 0
      else False,
      collate_fn=classification_collate_fn,
    )


# --- Main Demo / Test Script --- #
# Temporary cache directory for dummy data


def create_dummy_image(path: str):
  """Creates a small random JPEG image."""
  os.makedirs(os.path.dirname(path), exist_ok=True)
  img = Image.new(
    "RGB",
    (50, 50),
    color=(
      random.randint(0, 255),
      random.randint(0, 255),
      random.randint(0, 255),
    ),
  )
  img.save(path, "JPEG")


def demo_ilsvrc_subset():
  """Creates dummy data and tests the DataModule for the 'ilsvrc_subset' type."""
  print("\n" + "=" * 50)
  print("--- Running Demo for ILSVRC-style Subset ---")
  print("=" * 50)

  DATA_DIR = os.path.join(CACHE_DIR, "ilsvrc_subset")
  os.makedirs(DATA_DIR, exist_ok=True)

  # Create dummy data structure
  classes = [f"n0{100 + i}" for i in range(5)]  # 5 dummy classes
  dirs_to_create = [
    os.path.join(DATA_DIR, "train.X1"),
    os.path.join(DATA_DIR, "val.X"),
  ]

  img_counter = 0
  for base_dir in dirs_to_create:
    for class_name in classes:
      class_dir = os.path.join(base_dir, class_name)
      for i in range(10):  # 10 images per class per dir
        img_name = f"{class_name}_{img_counter}.JPEG"
        create_dummy_image(os.path.join(class_dir, img_name))
        img_counter += 1

  logger.info(f"Created dummy ILSVRC data in: {DATA_DIR}")

  # Configure and test the DataModule
  data_module = ImageClassificationDataModule(
    data_dir=DATA_DIR,
    dataset_type="ilsvrc_subset",
    image_size=(32, 32),
    train_val_test_split_ratio=(0.6, 0.2, 0.2),
    batch_size=4,
    num_workers=0,  # Use 0 for easier debugging
    random_seed=42,
  )

  data_module.prepare_data()
  data_module.setup(stage="fit")

  print(f"\nClass to Index Mapping: {data_module.class_to_idx}")
  print(f"Number of classes: {data_module.num_classes}")

  train_loader = data_module.train_dataloader()
  batch = next(iter(train_loader))

  print("\n--- Testing Train Dataloader (ILSVRC) ---")
  print(f"Batch keys: {batch.keys()}")
  print(f"Pixel values shape: {batch['pixel_values'].shape}")  # e.g., [4, 3, 32, 32]
  print(f"Class labels shape: {batch['labels'].shape}")  # e.g., [4]
  print(f"Sample labels: {batch['labels']}")


def demo_mini_imagenet():
  """Creates dummy data and tests the DataModule for the 'mini_imagenet' type."""
  print("\n" + "=" * 50)
  print("--- Running Demo for Mini-ImageNet style ---")
  print("=" * 50)

  DATA_DIR = os.path.join(CACHE_DIR, "mini_imagenet")
  IMAGE_DIR = os.path.join(DATA_DIR, "images")
  os.makedirs(IMAGE_DIR, exist_ok=True)

  # Create dummy data
  classes = [f"n0{200 + i}" for i in range(5)]  # 5 other dummy classes
  filenames = []
  labels = []
  for class_name in classes:
    for i in range(20):  # 20 images per class
      fname = f"{class_name}{i:08d}.jpg"
      create_dummy_image(os.path.join(IMAGE_DIR, fname))
      filenames.append(fname)
      labels.append(class_name)

  # Create dummy CSV files
  df = pd.DataFrame({"filename": filenames, "label": labels})
  train_df, val_test_df = train_test_split(
    df, test_size=0.4, random_state=42, stratify=df["label"]
  )
  val_df, test_df = train_test_split(
    val_test_df, test_size=0.5, random_state=42, stratify=val_test_df["label"]
  )

  train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
  val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
  test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

  logger.info(f"Created dummy Mini-ImageNet data in: {DATA_DIR}")

  # Configure and test the DataModule
  data_module = ImageClassificationDataModule(
    data_dir=DATA_DIR,
    dataset_type="mini_imagenet",
    image_size=(32, 32),
    batch_size=4,
    num_workers=0,
  )

  data_module.prepare_data()
  data_module.setup(stage="fit")

  print(f"\nClass to Index Mapping: {data_module.class_to_idx}")
  print(f"Number of classes: {data_module.num_classes}")

  train_loader = data_module.train_dataloader()
  batch = next(iter(train_loader))

  print("\n--- Testing Train Dataloader (Mini-ImageNet) ---")
  print(f"Batch keys: {batch.keys()}")
  print(f"Pixel values shape: {batch['pixel_values'].shape}")
  print(f"Class labels shape: {batch['labels'].shape}")
  print(f"Sample labels: {batch['labels']}")


if __name__ == "__main__":
  # Run the demos
  demo_ilsvrc_subset()
  demo_mini_imagenet()

  # You can clean up the dummy data afterwards if you wish
  # import shutil
  # if os.path.exists(CACHE_DIR):
  #   shutil.rmtree(CACHE_DIR)
  #   print(f"\nCleaned up dummy data directory: {CACHE_DIR}")

# fatuus._imagenet ends here
