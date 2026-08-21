from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.engine.trainer import BaseTrainer


# --- Configuration Management ---
def load_config(config_path: str) -> dict:
  """Load training configuration from YAML file"""
  with open(config_path) as f:
    return yaml.safe_load(f)


# --- Optimized Preprocessing Function ---
def preprocess_metal_image(
  img: Union[str, Path, np.ndarray], target_size: tuple
) -> np.ndarray:
  """Enhanced metal surface defect preprocessing optimized for YOLO"""
  if isinstance(img, (str, Path)):
    img = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
  elif img.ndim == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Processing pipeline
  denoised = cv2.fastNlMeansDenoising(
    img, h=5, templateWindowSize=5, searchWindowSize=15
  )
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  enhanced = clahe.apply(denoised)
  blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
  normalized = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

  resized = cv2.resize(normalized, target_size, interpolation=cv2.INTER_LINEAR_EXACT)
  return np.repeat(resized[..., np.newaxis], 3, axis=-1)


# --- Custom Dataset Class ---
class PreprocessedDataset(YOLODataset):
  def __init__(self, *args, preprocess_fn=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.preprocess_fn = preprocess_fn
    self.target_size = (self.imgsz, self.imgsz)

  def __getitem__(self, index):
    """Override to apply custom preprocessing before augmentation"""
    # Load original image
    image = self.load_image(index)

    # Apply custom preprocessing
    if self.preprocess_fn:
      image = self.preprocess_fn(image, self.target_size)
      image = image.astype(np.uint8)  # Ensure correct dtype

    # Process the rest normally
    return super().__getitem__(index)


# --- Custom Trainer Class ---
class CustomTrainer(BaseTrainer):
  def get_dataset(self, data, mode="train"):
    """Create dataset with custom preprocessing"""
    return PreprocessedDataset(
      img_path=data["train" if mode == "train" else "val"],
      imgsz=self.args.imgsz,
      batch_size=self.args.batch,
      augment=mode == "train",
      hyp=self.args,
      rect=False,
      cache=self.args.cache,
      preprocess_fn=preprocess_metal_image,
      prefix=f"{mode}: ",
    )


# --- Main Training Pipeline ---
def main(config_path: str = "config.yaml"):
  # Load configuration
  config = load_config(config_path)
  data_cfg = config["data"]
  train_cfg = config["train"]
  model_cfg = config["model"]

  # Create dataset YAML
  dataset_yaml = {
    "path": data_cfg["root"],
    "train": data_cfg["train"],
    "val": data_cfg["val"],
    "nc": data_cfg["nc"],
    "names": data_cfg["names"],
  }
  with open("dataset.yaml", "w") as f:
    yaml.dump(dataset_yaml, f)

  # Initialize model with custom trainer
  model = YOLO(model_cfg["architecture"])
  model.trainer = CustomTrainer(overrides=data_cfg)  # vars(model.args)

  # Training parameters
  train_params = {
    "data": "dataset.yaml",
    "epochs": train_cfg["epochs"],
    "batch": train_cfg["batch_size"],
    "imgsz": train_cfg["image_size"],
    "device": train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
    "optimizer": train_cfg.get("optimizer", "Adam"),
    "lr0": train_cfg["learning_rate"],
    "weight_decay": train_cfg["weight_decay"],
    "resume": train_cfg.get("resume", False),
    "verbose": train_cfg.get("verbose", True),
  }

  # Start training
  model.train(**train_params)

  # Validation and inference same as before...


if __name__ == "__main__":
  main()
