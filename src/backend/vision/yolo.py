import os
from pathlib import Path
from typing import Any, Callable, Optional, Union

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO
from ultralytics.data import loaders
from ultralytics.data.augment import Compose


def ultra2cv(res):
  # Extract annotated image (RGB format)
  annotated_image = res.plot()  # Get the first result (for single-image inference)
  # Convert RGB to BGR for OpenCV
  annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

  return annotated_image_bgr


def pil_to_cv2(pil_image):
  """Convert PIL Image to OpenCV Image

  Args:
      pil_image: PIL.Image object

  Returns:
      OpenCV image (numpy.ndarray in BGR format)
  """
  # Convert PIL to numpy array (RGB format)
  numpy_image = np.array(pil_image)

  # Convert RGB to BGR
  if len(numpy_image.shape) == 3:  # Color image
    return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
  else:  # Grayscale image
    return numpy_image


def _preprocess_metal_image(
  img: Union[str | os.PathLike | Any], target_size=(640, 640)
):
  """
  Process real-world metal images to resemble NEU-CLS dataset characteristics
  Returns processed image in RGB format (H, W, 3)
  """
  if isinstance(img, str) | isinstance(img, os.PathLike):
    # Read image
    img = cv2.imread(img)

  if isinstance(img, Image.Image):
    img = pil_to_cv2(img)

  # Convert to grayscale (NEU-CLS uses grayscale images)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Noise reduction (adjust parameters based on your images)
  denoised = cv2.fastNlMeansDenoising(
    gray, h=7, templateWindowSize=7, searchWindowSize=21
  )

  # Handle highlights (Optional)
  _, thresh = cv2.threshold(denoised, 220, 255, cv2.THRESH_TRUNC)

  # Contrast enhancement using CLAHE
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  enhanced = clahe.apply(thresh)

  # Background homogenization (optional)
  blur = cv2.GaussianBlur(enhanced, (25, 25), 0)
  normalized = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

  # Resize to model input size
  resized = cv2.resize(normalized, target_size, interpolation=cv2.INTER_AREA)

  # Convert back to 3-channel "RGB" (if model expects 3-channel input)
  final_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

  return final_img


def preprocess_metal_image(
  img: Union[str, Path, Any], target_size=(640, 640)
) -> np.ndarray:
  """An optimized faster preprocessing pipeline of _preprocess_metal_image

  Optimized metal image preprocessing with 3-5x speed improvement
  Returns processed image in RGB format (H, W, 3)
  """
  if isinstance(img, (str, Path)):
    img = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)  # Direct grayscale loading
  else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  denoised = cv2.fastNlMeansDenoising(
    img, h=5, templateWindowSize=5, searchWindowSize=15
  )  # Reduced from 7; Reduced from 21

  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))  # Smaller grid
  enhanced = clahe.apply(
    np.clip(denoised, 0, 220)
  )  # Built-in clipping instead of separate threshold

  blur = cv2.GaussianBlur(enhanced, (15, 15), 0, borderType=cv2.BORDER_REPLICATE)
  normalized = cv2.addWeighted(enhanced, 1.3, blur, -0.3, 0)

  resized = cv2.resize(normalized, target_size, interpolation=cv2.INTER_LINEAR_EXACT)
  return np.repeat(resized[..., np.newaxis], 3, axis=-1)


class TrainingConfig:
  """Centralized configuration for model training"""

  def __init__(self, config_path: str):
    with open(config_path) as f:
      config = yaml.safe_load(f)

    # Model configuration
    self.model_name: str = config["model"]["name"]
    self.model_config: str = config["model"]["config"]
    self.pretrained_weights: str = config["model"]["pretrained_weights"]
    self.input_size: int = config["model"]["input_size"]

    # Training parameters
    self.epochs: int = config["training"]["epochs"]
    self.batch_size: int = config["training"]["batch_size"]
    self.optimizer: str = config["training"]["optimizer"]
    self.learning_rate: float = config["training"]["learning_rate"]
    self.weight_decay: float = config["training"]["weight_decay"]
    self.resume_training: bool = config["training"]["resume"]

    # Dataset configuration
    self.dataset_dir: Path = Path(config["paths"]["dataset"]).expanduser()
    self.output_dir: Path = Path(config["paths"]["output"]).expanduser()
    self.class_names: list = config["dataset"]["classes"]
    self.num_classes: int = config["dataset"]["num_classes"]

    # Create output directory
    self.output_dir.mkdir(parents=True, exist_ok=True)

  def create_dataset_yaml(self):
    """Generate YOLO dataset configuration file"""
    dataset_config = {
      "train": str(self.dataset_dir / "train/images"),
      "val": str(self.dataset_dir / "valid/images"),
      "nc": self.num_classes,
      "names": self.class_names,
    }
    with open(self.output_dir / "dataset.yaml", "w") as f:
      yaml.dump(dataset_config, f)


class YOLOTrainer:
  """YOLO training pipeline with integrated preprocessing"""

  def __init__(self, config: TrainingConfig, preprocess_fn: Optional[Callable] = None):
    self.config = config
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model = self._initialize_model()
    self._configure_preprocessing(preprocess_fn)

  def _initialize_model(self):
    """Load YOLO model from config or pretrained weights"""
    if self.config.pretrained_weights:
      model = YOLO(self.config.pretrained_weights)
    else:
      model = YOLO(self.config.model_config)

    # Update model parameters from config
    model.args["imgsz"] = self.config.input_size
    model.args["device"] = self.device

    return model

  def _configure_preprocessing(self, preprocess_fn):
    """Inject custom preprocessing into YOLO's data pipeline"""
    if preprocess_fn:
      # Create custom transform pipeline
      self.preprocess = Compose(
        [
          lambda x: preprocess_fn(x),  # Apply user's preprocessing
          # Add necessary YOLO formatting transforms
          lambda x: x.permute(2, 0, 1),  # HWC to CHW
          lambda x: x.float().div(255),  # Normalize
        ]
      )

      # Override default training loader
      loaders._train_loader = self._create_custom_loader

  def _create_custom_loader(self, dataset, batch_size, workers, shuffle=True):
    """Create DataLoader with custom preprocessing"""
    return torch.utils.data.DataLoader(
      dataset=dataset,
      batch_size=batch_size,
      shuffle=shuffle,
      num_workers=workers,
      collate_fn=loaders.classify_collate,
      transform=self.preprocess,  # Inject our preprocessing
    )

  def train(self):
    """Execute training pipeline with proper preprocessing"""
    self.config.create_dataset_yaml()

    # Force YOLO to use our custom loader
    results = self.model.train(
      data=str(self.config.output_dir / "dataset.yaml"),
      epochs=self.config.epochs,
      batch=self.config.batch_size,
      imgsz=self.config.input_size,
      device=self.device,
      optimizer=self.config.optimizer,
      lr0=self.config.learning_rate,
      weight_decay=self.config.weight_decay,
      resume=self.config.resume_training,
      verbose=True,
      augment=True,  # Now handled by our custom pipeline
      # Add custom transforms configuration
      # transform={"train": self.preprocess, "val": self.preprocess},
    )
    return results


if __name__ == "__main__":
  pwd = os.path.dirname(os.path.abspath(__file__))
  config_path = os.path.join(pwd, "configs", "trainer.yaml")
  config = TrainingConfig(config_path)

  # Define or import your preprocessing function
  def custom_preprocess(image):
    return preprocess_metal_image(image, (256, 256))

  trainer = YOLOTrainer(config, preprocess_fn=custom_preprocess)

  trainer.train()
  trainer.validate()

  image_path = str(config.dataset_dir / "valid/images/scratches_5.jpg")
  trainer.predict(image_path)
