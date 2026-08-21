import os

import cv2
import numpy as np
from yolo import TrainingConfig, YOLOTrainer, preprocess_metal_image


def main():
  # Path to config
  pwd = os.path.dirname(os.path.abspath(__file__))
  config_path = os.path.join(pwd, "configs", "trainer.yaml")
  if not os.path.exists(config_path):
    print(f"Config not found: {config_path}")
    return

  config = TrainingConfig(config_path)

  # Define Preprocess
  def custom_preprocess(image):
    return preprocess_metal_image(image, (256, 256))

  # Initialize Trainer (Inference mode)
  print("Initializing YOLO DPU Inference...")
  try:
    # Force use_dpu=True for testing DPU
    trainer = YOLOTrainer(config, preprocess_fn=custom_preprocess, use_dpu=True)
  except ImportError as e:
    print(f"Failed to initialize DPU runner: {e}")
    print(
      "Please ensure 'vitis-ai-library' and 'python3-vitis-ai' are installed and 'xir', 'vitis_ai_vart' modules are available."
    )
    return
  except Exception as e:
    print(f"Initialization Error: {e}")
    return

  # Test Image
  # Use a dummy image if dataset not present
  image_path = str(config.dataset_dir / "valid/images/scratches_5.jpg")
  if not os.path.exists(image_path):
    print(f"Test image not found at {image_path}. Creating dummy image.")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    image_path = "dummy_test.jpg"
    cv2.imwrite(image_path, dummy)

  print(f"Running inference on {image_path}...")

  # Warmup
  print("Warmup...")
  trainer.predict(image_path)

  # Benchmark
  print("Benchmarking...")
  start = cv2.getTickCount()
  trainer.predict(image_path)
  end = cv2.getTickCount()

  t = (end - start) / cv2.getTickFrequency()
  print(f"Inference time: {t * 1000:.2f} ms")


if __name__ == "__main__":
  main()
