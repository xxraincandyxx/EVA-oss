import cv2
import numpy as np

# Check if Vitis AI is available
try:
  import vart
  import xir

  VITIS_AVAILABLE = True
except ImportError:
  VITIS_AVAILABLE = False
  print(
    "Warning: Vitis AI libraries (xir, vart) not found. DPU inference will not work."
  )


class DPURunner:
  def __init__(self, model_path):
    if not VITIS_AVAILABLE:
      raise ImportError("Vitis AI libraries not installed.")

    self.graph = xir.Graph.deserialize(model_path)
    root = self.graph.get_root_subgraph()
    # 兼容不同版本的 XIR API
    if hasattr(root, "get_top_topological_subgraphs"):
      child_subgraphs = root.get_top_topological_subgraphs()
    elif hasattr(root, "get_children"):
      child_subgraphs = root.get_children()
    elif hasattr(root, "children_topological_sort"):
      child_subgraphs = root.children_topological_sort()
    else:
      child_subgraphs = []

    self.dpu_subgraphs = [
      s
      for s in child_subgraphs
      if s.has_attr("device") and str(s.get_attr("device")).upper() == "DPU"
    ]

    if not self.dpu_subgraphs:
      raise ValueError("No DPU subgraph found in the model")

    # Create runner for the first DPU subgraph (usually there's one main one for YOLO)
    # VART 2.x Runner
    self.runner = vart.Runner.create_runner(self.dpu_subgraphs[0], "run")

    self.input_tensors = self.runner.get_input_tensors()
    self.output_tensors = self.runner.get_output_tensors()

    self.input_shape = tuple(self.input_tensors[0].dims)  # (B, H, W, C) or (B, C, H, W)
    self.output_shapes = [tuple(t.dims) for t in self.output_tensors]

    print(f"[DPU] Model Loaded: {model_path}")
    print(f"[DPU] Input Shape: {self.input_shape}")
    print(f"[DPU] Output Shapes: {self.output_shapes}")

  def preprocess(self, image):
    """
    Resize and normalize image for DPU.
    Assumes DPU expects:
    - Layout: NHWC (standard for Vitis AI)
    - Scale: Typically input is int8, so we might need to scale float 0-1 to int8 or keep as uint8 if model expects raw.
    - WARNING: Standard Vitis AI models often expect pre-quantized inputs (fix point).
      Usually: (image - mean) * scale.
      But for many raw DPU models, we just resize and pass int8.
      We will assume standard YOLO preprocessing (resize, pad) and verify quantization info if available.
    """
    # 支持 NHWC / NCHW 两种输入
    if len(self.input_shape) == 4:
      if self.input_shape[1] in [1, 3]:  # NCHW
        h, w = self.input_shape[2], self.input_shape[3]
        layout = "NCHW"
      else:  # NHWC
        h, w = self.input_shape[1], self.input_shape[2]
        layout = "NHWC"
    else:
      raise ValueError(f"Unexpected input shape: {self.input_shape}")

    # Letterbox resize (keep aspect ratio)
    img_h, img_w = image.shape[:2]
    scale = min(w / img_w, h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    # Create canvas
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)  # 114 is grey padding for YOLO

    # Center or Top-Left? YOLO usually centers or top-lefts. Ultralytics centers.
    dw = (w - new_w) // 2
    dh = (h - new_h) // 2
    canvas[dh : dh + new_h, dw : dw + new_w, :] = resized

    # For DPU, input is usually int8. If the model was quantized with scale, we need to apply it.
    # However, the Runner API often handles fix-point conversion if we pass float,
    # OR we perform fix-point manually using get_attr("fix_point").

    # Let's try basic normalization if float, or raw if int8 tensor.
    # Checking tensor type:
    # For simplicity in this template, we assume the runner handles standard image input
    # or we pass raw uint8 if the input layer is named 'input_0' etc.

    # NOTE: Most Vitis-AI YOLO 示例直接喂入量化前的图像数据，再由量化参数完成缩放。
    # 这里将数据转换为 int8，以匹配 xmodel 的 xint8 量化类型，避免 VART Python 端
    # 出现 unsupported data type（np.uint8 在某些版本中会被映射为 UNKNOWN）。
    if layout == "NCHW":
      canvas = np.transpose(canvas, (2, 0, 1))  # HWC -> CHW
    canvas = canvas.astype(np.int8)
    return np.expand_dims(canvas, axis=0), scale, (dw, dh)

  def run(self, image):
    input_data, scale, (dw, dh) = self.preprocess(image)

    # Create output buffers（使用 int8 与 xint8 对齐，后处理阶段可再转换为 float）
    output_data = []
    for t in self.output_tensors:
      shape = tuple(t.dims)
      output_data.append(np.empty(shape, dtype=np.int8, order="C"))

    # Execute
    job_id = self.runner.execute_async([input_data], output_data)
    self.runner.wait(job_id)

    # Postprocess
    # output_data contains the raw feature maps.
    # Decoding YOLO output from raw feature maps is complex (sigmoid, grid matching).
    # Ideally, we use a library or the model includes decoding (unlikely for standard DPU).

    return output_data


# ... (YOLO decoding logic would go here, typically adapting from ultralytics/utils/ops.py)
