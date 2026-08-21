import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 简单的 YOLOv3 VOC 配置（与 Vitis-AI tf_yolov3_3.5 工程一致）
NUM_CLASSES = 20
CLASS_NAMES = [
  "aeroplane",
  "bicycle",
  "bird",
  "boat",
  "bottle",
  "bus",
  "car",
  "cat",
  "chair",
  "cow",
  "diningtable",
  "dog",
  "horse",
  "motorbike",
  "person",
  "pottedplant",
  "sheep",
  "sofa",
  "train",
  "tvmonitor",
]
ANCHORS = np.array(
  [
    [10, 13],
    [16, 30],
    [33, 23],
    [30, 61],
    [62, 45],
    [59, 119],
    [116, 90],
    [156, 198],
    [373, 326],
  ],
  dtype=np.float32,
)
ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
OBJ_THRESH = 0.3
NMS_THRESH = 0.45
INPUT_SIZE = 416


def sigmoid(x: np.ndarray) -> np.ndarray:
  return 1.0 / (1.0 + np.exp(-x))


def _decode_single_output(
  feat: np.ndarray, anchors: np.ndarray, input_shape, image_shape
):
  """
  将单尺度特征图解码为原图坐标系下的候选框和得分
  feat: (H, W, 75)  -> (H, W, 3, 5+num_classes)
  anchors: (3, 2)
  input_shape: (h, w)
  image_shape: (ih, iw)
  """
  num_anchors = anchors.shape[0]
  grid_h, grid_w = feat.shape[:2]

  feat = feat.reshape((grid_h, grid_w, num_anchors, 5 + NUM_CLASSES))

  grid_y = np.arange(grid_h).reshape((grid_h, 1, 1, 1))
  grid_y = np.tile(grid_y, (1, grid_w, num_anchors, 1))
  grid_x = np.arange(grid_w).reshape((1, grid_w, 1, 1))
  grid_x = np.tile(grid_x, (grid_h, 1, num_anchors, 1))
  grid = np.concatenate([grid_x, grid_y], axis=-1).astype(np.float32)

  box_xy = (sigmoid(feat[..., 0:2]) + grid) / np.array(
    [grid_w, grid_h], dtype=np.float32
  )
  box_wh = (
    np.exp(feat[..., 2:4])
    * anchors.reshape((1, 1, num_anchors, 2))
    / np.array([input_shape[1], input_shape[0]], dtype=np.float32)
  )

  box_conf = sigmoid(feat[..., 4:5])
  box_prob = sigmoid(feat[..., 5:])

  # 映射回原图坐标（对应 Vitis-AI YOLO3 predictor 的 correct_boxes 逻辑）
  box_yx = box_xy[..., ::-1]
  box_hw = box_wh[..., ::-1]

  input_shape = np.array(input_shape, dtype=np.float32)
  image_shape = np.array(image_shape, dtype=np.float32)

  new_shape = np.round(image_shape * np.min(input_shape / image_shape))
  offset = (input_shape - new_shape) / 2.0 / input_shape
  scale = input_shape / new_shape

  box_yx = (box_yx - offset) * scale
  box_hw = box_hw * scale

  box_mins = box_yx - (box_hw / 2.0)
  box_maxes = box_yx + (box_hw / 2.0)

  boxes = np.concatenate(
    [
      box_mins[..., 0:1],
      box_mins[..., 1:2],
      box_maxes[..., 0:1],
      box_maxes[..., 1:2],
    ],
    axis=-1,
  )
  boxes *= np.concatenate([image_shape, image_shape], axis=-1)

  boxes = boxes.reshape((-1, 4))
  box_scores = (box_conf * box_prob).reshape((-1, NUM_CLASSES))

  return boxes, box_scores


def _nms_boxes(boxes, scores, iou_thresh, max_boxes=50):
  """
  简单 NMS，实现与 TF 版类似的效果
  """
  if boxes.size == 0:
    return []
  x1 = boxes[:, 0]
  y1 = boxes[:, 1]
  x2 = boxes[:, 2]
  y2 = boxes[:, 3]

  areas = (x2 - x1) * (y2 - y1)
  order = scores.argsort()[::-1]

  keep = []
  while order.size > 0 and len(keep) < max_boxes:
    i = order[0]
    keep.append(i)
    if order.size == 1:
      break
    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

    idxs = np.where(iou < iou_thresh)[0]
    order = order[idxs + 1]
  return keep


def postprocess_yolo(outputs, runner, image_shape):
  """
  将 DPU 输出特征图解码为最终检测结果
  outputs: DPURunner.run 的输出列表（int8）
  runner: DPURunner 实例，用于获取 fix_point 信息
  image_shape: (h, w)
  """
  tensors = runner.output_tensors
  all_boxes = []
  all_scores = []

  input_shape = (INPUT_SIZE, INPUT_SIZE)

  for i, (feat_int8, t) in enumerate(zip(outputs, tensors)):
    feat = feat_int8.astype(np.float32)
    if t.has_attr("fix_point"):
      fix_point = t.get_attr("fix_point")
      feat = feat * (2.0 ** (-fix_point))

    feat = feat[0]  # 去掉 batch 维
    anchors = ANCHORS[ANCHOR_MASKS[i]]
    boxes, box_scores = _decode_single_output(feat, anchors, input_shape, image_shape)
    all_boxes.append(boxes)
    all_scores.append(box_scores)

  boxes = np.concatenate(all_boxes, axis=0)
  box_scores = np.concatenate(all_scores, axis=0)

  mask = box_scores >= OBJ_THRESH

  final_boxes = []
  final_scores = []
  final_classes = []

  for c in range(NUM_CLASSES):
    class_boxes = boxes[mask[:, c]]
    class_scores = box_scores[mask[:, c], c]
    if class_boxes.size == 0:
      continue
    keep = _nms_boxes(class_boxes, class_scores, NMS_THRESH, max_boxes=50)
    if not keep:
      continue
    final_boxes.append(class_boxes[keep])
    final_scores.append(class_scores[keep])
    final_classes.append(np.full(len(keep), c, dtype=np.int32))

  if not final_boxes:
    return (
      np.zeros((0, 4)),
      np.zeros((0,), dtype=np.float32),
      np.zeros((0,), dtype=np.int32),
    )

  final_boxes = np.concatenate(final_boxes, axis=0)
  final_scores = np.concatenate(final_scores, axis=0)
  final_classes = np.concatenate(final_classes, axis=0)

  return final_boxes, final_scores, final_classes


def run_on_folder(model_path: str, image_dir: str):
  """
  在指定目录下随机抽取若干张图片，跑一轮 DPU YOLOv3 检测并打印结果概要
  """
  # 动态加载 yolo_dpu.DPURunner，避免依赖完整 eva 包
  vision_dir = Path(model_path).resolve().parent
  yolo_dpu_path = vision_dir / "yolo_dpu.py"
  spec = None
  if yolo_dpu_path.exists():
    import importlib.util

    spec = importlib.util.spec_from_file_location("yolo_dpu_mod", str(yolo_dpu_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["yolo_dpu_mod"] = mod
    spec.loader.exec_module(mod)
    DPURunner = mod.DPURunner
  else:
    raise RuntimeError(f"未找到 yolo_dpu.py: {yolo_dpu_path}")

  runner = DPURunner(model_path)

  image_dir = Path(image_dir)
  all_imgs = sorted(
    [p for p in image_dir.glob("*.JPEG")]
    + [p for p in image_dir.glob("*.jpg")]
    + [p for p in image_dir.glob("*.png")]
  )
  if not all_imgs:
    print(f"目录中未找到图片: {image_dir}")
    return

  print(f"共找到 {len(all_imgs)} 张图片，开始测试 YOLOv3 VOC DPU 检测...")

  for img_path in all_imgs:
    img = cv2.imread(str(img_path))
    if img is None:
      print(f"跳过无法读取的图片: {img_path.name}")
      continue
    ih, iw = img.shape[:2]
    t0 = time.time()
    outputs = runner.run(img)
    boxes, scores, classes = postprocess_yolo(outputs, runner, (ih, iw))
    t1 = time.time()

    print(
      f"\n图片: {img_path.name}, 大小: {iw}x{ih}, 推理耗时: {(t1 - t0) * 1000:.1f} ms"
    )
    if boxes.shape[0] == 0:
      print("  未检测到目标（阈值 OBJ_THRESH = %.2f）" % OBJ_THRESH)
      continue

    # 打印前几条检测结果
    for i in range(min(5, boxes.shape[0])):
      x1, y1, x2, y2 = boxes[i]
      score = scores[i]
      cls_id = int(classes[i])
      cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
      print(
        f"  [{i}] {cls_name:<12} "
        f"score={score:.3f} "
        f"box=({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
      )


if __name__ == "__main__":
  # 默认模型和测试集路径适配当前板端目录结构
  default_model = str(Path.home() / "eva/src/eva/vision/yolo.xmodel")
  default_images = str(Path.home() / "eva/test_images_imagenet")

  model = os.environ.get("YOLO_DPU_MODEL", default_model)
  img_dir = os.environ.get("YOLO_DPU_IMAGEDIR", default_images)

  print("使用模型:", model)
  print("测试图片目录:", img_dir)

  run_on_folder(model, img_dir)
