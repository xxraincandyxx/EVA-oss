import os
import sys
import time
from pathlib import Path

import cv2


def main():
  """
  在板端实时从摄像头采集图像，经 DPU YOLOv3 推理后画框，并定期保存结果帧。
  - 使用已有的 eva.vision.yolo_dpu.DPURunner 和 postprocess_yolo 逻辑
  - 不依赖完整 EVA 后端配置，避免 torch 等重依赖
  """
  # 为了避免触发 eva.__init__ 里的完整依赖，这里采用文件级加载
  import importlib.util

  repo_root = Path(__file__).resolve().parents[2]
  yolo_dpu_path = repo_root / "eva" / "vision" / "yolo_dpu.py"
  run_imagenet_path = repo_root / "eva" / "vision" / "run_yolov3_dpu_imagenet.py"

  if not yolo_dpu_path.exists() or not run_imagenet_path.exists():
    print(f"[DPU CAM] 找不到 YOLO DPU 模块文件: {yolo_dpu_path} 或 {run_imagenet_path}")
    return

  # 动态加载 DPURunner
  spec_yolo = importlib.util.spec_from_file_location("yolo_dpu_mod", str(yolo_dpu_path))
  yolo_mod = importlib.util.module_from_spec(spec_yolo)
  sys.modules["yolo_dpu_mod"] = yolo_mod
  spec_yolo.loader.exec_module(yolo_mod)
  DPURunner = yolo_mod.DPURunner

  # 动态加载 postprocess_yolo
  spec_post = importlib.util.spec_from_file_location(
    "run_yolov3_dpu_imagenet_mod", str(run_imagenet_path)
  )
  post_mod = importlib.util.module_from_spec(spec_post)
  sys.modules["run_yolov3_dpu_imagenet_mod"] = post_mod
  spec_post.loader.exec_module(post_mod)
  postprocess_yolo = post_mod.postprocess_yolo

  model_path = os.path.expanduser("~/eva/src/eva/vision/yolo.xmodel")
  if not os.path.exists(model_path):
    print(f"[DPU CAM] xmodel 不存在: {model_path}")
    return

  print(f"[DPU CAM] 使用模型: {model_path}")
  runner = DPURunner(model_path)

  # 打开摄像头：支持索引(int) 或 设备/URL 字符串
  cam_src = os.environ.get("YOLO_DPU_CAM_SRC", "0")
  try:
    cam_index = int(cam_src)
  except ValueError:
    cam_index = cam_src  # 可能是 '/dev/video2' 或 rtsp/http url

  print(f"[DPU CAM] 打开摄像头源: {cam_index}")
  cap = cv2.VideoCapture(cam_index)
  if not cap.isOpened():
    print(f"[DPU CAM] 无法打开摄像头索引 {cam_index}")
    return

  print(f"[DPU CAM] 摄像头 {cam_index} 已打开，开始推理...")

  out_dir = Path.home() / "eva" / "captures_dpu"
  out_dir.mkdir(parents=True, exist_ok=True)

  frame_id = 0

  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        print("[DPU CAM] 读取帧失败，结束。")
        break

      h, w = frame.shape[:2]
      t0 = time.time()
      outputs = runner.run(frame)
      boxes, scores, classes = postprocess_yolo(outputs, runner, (h, w))
      t1 = time.time()

      # 打印简要信息
      dt = (t1 - t0) * 1000.0
      msg = f"[DPU CAM] frame={frame_id} time={dt:.1f} ms, det={boxes.shape[0]}"
      print(msg)

      # 画框
      vis = frame.copy()
      for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i]
        cv2.rectangle(
          vis,
          (int(x1), int(y1)),
          (int(x2), int(y2)),
          (0, 255, 0),
          2,
        )

      # 每 10 帧保存一张结果图
      if frame_id % 10 == 0:
        out_path = out_dir / f"cam_dpu_{frame_id:06d}.jpg"
        cv2.imwrite(str(out_path), vis)

      frame_id += 1

      # 简单的退出条件：按 Ctrl+C 终止
  except KeyboardInterrupt:
    print("[DPU CAM] 收到中断信号，退出。")
  finally:
    cap.release()
    print("[DPU CAM] 摄像头已释放。")


if __name__ == "__main__":
  main()
