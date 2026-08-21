"""Command-line entry point for the EVA backend."""

import argparse
import multiprocessing as mp
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from .paths import MODEL_DIR


def _env_bool(name: str, default: bool) -> bool:
  value = os.getenv(name)
  if value is None:
    return default
  return value.casefold() in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Run the EVA backend services.")
  parser.add_argument("--host", default=os.getenv("EVA_HOST", "0.0.0.0"))
  parser.add_argument("--port", type=int, default=int(os.getenv("EVA_PORT", "8080")))
  parser.add_argument("--device", default=os.getenv("EVA_DEVICE", "/dev/ttyUSB0"))
  parser.add_argument("--camera-index", type=int, default=os.getenv("EVA_CAMERA_INDEX"))
  parser.add_argument(
    "--camera",
    action=argparse.BooleanOptionalAction,
    default=_env_bool("EVA_CAMERA", True),
  )
  parser.add_argument(
    "--vision",
    action=argparse.BooleanOptionalAction,
    default=_env_bool("EVA_VISION", False),
  )
  parser.add_argument(
    "--hardware-port",
    action=argparse.BooleanOptionalAction,
    default=_env_bool("EVA_HARDWARE_PORT", True),
  )
  parser.add_argument(
    "--debug", action="store_true", default=_env_bool("EVA_DEBUG", False)
  )
  parser.add_argument("--ssl-cert", type=Path, default=os.getenv("EVA_SSL_CERT"))
  parser.add_argument("--ssl-key", type=Path, default=os.getenv("EVA_SSL_KEY"))
  parser.add_argument(
    "--vision-weights",
    type=Path,
    default=Path(os.getenv("EVA_VISION_WEIGHTS", MODEL_DIR / "model_weights.ckpt")),
  )
  return parser


def _build_config(args: argparse.Namespace):
  from .config import EvaGlobalConfig

  model_config = None
  model_type = None
  if args.vision:
    from .models.transformers.focus.configuration_focus import FocusConfig

    model_config = FocusConfig(num_classes=6, operating_mode="light")
    model_type = "transformer"

  return EvaGlobalConfig(
    dev_path=args.device,
    camera_index=args.camera_index,
    enable_camera=args.camera,
    enable_vision=args.vision,
    enable_port=args.hardware_port,
    server_host=args.host,
    server_port=args.port,
    server_ssl_cert_path=args.ssl_cert,
    server_ssl_key_path=args.ssl_key,
    vision_model_type=model_type,
    vision_model_config=model_config,
    vision_pretrained_model_path=args.vision_weights if args.vision else None,
    debug=args.debug,
    _tracemalloc=args.debug,
    _display_img_size=(320, 240),
    _roboarm_init_thetas=[90.0, 42.5, -159.0, 270.0, 90.0, 0.0],
    _flash_interval=0.6,
    _use_ctrl_rot_fallback=True,
  )


def main(argv: Optional[Sequence[str]] = None) -> int:
  args = build_parser().parse_args(argv)
  mp.set_start_method("spawn", force=True)

  from .launcher import EvaLauncher

  EvaLauncher(config=_build_config(args)).run()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
