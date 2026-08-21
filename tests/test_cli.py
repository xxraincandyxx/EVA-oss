from backend.__main__ import _build_config, build_parser


def test_cli_builds_hardware_free_config():
  args = build_parser().parse_args(
    ["--no-camera", "--no-vision", "--no-hardware-port", "--port", "9000"]
  )

  config = _build_config(args)

  assert config.server_port == 9000
  assert config.enable_camera is False
  assert config.enable_vision is False
  assert config.enable_port is False
  assert config.vision_model_config is None


def test_cli_reads_environment(monkeypatch):
  monkeypatch.setenv("EVA_HOST", "127.0.0.1")
  monkeypatch.setenv("EVA_PORT", "9100")
  monkeypatch.setenv("EVA_CAMERA", "false")

  args = build_parser().parse_args([])

  assert args.host == "127.0.0.1"
  assert args.port == 9100
  assert args.camera is False
