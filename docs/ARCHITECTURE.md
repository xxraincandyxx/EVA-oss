# Architecture

EVA has three deployable layers with one-way ownership boundaries:

```text
React operator UI
  -> Socket.IO events and HTTP video streams
Python orchestration service
  -> pybind11 module
C++ control library
  -> serial and GPIO devices
```

## Components

| Path | Responsibility |
| --- | --- |
| `src/frontend` | Operator workflows, local UI state, 3D visualization, and backend transport |
| `src/backend/web` | HTTP routes and Socket.IO protocol adapters |
| `src/backend` | Service lifecycle, scheduling, device allocation, and runtime configuration |
| `src/backend/vision` | Camera capture and optional inference backends |
| `src/backend/models` | Optional Focus model implementation |
| `src/backend/dl_utils` | Training datasets and visualization utilities |
| `dynamo_` | Native kinematics and hardware control |

The frontend never imports backend implementation details. Web handlers adapt
transport payloads to backend services, while hardware operations remain behind
`EvaConsole` and the native `dynamo_` binding.

## Runtime

`python -m backend` parses CLI and `EVA_*` settings, builds one
`EvaGlobalConfig`, and starts `EvaLauncher`. The launcher owns the allocator and
its camera, vision, console, and web-service lifecycles. Shutdown is coordinated
from the launcher so resources are released once.

The React app creates one Socket.IO connection. Zustand owns shared server state
and listener cleanup; pages are loaded on demand. Control and FPV load the
Three.js simulation only when those workflows are opened.

## Filesystem

Source directories are read-only at runtime. Generated state belongs outside
the repository:

| Setting | Default |
| --- | --- |
| `EVA_DATA_DIR` | `$XDG_DATA_HOME/eva` or `~/.local/share/eva` |
| `EVA_STATE_DIR` | `$XDG_STATE_HOME/eva` or `~/.local/state/eva` |
| `EVA_MODEL_DIR` | `src/weights` for development |
| `EVA_TRAINING_DATA_DIR` | The EVA data directory under `training` |
| `EVA_TRAINING_CACHE_DIR` | The EVA data directory under `cache/training` |

Databases use the data directory. Logs use the state directory. TLS keys,
weights, captured frames, databases, and generated builds are never source
artifacts.

## Dependency Profiles

`requirements/base.txt` is the service runtime. `requirements/ml.txt` adds
training and transformer dependencies. `requirements/edge.txt` adds board-only
Vitis AI packages, and `requirements/dev.txt` contains the complete local test
and formatting toolchain.
