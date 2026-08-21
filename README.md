# EVA

EVA is an edge-deployed vision-language-action system for a six-axis robot.
The application combines a Python orchestration service, a C++ motion-control
library exposed through pybind11, and a React/Three.js operator interface.

## Repository layout

```text
dynamo_/               C++ kinematics and device-control library
src/backend/            Python application, web API, vision, and model code
src/frontend/           React operator interface
stubs/                  Type information for the native Python module
tests/                  Deterministic Python unit tests
experiments/            Reproducible training reports
requirements/           Runtime, ML, development, and edge dependency sets
docs/                   Architecture and deployment boundaries
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for component ownership,
runtime flow, and persistent-data locations.

Runtime data, credentials, model weights, captured images, build products, and
local databases are intentionally excluded from version control.

## Development setup

Python 3.9+, CMake 3.20+, a C++17 compiler, and Node.js 20.19+ or 22.12+
are required.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements/dev.txt
.venv/bin/pip install --no-deps -e .
npm --prefix src/frontend ci
PYTHON=.venv/bin/python ./build.sh
```

For a minimal KV260/Vitis AI installation, use `requirements/edge.txt` instead
of the development requirements. The Vitis `xir` and `vart` packages must be
available from the board's configured package source.

## Run

Start the services in separate terminals:

```bash
.venv/bin/python -m backend
VITE_EVA_BACKEND_URL=http://127.0.0.1:8080 npm --prefix src/frontend run dev
```

The backend listens on port `8080` by default and the Vite development server
uses port `5173`. A production frontend served by the backend uses the current
origin and does not need `VITE_EVA_BACKEND_URL`. Hardware devices, model paths,
and remote LLM credentials are deployment configuration and must not be
committed to the repository.

Run `.venv/bin/python -m backend --help` for device, camera, vision, TLS, and
network options. Every option also has an `EVA_*` environment variable for
service deployments.

## Checks

```bash
make lint          # Python and frontend static checks
make format-check  # Python and C++ formatting checks
make test          # Python, C++, and frontend build checks
make check         # Complete verification suite
```

The C++ suite can be built without Python bindings when only the control core
is needed:

```bash
cmake -S . -B build -DEVA_BUILD_PYTHON=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Security

Supply secrets through environment variables or a deployment secret manager.
Do not commit TLS keys, API tokens, device credentials, or generated databases.

## GitHub Pages

The `deploy-pages.yml` workflow publishes the React interface from `master`.
GitHub Pages is static hosting: robot control, camera streams, Socket.IO, and
agent requests remain offline unless the frontend is built with
`VITE_EVA_BACKEND_URL` pointing to a separately deployed HTTPS backend.
