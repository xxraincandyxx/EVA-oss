#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${project_root}/build"
python_bin="${PYTHON:-python3}"

cmake -S "${project_root}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="${python_bin}"
cmake --build "${build_dir}" --parallel
ctest --test-dir "${build_dir}" --output-on-failure

(
  cd "${build_dir}"
  "${python_bin}" -m pybind11_stubgen \
    --output-dir "${project_root}/stubs" \
    dynamo_
)

mkdir -p "${project_root}/src/backend/lib"
find "${build_dir}" -maxdepth 1 -type f \( -name 'dynamo_*.so' -o -name 'dynamo_*.pyd' \) \
  -exec ln -sf {} "${project_root}/src/backend/lib/" \;
ln -sfn "../../../stubs/dynamo_.pyi" "${project_root}/src/backend/lib/dynamo_.pyi"
ln -sf "${build_dir}/compile_commands.json" "${project_root}/compile_commands.json"
