#!/bin/bash
# Build script for zyx-tt-runtime
set -e

TT_METAL=${TT_METAL_HOME:-$HOME/Dev/cpp/tt-metal}
BUILD_DIR=${TT_METAL}/build_Release

CXX=g++
CXXFLAGS="-std=c++20 -Wall -Wextra -O3"

INCLUDES=(
    -I "${TT_METAL}/tt_metal/include"
    -I "${BUILD_DIR}/include"
    -I "${TT_METAL}/tt_metal/api/tt-metalium"
    -I "${TT_METAL}/src"
    -I "${BUILD_DIR}/include/spdlog/include"
    -I "${TT_METAL}/.cpmcache/spdlog/b1c2586bb5c35a7929362e87f62433eb68206873/include"
    -I "${BUILD_DIR}/include/fmt/include"
)

LIBS=(
    -L "${BUILD_DIR}/lib"
    -ltt_metal
    -ltt-umd
    -ltt_stl
    -lflatbuffers
    -lfmt
    -lspdlog
    -Wl,-rpath,"${BUILD_DIR}/lib"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE="${SCRIPT_DIR}/src/main.cpp"
OUTPUT="${SCRIPT_DIR}/zyx-tt-runtime"

echo "Building zyx-tt-runtime..."
$CXX $CXXFLAGS "${INCLUDES[@]}" "$SOURCE" "${LIBS[@]}" -o "$OUTPUT"
echo "Build successful: $OUTPUT"
