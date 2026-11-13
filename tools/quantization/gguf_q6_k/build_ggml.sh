#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
COMMIT="90e3841d9f35a5bf3c112b9d71502bce77de5b32"

mkdir -p "$BUILD_DIR/source"
if [ ! -f "$BUILD_DIR/llama.cpp.tar.gz" ]; then
    curl --fail --location --retry 3 \
        "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/$COMMIT" \
        -o "$BUILD_DIR/llama.cpp.tar.gz"
fi
echo "aa83cc47271ee1ce1af43fd658d60332b74822a145a2f26109d950f09ddeb5a6  $BUILD_DIR/llama.cpp.tar.gz" | sha256sum -c -
tar -xzf "$BUILD_DIR/llama.cpp.tar.gz" -C "$BUILD_DIR/source" --strip-components=1
cmake -S "$BUILD_DIR/source" -B "$BUILD_DIR/compiled" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA=OFF -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build "$BUILD_DIR/compiled" --target ggml-base --parallel 4
cp "$BUILD_DIR/compiled/bin/libggml-base.so" "$SCRIPT_DIR/libggml-base.so"
