#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$script_dir/../.." && pwd)"
build_dir="$root/build/vulkan-layer"
mkdir -p "$build_dir"

src="$script_dir/vd_capture_layer.cpp"
out_lib="$build_dir/libVkLayer_VD_capture.so"
out_manifest="$build_dir/VkLayer_VD_capture.json"

# Regenerate the embedded brightness shaders when glslc exists; the checked-in header is used otherwise.
if command -v glslc >/dev/null 2>&1; then
  tmp_vert="$build_dir/bright.vert.spv.inc"
  tmp_frag="$build_dir/bright.frag.spv.inc"
  glslc -O -mfmt=num "$script_dir/shaders/bright.vert" -o "$tmp_vert"
  glslc -O -mfmt=num "$script_dir/shaders/bright.frag" -o "$tmp_frag"
  {
    echo '// Generated from shaders/bright.{vert,frag} by build_layer.sh (glslc -O). Do not edit.'
    echo '#pragma once'
    echo '#include <cstdint>'
    echo 'static const uint32_t kBrightVertSpv[] = {'
    cat "$tmp_vert"
    echo '};'
    echo 'static const uint32_t kBrightFragSpv[] = {'
    cat "$tmp_frag"
    echo '};'
  } > "$build_dir/bright_spv.h"
  cmp -s "$build_dir/bright_spv.h" "$script_dir/bright_spv.h" || cp "$build_dir/bright_spv.h" "$script_dir/bright_spv.h"
fi

echo "[BUILD] Compiling Vulkan Implicit Layer..."
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -fPIC -shared \
    -fvisibility=hidden -pthread \
    -Wl,-Bsymbolic \
    -Wl,-soname,libVkLayer_VD_capture.so \
    -static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL \
    -I"$script_dir" \
    "$src" -o "$out_lib"

echo "[BUILD] Shared library compiled at: $out_lib"
sed "s|@LAYER_LIBRARY_PATH@|$out_lib|g" "$script_dir/VkLayer_VD_capture.json.in" > "$out_manifest"
echo "[BUILD] Manifest generated at: $out_manifest"
echo "[BUILD] Success!"
