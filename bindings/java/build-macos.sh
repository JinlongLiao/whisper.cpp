#!/bin/bash
# 构建 Java JAR 使用的 Apple Silicon 动态库；失败时不复制未完成的资源。
# 用法：bash bindings/java/build-macos.sh [独立的 CMake 构建目录]
set -euo pipefail

if [[ "$(uname -s)" != Darwin ]]; then
    echo "此脚本只能在 macOS 上运行。" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
build_dir="${1:-$repo_dir/build-java-macos}"
libraries=(ggml-base ggml-cpu ggml-blas ggml-metal ggml parakeet whisper)

# 不复用日常 build 的宿主机默认目标；Metal 源码嵌入库中，避免依赖构建目录的资源。
# darwin 资源当前只支持 arm64，不在同一路径写入 Intel 或宿主机专用指令集制品。
cmake -S "$repo_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=Apple \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_EXAMPLES=OFF
cmake --build "$build_dir" --config Release --parallel --target whisper parakeet

# 在临时目录修改副本，不破坏 CMake 维护的版本化文件与符号链接。
stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/whisper-java-macos.XXXXXX")"
trap 'rm -rf "$stage_dir"' EXIT
for library in "${libraries[@]}"; do
    cp -L "$build_dir/bin/lib${library}.dylib" "$stage_dir/"
done

for library in "${libraries[@]}"; do
    dylib="$stage_dir/lib${library}.dylib"
    if [[ "$(lipo -archs "$dylib")" != arm64 ]]; then
        echo "架构不符合 arm64 资源约定：$dylib" >&2
        exit 1
    fi
    if [[ "$(vtool -show-build "$dylib" | awk '$1 == "minos" {print $2}')" != 15.0 ]]; then
        echo "最低系统版本必须是 15.0，请检查构建配置：$dylib" >&2
        exit 1
    fi

    # 修改会使已有签名失效；完成所有 Mach-O 修改后重新进行 ad-hoc 签名。
    codesign --remove-signature "$dylib"
    install_name_tool -id "@loader_path/lib${library}.dylib" "$dylib"
    while IFS= read -r dependency; do
        name="${dependency##*/}"
        case "$name" in
            libggml*.dylib|libwhisper*.dylib|libparakeet*.dylib)
                # CMake 版本化依赖映射到 LocalLib 实际释放的无版本文件名。
                name="$(printf '%s\n' "$name" | sed -E 's/(\.[0-9]+)+\.dylib$/.dylib/')"
                if [[ ! -f "$stage_dir/$name" ]]; then
                    echo "缺少内部依赖：$dependency" >&2
                    exit 1
                fi
                install_name_tool -change "$dependency" "@loader_path/$name" "$dylib"
                ;;
            *)
                case "$dependency" in
                    /usr/lib/*|/System/Library/*) ;;
                    *) echo "存在未打包的外部依赖：$dependency" >&2; exit 1 ;;
                esac
                ;;
        esac
    done < <(otool -L "$dylib" | sed '1,2d; s/^[[:space:]]*//; s/ (compatibility version.*//')

    # 同目录依赖不再使用 RPATH；删除所有构建目录搜索路径，避免本机误加载掩盖缺陷。
    while IFS= read -r rpath; do
        install_name_tool -delete_rpath "$rpath" "$dylib"
    done < <(otool -l "$dylib" | awk '/cmd LC_RPATH/ {getline; getline; sub(/^ *path /, ""); sub(/ \(offset.*$/, ""); print}')
    codesign --force --sign - "$dylib"
    codesign --verify --strict "$dylib"
done

# 全部库加载成功才覆盖打包资源；此检查不等价于旧版 macOS 上的推理验证。
/usr/bin/python3 - "$stage_dir" <<'PY'
import ctypes
import pathlib
import sys

for library in ("ggml-base", "ggml-cpu", "ggml-blas", "ggml-metal", "ggml", "parakeet", "whisper"):
    ctypes.CDLL(str(pathlib.Path(sys.argv[1]) / ("lib" + library + ".dylib")))
    print("加载成功：" + library)
PY
cp "$stage_dir/"*.dylib "$script_dir/src/main/resources/darwin/"
echo "已更新 7 个 macOS arm64 原生库（最低 macOS 15.0）；可继续构建 Java JAR。"
