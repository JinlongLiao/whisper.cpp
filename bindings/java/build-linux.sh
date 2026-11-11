#!/bin/bash
# 在 Linux x86-64 上构建并替换 Java 原生资源；需 CMake、C/C++ 编译器、patchelf、readelf、Python 3。
# 用法：bash bindings/java/build-linux.sh [独立的 CMake 构建目录]
set -euo pipefail

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "请在 Linux x86-64 的低版本 GLIBC 构建环境中运行。" >&2
    exit 1
fi
for tool in cmake patchelf readelf python3; do
    command -v "$tool" >/dev/null || { echo "缺少工具：$tool" >&2; exit 1; }
done
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
build_dir="${1:-$repo_dir/build-java-linux}"
libraries=(ggml-base ggml-cpu ggml parakeet whisper)

# 与 LocalLib 的五库 CPU 资源契约保持一致；不链接未随 JAR 打包的可选后端。
cmake -S "$repo_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF -DGGML_CPU_ALL_VARIANTS=OFF \
    -DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
    -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=OFF
cmake --build "$build_dir" --parallel --target whisper parakeet

stage_dir="$(mktemp -d "${TMPDIR:-/tmp}/whisper-java-linux.XXXXXX")"
trap 'rm -rf "$stage_dir"' EXIT
for library in "${libraries[@]}"; do
    cp -L "$build_dir/bin/lib${library}.so" "$stage_dir/"
done

# 校验 ABI 后修复副本：DT_NEEDED、SONAME 与实际释放文件名一致，RUNPATH 仅指向同目录。
python3 - "$stage_dir" <<'PY'
import os
import pathlib
import re
import subprocess
import sys

directory = pathlib.Path(sys.argv[1])
for library in sorted(directory.glob('*.so')):
    header = subprocess.check_output(['readelf', '-h', str(library)], env=dict(os.environ, LC_ALL='C'), text=True)
    if not re.search(r'Machine:\s+Advanced Micro Devices X86-64', header) or 'ELF64' not in header:
        raise SystemExit('架构不是 Linux x86-64：' + str(library))
    versions = subprocess.check_output(['readelf', '--version-info', str(library)], text=True)
    required = [tuple(map(int, version.split('.'))) for version in re.findall(r'GLIBC_(\d+(?:\.\d+)+)', versions)]
    if required and max(required) > (2, 32):
        raise SystemExit(str(library) + ' 要求 GLIBC ' + '.'.join(map(str, max(required))) + '；请在 GLIBC <= 2.32 环境重新编译，不可修改版本标签绕过。')
    for dependency in subprocess.check_output(['patchelf', '--print-needed', str(library)], text=True).splitlines():
        name = re.sub(r'\.so(?:\.\d+)+$', '.so', pathlib.Path(dependency).name)
        if name.startswith(('libggml', 'libwhisper', 'libparakeet')):
            if not (directory / name).is_file():
                raise SystemExit('缺少内部依赖：' + dependency)
            subprocess.check_call(['patchelf', '--replace-needed', dependency, name, str(library)])
        elif '/' in dependency:
            raise SystemExit('存在外部绝对或相对路径依赖：' + dependency)
    subprocess.check_call(['patchelf', '--set-soname', library.name, '--set-rpath', '$ORIGIN', str(library)])
    print('ABI 和依赖修复完成：' + library.name)
PY

# 清除开发环境搜索路径，在新进程中检查五库是否可以从临时目录加载。
env -u LD_LIBRARY_PATH -u LD_PRELOAD python3 - "$stage_dir" <<'PY'
import ctypes
import pathlib
import sys

for library in ('ggml-base', 'ggml-cpu', 'ggml', 'parakeet', 'whisper'):
    ctypes.CDLL(str(pathlib.Path(sys.argv[1]) / ('lib' + library + '.so')))
    print('加载成功：' + library)
PY
cp "$stage_dir/"*.so "$script_dir/src/main/resources/linux-x86-64/"
echo "已更新 5 个 Linux x86-64 原生库；目标服务器的 C++/OpenMP 运行库及推理仍需验证。"
