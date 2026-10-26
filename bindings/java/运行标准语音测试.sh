#!/usr/bin/env bash
# 一键下载固定样本、运行五模型评测并生成本地报告；任一步骤失败均返回非零状态。
set -euo pipefail
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_directory"

if [[ $# -gt 1 ]] || [[ $# -eq 1 && "$1" != '--report-only' ]]; then
    echo '用法：bash 运行标准语音测试.sh [--report-only]' >&2
    exit 2
fi
command -v python3 >/dev/null || { echo '缺少 python3' >&2; exit 1; }
# mkdir 为原子操作；同一项目只能有一个入口生成结果，避免并发覆盖。
mkdir -p target
lock_directory='target/standard-automation.lock'
if ! mkdir "$lock_directory" 2>/dev/null; then
    echo '已有自动化任务占用结果目录。若上次进程被强制终止，请确认没有任务运行后删除 target/standard-automation.lock 空目录。' >&2
    exit 1
fi
trap 'rmdir "$lock_directory"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "${1:-}" == '--report-only' ]]; then
    python3 标准语音模型测试.py --report-only
    echo "报告已更新：$script_directory/测试结果/标准语音五模型测试报告.md"
    exit 0
fi

# 在网络下载前检查推理和解码工具，避免下载完成后才发现环境不可用。
if [[ ! -x ../../build/bin/whisper-cli ]]; then
    echo '缺少 ../../build/bin/whisper-cli，请按《标准语音自动化测试使用说明.md》先构建。' >&2
    exit 1
fi
if [[ -z "${FFMPEG:-}" ]]; then
    FFMPEG="$(command -v ffmpeg || true)"
fi
if [[ -z "$FFMPEG" || ! -x "$FFMPEG" ]]; then
    echo '请将 ffmpeg 加入 PATH，或设置 FFMPEG=/完整路径/ffmpeg。' >&2
    exit 1
fi
export FFMPEG
mkdir -p target
run_log="target/standard-automation-$(date '+%Y%m%d-%H%M%S').log"
# pipefail 保留 Python、下载器或模型推理的失败状态，tee 同步输出完整日志。
{
    for model in tiny base; do
        if [[ ! -s "../../models/ggml-$model.bin" ]]; then
            bash ../../models/download-ggml-model.sh "$model"
        fi
    done
    for quantization in q3_k q4_k q5_k; do
        # 既有下载器会校验大小与 SHA-256；已存在且校验通过时直接复用。
        bash download-dialect-model.sh "orbination-$quantization"
    done
    python3 下载标准语音样本.py
    python3 标准语音模型测试.py
    echo "测试完成，报告：$script_directory/测试结果/标准语音五模型测试报告.md"
    echo "逐轮数据：$script_directory/测试结果/标准语音五模型/逐轮结果.csv"
} 2>&1 | tee "$run_log"
echo "运行日志：$script_directory/$run_log"
