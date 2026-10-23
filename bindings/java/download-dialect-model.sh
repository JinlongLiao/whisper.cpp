#!/usr/bin/env bash
# 下载模型到脚本所在 Java 项目的 model/，与调用者当前目录无关。
# 模型来源、时间与使用限制见《粤语与台语模型说明.md》。
set -euo pipefail

# 输出支持的模型；无参数或 --list 时不访问网络。
list_dialect_models() {
    cat <<'MODELS'
用法：bash download-dialect-model.sh <模型编号>
下载目录：脚本所在项目的 model/

yue-base                 粤语 Base，148 MB
yue-small-q5_1           粤语 Small Q5_1，190 MB
zh-tw-turbo-q8_0         台湾华语 Turbo Q8_0，874 MB
zh-tw-turbo-f16          台湾华语 Turbo F16，1.62 GB
nan-breeze26-q5_0        台语 Breeze-ASR-26 Q5_0，1.08 GB
nan-breeze26-q4_0        台语 Breeze-ASR-26 Q4_0，889 MB
MODELS
}

if [[ $# -eq 0 ]]; then
    list_dialect_models
    exit 0
fi
if [[ $# -ne 1 ]]; then
    list_dialect_models >&2
    exit 2
fi
case "$1" in
    --list|--help|-h)
        list_dialect_models
        exit 0
        ;;
    yue-base)
        model_repository='wabisabisocial/whisper-base-cantonese-ggml'
        model_revision='6d3f8d2cc433703969c361c9b9db7f9894fec461'
        remote_filename='ggml-base-yue.bin'
        local_filename='ggml-base-yue.bin'
        expected_bytes=147951482
        expected_sha256='7a1c2555abd1645ba5a737efaad055393026ace58504ed37dc9724fdaea9c865'
        ;;
    yue-small-q5_1)
        model_repository='betteropts/whisper-small-cantonese-ggml-q5_1'
        model_revision='5cf63f89f6ad31f661267901fef2f7c4affb9786'
        remote_filename='ggml-small-cantonese-q5_1-v1.bin'
        local_filename='ggml-small-cantonese-q5_1-v1.bin'
        expected_bytes=190085487
        expected_sha256='157efd532a10c8b7d4351e46e514263bfe1c69072a525c58d3aed39b271e39c6'
        ;;
    zh-tw-turbo-q8_0)
        model_repository='illitan/whisper-large-v3-turbo-zh-TW-ggml'
        model_revision='8e8a099b61d4ec8ace5a5966e625efe2b2c08a20'
        remote_filename='ggml-whisper-large-v3-turbo-zh-TW-q8_0.bin'
        local_filename='ggml-whisper-large-v3-turbo-zh-TW-q8_0.bin'
        expected_bytes=874188075
        expected_sha256='b97084becc150cde926adfda323d66d89bf8a279eb94662e1bfad1b73ab90dbe'
        ;;
    zh-tw-turbo-f16)
        model_repository='illitan/whisper-large-v3-turbo-zh-TW-ggml'
        model_revision='8e8a099b61d4ec8ace5a5966e625efe2b2c08a20'
        remote_filename='ggml-whisper-large-v3-turbo-zh-TW.bin'
        local_filename='ggml-whisper-large-v3-turbo-zh-TW.bin'
        expected_bytes=1624555275
        expected_sha256='00a96ea038acdb7452ae02edc2e69f62e770b2d836fb3e05672adc1ede80cdff'
        ;;
    nan-breeze26-q5_0)
        model_repository='phate334/Breeze-ASR-26-GGML'
        model_revision='2665d49ba0025177bd81c61c1b0ec2cd875735a7'
        remote_filename='ggml-model-q5_0.bin'
        local_filename='ggml-breeze-asr-26-q5_0.bin'
        expected_bytes=1080732108
        expected_sha256='60f25e3a21feca12ec082e6d36f08f94455d9900d6343f7fcb2906f71cc7c449'
        ;;
    nan-breeze26-q4_0)
        model_repository='phate334/Breeze-ASR-26-GGML'
        model_revision='2665d49ba0025177bd81c61c1b0ec2cd875735a7'
        remote_filename='ggml-model-q4_0.bin'
        local_filename='ggml-breeze-asr-26-q4_0.bin'
        expected_bytes=888932908
        expected_sha256='0e08c5a2d3a5ba4bae7f85d02212e887537f78d7e623d8694313e2dd65de22ec'
        ;;
    *)
        printf '未知模型编号：%s\n' "$1" >&2
        list_dialect_models >&2
        exit 2
        ;;
esac

command -v curl >/dev/null 2>&1 || { printf '需要安装 curl。\n' >&2; exit 1; }
if command -v shasum >/dev/null 2>&1; then
    sha256_command=(shasum -a 256)
elif command -v sha256sum >/dev/null 2>&1; then
    sha256_command=(sha256sum)
else
    printf '需要 shasum 或 sha256sum 校验模型。\n' >&2
    exit 1
fi

project_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
model_directory="$project_directory/model"
mkdir -p "$model_directory"
model_destination="$model_directory/$local_filename"

# 先核对大小再计算完整 SHA-256；已有损坏文件不会被当作成功下载。
verify_dialect_model() {
    local candidate_path="$1"
    local actual_bytes actual_sha256
    [[ -f "$candidate_path" ]] || return 1
    actual_bytes="$(wc -c < "$candidate_path" | tr -d '[:space:]')"
    [[ "$actual_bytes" == "$expected_bytes" ]] || return 1
    actual_sha256="$("${sha256_command[@]}" "$candidate_path")"
    actual_sha256="${actual_sha256%% *}"
    [[ "$actual_sha256" == "$expected_sha256" ]]
}

if verify_dialect_model "$model_destination"; then
    printf '模型已存在且校验通过：%s\n' "$model_destination"
    exit 0
fi

# 临时文件与目标在同一目录，校验完成后再重命名，避免中断留下可加载的残缺模型。
download_temp="$(mktemp "$model_directory/.${local_filename}.part.XXXXXX")"
trap 'rm -f -- "$download_temp"' EXIT
trap 'exit 130' INT
trap 'exit 143' HUP TERM
model_url="https://huggingface.co/$model_repository/resolve/$model_revision/$remote_filename"
printf '下载：%s\n目标：%s\n' "$model_url" "$model_destination"
curl --fail --location --retry 3 --retry-delay 2 --connect-timeout 15 --max-time 1800 \
    --output "$download_temp" "$model_url"
if ! verify_dialect_model "$download_temp"; then
    printf '模型大小或 SHA-256 校验失败，保留原目标文件：%s\n' "$model_destination" >&2
    exit 1
fi
mv -f -- "$download_temp" "$model_destination"
printf '下载完成，大小与 SHA-256 校验通过：%s\n' "$model_destination"
