#!/usr/bin/env python3
"""下载中英日韩各三条 FLEURS 测试语音及一条粤语官方演示。

FLEURS 使用固定版本及归档顺序前三条匹配参考稿的音频，流式读取后关闭连接，
无需下载整个语料库。样本、参考稿、来源、许可证和 SHA-256 保存到脚本旁 model 下。
粤语示例未单独标明音频许可证，仅用于本地测试，不作为可再分发的标准数据集。
来源：https://huggingface.co/datasets/google/fleurs
粤语参考：https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/pretrained.html
"""
import concurrent.futures
import hashlib
import json
import pathlib
import tarfile
import urllib.request
import time

ROOT = pathlib.Path(__file__).resolve().parent / 'model/标准语音测试集'
REVISION = '70bb2e84b976b7e960aa89f1c648e09c59f894dd'
CANTONESE_REFERENCE = '今次寻寻觅觅，终于揾到my princess，肯借个场俾我哋玩。你知啦，喺香港地喺繁忙时间要揾个场嚟拍嘢系非常之难嘅。再一次多谢你哋，亦都好多谢片入边嘅每一个人。下一次我哋斗啲咩好？喺下面留言话我哋知啦。拜拜。'

def download_fleurs_samples(pair):
    """下载固定语种的前三条语音并保留发布方规范化参考稿；网络错误直接传播。"""
    code, config = pair
    metadata = ROOT / (code + '.json')
    if metadata.exists():
        cached = json.loads(metadata.read_text(encoding='utf-8'))
        if len(cached) == 3 and all(
                row.get('revision') == REVISION
                and (ROOT / pathlib.Path(row['file']).name).exists()
                and hashlib.sha256((ROOT / pathlib.Path(row['file']).name).read_bytes()).hexdigest() == row['sha256']
                for row in cached):
            print(code, '已校验并复用', flush=True)
            return cached
    base = 'https://huggingface.co/datasets/google/fleurs/resolve/' + REVISION + '/data/' + config + '/'
    with urllib.request.urlopen(base + 'test.tsv', timeout=60) as response:
        text = response.read().decode()
    rows = {row.split('\t')[1]: row.split('\t') for row in text.splitlines()}
    records = []
    with urllib.request.urlopen(base + 'audio/test.tar.gz', timeout=120) as response:
        with tarfile.open(fileobj=response, mode='r|gz') as archive:
            for member in archive:
                name = pathlib.Path(member.name).name
                if name not in rows or not member.isfile():
                    continue
                data = archive.extractfile(member).read()
                row = rows[name]
                dest = ROOT / (code + '-' + name)
                dest.write_bytes(data)
                records.append(dict(language=code, file='model/标准语音测试集/' + dest.name,
                                    reference=row[3], original_reference=row[2],
                                    source=base + 'audio/test.tar.gz', member=member.name,
                                    revision=REVISION, license='CC-BY-4.0',
                                    sha256=hashlib.sha256(data).hexdigest()))
                if len(records) == 3:
                    break
    if len(records) != 3:
        raise ValueError('未取得三条完整样本：' + config)
    (ROOT / (code + '.json')).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(code, len(records), flush=True)
    return records

def download_language_with_retry(pair):
    """对公开下载的临时网络错误最多重试三次；最终保留完整异常。"""
    for attempt in range(3):
        try:
            return download_fleurs_samples(pair)
        except (OSError, EOFError):
            if attempt == 2:
                raise
            time.sleep(1)

def main():
    """并行下载四个独立归档，最后写入完整清单；失败时不发布不完整清单。"""
    ROOT.mkdir(parents=True, exist_ok=True)
    pairs = [('zh', 'cmn_hans_cn'), ('en', 'en_us'), ('ja', 'ja_jp'), ('ko', 'ko_kr')]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        records = [record for batch in pool.map(download_language_with_retry, pairs) for record in batch]
    url = 'https://k2-fsa.github.io/sherpa/_static/qwen3-asr-0.6b/cantonese.wav'
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    (ROOT / 'yue-cantonese.wav').write_bytes(data)
    records.append(dict(language='yue', file='model/标准语音测试集/yue-cantonese.wav',
                        reference=CANTONESE_REFERENCE, source=url,
                        reference_source='https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/pretrained.html',
                        license='示例音频未单独标明许可证，仅本地评测',
                        sha256=hashlib.sha256(data).hexdigest()))
    # 参考文本紧邻音频保存，便于试听时直接对照，不必打开 JSON 清单。
    for record in records:
        audio_name = pathlib.Path(record['file']).stem
        (ROOT / (audio_name + '.参考文本.txt')).write_text(record['reference'] + '\n', encoding='utf-8')
    (ROOT / 'manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print('下载完成，共', len(records), '条')

if __name__ == '__main__':
    main()
