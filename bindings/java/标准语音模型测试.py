#!/usr/bin/env python3
"""用本地 whisper-cli 串行评测固定公开样本，保存原始输出及字符/词编辑距离。

从 bindings/java 运行；需已下载模型及 model/标准语音测试集/manifest.json。
每条语音独立启动进程三次；耗时取原生 total-load 的中位数，含初始化余项，
不等价于 Java 常驻模型的纯推理耗时。语种检测另运行一次，不指定真实标签。
"""
import json,pathlib,subprocess,time,re,statistics,unicodedata,wave,hashlib
import argparse
import os
import shutil
import platform
from datetime import datetime
ROOT=pathlib.Path('测试结果/标准语音五模型')
MODELS={'tiny':'../../models/ggml-tiny.bin','base':'../../models/ggml-base.bin',**{q:'model/ggml-orbination-turbo-'+q+'.bin' for q in ['q3_k','q4_k','q5_k']}}
def normalize(text):
    """只保留 Unicode 字母和数字，统一 NFKC 与大小写；不做简繁、数字读法转换。"""
    return ''.join(c if c.isalnum() else ' ' for c in unicodedata.normalize('NFKC',text).lower())
def distance(ref,hyp):
    """计算 Levenshtein 编辑距离；插入、删除、替换的代价均为一。"""
    row=list(range(len(hyp)+1))
    for i,a in enumerate(ref,1):
        nxt=[i]
        for j,b in enumerate(hyp,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(a!=b)))
        row=nxt
    return row[-1]
def timing(log,key):
    """读取 whisper_print_timings 原生毫秒，缺失即失败，不伪造耗时。"""
    return float(re.search(r'\b'+key+r' time\s*=\s*([\d.]+) ms',log).group(1))/1000

def write_report(results, cases):
    """生成逐语言汇总和完整逐条对照；文本分数使用第一轮，性能取三轮中位数。"""
    import csv
    names = {'zh': '普通话', 'en': '英语', 'yue': '粤语', 'ja': '日语', 'ko': '韩语'}
    lines = ['# 标准语音五模型测试报告', '',
             f"测试日期：{datetime.fromtimestamp((ROOT / 'results.json').stat().st_mtime).date()}；平台：{platform.system()}／{platform.machine()}，本项目已构建的 whisper-cli，8 线程、Beam Search 5、自动语种、原文转写、关闭 Flash Attention、不继承上下文。实际计算后端见每轮原生日志。", '',
             '样本共 13 条：Google FLEURS 测试集的普通话、英语、日语、韩语各 3 条（CC-BY-4.0）；另加 sherpa 官方演示的一条粤语样本及其 Ground truth。粤语示例不是 FLEURS 的同分布样本，音频未单独标明许可证，仅作本地诊断，不随报告再分发。', '',
             '来源：[Google FLEURS](https://huggingface.co/datasets/google/fleurs)，[粤语音频及参考稿](https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/pretrained.html)。固定样本文件名、参考稿、来源和文件 SHA-256 见 [样本清单](标准语音五模型/样本清单.tsv)。', '',
             '## 指标与边界', '',
             '- 文本：英语计算 WER；其他语种计算 CER。先做 Unicode NFKC、转小写、移除标点；CER 另移除空格，WER 按空格分词。不转换简繁、同义字或数字读法。编辑距离＝替换＋删除＋插入，除以参考单位数，越低越好；可能大于 100%，不等于人工语义准确率。',
             '- 汇总文本分数按参考字符／词数加权；取第一轮完整输出，另外两轮原文一并保存。FLEURS 使用官方 normalized transcription，未人工重新核听修订。粤语参考为简体，输出繁体也会计为字符差异，因此其 CER 需结合完整文本判断。',
             '- 语种：每样本另运行一次原生自动检测；精确匹配 zh／en／ja／ko／yue。候选概率不是准确率。Tiny／Base 不支持独立 yue 标签，粤语一栏标为不支持；Large-v3 系列按 yue 统计。',
             '- 性能：串行测试，每条转写独立进程重复三次，取原生 total time 减 load time 的中位数；包含语种检测及部分初始化，不含 FFmpeg 解码。另保存加载时间及端到端进程耗时；不是常驻 Java 模型的纯推理耗时，不能直接与旧报告比较。语种检测同样取独立 -dl 进程的 total-load，仅运行一次。',
             '- 仅 13 条短样本，适合冒烟对比，不能视为整个公开测试集的 benchmark 或泛化准确率。旧 cn_cantonese.opus 已由用户确认是空音频，未加入本次语音准确性评测。', '',
             '## 按语种文本错误率', '',
             '| 模型 | 普通话 CER | 英语 WER | 日语 CER | 韩语 CER | 粤语 CER（单条，含简繁差异） |',
             '|---|---:|---:|---:|---:|---:|']
    for model in MODELS:
        scores = []
        for lang in ['zh', 'en', 'ja', 'ko', 'yue']:
            group = [r['trials'][0] for r in results if r['model'] == model and r['expected_language'] == lang]
            scores.append(f"{100 * sum(t['errors'] for t in group) / sum(t['reference_units'] for t in group):.2f}%")
        lines.append('| ' + model + ' | ' + ' | '.join(scores) + ' |')
    lines += ['', '## 语种识别与性能', '',
              '| 模型 | 中英日韩语种命中 | 粤语首选语种／概率 | 粤语标签命中 | 13 条转写中位耗时之和 | 总音频时长 | RTF | 单条语种检测平均耗时 |',
              '|---|---:|---|---|---:|---:|---:|---:|']
    for model in MODELS:
        group = [r for r in results if r['model'] == model]
        common = [r for r in group if r['expected_language'] != 'yue']
        yue = next(r for r in group if r['expected_language'] == 'yue')
        total = sum(r['median_native_s'] for r in group)
        duration = sum(r['duration_s'] for r in group)
        hit = sum(r['expected_language'] == r['detected_language'] for r in common)
        yue_hit = '不支持独立标签' if model in ['tiny', 'base'] else str(int(yue['detected_language'] == 'yue')) + '/1'
        lines.append(f"| {model} | {hit}/12 | {yue['detected_language']}／{yue['probability']:.4f} | {yue_hit} | {total:.3f} 秒 | {duration:.3f} 秒 | {total/duration:.4f} | {statistics.mean(r['lid_s'] for r in group):.3f} 秒 |")
    varied = sum(len(set(t['text'] for t in r['trials'])) > 1 for r in results)
    lines += ['', f'65 个模型／样本组合中，有 {varied} 个组合的三轮转写文本不完全相同。以下使用第一轮；原始三轮 JSON、日志、文本均在 `标准语音五模型/<模型>/<样本>/`。', '', '## 完整逐条对照', '']
    for case in cases:
        lines += [f"### {names[case['language']]}：{case['id']}", '',
                  f"音频：{case['duration_s']:.3f} 秒。参考稿：{case['reference']}", '',
                  '| 模型 | 检测语种／概率 | 检测耗时 | 转写耗时（三轮中位） | CER／WER | 第一轮完整转写 |',
                  '|---|---|---:|---:|---:|---|']
        for r in [r for r in results if r['id'] == case['id']]:
            t = r['trials'][0]
            text = t['text'].replace('|', '\\|').replace('\n', ' ')
            lines.append(f"| {r['model']} | {r['detected_language']}／{r['probability']:.4f} | {r['lid_s']:.3f} 秒 | {r['median_native_s']:.3f} 秒 | {100*t['error_rate']:.2f}% | {text} |")
        lines.append('')
    lines += ['## 复现', '', '在 bindings/java 运行：', '', '```bash', 'python3 下载标准语音样本.py', 'python3 标准语音模型测试.py', '```', '',
              '五个模型须已存在于脚本声明的固定路径。下载脚本仅下载音频；测试不请求远程模型服务，失败会传播并保留日志。性能测试需允许 Metal 访问且避免其他 GPU 任务。']
    with (ROOT / '样本清单.tsv').open('w', encoding='utf-8-sig', newline='') as output:
        writer = csv.writer(output, delimiter='\t')
        writer.writerow(['样本', '参考语种', '时长秒', 'SHA256', '来源', '归档成员', '许可证', '参考稿'])
        for case in cases:
            writer.writerow([case['id'], case['language'], case['duration_s'], case['sha256'],
                             case['source'], case.get('member', ''), case['license'], case['reference']])
    (ROOT.parent / '标准语音五模型测试报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with (ROOT / '逐轮结果.csv').open('w', encoding='utf-8-sig', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['模型','样本','参考语种','检测语种','候选概率','检测耗时秒','轮次','音频秒','原生耗时秒','加载秒','进程秒','CER或WER','参考稿','转写文本'])
        for r in results:
            for i, t in enumerate(r['trials'], 1):
                writer.writerow([r['model'],r['id'],r['expected_language'],r['detected_language'],r['probability'],r['lid_s'],i,r['duration_s'],t['native_s'],t['load_s'],t['process_s'],t['error_rate'],r['reference'],t['text']])

def main():
    """完成五模型的同输入三轮测试；单次失败立即保留日志并终止。"""
    cases=json.loads(pathlib.Path('model/标准语音测试集/manifest.json').read_text())
    ffmpeg = os.environ.get('FFMPEG') or shutil.which('ffmpeg')
    if not ffmpeg:
        raise FileNotFoundError('未找到 ffmpeg，请将其加入 PATH 或设置 FFMPEG 为可执行文件路径')
    ROOT.mkdir(parents=True, exist_ok=True)
    pcm=pathlib.Path('target/standard-audio-pcm');pcm.mkdir(parents=True,exist_ok=True)
    for case in cases:
        source=pathlib.Path(case['file']);case['id']=source.stem
        actual_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_sha256 != case['sha256']:
            raise ValueError('音频 SHA-256 与下载清单不一致：' + str(source))
        dest=pcm/source.name
        subprocess.run([ffmpeg,'-v','error','-y','-i',str(source),'-ar','16000','-ac','1','-c:a','pcm_s16le',str(dest)],check=True)
        with wave.open(str(dest)) as audio:case['duration_s']=audio.getnframes()/audio.getframerate()
    (ROOT/'样本清单.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
    results=[]
    for model,filename in MODELS.items():
        for case in cases:
            out=ROOT/model/case['id'];out.mkdir(parents=True,exist_ok=True)
            base=['../../build/bin/whisper-cli','-m',filename,'-f',str(pcm/pathlib.Path(case['file']).name),'-l','auto','-t','8','-bs','5','-nfa','-mc','0']
            trials=[]
            for trial in range(1,4):
                prefix=out/str(trial);start=time.perf_counter()
                completed=subprocess.run(base+['-oj','-of',str(prefix)],capture_output=True,text=True)
                wall=time.perf_counter()-start;log=completed.stdout+'\n'+completed.stderr
                prefix.with_suffix('.log').write_text(log)
                completed.check_returncode()
                data=json.loads(prefix.with_suffix('.json').read_text())
                text=''.join(s['text'] for s in data['transcription']).strip()
                prefix.with_suffix('.txt').write_text(text)
                ref=normalize(case['reference']);hyp=normalize(text)
                ref=ref.split() if case['language']=='en' else ''.join(ref.split())
                hyp=hyp.split() if case['language']=='en' else ''.join(hyp.split())
                trials.append(dict(text=text,language=data['result']['language'],errors=distance(ref,hyp),reference_units=len(ref),error_rate=distance(ref,hyp)/len(ref),native_s=timing(log,'total')-timing(log,'load'),load_s=timing(log,'load'),process_s=wall))
            completed=subprocess.run(base+['-dl'],capture_output=True,text=True)
            log=completed.stdout+'\n'+completed.stderr;(out/'language.log').write_text(log);completed.check_returncode()
            detected=re.search(r'auto-detected language:\s*(\S+)\s*\(p = ([\d.]+)\)',log)
            if not detected:raise ValueError('语种检测输出无法解析: '+str(out))
            result=dict(model=model,id=case['id'],expected_language=case['language'],detected_language=detected[1],probability=float(detected[2]),lid_s=timing(log,'total')-timing(log,'load'),duration_s=case['duration_s'],reference=case['reference'],trials=trials,median_native_s=statistics.median(t['native_s'] for t in trials))
            results.append(result);(ROOT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
            print(model,case['id'],detected[1],round(trials[0]['error_rate'],3),round(result['median_native_s'],3),flush=True)
    write_report(results, cases)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='五模型语种、转写、性能评测与自动报告')
    parser.add_argument('--report-only', action='store_true', help='只从完整的本地结果重新生成报告，不运行模型')
    arguments = parser.parse_args()
    if arguments.report_only:
        saved_results = json.loads((ROOT / 'results.json').read_text(encoding='utf-8'))
        saved_cases = json.loads((ROOT / '样本清单.json').read_text(encoding='utf-8'))
        expected = {(model, case['id']) for model in MODELS for case in saved_cases}
        actual = {(result['model'], result['id']) for result in saved_results}
        if not saved_cases or len(actual) != len(saved_results) or actual != expected:
            raise ValueError('本地结果不完整或重复，不能生成完整报告')
        if any(len(result['trials']) != 3 for result in saved_results):
            raise ValueError('存在未完成三轮转写的结果，不能生成完整报告')
        write_report(saved_results, saved_cases)
    else:
        main()
