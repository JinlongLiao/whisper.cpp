package io.github.ggerganov.whispercpp;

import io.github.ggerganov.whispercpp.bean.WhisperLanguageProbability;
import io.github.ggerganov.whispercpp.bean.WhisperSegment;
import io.github.ggerganov.whispercpp.params.CBool;
import io.github.ggerganov.whispercpp.params.WhisperFullParams;
import io.github.ggerganov.whispercpp.params.WhisperSamplingStrategy;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 对用户提供的两段音频比较五个模型的自动转写、首窗口及分窗语种识别。
 * 先通过 FFmpeg 转成 target/orbination-audio/pcm/ 中的 16 kHz 单声道 PCM WAV。
 * 显式启用后执行；模型不下载，GPU 测试串行运行，无参考稿时不计算 CER 或 WER。
 */
@EnabledIfSystemProperty(named = "whisper.test.audioComparison", matches = "true")
class AudioModelComparisonTest {
    /** 比较原版 Tiny。 */
    @Test
    void testTinyAudioComparison() throws Exception {
        compareAudioModel("tiny", "../../models/ggml-tiny.bin");
    }

    /** 比较原版 Base。 */
    @Test
    void testBaseAudioComparison() throws Exception {
        compareAudioModel("base", "../../models/ggml-base.bin");
    }

    /** 比较 Orbination Q3_K。 */
    @Test
    void testQ3KAudioComparison() throws Exception {
        compareAudioModel("q3_k", "model/ggml-orbination-turbo-q3_k.bin");
    }

    /** 比较 Orbination Q4_K。 */
    @Test
    void testQ4KAudioComparison() throws Exception {
        compareAudioModel("q4_k", "model/ggml-orbination-turbo-q4_k.bin");
    }

    /** 比较 Orbination Q5_K。 */
    @Test
    void testQ5KAudioComparison() throws Exception {
        compareAudioModel("q5_k", "model/ggml-orbination-turbo-q5_k.bin");
    }

    /**
     * 对短音频增益 30 dB 的独立副本做诊断；不修改原文件，不替代原音频测试。
     * 对五个模型分别记录候选语种和三次自动转写，输出到 normalized 子目录。
     *
     * @throws Exception 模型、音频、原生推理或文件操作失败时传播
     */
    @Test
    void testNormalizedCantoneseAudioComparison() throws Exception {
        float[] samples = readAudioSamples("cn_cantonese_gain30");
        for (String modelName : Arrays.asList("tiny", "base", "q3_k", "q4_k", "q5_k")) {
            String filename;
            if (modelName.equals("tiny") || modelName.equals("base")) {
                filename = "../../models/ggml-" + modelName + ".bin";
            } else {
                filename = "model/ggml-orbination-turbo-" + modelName + ".bin";
            }
            Path output = Paths.get("target/audio-model-comparison/normalized", modelName);
            Files.createDirectories(output);
            try (WhisperCpp whisperCpp = new WhisperCpp()) {
                whisperCpp.initContext(filename);
                WhisperFullParams.ByValue params = whisperCpp.getFullDefaultParams(
                        WhisperSamplingStrategy.WHISPER_SAMPLING_BEAM_SEARCH);
                params.n_threads = 8;
                params.beam_search.beam_size = 5;
                params.language = "auto";
                params.translate = CBool.FALSE;
                params.no_context = CBool.TRUE;
                params.detectLanguage(false);
                StringBuilder languages = new StringBuilder("scope\tstart_s\tend_s\telapsed_s\tcandidates\n");
                StringBuilder metrics = new StringBuilder("trial\taudio_s\ttranscribe_s\trtf\tsegments\tdecoded_lang\n");
                for (int trial = 1; trial <= 3; trial++) {
                    appendLanguageCandidates(whisperCpp, samples, "gain30-trial" + trial, 0, languages);
                    long start = System.nanoTime();
                    List<WhisperSegment> segments = whisperCpp.fullTranscribeWithTime(params, samples);
                    double seconds = elapsedSeconds(start);
                    StringBuilder text = new StringBuilder();
                    for (WhisperSegment segment : segments) {
                        text.append(segment.getSentence());
                    }
                    writeResultText(output.resolve("trial-" + trial + ".txt"), text.toString());
                    String code = whisperCpp.getLib().whisper_lang_str(
                            whisperCpp.getLib().whisper_full_lang_id(whisperCpp.getCtx()));
                    metrics.append(String.format(Locale.ROOT, "%d\t%.6f\t%.6f\t%.6f\t%d\t%s%n",
                            trial, samples.length / 16000.0, seconds, seconds / (samples.length / 16000.0),
                            segments.size(), code));
                }
                writeResultText(output.resolve("languages.tsv"), languages.toString());
                writeResultText(output.resolve("metrics.tsv"), metrics.toString());
            }
        }
    }

    /**
     * 每模型加载一次，先用首段 5 秒预热，再分别对两段音频执行独立检测及三次转写。
     * 推理使用自动语种、8 线程、Beam Search 5；关闭上下文续接，防止不同轮次之间继承提示文本。
     * 输出 target/audio-model-comparison/ 下的完整文本、时间段、语种候选和耗时，不做 CER/WER 阈值断言；有语音的 cn_en 空分段在记录完成后使测试失败；已确认静音的短文件仅记录输出。
     *
     * @param modelName 结果目录使用的模型名称
     * @param modelFilename 已有本地模型路径
     * @throws Exception 输入缺失、原生调用或结果写入失败时传播
     */
    private static void compareAudioModel(String modelName, String modelFilename) throws Exception {
        Path outputDirectory = Paths.get("target/audio-model-comparison", modelName);
        Files.createDirectories(outputDirectory);
        long start = System.nanoTime();
        try (WhisperCpp whisperCpp = new WhisperCpp()) {
            whisperCpp.initContext(modelFilename);
            double loadSeconds = elapsedSeconds(start);
            boolean hasEmptyTranscription = false;
            WhisperFullParams.ByValue params = whisperCpp.getFullDefaultParams(
                    WhisperSamplingStrategy.WHISPER_SAMPLING_BEAM_SEARCH);
            params.n_threads = 8;
            params.beam_search.beam_size = 5;
            params.language = "auto";
            params.translate = CBool.FALSE;
            params.no_context = CBool.TRUE;
            params.detectLanguage(false);
            float[] warmup = readAudioSamples("cn_en");
            whisperCpp.fullTranscribeWithTime(params, Arrays.copyOf(warmup, Math.min(warmup.length, 16000 * 5)));
            writeResultText(outputDirectory.resolve("model.tsv"), "model\tbytes\tload_s\tn_vocab\n"
                    + modelName + "\t" + Files.size(Paths.get(modelFilename)) + "\t" + loadSeconds + "\t"
                    + whisperCpp.getLib().whisper_model_n_vocab(whisperCpp.getCtx()) + "\n");
            for (String audioName : Arrays.asList("cn_en", "cn_cantonese")) {
                float[] samples = readAudioSamples(audioName);
                Path audioDirectory = outputDirectory.resolve(audioName);
                Files.createDirectories(audioDirectory);
                StringBuilder languageReport = new StringBuilder("scope\tstart_s\tend_s\telapsed_s\tcandidates\n");
                // 原生语种检测只利用开头窗口；这里显式截取最多 30 秒，不将其视为全程语种比例。
                float[] firstWindow = Arrays.copyOf(samples, Math.min(samples.length, 16000 * 30));
                for (int trial = 1; trial <= 3; trial++) {
                    appendLanguageCandidates(whisperCpp, firstWindow, "first30-trial" + trial, 0, languageReport);
                }
                if (samples.length > 16000 * 10) {
                    for (int offset = 0; offset < samples.length; offset += 16000 * 10) {
                        int end = Math.min(samples.length, offset + 16000 * 10);
                        appendLanguageCandidates(whisperCpp, Arrays.copyOfRange(samples, offset, end),
                                "window10", offset / 16000.0, languageReport);
                    }
                }
                writeResultText(audioDirectory.resolve("languages.tsv"), languageReport.toString());
                StringBuilder metrics = new StringBuilder("trial\taudio_s\ttranscribe_s\trtf\tsegments\tdecoded_lang\n");
                for (int trial = 1; trial <= 3; trial++) {
                    start = System.nanoTime();
                    List<WhisperSegment> segments = whisperCpp.fullTranscribeWithTime(params, samples);
                    double seconds = elapsedSeconds(start);
                    double audioSeconds = samples.length / 16000.0;
                    hasEmptyTranscription |= "cn_en".equals(audioName) && segments.isEmpty();
                    StringBuilder transcript = new StringBuilder();
                    StringBuilder segmentReport = new StringBuilder("start_ms\tend_ms\ttext\n");
                    for (WhisperSegment segment : segments) {
                        transcript.append(segment.getSentence());
                        segmentReport.append(segment.getStart() * 10).append('\t').append(segment.getEnd() * 10)
                                .append('\t').append(segment.getSentence().replace('\n', ' ')).append('\n');
                    }
                    writeResultText(audioDirectory.resolve("trial-" + trial + ".txt"), transcript.toString());
                    writeResultText(audioDirectory.resolve("trial-" + trial + "-segments.tsv"), segmentReport.toString());
                    String decodedLanguage = whisperCpp.getLib().whisper_lang_str(
                            whisperCpp.getLib().whisper_full_lang_id(whisperCpp.getCtx()));
                    metrics.append(String.format(Locale.ROOT, "%d\t%.6f\t%.6f\t%.6f\t%d\t%s%n",
                            trial, audioSeconds, seconds, seconds / audioSeconds, segments.size(), decodedLanguage));
                    writeResultText(audioDirectory.resolve("metrics.tsv"), metrics.toString());
                    System.out.println("AUDIO_COMPARISON model=" + modelName + " audio=" + audioName
                            + " trial=" + trial + " seconds=" + seconds + " lang=" + decodedLanguage);
                }
            }
            // 仅对已确认有语音的长文件检查空输出；短文件已确认静音，空输出符合预期。
            assertFalse(hasEmptyTranscription, "有语音的 cn_en 至少一次未返回分段，详见各轮结果文件");
        }
    }

    /**
     * 记录本地库返回的前五个候选，保留原概率，不人为过滤模型不支持的语种后重新归一化。
     *
     * @param whisperCpp 已初始化的模型实例
     * @param samples 当前检测窗口的采样
     * @param scope 当前窗口或重复轮次的标识
     * @param startSeconds 窗口在原音频中的起点，单位秒
     * @param report 接收候选和检测耗时的结果文本
     * @throws Exception 原生检测失败时传播
     */
    private static void appendLanguageCandidates(WhisperCpp whisperCpp, float[] samples, String scope,
                                                 double startSeconds, StringBuilder report) throws Exception {
        long start = System.nanoTime();
        List<WhisperLanguageProbability> candidates = whisperCpp.detectAudioLanguageProbabilities(samples, 8);
        double seconds = elapsedSeconds(start);
        report.append(scope).append('\t').append(startSeconds).append('\t')
                .append(startSeconds + samples.length / 16000.0).append('\t').append(seconds).append('\t');
        for (WhisperLanguageProbability candidate : candidates.subList(0, Math.min(5, candidates.size()))) {
            assertTrue(Float.isFinite(candidate.getProbability()));
            report.append(candidate.getLanguageCode()).append(':').append(candidate.getProbability()).append(' ');
        }
        report.append('\n');
    }

    /**
     * 读取事先由 FFmpeg 解码的 WAV；校验格式后按 16 位小端 PCM 转换，不计入推理耗时。
     *
     * @param audioName 固定输入音频名称，不含扩展名
     * @return 16 kHz 单声道浮点采样
     * @throws Exception 文件缺失、格式不符或读取失败时传播
     */
    private static float[] readAudioSamples(String audioName) throws Exception {
        try (AudioInputStream input = AudioSystem.getAudioInputStream(
                Paths.get("target/orbination-audio/pcm", audioName + ".wav").toFile());
             ByteArrayOutputStream bytes = new ByteArrayOutputStream()) {
            assertEquals(AudioFormat.Encoding.PCM_SIGNED, input.getFormat().getEncoding());
            assertEquals(16000.0f, input.getFormat().getSampleRate());
            assertEquals(1, input.getFormat().getChannels());
            assertEquals(16, input.getFormat().getSampleSizeInBits());
            assertFalse(input.getFormat().isBigEndian());
            byte[] buffer = new byte[4096];
            int length;
            while ((length = input.read(buffer)) != -1) {
                bytes.write(buffer, 0, length);
            }
            byte[] pcm = bytes.toByteArray();
            assertTrue(pcm.length > 0 && pcm.length % 2 == 0);
            float[] samples = new float[pcm.length / 2];
            for (int index = 0; index < samples.length; index++) {
                samples[index] = (short) ((pcm[index * 2] & 0xff) | (pcm[index * 2 + 1] << 8)) / 32768.0f;
            }
            return samples;
        }
    }

    /**
     * 将单调时钟的纳秒差转换为性能记录使用的秒数。
     * @param start 纳秒时钟起点
     * @return 从起点至当前的墙钟秒数
     */
    private static double elapsedSeconds(long start) {
        return (System.nanoTime() - start) / 1_000_000_000.0;
    }

    /**
     * 以 UTF-8 保存完整结果；失败不吞掉异常。
     * @param path 结果文件路径，父目录已创建
     * @param text 完整结果文本
     * @throws Exception 文件写入失败时传播
     */
    private static void writeResultText(Path path, String text) throws Exception {
        Files.write(path, text.getBytes(StandardCharsets.UTF_8));
    }
}
