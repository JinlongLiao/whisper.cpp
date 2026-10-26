package io.github.ggerganov.whispercpp;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.util.List;
import java.util.Locale;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;

import io.github.ggerganov.whispercpp.bean.WhisperLanguageProbability;
import io.github.ggerganov.whispercpp.params.WhisperFullParams;
import io.github.ggerganov.whispercpp.params.WhisperSamplingStrategy;

/**
 * 显式启用后，用相同 JFK 音频逐一验证 Orbination 三个量化版本。
 * 模型预先由下载脚本放入 model/；本测试不下载、不录音，不支持并行执行 GPU 对比。
 * 单样本结果只能证明加载和基础推理可用，不能代表多语种识别质量或稳定性能基准。
 */
@EnabledIfSystemProperty(named = "whisper.test.orbination", matches = "true")
class OrbinationWhisperTest {
    /**
     * 验证 Q3_K 的加载、候选语种概率、转写及原生资源释放。
     */
    @Test
    void testOrbinationQ3KAudioRecognition() throws Exception {
        verifyOrbinationAudioRecognition("q3_k");
    }

    /**
     * 验证 Q4_K 的加载、候选语种概率、转写及原生资源释放。
     */
    @Test
    void testOrbinationQ4KAudioRecognition() throws Exception {
        verifyOrbinationAudioRecognition("q4_k");
    }

    /**
     * 验证 Q5_K 的加载、候选语种概率、转写及原生资源释放。
     */
    @Test
    void testOrbinationQ5KAudioRecognition() throws Exception {
        verifyOrbinationAudioRecognition("q5_k");
    }

    /**
     * 同一音频依次执行加载、语种检测、Beam Search 转写及关闭，输出各阶段墙钟耗时。
     * 验证输出非空、最高语种为英语、概率合法且归一化，并报告相对参考文本的 WER。
     * 关闭通过 try-with-resources 执行；原生崩溃须由独立测试 JVM 的退出状态判定。
     *
     * @param quantization 本次测试的量化标识，与本地模型文件名一致
     * @throws Exception 模型缺失、音频读取失败、原生调用失败或断言不满足时传播
     */
    private static void verifyOrbinationAudioRecognition(String quantization) throws Exception {
        File modelFile = new File("model/ggml-orbination-turbo-" + quantization + ".bin");
        assertTrue(modelFile.isFile(), "请先通过下载脚本获取模型：" + modelFile);
        float[] audioSamples = readJfkAudioSamples();
        long loadStart = System.nanoTime();
        try (WhisperCpp whisperCpp = new WhisperCpp()) {
            whisperCpp.initContext(modelFile.getAbsolutePath());
            double loadSeconds = (System.nanoTime() - loadStart) / 1_000_000_000.0;
            long detectionStart = System.nanoTime();
            List<WhisperLanguageProbability> candidates = whisperCpp.detectAudioLanguageProbabilities(audioSamples, 8);
            double detectionSeconds = (System.nanoTime() - detectionStart) / 1_000_000_000.0;
            assertEquals(whisperCpp.getLib().whisper_lang_max_id() + 1, candidates.size());
            assertEquals("en", candidates.get(0).getLanguageCode());
            double probabilitySum = 0;
            float previousProbability = 1;
            for (WhisperLanguageProbability candidate : candidates) {
                assertTrue(Float.isFinite(candidate.getProbability()));
                assertTrue(candidate.getProbability() >= 0 && candidate.getProbability() <= previousProbability);
                previousProbability = candidate.getProbability();
                probabilitySum += candidate.getProbability();
            }
            assertEquals(1, probabilitySum, 0.00001);
            WhisperFullParams.ByValue whisperFullParams = whisperCpp.getFullDefaultParams(
                    WhisperSamplingStrategy.WHISPER_SAMPLING_BEAM_SEARCH);
            whisperFullParams.n_threads = 8;
            whisperFullParams.beam_search.beam_size = 5;
            whisperFullParams.language = "en";
            long transcriptionStart = System.nanoTime();
            String transcript = whisperCpp.fullTranscribe(whisperFullParams, audioSamples);
            double transcriptionSeconds = (System.nanoTime() - transcriptionStart) / 1_000_000_000.0;
            assertFalse(transcript.trim().isEmpty(), "原生调用成功但转写为空");
            String reference = "and so my fellow americans ask not what your country can do for you "
                    + "ask what you can do for your country";
            String resultSummary = String.format(Locale.ROOT,
                    "ORBINATION_RESULT variant=%s bytes=%d audio_s=%.3f load_s=%.3f lid_s=%.3f "
                            + "transcribe_s=%.3f top_lang=%s probability=%.6f wer=%.4f transcript=%s%n",
                    quantization, modelFile.length(), audioSamples.length / 16000.0, loadSeconds,
                    detectionSeconds, transcriptionSeconds, candidates.get(0).getLanguageCode(),
                    candidates.get(0).getProbability(), calculateTranscriptWordErrorRate(reference, transcript), transcript);
            // 先构造整行再输出，避免格式化器多次写入时与原生 stderr 日志交错。
            System.out.print(resultSummary);
        }
    }

    /**
     * 循环读取仓库 JFK WAV，按有符号小端 16 位 PCM 转为浮点采样，不使用 available() 估算长度。
     *
     * @return 非空的 16 kHz 单声道采样数组
     * @throws Exception 音频文件缺失、格式不符或读取失败时传播
     */
    private static float[] readJfkAudioSamples() throws Exception {
        try (AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(new File("../../samples/jfk.wav"));
             ByteArrayOutputStream audioBytes = new ByteArrayOutputStream()) {
            AudioFormat format = audioInputStream.getFormat();
            assertEquals(AudioFormat.Encoding.PCM_SIGNED, format.getEncoding());
            assertEquals(16000.0f, format.getSampleRate());
            assertEquals(1, format.getChannels());
            assertEquals(16, format.getSampleSizeInBits());
            assertFalse(format.isBigEndian());
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = audioInputStream.read(buffer)) != -1) {
                audioBytes.write(buffer, 0, bytesRead);
            }
            byte[] pcmBytes = audioBytes.toByteArray();
            assertTrue(pcmBytes.length > 0);
            assertEquals(0, pcmBytes.length % 2);
            float[] samples = new float[pcmBytes.length / 2];
            for (int index = 0; index < samples.length; index++) {
                short sample = (short) ((pcmBytes[index * 2] & 0xff) | (pcmBytes[index * 2 + 1] << 8));
                samples[index] = sample / 32768.0f;
            }
            return samples;
        }
    }

    /**
     * 在忽略大小写和英文标点后，按单词 Levenshtein 编辑距离计算本样本 WER，仅用于结果报告。
     *
     * @param reference  非空英文参考文本
     * @param transcript 非空英文转写文本
     * @return 编辑距离除以参考词数，可能大于 1
     */
    private static double calculateTranscriptWordErrorRate(String reference, String transcript) {
        String[] expected = reference.split("\\s+");
        String[] actual = transcript.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9\\s]", "").trim().split("\\s+");
        int[] distances = new int[actual.length + 1];
        for (int index = 0; index <= actual.length; index++) {
            distances[index] = index;
        }
        for (int row = 1; row <= expected.length; row++) {
            int diagonal = distances[0];
            distances[0] = row;
            for (int column = 1; column <= actual.length; column++) {
                int previous = distances[column];
                distances[column] = Math.min(Math.min(distances[column] + 1, distances[column - 1] + 1),
                        diagonal + (expected[row - 1].equals(actual[column - 1]) ? 0 : 1));
                diagonal = previous;
            }
        }
        return distances[actual.length] / (double) expected.length;
    }
}
