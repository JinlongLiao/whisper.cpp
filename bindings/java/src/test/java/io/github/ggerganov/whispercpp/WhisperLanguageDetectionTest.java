package io.github.ggerganov.whispercpp;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import io.github.ggerganov.whispercpp.callbacks.Slf4jGgmlLogCallback;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.TargetDataLine;

import io.github.ggerganov.whispercpp.bean.WhisperLanguageProbability;
import io.github.ggerganov.whispercpp.params.WhisperFullParams;
import io.github.ggerganov.whispercpp.params.WhisperSamplingStrategy;

/**
 * 验证本地库语种名称查询，以及多语种模型对 WAV 和麦克风音频的自动检测。
 */
class WhisperLanguageDetectionTest {
    /** 当前类的业务和测试日志输出入口。 */
    private static final Logger logger = LoggerFactory.getLogger(WhisperLanguageDetectionTest.class);


    /** 在模型初始化前注册强引用单例，将后续原生全局日志转发给 SLF4J。 */
    @BeforeAll
    static void configureNativeLogging() {
        try (WhisperCpp whisperCpp = new WhisperCpp()) {
            whisperCpp.setLogCallback(Slf4jGgmlLogCallback.getInstance());
        }
    }

    /**
     * 无需模型即可验证语种 ID、代码和完整名称均由本地原生库提供。
     */
    @Test
    void testQueryLanguageNameFromNativeLibrary() {
        try (WhisperCpp whisperCpp = new WhisperCpp()) {
            WhisperCppJnaLibrary whisperCppJnaLibrary = whisperCpp.getLib();
            int languageId = whisperCppJnaLibrary.whisper_lang_id("en");
            assertTrue(languageId >= 0);
            String actual = whisperCppJnaLibrary.whisper_lang_str(languageId);
            assertEquals("en", actual);
            String actual1 = whisperCppJnaLibrary.whisper_lang_str_full(languageId);
            assertEquals("english", actual1);
        }
    }

    /**
     * 拒绝无效音频、线程数及未初始化的上下文，避免将无效指针传入原生库。
     */
    @Test
    void testValidateAudioLanguageProbabilityInputs() {
        try (WhisperCpp whisperCpp = new WhisperCpp()) {
            assertThrows(IllegalArgumentException.class,
                    () -> whisperCpp.detectAudioLanguageProbabilities(null, 1));
            assertThrows(IllegalArgumentException.class,
                    () -> whisperCpp.detectAudioLanguageProbabilities(new float[0], 1));
            assertThrows(IllegalArgumentException.class,
                    () -> whisperCpp.detectAudioLanguageProbabilities(new float[160], 0));
            assertThrows(IllegalStateException.class,
                    () -> whisperCpp.detectAudioLanguageProbabilities(new float[160], 1));
        }
    }

    /**
     * 按参考实现启用自动语种检测，随后直接从本地库获取检测结果的代码和名称。
     * 使用本地 ../../models/ggml-tiny.bin，不下载模型。
     * 模型缺失或加载失败时测试失败；英语专用模型不满足前置条件并使测试失败。
     * 同时验证多候选概率的数量、排序、范围、归一化和最高候选与单语种结果一致。
     * 音频限定为仓库提供的 16 kHz、单声道、16 位小端 PCM WAV；所有资源随测试释放。
     *
     * @throws Exception 音频读取、模型初始化或原生调用失败时传播给测试框架
     * @see WhisperCppJnaLibrary#whisper_full
     * @see WhisperCppJnaLibrary#whisper_full_lang_id
     */
    @Test
    void testDetectAudioLanguage() throws Exception {
        File modelFile = resolveLanguageDetectionModel().toFile();
        File audioFile = new File("../../samples/jfk.wav");

        try (AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(audioFile);
             ByteArrayOutputStream audioBytes = new ByteArrayOutputStream();
             WhisperCpp whisperCpp = new WhisperCpp()) {
            AudioFormat audioFormat = audioInputStream.getFormat();
            assertEquals(AudioFormat.Encoding.PCM_SIGNED, audioFormat.getEncoding());
            assertEquals(16000.0f, audioFormat.getSampleRate());
            assertEquals(1, audioFormat.getChannels());
            assertEquals(16, audioFormat.getSampleSizeInBits());
            assertFalse(audioFormat.isBigEndian());

            // 循环读至 EOF，避免 available() 或一次 read() 导致音频被截断。
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = audioInputStream.read(buffer)) != -1) {
                audioBytes.write(buffer, 0, bytesRead);
            }
            byte[] pcmBytes = audioBytes.toByteArray();
            assertTrue(pcmBytes.length > 0);
            assertEquals(0, pcmBytes.length % 2);
            float[] audioSamples = new float[pcmBytes.length / 2];
            for (int sampleIndex = 0; sampleIndex < audioSamples.length; sampleIndex++) {
                int byteIndex = sampleIndex * 2;
                short sample = (short) ((pcmBytes[byteIndex] & 0xFF) | (pcmBytes[byteIndex + 1] << 8));
                audioSamples[sampleIndex] = sample / 32768.0f;
            }

            whisperCpp.initContext(modelFile.getAbsolutePath());
            WhisperCppJnaLibrary whisperCppJnaLibrary = whisperCpp.getLib();
            assertEquals(1, whisperCppJnaLibrary.whisper_is_multilingual(whisperCpp.getCtx()),
                    "自动语种检测必须使用多语种模型，不能使用 .en 模型");
            WhisperFullParams.ByValue whisperFullParams = whisperCpp.getFullDefaultParams(
                    WhisperSamplingStrategy.WHISPER_SAMPLING_BEAM_SEARCH);
            whisperFullParams.n_threads = 8;
            whisperFullParams.temperature = 0.0f;
            whisperFullParams.temperature_inc = 0.2f;
            whisperFullParams.language = "auto";
            whisperFullParams.detectLanguage(true);

            assertEquals(0, whisperCppJnaLibrary.whisper_full(whisperCpp.getCtx(), whisperFullParams,
                    audioSamples, audioSamples.length), "原生语种检测调用失败");
            int languageId = whisperCppJnaLibrary.whisper_full_lang_id(whisperCpp.getCtx());
            assertTrue(languageId >= 0);
            String languageCode = whisperCppJnaLibrary.whisper_lang_str(languageId);
            String languageName = whisperCppJnaLibrary.whisper_lang_str_full(languageId);
            logger.info("音频语种：id={}, code={}, name={}", languageId, languageCode, languageName);
            assertEquals("en", languageCode);
            assertEquals("english", languageName);

            List<WhisperLanguageProbability> candidates = whisperCpp.detectAudioLanguageProbabilities(audioSamples, 8);
            assertEquals(whisperCppJnaLibrary.whisper_lang_max_id() + 1, candidates.size());
            assertTrue(candidates.size() > 1);
            assertEquals(languageId, candidates.get(0).getLanguageId());
            assertEquals(languageCode, candidates.get(0).getLanguageCode());
            assertEquals(languageName, candidates.get(0).getLanguageName());
            double probabilitySum = 0;
            float previousProbability = 1.0f;
            for (WhisperLanguageProbability candidate : candidates) {
                assertTrue(Float.isFinite(candidate.getProbability()));
                assertTrue(candidate.getProbability() >= 0 && candidate.getProbability() <= 1);
                assertTrue(candidate.getProbability() <= previousProbability);
                assertEquals(whisperCppJnaLibrary.whisper_lang_str(candidate.getLanguageId()), candidate.getLanguageCode());
                assertEquals(whisperCppJnaLibrary.whisper_lang_str_full(candidate.getLanguageId()), candidate.getLanguageName());
                probabilitySum += candidate.getProbability();
                previousProbability = candidate.getProbability();
            }
            assertEquals(1.0, probabilitySum, 0.00001);
            for (WhisperLanguageProbability candidate : candidates.subList(0, 5)) {
                logger.info("候选语种：code={}, name={}, probability={}", candidate.getLanguageCode(),
                        candidate.getLanguageName(), candidate.getProbability());
            }
        }
    }

    /**
     * 采集默认麦克风 5 秒音频，使用多语种 tiny 模型输出前 5 个候选语种及概率。
     * 仅在 whisper.test.microphone=true 时执行；运行前需允许当前 JVM 访问麦克风。
     * 先加载模型，再以 16 kHz、单声道、16 位有符号小端 PCM 录音；采集结束立即关闭设备。
     * 结果只校验概率契约，不断言固定语种，因为输入由现场说话内容决定。
     *
     * @throws Exception 麦克风不可用、权限不足、采集或模型推理失败时传播给测试框架
     */
    @Test
    @EnabledIfSystemProperty(named = "whisper.test.microphone", matches = "true")
    void testDetectMicrophoneAudioLanguage() throws Exception {
        Path modelPath = resolveLanguageDetectionModel();
        AudioFormat audioFormat = new AudioFormat(16000.0f, 16, 1, true, false);
        // 固定 5 秒采样量，读取范围有界；无需将麦克风原始录音写入磁盘。
        byte[] pcmBytes = new byte[16000 * 2 * 5];
        try (WhisperCpp whisperCpp = new WhisperCpp()) {
            whisperCpp.initContext(modelPath.toString());
            assertEquals(1, whisperCpp.getLib().whisper_is_multilingual(whisperCpp.getCtx()),
                    "麦克风语种检测必须使用多语种模型");
            try (TargetDataLine microphoneLine = AudioSystem.getTargetDataLine(audioFormat)) {
                microphoneLine.open(audioFormat);
                logger.info("麦克风已就绪，请连续说话 5 秒……");
                microphoneLine.start();
                int recordedBytes = 0;
                while (recordedBytes < pcmBytes.length) {
                    int bytesRead = microphoneLine.read(pcmBytes, recordedBytes, pcmBytes.length - recordedBytes);
                    if (bytesRead <= 0) {
                        throw new IOException("Microphone capture stopped: recorded=" + recordedBytes
                                + ", expected=" + pcmBytes.length);
                    }
                    recordedBytes += bytesRead;
                }
                microphoneLine.stop();
            }
            logger.info("录音结束，正在识别语种……");
            float[] audioSamples = new float[pcmBytes.length / 2];
            for (int sampleIndex = 0; sampleIndex < audioSamples.length; sampleIndex++) {
                int byteIndex = sampleIndex * 2;
                short sample = (short) ((pcmBytes[byteIndex] & 0xFF) | (pcmBytes[byteIndex + 1] << 8));
                audioSamples[sampleIndex] = sample / 32768.0f;
            }
            List<WhisperLanguageProbability> candidates = whisperCpp.detectAudioLanguageProbabilities(audioSamples, 8);
            assertEquals(whisperCpp.getLib().whisper_lang_max_id() + 1, candidates.size());
            double probabilitySum = 0;
            float previousProbability = 1.0f;
            for (WhisperLanguageProbability candidate : candidates) {
                assertTrue(Float.isFinite(candidate.getProbability()));
                assertTrue(candidate.getProbability() >= 0 && candidate.getProbability() <= previousProbability);
                assertNotNull(candidate.getLanguageCode());
                assertNotNull(candidate.getLanguageName());
                probabilitySum += candidate.getProbability();
                previousProbability = candidate.getProbability();
            }
            assertEquals(1.0, probabilitySum, 0.00001);
            for (WhisperLanguageProbability candidate : candidates.subList(0, Math.min(5, candidates.size()))) {
                logger.info("麦克风候选语种：code={}, name={}, probability={}", candidate.getLanguageCode(),
                        candidate.getLanguageName(), candidate.getProbability());
            }
        }
    }

    /**
     * 获取仓库中的本地多语种 tiny 模型，不执行网络请求或模型下载。
     *
     * @return 相对于 bindings/java 的模型路径 ../../models/ggml-tiny.bin
     * @throws IOException 模型文件不存在或为空时抛出
     */
    private static Path resolveLanguageDetectionModel() throws IOException {
        Path modelPath = Paths.get("../../models/ggml-tiny.bin");
        if (!Files.isRegularFile(modelPath) || Files.size(modelPath) == 0) {
            throw new IOException("Local model is missing or empty: " + modelPath.toAbsolutePath());
        }
        return modelPath;
    }

}
