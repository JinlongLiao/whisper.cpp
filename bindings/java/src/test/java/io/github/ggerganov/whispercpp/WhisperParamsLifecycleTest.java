package io.github.ggerganov.whispercpp;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import io.github.ggerganov.whispercpp.params.CBool;
import io.github.ggerganov.whispercpp.params.WhisperContextParams;
import io.github.ggerganov.whispercpp.params.WhisperFullParams;
import io.github.ggerganov.whispercpp.params.WhisperSamplingStrategy;

/**
 * 无需模型文件，验证真实原生参数的独立内存和配对释放生命周期。
 */
class WhisperParamsLifecycleTest {
    /**
     * 原生默认参数释放后，返回值仍可读写；重复获取及关闭实例不得使旧值失效。
     */
    @Test
    void contextDefaultsRemainIndependentAfterClose() {
        WhisperCpp whisperCpp = new WhisperCpp();
        WhisperContextParams.ByValue first = whisperCpp.getContextDefaultParams();
        first.useGpu(false);
        first.write();
        for (int index = 0; index < 1000; index++) {
            WhisperContextParams.ByValue next = whisperCpp.getContextDefaultParams();
            assertEquals(CBool.TRUE, next.use_gpu);
            assertNotEquals(first.getPointer(), next.getPointer());
        }
        whisperCpp.close();
        first.read();
        assertEquals(CBool.FALSE, first.use_gpu);
        assertEquals(0, first.dtw_aheads.n_heads.longValue());
        assertNull(first.dtw_aheads.heads);
        assertDoesNotThrow(whisperCpp::close);
    }

    /**
     * 两种策略各自缓存；关闭后清空指针，重复关闭安全，再次获取会重新创建原生参数。
     */
    @Test
    void cachedFullParamsCanBeReleasedAndRecreated() {
        try (WhisperCpp whisperCpp = new WhisperCpp()) {
            for (int index = 0; index < 100; index++) {
                for (WhisperSamplingStrategy strategy : WhisperSamplingStrategy.values()) {
                    WhisperFullParams.ByValue params = whisperCpp.getFullDefaultParams(strategy);
                    assertEquals(strategy.ordinal(), params.strategy);
                    assertTrue(params.n_threads > 0);
                }
                assertNotNull(whisperCpp.getGreedyParamsPointer());
                assertNotNull(whisperCpp.getBeamParamsPointer());
                whisperCpp.close();
                assertNull(whisperCpp.getGreedyParamsPointer());
                assertNull(whisperCpp.getBeamParamsPointer());
                assertDoesNotThrow(whisperCpp::close);
            }
        }
    }
}
