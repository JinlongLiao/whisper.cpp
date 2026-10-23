package io.github.ggerganov.whispercpp;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import io.github.ggerganov.whispercpp.callbacks.Slf4jGgmlLogCallback;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class WhisperJnaLibraryTest {
    /**
     * 当前类的业务和测试日志输出入口。
     */
    private static final Logger logger = LoggerFactory.getLogger(WhisperJnaLibraryTest.class);


    /**
     * 在系统信息查询触发 Metal 设备初始化之前注册全局原生日志回调。
     * 单独从 IDE 运行本测试时同样生效，不依赖其他测试类的初始化顺序。
     */
    @BeforeAll
    static void configureNativeLogging() {
        WhisperCppJnaLibrary.whisper.whisper_log_set(Slf4jGgmlLogCallback.getInstance(), null);
    }

    @Test
    void testWhisperPrint_system_info() {
        String systemInfo = WhisperCppJnaLibrary.whisper.whisper_print_system_info();
        // eg: "AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0
        //    | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | VSX = 0 | COREML = 0 | "
        logger.info("System info: {}", systemInfo);
        assertTrue(systemInfo.length() > 10);
    }
}
