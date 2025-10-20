package io.github.ggerganov.whispercpp.callbacks;

import com.sun.jna.Callback;

/**
 * Callback for GGML logging
 * Maps to the C typedef: void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data)
 */
public interface GgmlLogCallback extends Callback {
    /**
     * Called when a log message is generated
     * <p>
     * //  GGML_LOG_LEVEL_NONE  = 0,
     * //        GGML_LOG_LEVEL_DEBUG = 1,
     * //        GGML_LOG_LEVEL_INFO  = 2,
     * //        GGML_LOG_LEVEL_WARN  = 3,
     * //        GGML_LOG_LEVEL_ERROR = 4,
     * //        GGML_LOG_LEVEL_CONT  = 5,
     * // continue previous log
     *
     * @param level    Log level (0-5 corresponding to GGML_LOG_LEVEL_*)
     * @param text     Log message text
     * @param userData User data passed to the callback
     */
    void invoke(int level, String text, com.sun.jna.Pointer userData);
}
