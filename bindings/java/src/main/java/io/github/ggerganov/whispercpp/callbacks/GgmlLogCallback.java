package io.github.ggerganov.whispercpp.callbacks;

import com.sun.jna.Callback;

/**
 * Callback for GGML logging
 * Maps to the C typedef: void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data)
 */
public interface GgmlLogCallback extends Callback {
    /**
     * Called when a log message is generated
     *
     * @param level Log level (0-5 corresponding to GGML_LOG_LEVEL_*)
     * @param text Log message text
     * @param userData User data passed to the callback
     */
    void invoke(int level, String text, com.sun.jna.Pointer userData);
}
