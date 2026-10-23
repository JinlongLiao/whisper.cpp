package io.github.ggerganov.whispercpp.callbacks;

import com.sun.jna.Pointer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 将本地 GGML 日志原文转发给 SLF4J，不截断或过滤消息。
 * 无状态单例可被原生线程并发调用；静态强引用覆盖原生全局回调的生命周期，避免被 GC 回收。
 *
 * @see <a href="https://github.com/ggml-org/ggml/blob/master/include/ggml.h">ggml_log_level</a>
 */
public final class Slf4jGgmlLogCallback implements GgmlLogCallback {
    /**
     * 原生日志的 SLF4J 输出入口，级别及去向由调用方的日志实现配置。
     */
    private static final Logger logger = LoggerFactory.getLogger(Slf4jGgmlLogCallback.class);
    /**
     * 在当前类加载器生命周期内持有的唯一回调实例。
     */
    private static final Slf4jGgmlLogCallback INSTANCE = new Slf4jGgmlLogCallback();

    /**
     * 仅允许通过单例注册原生全局回调。
     */
    private Slf4jGgmlLogCallback() {
    }

    /**
     * @return 可安全传给 WhisperCpp.setLogCallback 的共享实例
     */
    public static Slf4jGgmlLogCallback getInstance() {
        return INSTANCE;
    }

    /**
     * 映射 DEBUG、WARN、ERROR 到同名级别，其余以 INFO 输出。
     * CONT 是原生续行消息，按 INFO 原样输出，不缓存或合并不同原生线程的日志。
     *
     * @param level    GGML 日志级别，1 为 DEBUG、2 为 INFO、3 为 WARN、4 为 ERROR、5 为 CONT
     * @param text     原生消息原文，保留已有换行及上下文
     * @param userData 原生透传数据，本实现不使用
     */
    @Override
    public void invoke(int level, String text, Pointer userData) {
        switch (level) {
            case 1:
                logger.debug("{}", text);
                break;
            case 3:
                logger.warn("{}", text);
                break;
            case 4:
                logger.error("{}", text);
                break;
            default:
                logger.info("{}", text);
                break;
        }
    }
}
