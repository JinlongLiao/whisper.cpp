package io.github.ggerganov.whispercpp.callbacks;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.ConsoleAppender;
import ch.qos.logback.core.read.ListAppender;
import ch.qos.logback.core.spi.FilterReply;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/** 验证原生日志的级别、完整内容及测试控制台输出流配置。 */
class Slf4jGgmlLogCallbackTest {
    /**
     * 捕获回调产生的日志事件，校验多行消息与占位符原文未被改写。
     * 同时核对实际加载的控制台配置：仅 ERROR 走标准错误流，其余已启用级别走标准输出。
     * 测试结束仅移除自身添加的事件捕获器，不改变全局输出流或日志级别。
     */
    @Test
    void testForwardNativeLogLevelsAndCompleteText() {
        Logger callbackLogger = (Logger) LoggerFactory.getLogger(Slf4jGgmlLogCallback.class);
        Logger rootLogger = (Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
        ConsoleAppender<ILoggingEvent> stdoutAppender =
                (ConsoleAppender<ILoggingEvent>) rootLogger.getAppender("STDOUT");
        ConsoleAppender<ILoggingEvent> stderrAppender =
                (ConsoleAppender<ILoggingEvent>) rootLogger.getAppender("STDERR");
        assertNotNull(stdoutAppender);
        assertNotNull(stderrAppender);
        assertEquals("System.out", stdoutAppender.getTarget());
        assertEquals("System.err", stderrAppender.getTarget());

        ListAppender<ILoggingEvent> capturedEvents = new ListAppender<>();
        capturedEvents.setContext(callbackLogger.getLoggerContext());
        capturedEvents.start();
        callbackLogger.addAppender(capturedEvents);
        try {
            Slf4jGgmlLogCallback callback = Slf4jGgmlLogCallback.getInstance();
            callback.invoke(2, "模型信息：request=123\n完整第二行 {}", null);
            callback.invoke(3, "native-warning", null);
            callback.invoke(4, "native-error", null);
            callback.invoke(5, "native-continuation", null);
            assertEquals(4, capturedEvents.list.size());
            assertEquals(Level.INFO, capturedEvents.list.get(0).getLevel());
            assertEquals("模型信息：request=123\n完整第二行 {}", capturedEvents.list.get(0).getFormattedMessage());
            assertEquals(Level.WARN, capturedEvents.list.get(1).getLevel());
            assertEquals("native-warning", capturedEvents.list.get(1).getFormattedMessage());
            assertEquals(Level.ERROR, capturedEvents.list.get(2).getLevel());
            assertEquals("native-error", capturedEvents.list.get(2).getFormattedMessage());
            assertEquals(Level.INFO, capturedEvents.list.get(3).getLevel());
            assertEquals("native-continuation", capturedEvents.list.get(3).getFormattedMessage());
            for (ILoggingEvent event : capturedEvents.list) {
                if (event.getLevel() == Level.ERROR) {
                    assertEquals(FilterReply.DENY, stdoutAppender.getFilterChainDecision(event));
                    assertEquals(FilterReply.ACCEPT, stderrAppender.getFilterChainDecision(event));
                } else {
                    assertNotEquals(FilterReply.DENY, stdoutAppender.getFilterChainDecision(event));
                    assertEquals(FilterReply.DENY, stderrAppender.getFilterChainDecision(event));
                }
            }
        } finally {
            callbackLogger.detachAppender(capturedEvents);
            capturedEvents.stop();
        }
    }
}
