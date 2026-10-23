package io.github.ggerganov.whispercpp;

/**
 * 用于 DTW token 时间戳对齐的注意力头预设常量，保留当前 Java 绑定中的历史编号。
 *
 * <p>DTW 根据指定文本解码器层的交叉注意力头估算 token 的时间位置。
 * 调用方需在初始化上下文前启用 DTW，并选择与模型匹配的预设；本类常量不会自行启用对齐。
 *
 * <p><strong>版本边界：</strong>当前仓库 include/whisper.h 将 N_TOP_MOST、CUSTOM 放在 NONE 之后，
 * 与本类历史编号顺序不同。除 NONE 外，不应直接假定本类数值等于当前原生枚举值。
 * 本次注释保留现有数值；设置 WhisperContextParams.dtw_aheads_preset 前必须核对实际动态库版本。
 *
 * @see <a href="https://github.com/ggml-org/whisper.cpp/blob/master/include/whisper.h">whisper_alignment_heads_preset</a>
 * @see io.github.ggerganov.whispercpp.params.WhisperContextParams#setDtwAheadsPreset(int)
 */
public class WhisperConstants {
    // Alignment heads presets
    /**
     * 不选择预置对齐头；该值本身不代表已启用或禁用 dtw_token_timestamps。
     * 此值为类加载时确定的不可变整数，无实例生命周期或可变并发状态。
     */
    public static final int WHISPER_AHEADS_NONE = 0;
    /**
     * 适用于 tiny.en 模型的预置 DTW 对齐头组合；只能用于匹配的模型结构。
     * 本类历史编号为 1；当前仓库原生枚举编号为 3，使用前须核对版本。
     */
    public static final int WHISPER_AHEADS_TINY_EN = 1;
    /**
     * 适用于 tiny 模型的预置 DTW 对齐头组合；只能用于匹配的模型结构。
     * 本类历史编号为 2；当前仓库原生枚举编号为 4，使用前须核对版本。
     */
    public static final int WHISPER_AHEADS_TINY = 2;
    /**
     * 适用于 base.en 模型的预置 DTW 对齐头组合；只能用于匹配的模型结构。
     * 本类历史编号为 3；当前仓库原生枚举编号为 5，使用前须核对版本。
     */
    public static final int WHISPER_AHEADS_BASE_EN = 3;
    /**
     * 适用于 base 模型的预置 DTW 对齐头组合；只能用于匹配的模型结构。
     * 本类历史编号为 4；当前仓库原生枚举编号为 6，使用前须核对版本。
     */
    public static final int WHISPER_AHEADS_BASE = 4;
    /**
     * 适用于 small.en 模型的预置 DTW 对齐头组合；只能用于匹配的模型结构。
     * 本类历史编号为 5；当前仓库原生枚举编号为 7，使用前须核对版本。
     */
    public static final int WHISPER_AHEADS_SMALL_EN = 5;
    /**
     * 适用于 small 模型的预置 DTW 对齐头组合；只能用于匹配的模型结构。
     * 本类历史编号为 6；当前仓库原生枚举编号为 8，使用前须核对版本。
     */
    public static final int WHISPER_AHEADS_SMALL = 6;
    /**
     * 适用于 medium.en 模型的预置 DTW 对齐头组合；只能用于匹配的模型结构。
     * 本类历史编号为 7；当前仓库原生枚举编号为 9，使用前须核对版本。
     */
    public static final int WHISPER_AHEADS_MEDIUM_EN = 7;
    /**
     * 适用于 medium 模型的预置 DTW 对齐头组合；只能用于匹配的模型结构。
     * 本类历史编号为 8；当前仓库原生枚举编号为 10，使用前须核对版本。
     */
    public static final int WHISPER_AHEADS_MEDIUM = 8;
    /**
     * 适用于 large-v1 模型的预置 DTW 对齐头组合；只能用于匹配的模型结构。
     * 本类历史编号为 9；当前仓库原生枚举编号为 11，使用前须核对版本。
     */
    public static final int WHISPER_AHEADS_LARGE_V1 = 9;
    /**
     * 适用于 large-v2 模型的预置 DTW 对齐头组合；只能用于匹配的模型结构。
     * 本类历史编号为 10；当前仓库原生枚举编号为 12，使用前须核对版本。
     */
    public static final int WHISPER_AHEADS_LARGE_V2 = 10;
    /**
     * 适用于 large-v3 模型的预置 DTW 对齐头组合；只能用于匹配的模型结构。
     * 本类历史编号为 11；当前仓库原生枚举编号为 13，使用前须核对版本。
     */
    public static final int WHISPER_AHEADS_LARGE_V3 = 11;
    /**
     * 适用于 large-v3-turbo 模型的预置 DTW 对齐头组合；只能用于匹配的模型结构。
     * 本类历史编号为 12；当前仓库原生枚举编号为 14，使用前须核对版本。
     */
    public static final int WHISPER_AHEADS_LARGE_V3_TURBO = 12;
    /**
     * 使用调用方通过 dtw_aheads 提供的文本层和注意力头组合。
     * 本类历史编号为 13；当前仓库原生枚举编号为 2，使用前须核对版本。
     */
    public static final int WHISPER_AHEADS_CUSTOM = 13;
    /**
     * 使用最后 dtw_n_top 个文本解码器层的全部注意力头。
     * 本类历史编号为 14；当前仓库原生枚举编号为 1，使用前须核对版本。
     */
    public static final int WHISPER_AHEADS_N_TOP_MOST = 14;
    /**
     * 本 Java 类的预设数量哨兵，不是可传入原生接口的有效预设。
     * 当前原生头文件未声明此计数项；不能以该值推断原生枚举的最大有效编号。
     */
    public static final int WHISPER_AHEADS_COUNT = 15;
}
