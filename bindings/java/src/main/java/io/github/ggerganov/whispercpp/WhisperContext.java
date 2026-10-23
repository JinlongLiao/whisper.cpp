package io.github.ggerganov.whispercpp;

import com.sun.jna.NativeLong;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import io.github.ggerganov.whispercpp.ggml.GgmlType;
import io.github.ggerganov.whispercpp.params.WhisperContextParams;

import java.util.Arrays;
import java.util.List;

/**
 * 保留旧版 Java 绑定对 whisper_context 内部成员的描述，不负责创建、初始化或释放原生模型。
 *
 * <p><strong>不能直接当作当前原生上下文的可靠内存映射。</strong>
 * 当前 src/whisper.cpp 中的 whisper_context 是 C++ 私有结构：model、vocab 为内嵌对象，
 * path_model 为 std::string，均非本类声明的简单指针；时间字段为固定宽度 int64_t。
 * 本类的 NativeLong、Java 枚举及 Pointer 字段并未建立这些 C++ 成员的 ABI 映射。
 * 因此，在未按实际动态库核验布局之前，不应对 whisper_init 返回的指针执行本类的 read/write。
 *
 * <p>业务调用应将上下文保持为不透明 Pointer，通过 WhisperCppJnaLibrary 的公开查询接口读取信息。
 * JNA 结构体的 Java 字段不具备原生资源所有权；即使完成布局适配，也不能与推理或释放并发读写。
 *
 * @see WhisperCppJnaLibrary
 * @see <a href="https://github.com/ggml-org/whisper.cpp/blob/master/src/whisper.cpp">原生 whisper_context 定义</a>
 */
public class WhisperContext extends Structure {
    /**
     * 模型加载耗时的历史映射，单位为微秒；原生字段为 int64_t。
     * Java 初始值为 null；NativeLong 的宽度随平台变化，不能自动保证与原生 64 位整数一致。
     */
    public NativeLong t_load_us;
    /**
     * 原生上下文记录的起始计时值，单位为微秒，用于性能统计，不是 Unix 时间戳。
     * Java 初始值为 null；原生为 int64_t，读取前须验证 NativeLong 的平台宽度及结构布局。
     */
    public NativeLong t_start_us;

    /**
     * 模型权重存储类型的历史映射，Java 默认值为 F16。
     * 原生字段为 ggml_type 整数枚举；Java 枚举对象不能在未提供类型映射时等同于该整数。
     */
    public GgmlType wtype = GgmlType.GGML_TYPE_F16;
    /**
     * 中间计算类型的历史映射，Java 默认值为 F16；实际值由原生模型及后端决定。
     * Java 枚举声明不构成原生 ggml_type 的自动 ABI 适配。
     */
    public GgmlType itype = GgmlType.GGML_TYPE_F16;

    /**
     * 上下文初始化配置的按值结构体字段，Java 初始值为 null。
     * 字段布局依赖 WhisperContextParams 与动态库版本；不能据此绕过原生初始化流程。
     */
    public WhisperContextParams.ByValue params;

    /**
     * 模型成员的历史指针占位，默认 null；不拥有模型权重。
     * 当前原生 model 是内嵌 whisper_model，不是指针，该声明不能直接读取当前模型。
     */
    public Pointer model;
    /**
     * 词表成员的历史指针占位，默认 null；不拥有词表。
     * 当前原生 vocab 是内嵌 whisper_vocab，不是指针，应通过公开词表接口查询。
     */
    public Pointer vocab;
    /**
     * 默认推理状态的历史指针字段，默认 null；资源所有权属于原生上下文。
     * 有效状态在上下文释放后失效；本类前置成员布局未适配，不能保证该字段偏移正确。
     */
    public Pointer state;

    /**
     * 模型路径成员的历史指针占位，默认 null；不应按 C 字符串指针直接解引用。
     * 当前原生 path_model 为 std::string，仅文件加载入口会记录来源路径。
     */
    public Pointer path_model;

    /**
     * 声明本 Java Structure 的字段顺序，供 JNA 计算本类自身的内存布局。
     * 该顺序仅保留历史声明，不证明其尺寸、对齐和成员类型与当前 C++ 私有结构一致。
     *
     * @return 按声明顺序排列的字段名称列表；每次调用返回独立的定长列表
     */
    @Override
    protected List<String> getFieldOrder() {
        return Arrays.asList("t_load_us", "t_start_us", "wtype", "itype",
                "params", "model", "vocab", "state", "path_model");
    }
}
