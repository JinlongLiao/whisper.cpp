package io.github.ggerganov.whispercpp;

import com.sun.jna.Native;
import com.sun.jna.Pointer;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

import io.github.ggerganov.whispercpp.bean.WhisperLanguageProbability;
import io.github.ggerganov.whispercpp.bean.WhisperSegment;
import io.github.ggerganov.whispercpp.callbacks.GgmlLogCallback;
import io.github.ggerganov.whispercpp.params.WhisperContextParams;
import io.github.ggerganov.whispercpp.params.WhisperFullParams;
import io.github.ggerganov.whispercpp.params.WhisperSamplingStrategy;
import io.github.ggerganov.whispercpp.plugins.*;

/**
 * 管理单个 Whisper 模型上下文，封装默认参数、完整转写和候选语种概率查询。
 *
 * <p>典型流程为创建实例 → initContext 加载本地模型 → 转写或语种检测 → close。
 * 建议使用 try-with-resources；模型、GPU 缓冲区及原生参数不会仅因 Java 实例失去引用而自动释放。
 * 系统信息、基准及默认参数查询不要求先加载模型；转写与语种检测必须持有有效上下文。
 *
 * <p>实例没有同步或并发可见性保证。同一实例的初始化、推理、参数缓存读写与关闭应串行执行。
 * 日志回调实际为原生库的全局配置，多个实例互相覆盖；实例字段仅为 Java 强引用保留用途。
 * 低层指针通过 getter 暴露，调用方不得绕过本类重复释放其拥有的资源。
 *
 * <p>类初始化时会加载 GGML、CPU、Whisper、Parakeet 以及 macOS 专用后端；
 * 动态库缺失、架构不匹配或原生符号不兼容可能导致类初始化失败，而不是模型文件异常。
 *
 * @see WhisperCppJnaLibrary
 * @see <a href="https://github.com/ggml-org/whisper.cpp/blob/master/include/whisper.h">原生 C API 和资源契约</a>
 */
public class WhisperCpp implements AutoCloseable {

    // 按现有顺序触发各 JNA 接口的原生库加载；这些局部引用只用于确保类初始化发生。
    static {
        GGMLBaseLibrary ggmlBaseLibrary = GGMLBaseLibrary.GGML_BASE_LIBRARY;
        GGMLCpuLibrary ggmlCpuLibrary = GGMLCpuLibrary.GGML_CPU_LIBRARY;
        GGMLLibrary ggmlLibrary = GGMLLibrary.GGML_LIBRARY;
        WhisperCppJnaLibrary whisper = WhisperCppJnaLibrary.whisper;
        GGMLParakeetLibrary parakeetLibrary = GGMLParakeetLibrary.GGML_PARAKEET_LIBRARY;
        if (LocalLib.isMac) {
            GGMLMetaLibrary metaLibrary = GGMLMetaLibrary.GGML_META_LIBRARY;
            GGMLBlasLibrary blasLibrary = GGMLBlasLibrary.GGML_BLAS_LIBRARY;
        }
    }

    /**
     * 所有实例共享的 JNA 接口代理，不代表各实例上下文之间具有自动同步。
     */
    private static final WhisperCppJnaLibrary whisper = WhisperCppJnaLibrary.whisper;
    /**
     * 当前实例拥有的原生模型上下文，初始为 null；成功初始化后有效，close 后清空。
     * 仅限实例操作串行访问，禁止外部通过 getCtx 返回值自行释放后继续使用本实例。
     */
    private Pointer ctx = null;
    /**
     * 历史参数指针槽，初始为 null；当前类没有为它分配或赋值的流程。
     * 保留现有 getter 和关闭清理分支，不应将其当作已分配的默认上下文参数。
     */
    private Pointer paramsPointer = null;
    /**
     * 贪心策略默认参数的原生缓存，按需分配；缓存生命周期持续到 close。
     * 相同实例多次获取会复用同一地址，非线程安全，不得由调用方独立释放。
     */
    private Pointer greedyParamsPointer = null;
    /**
     * 束搜索策略默认参数的原生缓存，按需分配；缓存生命周期持续到 close。
     * 相同实例多次获取会复用同一地址，非线程安全，不得由调用方独立释放。
     */
    private Pointer beamParamsPointer = null;
    /**
     * 当前实例最近设置的日志回调强引用，初始为 null，用于阻止回调对象被 GC 回收。
     * 原生注册为进程内全局设置；close 不注销全局回调，原生仍可能调用时须继续保留强引用。
     */
    private GgmlLogCallback logCallback = null;

    /**
     * 计算仅提供模型名称时使用的本地模型查找目录，不创建目录或下载模型。
     * 优先读取 XDG_CACHE_HOME；变量不存在时使用 user.home 下的 .cache，再追加 whisper。
     * 空字符串环境变量按现有实现原样参与路径拼接，不会自动回退。
     *
     * @return 本地模型缓存目录的 File 表示，目录可能尚不存在
     */
    public File modelDir() {
        String modelDirPath = System.getenv("XDG_CACHE_HOME");
        if (modelDirPath == null) {
            modelDirPath = System.getProperty("user.home") + "/.cache";
        }

        return new File(modelDirPath, "whisper");
    }

    /**
     * 使用原生默认上下文参数加载本地模型，并创建默认推理状态。
     * 调用前应确保该实例没有正在运行的推理；重载模型会先释放旧上下文，不保留旧模型回退。
     * 仅提供名称时由 modelDir 定位；本方法不会下载模型。
     *
     * @param modelPath 非 null 的模型路径或名称，例如 base、base-en、base.en 或 ggml-base.en.bin
     * @throws FileNotFoundException 原生初始化返回 null 时抛出；也可能由格式、后端或内存问题导致，并非仅文件缺失
     * @see #initContext(String, WhisperContextParams.ByValue)
     * @see #getContextDefaultParams()
     */
    public void initContext(String modelPath) throws FileNotFoundException {
        initContextImpl(modelPath, getContextDefaultParams());
    }

    /**
     * 按调用方指定的 GPU、设备和注意力等配置加载本地模型。
     * 顺序为释放旧上下文、解析模型路径、创建新上下文；失败不会恢复旧模型。
     * params 按值传入，其字段布局须与实际动态库一致；本类不接管外部参数内存的释放。
     *
     * @param modelPath 非 null 的模型路径或名称；含斜杠或反斜杠时按路径原样传入
     * @param params    非 null 的上下文初始化配置，调用前应完成字段设置
     * @throws FileNotFoundException 原生初始化返回 null 时抛出，详细原生原因需结合日志判断
     */
    public void initContext(String modelPath, WhisperContextParams.ByValue params) throws FileNotFoundException {
        initContextImpl(modelPath, params);
    }

    /**
     * 执行单实例的模型替换和路径解析，不提供并发保护或失败回滚。
     * 无目录分隔符时，未以 .bin 结尾的名称会将连字符替换为点并补齐 ggml- 前缀与 .bin 后缀。
     * 随后在 modelDir 中定位文件；含目录分隔符的路径直接交给原生库。
     * 旧上下文在路径解析之前释放，调用方必须先保证参数有效，不能依赖失败后继续使用旧上下文。
     *
     * @param modelPath 非 null 的本地模型名称或路径
     * @param params    与当前原生布局一致的按值初始化配置
     * @throws FileNotFoundException 模型初始化返回 null 时抛出
     */
    private void initContextImpl(String modelPath, WhisperContextParams.ByValue params) throws FileNotFoundException {
        if (ctx != null) {
            whisper.whisper_free(ctx);
        }

        if (!modelPath.contains("/") && !modelPath.contains("\\")) {
            if (!modelPath.endsWith(".bin")) {
                modelPath = "ggml-" + modelPath.replace("-", ".") + ".bin";
            }

            modelPath = new File(modelDir(), modelPath).getAbsolutePath();
        }

        ctx = whisper.whisper_init_from_file_with_params(modelPath, params);

        if (ctx == null) {
            throw new FileNotFoundException(modelPath);
        }
    }

    /**
     * 从原生分配的默认上下文参数读取字段，返回用于按值传参的 JNA 结构体视图。
     * 本方法不要求模型已加载；每次调用会通过 by_ref 入口新分配一份原生参数。
     * 当前实现未将该原生地址登记到 paramsPointer，close 不会释放这份分配；
     * 调用方应在所有使用结束后通过 whisper_free_context_params 配对释放其底层指针。
     *
     * @return 已执行 read 的上下文默认参数，字段初始值以实际动态库为准
     * @see WhisperCppJnaLibrary#whisper_context_default_params_by_ref()
     * @see WhisperCppJnaLibrary#whisper_free_context_params(Pointer)
     */
    public WhisperContextParams.ByValue getContextDefaultParams() {
        WhisperContextParams.ByValue valueParams = new WhisperContextParams.ByValue(
                whisper.whisper_context_default_params_by_ref());
        valueParams.read();
        return valueParams;
    }

    /**
     * 按解码策略获取原生默认完整推理参数，并为每种策略最多缓存一个原生参数地址。
     * 不要求加载模型；返回结构体由对应缓存地址构造并读取，调用方不应单独释放该缓存。
     * 同一策略的底层地址会复用，不保证多次返回值与底层写入互相隔离；本方法不支持并发使用。
     * 缓存参数在 close 后失效，后续使用必须重新获取。
     *
     * @param strategy 非 null 的 GREEDY 或 BEAM_SEARCH 采样策略
     * @return 已读取原生默认值的按值推理参数结构体
     * @see #freeParams()
     * @see WhisperCppJnaLibrary#whisper_full_default_params_by_ref(int)
     */
    public WhisperFullParams.ByValue getFullDefaultParams(WhisperSamplingStrategy strategy) {
        Pointer pointer;

        // 每种策略复用一个原生参数地址，缓存由本实例关闭流程管理，不允许并发初始化或外部释放。
        if (strategy == WhisperSamplingStrategy.WHISPER_SAMPLING_GREEDY) {
            if (greedyParamsPointer == null) {
                greedyParamsPointer = whisper.whisper_full_default_params_by_ref(strategy.ordinal());
            }
            pointer = greedyParamsPointer;
        } else {
            if (beamParamsPointer == null) {
                beamParamsPointer = whisper.whisper_full_default_params_by_ref(strategy.ordinal());
            }
            pointer = beamParamsPointer;
        }

        WhisperFullParams.ByValue params = new WhisperFullParams.ByValue(pointer);
        params.read();
        return params;
    }

    /**
     * 依次释放当前模型上下文和缓存的完整推理参数，再向已保存的日志回调发出 DEBUG 关闭消息。
     * 资源指针清空后重复关闭不会再次释放同一地址，但每次调用仍会触发关闭日志回调。
     * 本方法不注销原生全局回调，也不永久禁止后续重新初始化实例；返回的旧上下文及参数视图不再有效。
     * 调用前必须等待该实例的初始化、转写和检测结束；不支持并发关闭。
     * 原生失败或回调抛出的运行时异常不会被吞掉，后续清理步骤可能因此无法执行。
     *
     * @see #freeContext()
     * @see #freeParams()
     * @see #setLogCallback(GgmlLogCallback)
     */
    @Override
    public void close() {
        freeContext();
        freeParams();
        if (Objects.nonNull(logCallback)) {
            logCallback.invoke(1, "Whisper closed", null);
        }
    }

    /**
     * 在上下文非空时释放模型及默认状态的后端资源，并在原生释放返回后清空 ctx。
     * 正常返回后可重复调用；不与推理或其他释放操作并发执行。
     * 此处仅管理上下文拥有的默认状态，不跟踪外部通过低层接口创建的独立状态。
     */
    private void freeContext() {
        if (Objects.nonNull(ctx)) {
            whisper.whisper_free(ctx);
            ctx = null;
        }
    }

    /**
     * 按历史参数槽、贪心缓存、束搜索缓存的顺序释放非空指针，并逐个清空引用。
     * 现有实现使用 Native.free；原生 by_ref 参数实际由 C++ new 分配，配对释放 API 为 whisper_free_params。
     * 此注释说明当前实现的分配释放兼容性边界，不代表已修复该历史释放方式。
     * 不负责释放 getContextDefaultParams 每次独立分配的上下文参数。
     */
    private void freeParams() {
        if (paramsPointer != null) {
            Native.free(Pointer.nativeValue(paramsPointer));
            paramsPointer = null;
        }
        if (greedyParamsPointer != null) {
            Native.free(Pointer.nativeValue(greedyParamsPointer));
            greedyParamsPointer = null;
        }
        if (beamParamsPointer != null) {
            Native.free(Pointer.nativeValue(beamParamsPointer));
            beamParamsPointer = null;
        }
    }

    /**
     * 执行完整音频推理，按原生结果顺序拼接所有分段文本，并去除最终字符串两端空白。
     * 调用前须加载模型；输入为 16 kHz 单声道浮点 PCM，不在此处解码文件或重采样。
     * 各分段原文还会按当前实现写入标准输出；结果保存在上下文默认状态并被下一次推理覆盖。
     * 不支持同一实例并发推理或关闭；回调和推理参数引用的原生内存在调用期间必须有效。
     *
     * @param whisperParams 非 null 的完整推理参数；策略及自动检测等选项由调用方配置
     * @param audioData     非 null 的 PCM 数组，通常归一化至 [-1, 1]；处理长度为整个数组
     * @return 拼接并 trim 后的识别文本；无分段时为空字符串
     * @throws IllegalStateException 模型未初始化或已关闭时抛出
     * @throws IOException           whisper_full 返回非零状态时抛出，原生错误详情由日志给出
     * @see WhisperCppJnaLibrary#whisper_full(Pointer, WhisperFullParams.ByValue, float[], int)
     */
    public String fullTranscribe(WhisperFullParams.ByValue whisperParams, float[] audioData) throws IOException {
        if (ctx == null) {
            throw new IllegalStateException("Model not initialised");
        }

        /*
        WhisperFullParams.ByValue valueParams = new WhisperFullParams.ByValue(
            lib.whisper_full_default_params_by_ref(WhisperSamplingStrategy.WHISPER_SAMPLING_BEAM_SEARCH.ordinal()));
        valueParams.read();
        */

        if (whisper.whisper_full(ctx, whisperParams, audioData, audioData.length) != 0) {
            throw new IOException("Failed to process audio");
        }

        int nSegments = whisper.whisper_full_n_segments(ctx);

        StringBuilder str = new StringBuilder();

        for (int i = 0; i < nSegments; i++) {
            String text = whisper.whisper_full_get_segment_text(ctx, i);
            System.out.println("Segment:" + text);
            str.append(text);
        }

        return str.toString().trim();
    }

    /**
     * 计算音频开头窗口的候选语种概率，按概率降序返回本地库支持的全部语种。
     * 必须先初始化多语种模型；输入为 16 kHz 单声道 PCM，采样值通常在 [-1, 1]。
     * 先计算 Mel，再执行一次原生语种检测；代码和完整英文名称均从本地库查询。
     * 概率是候选语种分布，不表示音频中各语种的时长占比，也不执行分段语种识别。
     * 检测从偏移 0 开始使用模型音频窗口（标准 Whisper 通常最多 30 秒），不自动遍历全部长音频。
     * 返回对象保存 Java 值，不依赖原生数组存活；列表包含全部语种且未做阈值过滤。
     * 调用会修改上下文的 Mel、编码器和解码器状态，不支持同一实例上的并发操作。
     *
     * @param audioSamples 非 null、非空的 PCM 采样数组
     * @param threadCount  原生计算线程数，必须大于 0
     * @return 概率从高到低排列的候选列表，概率单位为 0 至 1
     * @throws IllegalArgumentException 音频为空或线程数非法时抛出
     * @throws IllegalStateException    模型未初始化或模型仅支持英语时抛出
     * @throws IOException              Mel 计算或原生语种检测失败时抛出，包含失败阶段及原生返回码
     * @see WhisperCppJnaLibrary#whisper_pcm_to_mel
     * @see WhisperCppJnaLibrary#whisper_lang_auto_detect
     */
    public List<WhisperLanguageProbability> detectAudioLanguageProbabilities(float[] audioSamples, int threadCount)
            throws IOException {
        if (Objects.isNull(audioSamples) || audioSamples.length == 0) {
            throw new IllegalArgumentException("Audio samples must not be empty");
        }
        if (threadCount <= 0) {
            throw new IllegalArgumentException("Thread count must be greater than zero");
        }
        if (Objects.isNull(ctx)) {
            throw new IllegalStateException("Model not initialised");
        }
        if (whisper.whisper_is_multilingual(ctx) == 0) {
            throw new IllegalStateException("Language detection requires a multilingual model");
        }
        int melResult = whisper.whisper_pcm_to_mel(ctx, audioSamples, audioSamples.length, threadCount);
        if (melResult != 0) {
            throw new IOException("Failed to compute audio mel: result=" + melResult
                    + ", samples=" + audioSamples.length + ", threads=" + threadCount);
        }
        float[] languageProbabilities = new float[whisper.whisper_lang_max_id() + 1];
        int languageId = whisper.whisper_lang_auto_detect(ctx, 0, threadCount, languageProbabilities);
        if (languageId < 0) {
            throw new IOException("Failed to detect audio languages: result=" + languageId
                    + ", samples=" + audioSamples.length + ", threads=" + threadCount);
        }
        List<WhisperLanguageProbability> candidates = new ArrayList<>(languageProbabilities.length);
        for (int candidateId = 0; candidateId < languageProbabilities.length; candidateId++) {
            candidates.add(new WhisperLanguageProbability(candidateId, whisper.whisper_lang_str(candidateId),
                    whisper.whisper_lang_str_full(candidateId), languageProbabilities[candidateId]));
        }
        candidates.sort(Comparator.comparingDouble(WhisperLanguageProbability::getProbability).reversed());
        return candidates;
    }

    /**
     * 执行完整音频推理，返回包含分段起止时间和原文的 Java 结果列表。
     * 输入要求与 fullTranscribe 相同，不执行文件解码、重采样或额外的语种检测。
     * 起止时间原样保留原生单位：每单位为 10 毫秒；此方法不会乘 10 转为毫秒。
     * 结果是本次原生分段的 Java 值拷贝，按分段顺序排列；原生状态仍会被后续推理覆盖。
     * 同一实例不得并发转写、初始化或释放。
     *
     * @param whisperParams 非 null 的完整推理配置；时间戳输出行为受原生参数影响
     * @param audioData     非 null 的 16 kHz 单声道浮点 PCM 数组，处理范围为整个数组
     * @return 含起始时间、结束时间和文本的可变列表；无分段时为空列表
     * @throws IllegalStateException 模型未初始化或已关闭时抛出
     * @throws IOException           原生完整推理返回非零状态时抛出
     * @see #fullTranscribe(WhisperFullParams.ByValue, float[])
     */
    public List<WhisperSegment> fullTranscribeWithTime(WhisperFullParams.ByValue whisperParams, float[] audioData) throws IOException {
        if (ctx == null) {
            throw new IllegalStateException("Model not initialised");
        }

        if (whisper.whisper_full(ctx, whisperParams, audioData, audioData.length) != 0) {
            throw new IOException("Failed to process audio");
        }

        int nSegments = whisper.whisper_full_n_segments(ctx);
        List<WhisperSegment> segments = new ArrayList<>(nSegments);

        for (int i = 0; i < nSegments; i++) {
            long t0 = whisper.whisper_full_get_segment_t0(ctx, i);
            String text = whisper.whisper_full_get_segment_text(ctx, i);
            long t1 = whisper.whisper_full_get_segment_t1(ctx, i);
            segments.add(new WhisperSegment(t0, t1, text));
        }

        return segments;
    }

//    public int getTextSegmentCount(Pointer ctx) {
//        return lib.whisper_full_n_segments(ctx);
//    }
//    public String getTextSegment(Pointer ctx, int index) {
//        return lib.whisper_full_get_segment_text(ctx, index);
//    }

    /**
     * 查询原生库的构建选项和当前可用的计算后端，不要求先加载模型。
     * 该查询可能触发 Metal 等设备初始化；如需捕获初始化日志，应先调用 setLogCallback。
     * 原生使用静态报告缓冲区，不保证多个实例并发查询时互相隔离。
     *
     * @return 原生系统与后端信息文本
     * @see WhisperCppJnaLibrary#whisper_print_system_info()
     */
    public String getSystemInfo() {
        return whisper.whisper_print_system_info();
    }

    /**
     * 运行原生内存复制基准；无需模型，但会实际分配较大内存并消耗计算资源。
     * 报告由原生直接写入 stderr，不通过日志回调；返回值不是吞吐率。
     *
     * @param nthread 基准线程数，应为正数
     * @return 当前原生实现返回 0，性能值见标准错误输出
     * @see WhisperCppJnaLibrary#whisper_bench_memcpy(int)
     */
    public int benchMemcpy(int nthread) {
        return whisper.whisper_bench_memcpy(nthread);
    }

    /**
     * 运行原生 GGML 矩阵乘法基准；无需模型，但会分配矩阵缓冲区并持续执行计算。
     * 报告由原生直接写入 stderr，不通过日志回调；此接口不适合高频业务调用。
     *
     * @param nthread 基准线程数，应为正数
     * @return 当前原生实现返回 0，性能值见标准错误输出
     * @see WhisperCppJnaLibrary#whisper_bench_ggml_mul_mat(int)
     */
    public int benchGgmlMulMat(int nthread) {
        return whisper.whisper_bench_ggml_mul_mat(nthread);
    }

    /**
     * 保存 Java 回调强引用，并设置 Whisper/GGML 的原生全局日志回调，userData 固定为 null。
     * 应在模型或设备初始化之前设置；后续其他实例注册会覆盖当前全局回调，非实例隔离。
     * 回调可能由原生工作线程执行，必须线程安全，且不应向 JNA 边界抛出 Java 异常。
     * close 不注销原生全局回调；原生仍可能输出时，调用方须保证该实例或回调另有强引用。
     *
     * @param callback 新日志回调；传 null 恢复默认原生日志输出
     * @see WhisperCppJnaLibrary#whisper_log_set(GgmlLogCallback, Pointer)
     */
    public void setLogCallback(GgmlLogCallback callback) {
        // 保存回调引用，防止被垃圾回收
        this.logCallback = callback;
        // 调用原生函数设置日志回调
        if (callback != null) {
            // 这里可以传递用户数据，但为简单起见设为null
            whisper.whisper_log_set(callback, null);
        } else {
            // 重置为默认日志处理
            whisper.whisper_log_set(null, null);
        }
    }

    /**
     * 返回共享原生接口代理，是 getWhisper 的兼容别名，不要求模型已加载。
     *
     * @return 当前类加载器使用的 WhisperCppJnaLibrary 代理
     * @see #getWhisper()
     */
    public WhisperCppJnaLibrary getLib() {
        return getWhisper();
    }

    /**
     * 返回当前类加载器共享的低层 JNA 接口，不转移任何上下文或参数的所有权。
     * 调用方直接使用代理时，须自行遵守指针有效期、结构体布局和同状态串行执行的要求。
     *
     * @return 共享的原生 Whisper 接口代理
     */
    public WhisperCppJnaLibrary getWhisper() {
        return whisper;
    }

    /**
     * 获取当前模型上下文的借用指针，不转移所有权，也不保证返回后资源仍可并发使用。
     * 不得由外部独立释放；在重新初始化或 close 后，先前保存的指针失效。
     *
     * @return 当前有效上下文指针；尚未初始化、初始化返回 null 或已关闭时为 null
     */
    public Pointer getCtx() {
        return ctx;
    }

    /**
     * 返回保留的历史参数指针槽，不分配参数，也不等同于 getContextDefaultParams 的返回内存。
     *
     * @return 当前实现未赋值的参数槽，正常路径下为 null
     */
    public Pointer getParamsPointer() {
        return paramsPointer;
    }

    /**
     * 借用贪心默认参数的缓存地址，仅在首次请求该策略后分配。
     * 指针由本实例管理，外部不得释放；close 后已保存的地址失效。
     *
     * @return 贪心参数缓存指针；未分配或已清理时为 null
     */
    public Pointer getGreedyParamsPointer() {
        return greedyParamsPointer;
    }

    /**
     * 借用束搜索默认参数的缓存地址，仅在首次请求该策略后分配。
     * 指针由本实例管理，外部不得释放；close 后已保存的地址失效。
     *
     * @return 束搜索参数缓存指针；未分配或已清理时为 null
     */
    public Pointer getBeamParamsPointer() {
        return beamParamsPointer;
    }
}
