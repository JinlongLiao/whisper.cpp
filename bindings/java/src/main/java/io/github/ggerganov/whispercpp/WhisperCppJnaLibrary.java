package io.github.ggerganov.whispercpp;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

import io.github.ggerganov.whispercpp.callbacks.GgmlLogCallback;
import io.github.ggerganov.whispercpp.model.WhisperModelLoader;
import io.github.ggerganov.whispercpp.model.WhisperTokenData;
import io.github.ggerganov.whispercpp.params.WhisperContextParams;
import io.github.ggerganov.whispercpp.params.WhisperFullParams;
import io.github.ggerganov.whispercpp.plugins.LocalLib;

/**
 * whisper.cpp 原生 C API 的 JNA 声明，语义以当前仓库 include/whisper.h 和 src/whisper.cpp 为准。
 *
 * <p>常规推理流程：加载模型 → 设置参数 → 完整推理 → 读取结果 → 释放原生资源。
 * 低层推理流程为 PCM → Mel → 编码 → 解码；语种检测只需先准备 Mel，再调用自动检测。
 * 不带 state 的处理方法使用上下文默认状态；no_state 初始化后须显式创建并管理独立状态。
 * 同一推理状态不支持并发修改，释放前必须停止所有关联操作；纯查询也不得与资源释放并发。
 *
 * <p>本接口直接暴露原生指针、长度和返回码，不提供统一的 Java 参数校验或异常转换。
 * 无效指针、越界长度或不匹配的结构体布局可能导致原生进程崩溃，而不是可捕获的 Java 异常。
 * 原生堆分配的对象必须通过配对的 whisper_free 系列方法释放，Java GC 不负责释放这些资源。
 *
 * <p>单位约定：PCM 长度按 float 采样点计，Mel 偏移按帧计，自动检测偏移按毫秒计，
 * 分段及 token 时间戳按 10 毫秒计；各参数的单位不能混用。String 返回值由 JNA 转换为 Java 值，
 * 借用的原生内存不应自行释放；Structure 和 Pointer 参数的布局与生命周期仍由调用方保证。
 *
 * <p>日志回调是原生全局设置而非上下文属性；单独运行测试时也须在首次设备查询前显式注册。
 * 原生返回 null 或非零状态码时，应结合相同调用的原生日志判断原因，不能统一归因为文件缺失。
 * 本接口不绑定 SLF4J 或其他日志实现，接收方可以在调用层注册所需回调。
 *
 * @see <a href="https://github.com/ggml-org/whisper.cpp/blob/master/include/whisper.h">官方 C API 声明</a>
 * @see <a href="https://github.com/ggml-org/whisper.cpp/blob/master/src/whisper.cpp">官方原生实现</a>
 */
public interface WhisperCppJnaLibrary extends Library {

    /**
     * 当前类加载器共享的原生接口代理，首次访问时由 LocalLib 提供本地动态库路径。
     * 仅共享接口代理不代表共享上下文可并发推理；调用方仍须管理各上下文与状态的生命周期。
     */
    WhisperCppJnaLibrary whisper = Native.load(LocalLib.getLibPath(-1), WhisperCppJnaLibrary.class);

    /**
     * 查询当前原生构建及可用计算后端的信息。
     * 可能触发后端设备初始化；若需接管初始化日志，应在调用前注册 whisper_log_set。
     * 原生实现复用静态字符串，不应并发调用此查询。
     *
     * @return 系统信息文本，不需要调用方释放
     */
    String whisper_print_system_info();

    /**
     * 使用原生默认上下文参数从模型文件加载模型并创建默认推理状态。
     * 此入口已被原生 API 标记为弃用，新增调用优先选择对应的 with_params 入口。
     * 成功后必须调用 whisper_free 释放上下文。
     *
     * @param path_model 本地 GGML 模型文件路径；路径字符串编码须与当前原生平台一致
     * @return 成功返回上下文指针，加载失败返回 null
     */
    Pointer whisper_init_from_file(String path_model);

    /**
     * 在原生堆上分配并返回默认上下文参数。
     * 调用方读取或复制后必须使用 whisper_free_context_params 释放，不能依赖 Java GC。
     * 原生使用 C++ new 分配参数，应使用配对释放接口，不使用 Native.free 替代。
     *
     * @return 新分配的 whisper_context_params 指针
     */
    Pointer whisper_context_default_params_by_ref();

    /**
     * 释放由 whisper_context_default_params_by_ref 分配的参数。
     * 释放后该指针及基于该内存的结构体视图均失效，不得重复释放。
     *
     * @param params 原生分配的上下文参数指针
     */
    void whisper_free_context_params(Pointer params);

    /**
     * 按指定上下文参数加载模型文件，并创建默认推理状态。
     * params 按值传入，必须与当前原生 whisper_context_params 的字段布局一致。
     * 成功后由调用方在所有推理结束后调用 whisper_free 释放上下文。
     *
     * @param path_model 本地 GGML 模型文件路径；路径字符串编码须与当前原生平台一致
     * @param params     按值传递的上下文配置，包含 GPU、设备及注意力等设置
     * @return 成功返回上下文指针，加载失败返回 null
     */
    Pointer whisper_init_from_file_with_params(String path_model, WhisperContextParams.ByValue params);

    /**
     * 使用原生默认上下文参数从内存中的模型加载模型并创建默认推理状态。
     * 此入口已被原生 API 标记为弃用，新增调用优先选择对应的 with_params 入口。
     * 成功后必须调用 whisper_free 释放上下文。
     *
     * @param buffer      指向完整模型数据的原生内存，至少包含 buffer_size 字节，加载期间必须有效
     * @param buffer_size 模型数据字节数；当前 Java 签名为 int，而原生为 size_t，跨平台使用须核对 ABI
     * @return 成功返回上下文指针，加载失败返回 null
     */
    Pointer whisper_init_from_buffer(Pointer buffer, int buffer_size);

    /**
     * 使用原生默认上下文参数从自定义模型读取器加载模型并创建默认推理状态。
     * 此入口已被原生 API 标记为弃用，新增调用优先选择对应的 with_params 入口。
     * 成功后必须调用 whisper_free 释放上下文。
     *
     * @param loader 自定义原生模型读取器，结构体及其 Java 回调在同步加载完成前必须保持有效
     * @return 成功返回上下文指针，加载失败返回 null
     */
    Pointer whisper_init(WhisperModelLoader loader);

    /**
     * 使用原生默认上下文参数从模型文件加载模型，不自动创建推理状态。
     * 此入口已被原生 API 标记为弃用，新增调用优先选择对应的 with_params 入口。
     * 成功后必须调用 whisper_free 释放上下文。
     * 推理前须调用 whisper_init_state 创建独立状态，并使用带 state 的接口；独立状态需自行释放。
     *
     * @param path_model 本地 GGML 模型文件路径；路径字符串编码须与当前原生平台一致
     * @return 成功返回上下文指针，加载失败返回 null
     */
    Pointer whisper_init_from_file_no_state(String path_model);

    /**
     * 使用原生默认上下文参数从内存中的模型加载模型，不自动创建推理状态。
     * 此入口已被原生 API 标记为弃用，新增调用优先选择对应的 with_params 入口。
     * 成功后必须调用 whisper_free 释放上下文。
     * 推理前须调用 whisper_init_state 创建独立状态，并使用带 state 的接口；独立状态需自行释放。
     *
     * @param buffer      指向完整模型数据的原生内存，至少包含 buffer_size 字节，加载期间必须有效
     * @param buffer_size 模型数据字节数；当前 Java 签名为 int，而原生为 size_t，跨平台使用须核对 ABI
     * @return 成功返回上下文指针，加载失败返回 null
     */
    Pointer whisper_init_from_buffer_no_state(Pointer buffer, int buffer_size);

    /**
     * 使用原生默认上下文参数从自定义模型读取器加载模型，不自动创建推理状态。
     * 此入口已被原生 API 标记为弃用，新增调用优先选择对应的 with_params 入口。
     * 成功后必须调用 whisper_free 释放上下文。
     * 推理前须调用 whisper_init_state 创建独立状态，并使用带 state 的接口；独立状态需自行释放。
     *
     * @param loader 自定义原生模型读取器，结构体及其 Java 回调在同步加载完成前必须保持有效
     * @return 成功返回上下文指针，加载失败返回 null
     */
    Pointer whisper_init_no_state(WhisperModelLoader loader);

    /**
     * 为已加载的模型创建独立的原生推理状态。
     * 状态只适用于创建它的上下文；调用方须先释放独立状态，再释放关联上下文。
     * 同一状态的推理与释放不得并发执行。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 成功返回状态指针，初始化失败返回 null
     */
    Pointer whisper_init_state(Pointer ctx);

    /**
     * 释放模型上下文及其拥有的默认状态和计算后端资源。
     * 调用前必须结束使用该上下文的所有操作，并先释放自行创建的独立状态。
     * 释放后原指针失效；该操作不对悬空指针提供幂等保证，不得重复释放。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     */
    void whisper_free(Pointer ctx);

    /**
     * 释放 whisper_init_state 创建的独立推理状态。
     * 不得单独释放由上下文拥有的默认状态，避免随后释放上下文时重复释放。
     * 释放后该状态及其结果视图失效；不得与使用该状态的推理操作并发执行。
     *
     * @param state 由同一模型上下文创建且尚未释放的独立推理状态
     */
    void whisper_free_state(Pointer state);

    /**
     * 将 16 kHz 单声道浮点 PCM 转换为对数 Mel 频谱，保存到上下文默认状态。
     * 会覆盖状态中的频谱数据；同一状态不得并发计算或读取中间结果。
     *
     * @param ctx       已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param samples   16 kHz 单声道浮点 PCM 数组，通常归一化至 [-1, 1]，长度至少为 n_samples
     * @param n_samples 本次处理的 PCM 采样点数，非负且不得超过 samples.length，不是字节数
     * @param n_threads 原生计算线程数，应为正数
     * @return 成功返回 0，失败返回非零值
     */
    int whisper_pcm_to_mel(Pointer ctx, final float[] samples, int n_samples, int n_threads);

    /**
     * 将 16 kHz 单声道浮点 PCM 转换为对数 Mel 频谱，保存到指定状态。
     * 会覆盖状态中的频谱数据；同一状态不得并发计算或读取中间结果。
     *
     * @param ctx       已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param state     由同一模型上下文创建且尚未释放的独立推理状态
     * @param samples   16 kHz 单声道浮点 PCM 数组，通常归一化至 [-1, 1]，长度至少为 n_samples
     * @param n_samples 本次处理的 PCM 采样点数，非负且不得超过 samples.length，不是字节数
     * @param n_threads 原生计算线程数，应为正数
     * @return 成功返回 0，失败返回非零值
     */
    int whisper_pcm_to_mel_with_state(Pointer ctx, Pointer state, final float[] samples, int n_samples, int n_threads);

    /**
     * 将调用方提供的对数 Mel 频谱复制到上下文默认状态。
     * 与 whisper_pcm_to_mel 二选一，调用后可执行编码或语种检测。
     * data 至少包含 n_len * n_mel 个 float；n_mel 必须等于 whisper_model_n_mels(ctx)。
     *
     * @param ctx   已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param data  按原生 Mel 数据布局排列的一维频谱数组
     * @param n_len 频谱帧数，非负
     * @param n_mel 模型要求的频带数；通常为 80，large-v3 系列为 128
     * @return 成功返回 0，Mel 频带数与模型不匹配时返回 -1
     */
    int whisper_set_mel(Pointer ctx, final float[] data, int n_len, int n_mel);

    /**
     * 将调用方提供的对数 Mel 频谱复制到指定状态。
     * 与 whisper_pcm_to_mel 二选一，调用后可执行编码或语种检测。
     * data 至少包含 n_len * n_mel 个 float；n_mel 必须等于 whisper_model_n_mels(ctx)。
     *
     * @param ctx   已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param state 由同一模型上下文创建且尚未释放的独立推理状态
     * @param data  按原生 Mel 数据布局排列的一维频谱数组
     * @param n_len 频谱帧数，非负
     * @param n_mel 模型要求的频带数；通常为 80，large-v3 系列为 128
     * @return 成功返回 0，Mel 频带数与模型不匹配时返回 -1
     */
    int whisper_set_mel_with_state(Pointer ctx, Pointer state, final float[] data, int n_len, int n_mel);

    /**
     * 对上下文默认状态中已准备的 Mel 频谱执行编码。
     * 必须先调用对应状态的 whisper_pcm_to_mel 或 whisper_set_mel。
     * 编码结果保存在该状态中，供后续 whisper_decode 使用。
     *
     * @param ctx       已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param offset    频谱起始帧偏移，单位为 Mel 帧（每帧约 10 毫秒），不是毫秒
     * @param n_threads 原生计算线程数，应为正数
     * @return 成功返回 0，失败返回非零值
     */
    int whisper_encode(Pointer ctx, int offset, int n_threads);

    /**
     * 对指定状态中已准备的 Mel 频谱执行编码。
     * 必须先调用对应状态的 whisper_pcm_to_mel 或 whisper_set_mel。
     * 编码结果保存在该状态中，供后续 whisper_decode 使用。
     *
     * @param ctx       已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param state     由同一模型上下文创建且尚未释放的独立推理状态
     * @param offset    频谱起始帧偏移，单位为 Mel 帧（每帧约 10 毫秒），不是毫秒
     * @param n_threads 原生计算线程数，应为正数
     * @return 成功返回 0，失败返回非零值
     */
    int whisper_encode_with_state(Pointer ctx, Pointer state, int offset, int n_threads);

    /**
     * 基于上下文默认状态中的编码结果和 token 上下文执行解码。
     * 必须先完成相同状态的编码；该调用更新解码缓存及 logits。
     * tokens 指向至少 n_tokens 个连续 32 位原生 whisper_token；内存由调用方管理。
     *
     * @param ctx       已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param tokens    输入 token ID 的原生内存指针，调用期间必须有效
     * @param n_tokens  输入 token 数量，不得超过 tokens 的分配容量
     * @param n_past    复用之前解码缓存的 token 数量，必须与当前状态的解码历史一致
     * @param n_threads 原生计算线程数，应为正数
     * @return 成功返回 0，失败返回非零值
     */
    int whisper_decode(Pointer ctx, Pointer tokens, int n_tokens, int n_past, int n_threads);

    /**
     * 基于指定状态中的编码结果和 token 上下文执行解码。
     * 必须先完成相同状态的编码；该调用更新解码缓存及 logits。
     * tokens 指向至少 n_tokens 个连续 32 位原生 whisper_token；内存由调用方管理。
     *
     * @param ctx       已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param state     由同一模型上下文创建且尚未释放的独立推理状态
     * @param tokens    输入 token ID 的原生内存指针，调用期间必须有效
     * @param n_tokens  输入 token 数量，不得超过 tokens 的分配容量
     * @param n_past    复用之前解码缓存的 token 数量，必须与当前状态的解码历史一致
     * @param n_threads 原生计算线程数，应为正数
     * @return 成功返回 0，失败返回非零值
     */
    int whisper_decode_with_state(Pointer ctx, Pointer state, Pointer tokens, int n_tokens, int n_past, int n_threads);

    /**
     * 使用当前模型词表将文本转换为 token ID。
     * tokens 至少分配 n_max_tokens * 4 字节，原生不会为输出分配内存。
     * 可用 tokens=null、n_max_tokens=0 查询所需容量；空文本返回 0。
     *
     * @param ctx          已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param text         待分词文本；字符串编码须与模型预期的 UTF-8 一致
     * @param tokens       调用方分配的连续 32 位 token 输出内存；容量查询时可为 null
     * @param n_max_tokens 输出缓冲区可容纳的 token 数，不是字节数
     * @return 成功返回写入的 token 数；容量不足返回所需 token 数的负值
     */
    int whisper_tokenize(Pointer ctx, String text, Pointer tokens, int n_max_tokens);

    /**
     * 查询当前原生库语种表中的最大语种 ID。
     * 这是原生语种表上界，不等同于所加载模型的语种检测能力。
     *
     * @return 最大语种 ID；概率输出数组容量应至少为该值加 1
     */
    int whisper_lang_max_id();

    /**
     * 按语种代码或完整英文名称查询原生语种 ID。
     *
     * @param lang 原生表中的语种代码或英文名称，例如 de 或 german；需匹配原生表的大小写
     * @return 找到时返回语种 ID，未知语种返回 -1
     */
    int whisper_lang_id(String lang);

    /**
     * 按原生语种 ID 获取短代码。
     *
     * @param id 原生语种 ID，通常位于 0 至 whisper_lang_max_id() 范围内
     * @return 语种代码，例如 en；未知 ID 返回 null
     */
    String whisper_lang_str(int id);

    /**
     * 按原生语种 ID 获取完整英文名称，不使用 Java 侧名称映射。
     *
     * @param id 原生语种 ID，通常位于 0 至 whisper_lang_max_id() 范围内
     * @return 完整英文名称，例如 english；未知 ID 返回 null
     */
    String whisper_lang_str_full(int id);

    /**
     * 使用默认状态中的 Mel 频谱计算候选语种概率。
     * 必须先准备同一状态的 Mel 频谱，且应使用多语种模型；调用会修改编码和解码状态。
     * lang_probs 以原生语种 ID 为索引，写入 softmax 概率；它表示候选分布，不是语种时长占比。
     * 每次检测指定偏移处的模型音频窗口，不会自动扫描整段音频并识别语种切换。
     *
     * @param ctx        已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param offset_ms  检测窗口起点，单位为毫秒，必须位于已准备的音频范围内
     * @param n_threads  原生计算线程数，应为正数
     * @param lang_probs 可为 null；非 null 时长度至少为 whisper_lang_max_id() + 1，概率范围为 [0, 1]
     * @return 概率最高的语种 ID；偏移无效、编码或解码失败时返回负值
     */
    int whisper_lang_auto_detect(Pointer ctx, int offset_ms, int n_threads, float[] lang_probs);

    /**
     * 使用指定状态中的 Mel 频谱计算候选语种概率。
     * 必须先准备同一状态的 Mel 频谱，且应使用多语种模型；调用会修改编码和解码状态。
     * lang_probs 以原生语种 ID 为索引，写入 softmax 概率；它表示候选分布，不是语种时长占比。
     * 每次检测指定偏移处的模型音频窗口，不会自动扫描整段音频并识别语种切换。
     *
     * @param ctx        已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param state      由同一模型上下文创建且尚未释放的独立推理状态
     * @param offset_ms  检测窗口起点，单位为毫秒，必须位于已准备的音频范围内
     * @param n_threads  原生计算线程数，应为正数
     * @param lang_probs 可为 null；非 null 时长度至少为 whisper_lang_max_id() + 1，概率范围为 [0, 1]
     * @return 概率最高的语种 ID；偏移无效、编码或解码失败时返回负值
     */
    int whisper_lang_auto_detect_with_state(Pointer ctx, Pointer state, int offset_ms, int n_threads, float[] lang_probs);

    /**
     * 查询默认状态中已准备的原始 Mel 频谱帧数。
     * 应先计算或设置 Mel 频谱；返回值不是 PCM 采样点数。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 未按模型窗口补齐之前的 Mel 帧数，每帧约 10 毫秒
     */
    int whisper_n_len(Pointer ctx);

    /**
     * 查询独立状态中已准备的原始 Mel 频谱帧数。
     * 应先计算或设置 Mel 频谱；返回值不是 PCM 采样点数。
     *
     * @param state 由同一模型上下文创建且尚未释放的独立推理状态
     * @return 未按模型窗口补齐之前的 Mel 帧数，每帧约 10 毫秒
     */
    int whisper_n_len_from_state(Pointer state);

    /**
     * 读取已加载模型的词表大小。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 词表 token 数量
     */
    int whisper_n_vocab(Pointer ctx);

    /**
     * 读取已加载模型的文本上下文长度。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 文本上下文可容纳的 token 数量
     */
    int whisper_n_text_ctx(Pointer ctx);

    /**
     * 读取已加载模型的音频编码器上下文长度。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 音频编码器的位置数量，不是 PCM 采样点数
     */
    int whisper_n_audio_ctx(Pointer ctx);

    /**
     * 查询当前模型是否为多语种模型。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 多语种模型返回 1，英语专用模型返回 0
     */
    int whisper_is_multilingual(Pointer ctx);

    /**
     * 读取模型文件声明的词表大小。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 模型超参数中的词表 token 数量
     */
    int whisper_model_n_vocab(Pointer ctx);

    /**
     * 读取模型声明的音频编码器上下文长度。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 模型超参数中的音频位置数量
     */
    int whisper_model_n_audio_ctx(Pointer ctx);

    /**
     * 读取已加载模型的音频编码器隐藏维度。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 每个音频位置的隐藏向量维数
     */
    int whisper_model_n_audio_state(Pointer ctx);

    /**
     * 读取已加载模型的音频编码器注意力头数。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 每层音频注意力头数量
     */
    int whisper_model_n_audio_head(Pointer ctx);

    /**
     * 读取已加载模型的音频编码器层数。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 音频编码器层数量
     */
    int whisper_model_n_audio_layer(Pointer ctx);

    /**
     * 读取模型声明的文本上下文长度。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 模型超参数中的文本 token 容量
     */
    int whisper_model_n_text_ctx(Pointer ctx);

    /**
     * 读取已加载模型的文本解码器隐藏维度。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 每个文本位置的隐藏向量维数
     */
    int whisper_model_n_text_state(Pointer ctx);

    /**
     * 读取已加载模型的文本解码器注意力头数。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 每层文本注意力头数量
     */
    int whisper_model_n_text_head(Pointer ctx);

    /**
     * 读取已加载模型的文本解码器层数。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 文本解码器层数量
     */
    int whisper_model_n_text_layer(Pointer ctx);

    /**
     * 读取模型要求的 Mel 频带数。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return Mel 频带数量，例如 80 或 128
     */
    int whisper_model_n_mels(Pointer ctx);

    /**
     * 读取模型文件的数值存储类型。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 原生模型 ftype 编码，表示模型存储精度或量化类型
     */
    int whisper_model_ftype(Pointer ctx);

    /**
     * 读取已加载模型的规模类型。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 原生 e_model 枚举数值，不是 Java 枚举序号
     */
    int whisper_model_type(Pointer ctx);

    /**
     * 访问最近一次解码生成的原生 token logits。
     * 原生数据按 token 行、词表列排列；末行对应最后一个输入 token，元素是未归一化 logits。
     * 原生内存属于状态，下次解码或状态释放后可能失效，不得由调用方释放。
     * 注意：当前 float[] 返回声明没有携带原生数组长度，不能据此保证数组映射正确；使用前须核对 JNA 映射。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 当前 JNA 声明映射出的 float 数组；原生实际返回借用的 float 指针
     */
    float[] whisper_get_logits(Pointer ctx);

    /**
     * 访问最近一次解码生成的原生 token logits。
     * 原生数据按 token 行、词表列排列；末行对应最后一个输入 token，元素是未归一化 logits。
     * 原生内存属于状态，下次解码或状态释放后可能失效，不得由调用方释放。
     * 注意：当前 float[] 返回声明没有携带原生数组长度，不能据此保证数组映射正确；使用前须核对 JNA 映射。
     *
     * @param state 由同一模型上下文创建且尚未释放的独立推理状态
     * @return 当前 JNA 声明映射出的 float 数组；原生实际返回借用的 float 指针
     */
    float[] whisper_get_logits_from_state(Pointer state);

    /**
     * 使用当前模型词表查询 token 对应的原文片段。
     * token 必须是当前词表内的有效 ID，原生不会统一转换越界错误为 Java 异常。
     *
     * @param ctx   已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param token 当前模型的有效 token ID
     * @return token 对应的词表字符串，不保证单个 token 是完整单词或完整 Unicode 字符
     */
    String whisper_token_to_str(Pointer ctx, int token);

    /**
     * 读取已加载模型的可读模型规模名称。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 原生模型名称，例如 tiny 或 base
     */
    String whisper_model_type_readable(Pointer ctx);

    /**
     * 查询当前模型的文本结束特殊 token。
     * 特殊 token ID 依赖模型词表，不应在 Java 侧硬编码。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 该特殊 token 在当前模型词表中的 ID
     */
    int whisper_token_eot(Pointer ctx);

    /**
     * 查询当前模型的转写序列开始特殊 token。
     * 特殊 token ID 依赖模型词表，不应在 Java 侧硬编码。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 该特殊 token 在当前模型词表中的 ID
     */
    int whisper_token_sot(Pointer ctx);

    /**
     * 查询当前模型的前文提示开始特殊 token。
     * 特殊 token ID 依赖模型词表，不应在 Java 侧硬编码。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 该特殊 token 在当前模型词表中的 ID
     */
    int whisper_token_prev(Pointer ctx);

    /**
     * 查询当前模型的说话人轮换（tinydiarize）特殊 token。
     * 特殊 token ID 依赖模型词表，不应在 Java 侧硬编码。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 该特殊 token 在当前模型词表中的 ID
     */
    int whisper_token_solm(Pointer ctx);

    /**
     * 查询当前模型的禁用时间戳特殊 token。
     * 特殊 token ID 依赖模型词表，不应在 Java 侧硬编码。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 该特殊 token 在当前模型词表中的 ID
     */
    int whisper_token_not(Pointer ctx);

    /**
     * 查询当前模型的时间戳区间起始特殊 token。
     * 特殊 token ID 依赖模型词表，不应在 Java 侧硬编码。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 该特殊 token 在当前模型词表中的 ID
     */
    int whisper_token_beg(Pointer ctx);

    /**
     * 查询当前模型中指定语种的特殊 token。
     * 应使用支持该语种的多语种模型；原生计算 token ID 时不负责验证语种 ID 合法性。
     *
     * @param ctx     已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param lang_id 原生语种 ID，可通过 whisper_lang_id 查询
     * @return 该语种对应的 token ID
     */
    int whisper_token_lang(Pointer ctx, int lang_id);

    /**
     * 查询当前模型的翻译任务特殊 token。
     * 特殊 token ID 依赖模型词表，不应在 Java 侧硬编码。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 该特殊 token 在当前模型词表中的 ID
     */
    int whisper_token_translate(Pointer ctx);

    /**
     * 查询当前模型的转写任务特殊 token。
     * 特殊 token ID 依赖模型词表，不应在 Java 侧硬编码。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 该特殊 token 在当前模型词表中的 ID
     */
    int whisper_token_transcribe(Pointer ctx);

    /**
     * 通过原生日志回调输出模型加载及默认状态的性能统计。
     * 统计值包含原生采样、编码、解码等阶段耗时，单位为毫秒。
     * 需要统一日志输出时，应先注册 whisper_log_set；不得与默认状态推理并发读取统计。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     */
    void whisper_print_timings(Pointer ctx);

    /**
     * 重置默认状态的累计推理计时和相关计数。
     * 用于开始新一轮性能统计，不释放模型或推理结果；不得与同一状态推理并发执行。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     */
    void whisper_reset_timings(Pointer ctx);

    /**
     * 按采样策略在原生堆上分配默认完整推理参数。
     * 该入口避免直接按值返回复杂参数结构体；参数字段布局仍须与当前原生库一致。
     * 调用方必须使用 whisper_free_params 配对释放，不应使用 Native.free 释放 C++ new 分配的参数。
     *
     * @param strategy 原生采样策略枚举值，例如 GREEDY=0 或 BEAM_SEARCH=1
     * @return 新分配的 whisper_full_params 指针
     */
    Pointer whisper_full_default_params_by_ref(int strategy);

    /**
     * 释放 whisper_full_default_params_by_ref 分配的完整推理参数。
     * 仅释放参数结构体本身，不接管调用方传入的音频、外部指针或回调对象。
     * 释放后参数内存及基于它创建的结构体视图失效，不得重复释放。
     *
     * @param params 需要释放的原生完整推理参数指针
     */
    void whisper_free_params(Pointer params);

    /**
     * 使用默认状态执行完整音频推理。
     * 执行 PCM → Mel → 编码 → 解码流程；自动检测语种等提前返回行为由 params 控制。
     * params 按值传入；其中引用的原生指针和 Java 回调必须在调用期间保持有效。
     * 结果存入上下文默认状态；同一上下文的默认状态不得被并发推理或释放。
     *
     * @param ctx       已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param params    按值传入的完整推理配置，必须与当前原生结构体布局一致
     * @param samples   16 kHz 单声道浮点 PCM 数组，通常归一化至 [-1, 1]，长度至少为 n_samples
     * @param n_samples 本次处理的 PCM 采样点数，非负且不得超过 samples.length，不是字节数
     * @return 成功返回 0，失败返回非零原生状态码；具体原因通过原生日志输出
     */
    int whisper_full(Pointer ctx, WhisperFullParams.ByValue params, final float[] samples, int n_samples);

    /**
     * 使用指定独立状态执行完整音频推理。
     * 执行 PCM → Mel → 编码 → 解码流程；自动检测语种等提前返回行为由 params 控制。
     * params 按值传入；其中引用的原生指针和 Java 回调必须在调用期间保持有效。
     * 结果存入指定状态，须使用对应的 from_state 查询接口；同一状态不支持并发执行。
     *
     * @param ctx       已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param state     由同一模型上下文创建且尚未释放的独立推理状态
     * @param params    按值传入的完整推理配置，必须与当前原生结构体布局一致
     * @param samples   16 kHz 单声道浮点 PCM 数组，通常归一化至 [-1, 1]，长度至少为 n_samples
     * @param n_samples 本次处理的 PCM 采样点数，非负且不得超过 samples.length，不是字节数
     * @return 成功返回 0，失败返回非零原生状态码；具体原因通过原生日志输出
     */
    public int whisper_full_with_state(Pointer ctx, Pointer state, WhisperFullParams.ByValue params, float[] samples, int n_samples);

    /**
     * 使用默认状态执行分块并行音频推理。
     * 执行 PCM → Mel → 编码 → 解码流程；自动检测语种等提前返回行为由 params 控制。
     * params 按值传入；其中引用的原生指针和 Java 回调必须在调用期间保持有效。
     * 结果存入上下文默认状态；同一上下文的默认状态不得被并发推理或释放。
     * 内部将音频分块并使用多个状态处理；块边界处的识别准确率可能下降，需实测速度收益。
     *
     * @param ctx          已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param params       按值传入的完整推理配置，必须与当前原生结构体布局一致
     * @param samples      16 kHz 单声道浮点 PCM 数组，通常归一化至 [-1, 1]，长度至少为 n_samples
     * @param n_samples    本次处理的 PCM 采样点数，非负且不得超过 samples.length，不是字节数
     * @param n_processors 并行分块数量，至少为 1；并非单次推理内部的线程数
     * @return 成功返回 0，失败返回非零原生状态码；具体原因通过原生日志输出
     */
    int whisper_full_parallel(Pointer ctx, WhisperFullParams.ByValue params, final float[] samples, int n_samples, int n_processors);

    /**
     * 读取默认状态中最近一次完整推理的文本分段数量。
     * 应先完成该状态的完整推理，结果被下一次推理覆盖。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 分段数量；无结果时为 0
     */
    int whisper_full_n_segments(Pointer ctx);

    /**
     * 读取指定状态中最近一次完整推理的文本分段数量。
     * 应先完成该状态的完整推理，结果被下一次推理覆盖。
     *
     * @param state 由同一模型上下文创建且尚未释放的独立推理状态
     * @return 分段数量；无结果时为 0
     */
    int whisper_full_n_segments_from_state(Pointer state);

    /**
     * 读取默认状态在完整推理中使用或自动检测出的语种 ID。
     * 应先完成该状态的完整推理；可交给 whisper_lang_str 或 whisper_lang_str_full 查询名称。
     * 本方法只返回已记录的结果，不会重新执行语种检测。
     *
     * @param ctx 已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @return 该状态记录的原生语种 ID
     */
    int whisper_full_lang_id(Pointer ctx);

    /**
     * 读取指定状态在完整推理中使用或自动检测出的语种 ID。
     * 应先完成该状态的完整推理；可交给 whisper_lang_str 或 whisper_lang_str_full 查询名称。
     * 本方法只返回已记录的结果，不会重新执行语种检测。
     *
     * @param state 由同一模型上下文创建且尚未释放的独立推理状态
     * @return 该状态记录的原生语种 ID
     */
    int whisper_full_lang_id_from_state(Pointer state);

    /**
     * 读取默认状态指定分段的开始时间。
     * 索引必须位于最近一次结果范围内；启用 VAD 时，当前原生实现会映射回原音频时间轴。
     *
     * @param ctx       已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param i_segment 分段索引，从 0 开始，小于对应状态的 whisper_full_n_segments 返回值
     * @return 相对音频时间轴的开始时间，单位为 10 毫秒；乘 10 可换算为毫秒
     */
    long whisper_full_get_segment_t0(Pointer ctx, int i_segment);

    /**
     * 读取指定状态指定分段的开始时间。
     * 索引必须位于最近一次结果范围内；启用 VAD 时，当前原生实现会映射回原音频时间轴。
     *
     * @param state     由同一模型上下文创建且尚未释放的独立推理状态
     * @param i_segment 分段索引，从 0 开始，小于对应状态的 whisper_full_n_segments 返回值
     * @return 相对音频时间轴的开始时间，单位为 10 毫秒；乘 10 可换算为毫秒
     */
    long whisper_full_get_segment_t0_from_state(Pointer state, int i_segment);

    /**
     * 读取默认状态指定分段的结束时间。
     * 索引必须位于最近一次结果范围内；启用 VAD 时，当前原生实现会映射回原音频时间轴。
     *
     * @param ctx       已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param i_segment 分段索引，从 0 开始，小于对应状态的 whisper_full_n_segments 返回值
     * @return 相对音频时间轴的结束时间，单位为 10 毫秒；乘 10 可换算为毫秒
     */
    long whisper_full_get_segment_t1(Pointer ctx, int i_segment);

    /**
     * 读取指定状态指定分段的结束时间。
     * 索引必须位于最近一次结果范围内；启用 VAD 时，当前原生实现会映射回原音频时间轴。
     *
     * @param state     由同一模型上下文创建且尚未释放的独立推理状态
     * @param i_segment 分段索引，从 0 开始，小于对应状态的 whisper_full_n_segments 返回值
     * @return 相对音频时间轴的结束时间，单位为 10 毫秒；乘 10 可换算为毫秒
     */
    long whisper_full_get_segment_t1_from_state(Pointer state, int i_segment);

    /**
     * 读取默认状态指定分段的识别文本。
     * 应先完成完整推理，分段索引必须有效。
     *
     * @param ctx       已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param i_segment 分段索引，从 0 开始，小于对应状态的 whisper_full_n_segments 返回值
     * @return 分段文本，可能包含前导空格；无需调用方释放
     */
    String whisper_full_get_segment_text(Pointer ctx, int i_segment);

    /**
     * 读取指定状态指定分段的识别文本。
     * 应先完成完整推理，分段索引必须有效。
     *
     * @param state     由同一模型上下文创建且尚未释放的独立推理状态
     * @param i_segment 分段索引，从 0 开始，小于对应状态的 whisper_full_n_segments 返回值
     * @return 分段文本，可能包含前导空格；无需调用方释放
     */
    String whisper_full_get_segment_text_from_state(Pointer state, int i_segment);

    /**
     * 读取默认状态指定分段包含的 token 数量。
     * 分段索引必须位于最近一次完整推理的结果范围内。
     *
     * @param ctx       已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param i_segment 分段索引，从 0 开始，小于对应状态的 whisper_full_n_segments 返回值
     * @return 该分段的 token 数，可能包含特殊 token
     */
    int whisper_full_n_tokens(Pointer ctx, int i_segment);

    /**
     * 读取指定状态指定分段包含的 token 数量。
     * 分段索引必须位于最近一次完整推理的结果范围内。
     *
     * @param state     由同一模型上下文创建且尚未释放的独立推理状态
     * @param i_segment 分段索引，从 0 开始，小于对应状态的 whisper_full_n_segments 返回值
     * @return 该分段的 token 数，可能包含特殊 token
     */
    int whisper_full_n_tokens_from_state(Pointer state, int i_segment);

    /**
     * 读取默认状态指定分段内 token 的词表文本。
     * 必须先完成对应状态的完整推理，分段与 token 索引均不得越界。
     *
     * @param ctx       已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param i_segment 分段索引，从 0 开始，小于对应状态的 whisper_full_n_segments 返回值
     * @param i_token   分段内 token 索引，从 0 开始，小于对应分段的 whisper_full_n_tokens 返回值
     * @return 该 token 的词表文本，可能不是完整单词或完整 Unicode 字符
     */
    String whisper_full_get_token_text(Pointer ctx, int i_segment, int i_token);

    /**
     * 读取指定状态指定分段内 token 的词表文本。
     * 必须先完成对应状态的完整推理，分段与 token 索引均不得越界。
     *
     * @param ctx       已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param state     由同一模型上下文创建且尚未释放的独立推理状态
     * @param i_segment 分段索引，从 0 开始，小于对应状态的 whisper_full_n_segments 返回值
     * @param i_token   分段内 token 索引，从 0 开始，小于对应分段的 whisper_full_n_tokens 返回值
     * @return 该 token 的词表文本，可能不是完整单词或完整 Unicode 字符
     */
    String whisper_full_get_token_text_from_state(Pointer ctx, Pointer state, int i_segment, int i_token);

    /**
     * 读取默认状态指定分段内 token 的token ID。
     * 必须先完成对应状态的完整推理，分段与 token 索引均不得越界。
     *
     * @param ctx       已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param i_segment 分段索引，从 0 开始，小于对应状态的 whisper_full_n_segments 返回值
     * @param i_token   分段内 token 索引，从 0 开始，小于对应分段的 whisper_full_n_tokens 返回值
     * @return 当前模型词表中的 token ID
     */
    int whisper_full_get_token_id(Pointer ctx, int i_segment, int i_token);

    /**
     * 读取指定状态指定分段内 token 的token ID。
     * 必须先完成对应状态的完整推理，分段与 token 索引均不得越界。
     *
     * @param state     由同一模型上下文创建且尚未释放的独立推理状态
     * @param i_segment 分段索引，从 0 开始，小于对应状态的 whisper_full_n_segments 返回值
     * @param i_token   分段内 token 索引，从 0 开始，小于对应分段的 whisper_full_n_tokens 返回值
     * @return 当前模型词表中的 token ID
     */
    int whisper_full_get_token_id_from_state(Pointer state, int i_segment, int i_token);

    /**
     * 读取默认状态指定分段内 token 的完整 token 数据。
     * 必须先完成对应状态的完整推理，分段与 token 索引均不得越界。
     * token 时间戳仅在启用 token_timestamps 时有意义，时间字段单位为 10 毫秒。
     * 原生按值返回结构体；使用前需确认 WhisperTokenData 的 JNA 按值返回映射与当前 ABI 一致。
     * 启用 VAD 时此结构体的时间字段保留 VAD 处理后的时间轴，不能直接当作原音频时间。
     *
     * @param ctx       已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param i_segment 分段索引，从 0 开始，小于对应状态的 whisper_full_n_segments 返回值
     * @param i_token   分段内 token 索引，从 0 开始，小于对应分段的 whisper_full_n_tokens 返回值
     * @return 原生 token 数据结构，包含 ID、概率及时间信息
     */
    WhisperTokenData whisper_full_get_token_data(Pointer ctx, int i_segment, int i_token);

    /**
     * 读取指定状态指定分段内 token 的完整 token 数据。
     * 必须先完成对应状态的完整推理，分段与 token 索引均不得越界。
     * token 时间戳仅在启用 token_timestamps 时有意义，时间字段单位为 10 毫秒。
     * 原生按值返回结构体；使用前需确认 WhisperTokenData 的 JNA 按值返回映射与当前 ABI 一致。
     * 启用 VAD 时此结构体的时间字段保留 VAD 处理后的时间轴，不能直接当作原音频时间。
     *
     * @param state     由同一模型上下文创建且尚未释放的独立推理状态
     * @param i_segment 分段索引，从 0 开始，小于对应状态的 whisper_full_n_segments 返回值
     * @param i_token   分段内 token 索引，从 0 开始，小于对应分段的 whisper_full_n_tokens 返回值
     * @return 原生 token 数据结构，包含 ID、概率及时间信息
     */
    WhisperTokenData whisper_full_get_token_data_from_state(Pointer state, int i_segment, int i_token);

    /**
     * 读取默认状态指定分段内 token 的采样概率。
     * 必须先完成对应状态的完整推理，分段与 token 索引均不得越界。
     *
     * @param ctx       已成功加载且尚未释放的模型上下文；默认状态要求见接口级调用约定
     * @param i_segment 分段索引，从 0 开始，小于对应状态的 whisper_full_n_segments 返回值
     * @param i_token   分段内 token 索引，从 0 开始，小于对应分段的 whisper_full_n_tokens 返回值
     * @return 该 token 在解码时记录的概率，通常位于 [0, 1]
     */
    float whisper_full_get_token_p(Pointer ctx, int i_segment, int i_token);

    /**
     * 读取指定状态指定分段内 token 的采样概率。
     * 必须先完成对应状态的完整推理，分段与 token 索引均不得越界。
     *
     * @param state     由同一模型上下文创建且尚未释放的独立推理状态
     * @param i_segment 分段索引，从 0 开始，小于对应状态的 whisper_full_n_segments 返回值
     * @param i_token   分段内 token 索引，从 0 开始，小于对应分段的 whisper_full_n_tokens 返回值
     * @return 该 token 在解码时记录的概率，通常位于 [0, 1]
     */
    float whisper_full_get_token_p_from_state(Pointer state, int i_segment, int i_token);

    /**
     * 运行内存复制性能基准并将报告直接写入原生 stderr。
     * 该入口使用原生标准错误流，不经过 whisper_log_set；若需自行记录报告，使用对应的 _str 入口。
     * 基准可能分配较大内存并持续占用计算资源，不适合作为普通业务请求的一部分。
     *
     * @param nThreads 基准使用的线程数，应为正数
     * @return 当前原生实现返回 0；性能数值在输出报告中，不是返回码
     */
    int whisper_bench_memcpy(int nThreads);

    /**
     * 运行内存复制性能基准并获取文本报告。
     * 原生复用静态报告字符串，不应并发调用同一基准函数。
     * 基准会实际分配内存并执行计算；调用方可选择将返回报告写入自身日志。
     *
     * @param nThreads 基准使用的线程数，应为正数
     * @return 包含性能结果的报告字符串，不需要调用方释放
     */
    String whisper_bench_memcpy_str(int nThreads);

    /**
     * 运行GGML 矩阵乘法性能基准并将报告直接写入原生 stderr。
     * 该入口使用原生标准错误流，不经过 whisper_log_set；若需自行记录报告，使用对应的 _str 入口。
     * 基准可能分配较大内存并持续占用计算资源，不适合作为普通业务请求的一部分。
     *
     * @param nThreads 基准使用的线程数，应为正数
     * @return 当前原生实现返回 0；性能数值在输出报告中，不是返回码
     */
    int whisper_bench_ggml_mul_mat(int nThreads);

    /**
     * 运行GGML 矩阵乘法性能基准并获取文本报告。
     * 原生复用静态报告字符串，不应并发调用同一基准函数。
     * 基准会实际分配内存并执行计算；调用方可选择将返回报告写入自身日志。
     *
     * @param nThreads 基准使用的线程数，应为正数
     * @return 包含性能结果的报告字符串，不需要调用方释放
     */
    String whisper_bench_ggml_mul_mat_str(int nThreads);

    /**
     * 设置进程内 Whisper 及 GGML 的全局日志回调。
     * 应在查询系统信息、初始化模型或设备之前注册，才能接收随后产生的初始化日志。
     * 该设置不属于某个模型实例，后续注册会覆盖此前回调，传入 null 恢复默认原生日志输出。
     * 必须在原生库仍可能调用回调期间持有 Java 强引用，包含模型释放及后台线程退出阶段。
     * 回调可能由不同原生线程调用，必须线程安全且不向原生边界抛出 Java 异常。
     * 不拦截使用 fputs 等方式直接写入原生 stdout/stderr 的输出。
     *
     * @param ggmlLogCallback 日志回调；null 表示恢复原生默认实现
     * @param userData        传给每次回调的原生用户数据指针；无需透传时为 null，指向的内存须持续有效
     */
    void whisper_log_set(GgmlLogCallback ggmlLogCallback, Pointer userData);
}
