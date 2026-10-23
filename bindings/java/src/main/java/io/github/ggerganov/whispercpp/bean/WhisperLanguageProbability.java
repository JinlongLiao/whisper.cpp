package io.github.ggerganov.whispercpp.bean;

/**
 * 原生语种检测返回的单个候选；不可变对象，可在线程之间安全共享。
 */
public final class WhisperLanguageProbability {
    /**
     * 原生库支持的语种 ID，与当前库的语种表一致。
     */
    private final int languageId;
    /**
     * 原生库返回的语种代码，例如 en。
     */
    private final String languageCode;
    /**
     * 原生库返回的完整英文名称，例如 english。
     */
    private final String languageName;
    /**
     * 此候选在当前检测窗口中的概率，范围为 [0, 1]，不是时长占比。
     */
    private final float probability;

    /**
     * 保存一次检测中的原生候选值，不执行额外的语种名称映射。
     *
     * @param languageId   原生语种 ID
     * @param languageCode 原生语种代码
     * @param languageName 原生完整英文名称
     * @param probability  原生 softmax 概率，范围为 [0, 1]
     */
    public WhisperLanguageProbability(int languageId, String languageCode, String languageName, float probability) {
        this.languageId = languageId;
        this.languageCode = languageCode;
        this.languageName = languageName;
        this.probability = probability;
    }

    /**
     * @return 原生语种 ID
     */
    public int getLanguageId() {
        return languageId;
    }

    /**
     * @return 原生语种代码
     */
    public String getLanguageCode() {
        return languageCode;
    }

    /**
     * @return 原生完整英文名称
     */
    public String getLanguageName() {
        return languageName;
    }

    /**
     * @return 范围为 [0, 1] 的候选概率
     */
    public float getProbability() {
        return probability;
    }
}
