package io.github.ggerganov.whispercpp.plugins;

import com.sun.jna.Library;
import com.sun.jna.Native;

// 定义空接口用于加载依赖库（仅为加载库本身，无需声明函数）
public interface GGMLMetaLibrary extends Library {
    GGMLMetaLibrary GGML_META_LIBRARY = Native.load(LocalLib.getLibPath(4), GGMLMetaLibrary.class);
}
