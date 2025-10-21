package io.github.ggerganov.whispercpp.plugins;

import com.sun.jna.Library;
import com.sun.jna.Native;

// 定义空接口用于加载依赖库（仅为加载库本身，无需声明函数）
public  interface GGMLBaseLibrary extends Library {
    GGMLBaseLibrary GGML_BASE_LIBRARY = Native.load(LocalLib.getLibPath(0), GGMLBaseLibrary.class);
}
