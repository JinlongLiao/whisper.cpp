package io.github.ggerganov.whispercpp.plugins;

import com.sun.jna.Library;
import com.sun.jna.Native;

public interface GGMLLibrary extends Library {
    GGMLLibrary GGML_LIBRARY = Native.load(LocalLib.getLibPath(2), GGMLLibrary.class);
}
