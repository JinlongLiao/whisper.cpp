package io.github.ggerganov.whispercpp.plugins;

import com.sun.jna.Library;
import com.sun.jna.Native;

public interface GGMLCpuLibrary extends Library {
        GGMLCpuLibrary GGML_CPU_LIBRARY = Native.load(LocalLib.getLibPath(1), GGMLCpuLibrary.class);
    }
