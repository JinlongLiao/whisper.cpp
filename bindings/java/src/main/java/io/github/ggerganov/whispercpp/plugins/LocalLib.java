package io.github.ggerganov.whispercpp.plugins;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Objects;

import io.github.ggerganov.whispercpp.WhisperCppJnaLibrary;

/**
 * @author liaojinlong
 * @date 2025/10/21 21:34
 */
public final class LocalLib {

    public static String getLibPath(int init) {
        File dirFolder = new File(System.getProperty("log.path", System.getProperty("java.io.tmpdir")), "whisper/");
        if (!dirFolder.exists()) {
            dirFolder.mkdirs();
        }
        String os = System.getProperty("os.name").toLowerCase();
        boolean isWindows = os.contains("windows");
        boolean isMac = os.contains("mac");
        File ggml;
        File ggmlBase;
        File ggmlCpu;
        File whisper;
        File blas = null;
        File metal = null;
        String path;
        boolean isLinux = false;
        if (isWindows) {
            ggml = new File(dirFolder, "ggml.dll");
            ggmlBase = new File(dirFolder, "ggml-base.dll");
            ggmlCpu = new File(dirFolder, "ggml-cpu.dll");
            whisper = new File(dirFolder, "whisper.dll");
            path = "/win32-x86-64";
        } else if (isMac) {
            ggml = new File(dirFolder, "libggml.dylib");
            ggmlBase = new File(dirFolder, "libggml-base.dylib");
            ggmlCpu = new File(dirFolder, "libggml-cpu.dylib");
            whisper = new File(dirFolder, "libwhisper.dylib");
            blas = new File(dirFolder, "libggml-blas.dylib");
            metal = new File(dirFolder, "libggml-metal.dylib");
            path = "/darwin";
        } else {
            ggml = new File(dirFolder, "libggml.so");
            ggmlBase = new File(dirFolder, "libggml-base.so");
            ggmlCpu = new File(dirFolder, "libggml-cpu.so");
            whisper = new File(dirFolder, "libwhisper.so");
            path = "/linux-x86-64";
            isLinux = true;
        }
        if (!whisper.exists()) {
            copyFile(path, ggml);
            copyFile(path, ggmlBase);
            copyFile(path, ggmlCpu);
            copyFile(path, whisper);
            if (Objects.nonNull(blas)) {
                copyFile(path, blas);
            }
            if (Objects.nonNull(metal)) {
                copyFile(path, metal);
            }
        }
        if (init == 0) {
            return ggmlBase.getAbsolutePath();
        }
        if (init == 1) {
            return ggmlCpu.getAbsolutePath();
        }
        if (init == 2) {
            return ggml.getAbsolutePath();
        }
        if (init == 3) {
            if (Objects.isNull(blas)) {
                return null;
            }
            return blas.getAbsolutePath();
        }
        if (init == 4) {
            if (Objects.isNull(metal)) {
                return null;
            }
            return metal.getAbsolutePath();
        }
        return whisper.getAbsolutePath();
    }

    /**
     * Copies a file bundled in the package to the supplied destination.
     *
     * @param path The name of the bundled file.
     * @param dest The destination.
     * @throws RuntimeException If an unexpected error occurs.
     */
    static void copyFile(String path, File dest) {
        try {
            InputStream is = WhisperCppJnaLibrary.class.getResourceAsStream(path + "/" + dest.getName());
            if (is != null) {
                try {
                    dest.createNewFile();
                    dest.deleteOnExit();
                    copy(is, dest.getAbsolutePath());

                    is.close();
                } catch (IOException ioex) {
                }
            }
        } catch (NullPointerException ex) {

            throw ex;
        }
    }

    /**
     * Copy a file from source to destination.
     *
     * @param source      The name of the bundled file.
     * @param destination the destination
     * @return True if succeeded , False if not
     */
    static boolean copy(InputStream source, String destination) {
        boolean success = true;

        try {
            Files.copy(source, Paths.get(destination), StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException ex) {
            success = false;
        }

        return success;
    }

}
