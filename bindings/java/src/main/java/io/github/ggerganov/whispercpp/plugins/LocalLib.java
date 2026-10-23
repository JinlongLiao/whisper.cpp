package io.github.ggerganov.whispercpp.plugins;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
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
    public static boolean isMac;
    public static boolean isWindows;
    public static boolean isLinux;

    /**
     * 当前类加载器是否已完成原生库刷新；仅在 getLibPath 的类锁内访问。
     * 全部复制成功后才置为 true，避免加载时再次删除已使用的库。
     */
    private static boolean nativeLibrariesInitialized;

    /**
     * 首次调用删除并重新释放本平台原生库，后续调用复用已释放的文件。
     * 同步初始化保证并发调用不会在其他线程加载库时再次覆盖文件。
     *
     * @param init 库选择编号：0 base、1 cpu、2 ggml、3 blas、4 metal、5 parakeet，其他值为 whisper
     * @return 原生库绝对路径；平台不支持的可选库返回 null
     * @throws UncheckedIOException  删除或复制原生库失败时抛出
     * @throws IllegalStateException 打包资源不存在时抛出
     */
    public static synchronized String getLibPath(int init) {
        File dirFolder = new File(System.getProperty("log.path", System.getProperty("java.io.tmpdir")), "whisper/");
        if (!dirFolder.exists()) {
            dirFolder.mkdirs();
        }
        String os = System.getProperty("os.name").toLowerCase();
        isWindows = os.contains("windows");
        isMac = os.contains("mac");
        File ggml;
        File ggmlBase;
        File ggmlCpu;
        File whisper;
        File blas = null;
        File metal = null;
        File parakeet = null;
        String path;
        isLinux = false;
        if (isWindows) {
            ggml = new File(dirFolder, "ggml.dll");
            ggmlBase = new File(dirFolder, "ggml-base.dll");
            ggmlCpu = new File(dirFolder, "ggml-cpu.dll");
            whisper = new File(dirFolder, "whisper.dll");
            parakeet = new File(dirFolder, "parakeet.dll");
            path = "/win32-x86-64";
        } else if (isMac) {
            ggml = new File(dirFolder, "libggml.dylib");
            ggmlBase = new File(dirFolder, "libggml-base.dylib");
            ggmlCpu = new File(dirFolder, "libggml-cpu.dylib");
            whisper = new File(dirFolder, "libwhisper.dylib");
            blas = new File(dirFolder, "libggml-blas.dylib");
            metal = new File(dirFolder, "libggml-metal.dylib");
            parakeet = new File(dirFolder, "libparakeet.dylib");
            path = "/darwin";
        } else {
            ggml = new File(dirFolder, "libggml.so");
            ggmlBase = new File(dirFolder, "libggml-base.so");
            ggmlCpu = new File(dirFolder, "libggml-cpu.so");
            whisper = new File(dirFolder, "libwhisper.so");
            parakeet = new File(dirFolder, "libparakeet.so");
            path = "/linux-x86-64";
            isLinux = true;
        }
        if (!nativeLibrariesInitialized) {
            copyFile(path, ggml);
            copyFile(path, ggmlBase);
            copyFile(path, ggmlCpu);
            copyFile(path, whisper);
            copyFile(path, parakeet);
            if (Objects.nonNull(blas)) {
                copyFile(path, blas);
            }
            if (Objects.nonNull(metal)) {
                copyFile(path, metal);
            }
            nativeLibrariesInitialized = true;
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
        if (init == 5) {
            return parakeet.getAbsolutePath();
        }
        return whisper.getAbsolutePath();
    }

    /**
     * 删除旧缓存并从打包资源重新释放指定原生库；失败向调用方传播，阻止加载残缺文件。
     *
     * @param path 平台资源目录
     * @param dest 原生库缓存文件
     * @throws UncheckedIOException  删除、复制或关闭资源失败时抛出
     * @throws IllegalStateException 打包资源不存在时抛出
     */
    static void copyFile(String path, File dest) {
        String resourcePath = path + "/" + dest.getName();
        try (InputStream inputStream = WhisperCppJnaLibrary.class.getResourceAsStream(resourcePath)) {
            if (Objects.isNull(inputStream)) {
                throw new IllegalStateException("Native library resource not found: " + resourcePath);
            }
            Files.deleteIfExists(dest.toPath());
            Files.copy(inputStream, dest.toPath(), StandardCopyOption.REPLACE_EXISTING);
            dest.deleteOnExit();
        } catch (IOException exception) {
            throw new UncheckedIOException("Failed to refresh native library " + resourcePath
                    + " at " + dest.getAbsolutePath(), exception);
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
