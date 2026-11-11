# Java JNI bindings for Whisper

This package provides Java JNI bindings for whisper.cpp. They have been tested on:

  * <strike>Darwin (OS X) 12.6 on x64_64</strike>
  * Ubuntu on x86_64
  * Windows on x86_64

The "low level" bindings are in `WhisperCppJnaLibrary`. The most simple usage is as follows:

JNA will attempt to load the `whispercpp` shared library from:

- jna.library.path
- jna.platform.library
- ~/Library/Frameworks
- /Library/Frameworks
- /System/Library/Frameworks
- classpath

```java
import io.github.ggerganov.whispercpp.WhisperCpp;

public class Example {

    public static void main(String[] args) {
        
        WhisperCpp whisper = new WhisperCpp();
        try {
            // By default, models are loaded from ~/.cache/whisper/ and are usually named "ggml-${name}.bin"
            // or you can provide the absolute path to the model file.
            whisper.initContext("../ggml-base.en.bin"); 
            WhisperFullParams.ByValue whisperParams = whisper.getFullDefaultParams(WhisperSamplingStrategy.WHISPER_SAMPLING_BEAM_SEARCH); 
            
            // custom configuration if required      
            //whisperParams.n_threads = 8;
            whisperParams.temperature = 0.0f;
            whisperParams.temperature_inc = 0.2f;
            //whisperParams.language = "en";
                            
            float[] samples = readAudio(); // divide each value by 32767.0f
            List<WhisperSegment> whisperSegmentList = whisper.fullTranscribeWithTime(whisperParams, samples);
            
            for (WhisperSegment whisperSegment : whisperSegmentList) {

                long start = whisperSegment.getStart();
                long end = whisperSegment.getEnd();

                String text = whisperSegment.getSentence();
                    
                System.out.println("start: "+start);
                System.out.println("end: "+end);
                System.out.println("text: "+text);
                
            }
    
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            whisper.close();
        }
        
     }
}
```

## Building & Testing

In order to build, you need to have the JDK 8 or higher installed. Run the tests with:

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp/bindings/java

./gradlew build
```

You need to have the `whisper` library in your [JNA library path](https://java-native-access.github.io/jna/4.2.1/com/sun/jna/NativeLibrary.html). On Windows the dll is included in the jar and you can update it:

```bash
copy /y ..\..\build\bin\Release\whisper.dll build\generated\resources\main\win32-x86-64\whisper.dll
```


## macOS 原生库构建与打包

当前 `darwin/` 资源仅提供 Apple Silicon（arm64）库，最低系统目标为 macOS 15.0。
需要安装 CMake、Xcode 命令行工具及 Metal 编译工具。从仓库根目录运行：

```bash
bash bindings/java/build-macos.sh
mvn -f bindings/java/pom.xml clean package -DskipTests -Dgpg.skip=true
```

第一条命令在独立的 `build-java-macos/` 目录编译，不复用原 `build/` 的系统版本配置；
也可通过脚本的第一个参数指定其他构建目录。脚本不会发布 Maven 制品。

脚本将 7 个动态库复制到临时目录，把内部版本化依赖统一改为
`@loader_path/lib名称.dylib`，删除构建目录 RPATH，并重新进行 ad-hoc 签名。
检查 arm64 架构、最低系统版本、依赖范围、签名及本机加载成功后，才更新
`src/main/resources/darwin/`。Java 会把这些库释放到同一目录，无需修改 `LocalLib`。
不要直接从 CMake 构建目录复制 `.dylib` 到资源目录，否则会再次引入版本化依赖与本机路径。

这里的 ad-hoc 签名不等同于 Developer ID 签名或公证。构建机加载成功也不能代替
macOS 15.x、26.x、27.x 上的真实 Java 加载及 CPU/Metal 推理验证；发布前须分别验收。
Linux、Windows 资源不由此脚本处理。

## Linux 与 Windows 原生库构建与打包

这两个脚本分别在对应系统上运行，编译并校验 `ggml-base`、`ggml-cpu`、`ggml`、
`parakeet`、`whisper` 五个 x86-64 CPU 库，全部加载成功后替换对应资源目录。
不修改其他平台的资源，不自动发布 JAR。各平台资源更新完成后再统一打包。

Linux 需要 CMake、C/C++ 编译器、`patchelf`、`readelf` 和 Python 3：

```bash
bash bindings/java/build-linux.sh
```

默认构建目录是 `build-java-linux/`，可通过第一个参数指定其他目录。
脚本将 SONAME、内部依赖统一为实际打包的无版本 `.so` 文件名，将 RUNPATH
改为 `$ORIGIN`，并检查五库引用的 GLIBC 符号版本不高于 2.32。
必须在 GLIBC 不高于 2.32 的构建环境中编译，不能在高版本环境中靠修改标签降低要求。
这个检查不涵盖目标服务器的全部传递依赖；C++ 标准库的 `GLIBCXX`/`CXXABI`、
OpenMP 运行库、CPU 指令集及实际推理仍需在目标 Linux 验证。

Windows 在 64 位 **Developer PowerShell for VS 2022** 中运行，需要安装 C++ 工具链、
Windows SDK 和 CMake：

```powershell
& .\bindings\java\build-windows.ps1
```

默认构建目录是 `build-java-windows/`，可通过 `-BuildDirectory` 指定其他目录。
脚本使用 MSVC x64 Release 构建，检查 DLL 架构，并限定依赖从库所在目录与
System32 加载，避免开发机 PATH 掩盖缺失依赖。保留默认 OpenMP，运行机需要
对应的 Microsoft Visual C++ x64 运行库；不会把开发机运行库自动复制进 JAR。
Windows 版本兼容性、Java 加载及实际推理需在目标机器验证。

完成所需平台的资源更新后，在安装了 Maven 和 JDK 的环境执行：

```bash
mvn -f bindings/java/pom.xml clean package -DskipTests -Dgpg.skip=true
```

以上脚本均不启用宿主机专用 CPU 优化，也不提供跨系统编译；不要将同一个 CMake
构建目录用于不同平台或不同工具链。

## 原生参数生命周期

`WhisperCpp.getContextDefaultParams()` 返回 JNA 管理的独立参数副本，原始 C++ 参数在复制后立即释放。返回值可在实例关闭后继续读写，调用方不要对其底层指针执行原生释放。

`getFullDefaultParams()` 的参数仍由实例缓存，在 `close()` 时通过 `whisper_free_params()` 释放；关闭后不得继续使用旧的参数视图。直接使用低层 `*_default_params_by_ref()` 接口时，调用方仍须通过对应的 `whisper_free_context_params()` 或 `whisper_free_params()` 配对释放。

## License

The license for the Java bindings is the same as the license for the rest of the whisper.cpp project, which is the MIT License. See the `LICENSE` file for more details.

