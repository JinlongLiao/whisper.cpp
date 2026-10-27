# Linux 原生库兼容与打包说明

本文记录 2026-10-27 排查的 Whisper Java 绑定原生库加载问题，以及后续固定构建环境的建议。适用目录为本仓库的 `bindings/java`。

## 问题与原因

目标服务器环境为：

- Alibaba Cloud Linux 3.2104 U12.3（OpenAnolis Edition）。
- CPU 架构：`x86_64`。
- GLIBC：`2.32`。

启动时出现：

```text
java.lang.UnsatisfiedLinkError:
/lib64/libm.so.6: version `GLIBC_2.43' not found
(required by /data/java_log/tws_websocket/whisper/libggml-base.so)
```

原生库要求的符号版本高于服务器提供的版本，导致 JNA 加载失败，继而影响 `WhisperJnaPool` 和 Spring Bean 初始化。JNA 随后的 `not found in resource path` 属于回退加载错误，不能单独据此判断最初的磁盘文件不存在。

此次另一个构建候选环境是 Ubuntu 26.04.1 WSL、`x86_64`、GLIBC `2.43`。直接链接该环境的系统库，可能使产物依赖高版本 GLIBC。仅降低 GCC 版本或修改 `.so` 内的版本字符串，不能可靠实现兼容。

`LocalLib` 首次初始化会从 JAR 的平台资源中释放原生库，并对文件调用 `deleteOnExit()`；JVM 正常退出时文件可能被删除，下次启动也会重新覆盖。因此应更新 JAR 内的资源，而不是仅手工替换日志目录中的库。

## 推荐固定的编译环境

| 环境 | GLIBC 基线 | 使用建议 |
| --- | --- | --- |
| Alibaba Cloud Linux 3 x86_64 | 2.32 | 首选。与本次目标服务器一致，已用于此次构建及用户反馈的运行测试。 |
| Ubuntu 20.04 LTS x86_64 | 2.31 | 备选构建基线；需重新验证 GLIBCXX、CXXABI、系统依赖和模型识别。 |
| Ubuntu 22.04 LTS x86_64 | 2.35 | 不宜直接使用其系统库构建面向 GLIBC 2.32 的产物。 |
| 高版本 WSL / Linux | 取决于系统 | 可作为宿主，但应在固定低版本容器中构建，或使用配套低版本 sysroot 的交叉工具链。 |

本次成功使用的工具版本为：

```text
GLIBC 2.32
GCC / G++ 10.2.1 20200825 (Alibaba 10.2.1-3.9 2.32)
CMake 3.26.5
```

固定环境时应记录完整镜像版本或 digest、GCC/G++、链接器、CMake 版本和构建参数，不能只固定发行版名称。Ubuntu 20.04 已于 2025 年 5 月结束标准支持；长期使用应考虑 Ubuntu Pro 扩展维护。不能为了兼容而直接覆盖运行服务器的系统 GLIBC。

参考来源：

- [Ubuntu 20.04 GLIBC 2.31 官方包记录](https://code.launchpad.net/ubuntu/focal/+package/libc6-dev-x32)
- [Ubuntu 22.04 libc6 版本](https://packages.ubuntu.com/jammy/libc6)
- [Ubuntu 20.04 维护状态](https://ubuntu.com/20-04)

## 原生库编译与检查

本次源码包为提交 `573df6fa817c2834757cff1f7814405eaf4e8245` 的原样导出，没有为降低 GLIBC 修改 C/C++ 源码。后续可以使用仓库源码，在上述兼容环境中重新编译；源码更新后仍需重新验证。

在 Linux x86_64 构建机的仓库根目录执行：

```bash
cmake -S . -B build-linux-glibc232 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=OFF \
  -DGGML_NATIVE=OFF -DGGML_SSE42=OFF \
  -DGGML_AVX=OFF -DGGML_AVX2=OFF \
  -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF \
  -DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
  -DWHISPER_BUILD_TESTS=OFF \
  -DWHISPER_BUILD_EXAMPLES=OFF -DWHISPER_BUILD_SERVER=OFF

cmake --build build-linux-glibc232 \
  --target whisper parakeet --parallel 2
```

以上是本次采用的保守 CPU 指令集配置，性能需按实际音频与并发场景测试；不能仅为提速开启构建主机专用指令后直接分发。

需要配套更新的五个资源为：

```text
bindings/java/src/main/resources/linux-x86-64/libggml-base.so
bindings/java/src/main/resources/linux-x86-64/libggml-cpu.so
bindings/java/src/main/resources/linux-x86-64/libggml.so
bindings/java/src/main/resources/linux-x86-64/libwhisper.so
bindings/java/src/main/resources/linux-x86-64/libparakeet.so
```

CMake 产物可能包含 `.so` 符号链接和带版本的真实文件。收集时使用 `cp -L` 读取真实内容；使用 `find` 时需要考虑 `find -L`，否则 `-type f` 可能漏掉 `.so` 链接。不要混用不同源码版本生成的库。

逐库检查 ELF 架构、版本要求和动态依赖：

```bash
file build-linux-glibc232/bin/lib*.so
readelf --version-info --wide build-linux-glibc232/bin/libggml-cpu.so
readelf -d build-linux-glibc232/bin/libwhisper.so
ldd build-linux-glibc232/bin/libwhisper.so
```

所有五个库均需检查，GLIBC 要求不能高于 `2.32`；还要核对 `GLIBCXX`、`CXXABI`、`libgomp` 等依赖。`ldd` 若解析到构建目录，只能证明构建机当前可解析依赖，不能证明可独立分发。

本次产物仍带有构建目录绝对 RPATH，且依赖名含 `libggml-base.so.0`、`libggml-cpu.so.0`、`libggml.so.0`。当前 Java 绑定按 base → CPU → ggml → Whisper → Parakeet 顺序预加载。用户反馈本次测试正常，但没有单独验证移除构建目录后的加载；后续应处理绝对 RPATH，并在隔离目录或独立环境中验证依赖解析，避免依赖构建机残留文件。

校验脚本如需支持 Alibaba Cloud Linux 3 自带的 Python 3.6，应使用 `subprocess.check_output(..., universal_newlines=True)`，不能使用 Python 3.7 才支持的 `text=True`。

## Java JAR 打包

Maven 打包不会重新编译上述 C/C++ 原生库。更新资源后，可以在 Mac 上打包同时携带各平台资源的 Whisper JAR：

```bash
cd bindings/java
mvn clean install -DskipTests -Dgpg.skip=true
```

Maven 本地仓库使用 `~/.m2/settings.xml` 的 `localRepository` 配置；没有配置时使用默认仓库，不另行覆盖路径。

本地 `install` 仅更新本机 Maven 仓库。其他开发机或 CI 使用更新后的依赖，需要按项目既有发布流程将 `whispercpp` 发布至 Maven 私服；Git 推送不等于 Maven 制品发布。

最终服务 `tws_websocket` 的平台 profile 属于业务项目，不属于本仓库：

```bash
mvn -U clean package -Plinux  # Linux x64，默认目标
mvn -U clean package -Pmac    # macOS Apple Silicon
mvn -U clean package -Pwin    # Windows x64
```

每次只选择一个目标平台。Sherpa 和 JAVE 应一起切换，不能根据打包主机系统混入其他平台的原生依赖。Sherpa 所用的 JitPack 依赖需在最终项目配置 `https://jitpack.io` 仓库。

Linux 服务包应包含：

```text
jave-nativebin-linux64-3.5.1-wlzn-SNAPSHOT.jar
sherpa-onnx-native-lib-linux-x64-v1.13.8.jar
whispercpp-1.8.2-SNAPSHOT.jar
```

本次检查的业务服务字节码要求 Java 21 或更高；该要求来自业务服务，不代表 Whisper Java 绑定本身必须使用 Java 21。

## 验证边界

本次已确认：五个兼容库已进入最终业务 JAR，最高 GLIBC 要求为 `2.32`；Sherpa 与 JAVE 已切换为 Linux 包，用户反馈测试正常。

后续每次原生库更新仍应分别完成依赖检查、Java/JNA 加载、模型初始化、实际音频识别和必要的性能验证。Maven 构建成功、Git 推送成功、制品发布成功和业务运行通过是不同的验收阶段。
