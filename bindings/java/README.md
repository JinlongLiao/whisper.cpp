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


## 原生参数生命周期

`WhisperCpp.getContextDefaultParams()` 返回 JNA 管理的独立参数副本，原始 C++ 参数在复制后立即释放。返回值可在实例关闭后继续读写，调用方不要对其底层指针执行原生释放。

`getFullDefaultParams()` 的参数仍由实例缓存，在 `close()` 时通过 `whisper_free_params()` 释放；关闭后不得继续使用旧的参数视图。直接使用低层 `*_default_params_by_ref()` 接口时，调用方仍须通过对应的 `whisper_free_context_params()` 或 `whisper_free_params()` 配对释放。

## License

The license for the Java bindings is the same as the license for the rest of the whisper.cpp project, which is the MIT License. See the `LICENSE` file for more details.

