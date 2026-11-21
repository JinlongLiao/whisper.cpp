# 在 Windows x64 或 ARM64 的 Developer PowerShell for VS 中构建 Java 原生资源。
# 需要与架构匹配的 MSVC、Windows SDK 和 CMake；脚本负责构建和加载检查。
param(
    [string]$BuildDirectory = "",
    [ValidateSet("x64", "ARM64")][string]$Architecture = "x64",
    [string]$Generator = "Visual Studio 17 2022"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if ($env:OS -ne "Windows_NT" -or [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString() -ne $Architecture) {
    throw "请在与目标架构 $Architecture 一致的 Windows Developer PowerShell for VS 中运行。"
}
Get-Command cmake, dumpbin -ErrorAction Stop | Out-Null
$RepositoryDirectory = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
if ([string]::IsNullOrEmpty($BuildDirectory)) {
    $BuildDirectory = Join-Path $RepositoryDirectory "build-java-windows"
}
$Libraries = @("ggml-base", "ggml-cpu", "ggml", "parakeet", "whisper")

$Resource = "win32-x86-64"
$Machine = "8664"
$OpenMP = "ON"
if ($Architecture -eq "ARM64") {
    $Resource = "win32-aarch64"
    $Machine = "AA64"
    $OpenMP = "OFF"
}
# 保留 x64 的 OpenMP 配置，ARM64 使用 MSVC 基础运行库。
& cmake -S $RepositoryDirectory -B $BuildDirectory -G $Generator -A $Architecture `
    -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF `
    -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF "-DGGML_OPENMP=$OpenMP" `
    -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=OFF
if ($LASTEXITCODE -ne 0) { throw "CMake 配置失败。" }
& cmake --build $BuildDirectory --config Release --parallel --target whisper parakeet
if ($LASTEXITCODE -ne 0) { throw "原生库编译失败。" }

$StageDirectory = Join-Path ([IO.Path]::GetTempPath()) ("whisper-java-windows-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $StageDirectory | Out-Null
try {
    foreach ($Library in $Libraries) {
        $Source = Join-Path $BuildDirectory ("bin/Release/" + $Library + ".dll")
        $Headers = & dumpbin /headers $Source 2>&1
        if ($LASTEXITCODE -ne 0 -or ($Headers -join "`n") -notmatch "\b$Machine machine") {
            throw "DLL 架构与 $Architecture 不符或无法读取：$Source"
        }
        Copy-Item -LiteralPath $Source -Destination $StageDirectory
    }

    # 限定依赖查找为 DLL 所在目录与 System32，不使用开发机 PATH 掩盖缺失依赖。
    # 验证后释放所有句柄，再复制文件；该检查不代表 Java 推理或旧版 Windows 验收。
    if (-not ("WhisperNativeLoadCheck" -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class WhisperNativeLoadCheck {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern IntPtr LoadLibraryExW(string fileName, IntPtr file, uint flags);
    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool FreeLibrary(IntPtr module);
}
'@
    }
    $Handles = [Collections.Generic.List[IntPtr]]::new()
    try {
        foreach ($Library in $Libraries) {
            $Dll = Join-Path $StageDirectory ($Library + ".dll")
            $Module = [WhisperNativeLoadCheck]::LoadLibraryExW($Dll, [IntPtr]::Zero, 0x00000900)
            if ($Module -eq [IntPtr]::Zero) {
                $NativeError = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
                throw "加载 $Dll 失败，Win32 错误码：$NativeError；请检查 MSVC/OpenMP 运行库。"
            }
            $Handles.Add($Module)
            Write-Host "加载成功：$Library"
        }
    } finally {
        for ($Index = $Handles.Count - 1; $Index -ge 0; $Index--) {
            [WhisperNativeLoadCheck]::FreeLibrary($Handles[$Index]) | Out-Null
        }
    }
    $ResourceDirectory = Join-Path $PSScriptRoot "src/main/resources/$Resource"
    New-Item -ItemType Directory -Path $ResourceDirectory -Force | Out-Null
    foreach ($Library in $Libraries) {
        Copy-Item -LiteralPath (Join-Path $StageDirectory ($Library + ".dll")) `
            -Destination $ResourceDirectory -Force
    }
    Write-Host "已更新 5 个 Windows $Architecture 原生库；可继续构建 Java JAR。"
} finally {
    Remove-Item -LiteralPath $StageDirectory -Recurse -Force
}
