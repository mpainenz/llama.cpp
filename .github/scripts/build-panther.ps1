$ErrorActionPreference = 'Stop'
if ($env:GITHUB_REPOSITORY -ne 'mpainenz/llama.cpp' -or $env:GITHUB_ACTOR -ne 'mpainenz' -or
    $env:GITHUB_EVENT_NAME -notin @('push', 'workflow_dispatch') -or !$env:GITHUB_REF.StartsWith('refs/heads/')) {
    throw 'Panther only accepts owner-started branch builds in this fork'
}
$mutex = New-Object Threading.Mutex($false, 'Global\TensorRelayNativeBuild')
$locked = $false
try {
    try { $locked = $mutex.WaitOne([TimeSpan]::FromMinutes(90)) }
    catch [Threading.AbandonedMutexException] { $locked = $true }
    if (!$locked) { throw 'Another native build is still using Panther' }
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (!$vs) { throw 'Visual C++ build tools are missing' }
    & cmd.exe /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
    }
    if ($LASTEXITCODE -ne 0) { throw 'Visual C++ environment failed' }
    & nvcc --version
    if ($LASTEXITCODE -ne 0) { throw 'CUDA compiler is unavailable' }
    & glslc --version
    if ($LASTEXITCODE -ne 0) { throw 'Vulkan compiler is unavailable' }
    $revision = (& git rev-parse HEAD).Trim()
    if ($revision -notmatch '^[0-9a-f]{40}$') { throw 'Invalid source revision' }
    $build = (& python .github/scripts/select-panther-cache.py (Get-Location).Path 'E:\TensorRelay\BuildCache').Trim()
    if ($LASTEXITCODE -ne 0 -or !$build) { throw 'Build cache selection failed' }
    Write-Host "Executor build directory: $build"
    New-Item -ItemType Directory -Force $build | Out-Null
    $common = $env:CMAKE_COMMON.Trim() -split '\s+'
    & cmake -S . -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release `
        -DGGML_CUDA=ON -DGGML_VULKAN=ON -DGGML_CUDA_CUB_3DOT2=ON `
        '-DCMAKE_CUDA_ARCHITECTURES=75;80;86;89;100;120' @common
    if ($LASTEXITCODE -ne 0) { throw 'Executor configuration failed' }
    & cmake --build $build --target tensorrelay_stage_executor --parallel 4
    if ($LASTEXITCODE -ne 0) { throw 'Executor compilation failed' }
    Add-Content $env:GITHUB_ENV "TENSORRELAY_BUILD_BIN=$build\bin"
} finally {
    if ($locked) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
