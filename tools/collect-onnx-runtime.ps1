# Copies the ONNX Runtime and Qualcomm QNN libraries the `onnx` engine needs
# from the pip packages into src-tauri/runtime/. tauri.onnx.conf.json bundles
# that folder beside the executable; it is a separate config because a
# resource glob that matches nothing fails every build, and the default build
# has no DLLs.
#
#     python -m pip install onnxruntime-qnn
#     .\tools\collect-onnx-runtime.ps1
#     npm run tauri build -- --features onnx,whisper --config src-tauri/tauri.onnx.conf.json
#
# The QNN files are covered by Qualcomm's AI Stack licence (Qualcomm_LICENSE.pdf
# in the package): redistribution is allowed only in object form as part of an
# application, which is what this is. Its notices are copied alongside.
# HTP V73 is the Snapdragon X Elite / X Plus; V81 is the next generation.
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$dest = Join-Path $root "src-tauri\runtime"

$site = & python -c "import onnxruntime_qnn, os; print(os.path.dirname(os.path.dirname(onnxruntime_qnn.__file__)))"
if (-not $site) { throw "onnxruntime_qnn is not installed: python -m pip install onnxruntime-qnn" }

$files = @(
  "onnxruntime\capi\onnxruntime.dll",
  "onnxruntime\capi\onnxruntime_providers_shared.dll",
  "onnxruntime_qnn\onnxruntime_providers_qnn.dll",
  "onnxruntime_qnn\QnnSystem.dll",
  "onnxruntime_qnn\QnnHtp.dll",
  "onnxruntime_qnn\QnnHtpPrepare.dll",
  "onnxruntime_qnn\QnnHtpV73Stub.dll",
  "onnxruntime_qnn\libQnnHtpV73Skel.so",
  "onnxruntime_qnn\libqnnhtpv73.cat",
  "onnxruntime_qnn\QnnHtpV81Stub.dll",
  "onnxruntime_qnn\libQnnHtpV81Skel.so",
  "onnxruntime_qnn\libqnnhtpv81.cat",
  "onnxruntime_qnn\LICENSE",
  "onnxruntime_qnn\Qualcomm_LICENSE.pdf",
  "onnxruntime_qnn\ThirdPartyNotices.txt"
)

New-Item -ItemType Directory -Force $dest | Out-Null
$total = 0
foreach ($f in $files) {
  $src = Join-Path $site $f
  if (-not (Test-Path $src)) { throw "missing $src" }
  Copy-Item $src $dest -Force
  $total += (Get-Item $src).Length
}
"{0} files, {1:N0} MB -> {2}" -f $files.Count, ($total / 1MB), $dest
