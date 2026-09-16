<#
.SYNOPSIS
    Downloads a whisper ONNX model for Cooee and puts it in the layout the
    engine expects.

.DESCRIPTION
    Cooee ships without a model. That is not an oversight and not laziness
    about installer size: the app has no HTTP client and no TLS stack compiled
    into it, which is the whole basis of the local-only claim and is checked by
    tools/no-network.ps1. Giving it the ability to fetch its own model would
    mean linking one in, and the claim would be gone.

    So the download lives out here instead, in a script you can read before you
    run it. It fetches from Hugging Face over HTTPS, writes into
    models/whisper-<size>.en-onnx, and the app never opens a socket.

    Two files do the work:

      onnx/encoder_model.onnx           runs on the Hexagon NPU
      onnx/decoder_model_merged.onnx    runs on the CPU

    plus five small JSON/text files for the tokenizer and generation config.

    The decoder can also run on the NPU, which is worth about 0.4 s an
    utterance, but that needs static-shape graphs built from these weights.
    They are generated locally, not downloaded - see -WithNpuDecoder.

.PARAMETER Size
    base (about 280 MB) or small (about 925 MB).

    small is the one to use on a Snapdragon machine: on the NPU it is faster
    than base is on the CPU, and it is the model that hears "Cooee" as one
    word. base is the smaller download if you just want to try it.

.PARAMETER WithNpuDecoder
    After downloading, build the static-shape decoder graphs so the decoder
    runs on the NPU as well. Needs Python with onnx and numpy, and adds a few
    minutes and roughly 1.4 GB for small.

.EXAMPLE
    .\tools\get-model.ps1
    .\tools\get-model.ps1 -Size base
    .\tools\get-model.ps1 -Size small -WithNpuDecoder
#>
[CmdletBinding()]
param(
    [ValidateSet("base", "small")]
    [string] $Size = "small",

    [switch] $WithNpuDecoder,

    [string] $Destination = (Join-Path $PSScriptRoot "..\models")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repo = "onnx-community/whisper-$Size.en"
$base = "https://huggingface.co/$repo/resolve/main"

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
# Resolve so the path printed at the end is one you can paste into the picker,
# rather than one with a \..\ in the middle of it.
$Destination = (Resolve-Path $Destination).Path
$dir = Join-Path $Destination "whisper-$Size.en-onnx"

# Relative path -> approximate megabytes, for the progress report only.
$files = [ordered]@{
    "config.json"                      = 1
    "generation_config.json"           = 1
    "vocab.json"                       = 1
    "merges.txt"                       = 1
    "added_tokens.json"                = 1
    "onnx/encoder_model.onnx"          = if ($Size -eq "small") { 336 } else { 79 }
    "onnx/decoder_model_merged.onnx"   = if ($Size -eq "small") { 587 } else { 199 }
}
$totalMb = ($files.Values | Measure-Object -Sum).Sum

Write-Host ""
Write-Host "Cooee - whisper $Size.en (ONNX)" -ForegroundColor Cyan
Write-Host ("=" * 58)
Write-Host "  from    huggingface.co/$repo"
Write-Host "  into    $dir"
Write-Host ("  size    about {0} MB" -f $totalMb)
Write-Host ""

New-Item -ItemType Directory -Force -Path (Join-Path $dir "onnx") | Out-Null

$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
$done = 0
# Bytes actually pulled over the wire this run, so the summary can tell
# "downloaded it" apart from "it was already here".
$fetched = 0

foreach ($rel in $files.Keys) {
    $target = Join-Path $dir ($rel -replace '/', '\')
    $mb = $files[$rel]

    if ((Test-Path $target) -and (Get-Item $target).Length -gt 0) {
        Write-Host ("  [skip] {0,-32} already present" -f $rel) -ForegroundColor DarkGray
        $done += $mb
        continue
    }

    Write-Host ("  [get ] {0,-32} ~{1} MB" -f $rel, $mb)
    $url = "$base/$rel"

    try {
        if ($curl) {
            # curl.exe follows the CDN redirect and shows its own progress,
            # which matters when a single file is half a gigabyte.
            & curl.exe -L --fail --progress-bar -o $target $url
            if ($LASTEXITCODE -ne 0) { throw "curl exited $LASTEXITCODE" }
        }
        else {
            Invoke-WebRequest -Uri $url -OutFile $target -MaximumRedirection 5
        }
    }
    catch {
        # A partial file would look present to the skip check above and then
        # fail to load with something far less obvious.
        if (Test-Path $target) { Remove-Item $target -Force }
        Write-Host ""
        Write-Host "  Failed on $rel : $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "  Re-run to resume; finished files are skipped." -ForegroundColor Red
        exit 1
    }
    $fetched += (Get-Item $target).Length
    $done += $mb
}

Write-Host ""
# Only the files this script is responsible for. The folder may hold more:
# the static decoder graphs, or quantised variants pulled down by hand.
$needed = 0
foreach ($rel in $files.Keys) {
    $p = Join-Path $dir ($rel -replace '/', '\')
    if (Test-Path $p) { $needed += (Get-Item $p).Length }
}
if ($fetched -eq 0) {
    Write-Host ("Nothing to do. All {0:N0} MB already present." -f ($needed / 1MB)) -ForegroundColor Green
} else {
    Write-Host ("Done. {0:N0} MB fetched, {1:N0} MB in place." -f ($fetched / 1MB), ($needed / 1MB)) -ForegroundColor Green
}

if ($WithNpuDecoder) {
    Write-Host ""
    Write-Host "Building static-shape decoder graphs for the NPU" -ForegroundColor Cyan
    $script = Join-Path $PSScriptRoot "..\bench\static_decoder.py"
    if (-not (Test-Path $script)) {
        Write-Host "  bench/static_decoder.py not found; skipping." -ForegroundColor Yellow
    }
    elseif (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Host "  python not on PATH; skipping." -ForegroundColor Yellow
        Write-Host "  Install it, then: python -m pip install onnx numpy" -ForegroundColor Yellow
        Write-Host "  and: python bench\static_decoder.py `"$dir`"" -ForegroundColor Yellow
    }
    else {
        python -m pip install --quiet onnx numpy
        python $script $dir
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  Graph build failed. The decoder will run on the CPU," -ForegroundColor Yellow
            Write-Host "  which costs about 0.4 s an utterance and nothing else." -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "Next:" -ForegroundColor Cyan
Write-Host "  1. Open Cooee from the tray icon, then the cog."
Write-Host "  2. Model, then Folder..., and choose:"
Write-Host "       $dir"
Write-Host "  3. Save. The first load compiles the model for the NPU and takes"
Write-Host "     about 20 seconds; it is cached, so later launches are quick."
Write-Host ""
Write-Host "  Then hold Ctrl+Win and speak." -ForegroundColor Green
Write-Host ""
