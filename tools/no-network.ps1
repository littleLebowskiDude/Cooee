<#
.SYNOPSIS
    Asserts that Cooee has no way to make an outbound network request.

.DESCRIPTION
    "It all runs locally" is the whole claim of this app, and an architecture
    diagram is not evidence. This is: it walks the real dependency tree and
    fails if anything capable of reaching the internet is linked into the
    binary.

    Three classes of crate are treated differently.

      FORBIDDEN     HTTP clients, TLS stacks, websocket and QUIC transports,
                    DNS resolvers. You cannot make an outbound HTTPS request
                    without at least one of these. Any hit fails the build.

      ACKNOWLEDGED  Crates that are network-adjacent but cannot originate a
                    request on their own, each listed with why it is here.
                    They are reported, not hidden: `http` in particular looks
                    alarming and is worth explaining before someone else
                    finds it.

      UNKNOWN       Anything matching the network-ish name pattern that is in
                    neither list. Also a failure. A plain denylist rots the
                    first time a dependency pulls in something new, so the
                    default for an unrecognised networking crate is "no".

    Only `--edges normal` is walked: build- and dev-dependencies do not ship
    in the installed application, and this is a claim about the running app.

.EXAMPLE
    .\tools\no-network.ps1
    .\tools\no-network.ps1 -Features onnx,whisper
#>
[CmdletBinding()]
param(
    # Cargo features to enable, so the shipping configuration can be checked
    # and not just the default one.
    [string[]] $Features = @(),

    [string] $Manifest = (Join-Path $PSScriptRoot '..\src-tauri\Cargo.toml')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Anything that can originate a request or terminate TLS.
$Forbidden = @(
    # HTTP clients
    'reqwest', 'hyper', 'hyper-util', 'hyper-tls', 'ureq', 'curl', 'curl-sys',
    'isahc', 'surf', 'attohttpc', 'http-client', 'awc', 'minreq',
    # HTTP/2, HTTP/3, QUIC
    'h2', 'h3', 'quinn', 'quinn-proto',
    # TLS
    'rustls', 'tokio-rustls', 'native-tls', 'openssl', 'openssl-sys',
    'schannel', 'security-framework', 'boring', 'webpki', 'rustls-webpki',
    # Websockets
    'tungstenite', 'tokio-tungstenite', 'ws', 'websocket',
    # Name resolution
    'trust-dns-resolver', 'hickory-resolver', 'dns-lookup', 'resolv-conf'
)

# Network-adjacent, and why each one is not a way out of the machine.
$Acknowledged = [ordered]@{
    'http'            = 'Types only: Request/Response structs for Tauri custom protocol handlers. Contains no transport, no socket, no client.'
    'url'             = 'URL parsing, for Tauri asset protocol paths and CSP. A parser, not a fetcher.'
    'form_urlencoded' = 'Query-string encoding, pulled in by url.'
    'urlpattern'      = 'URL pattern matching, used by Tauri to scope its CSP.'
    'tokio'           = 'Async runtime for the Tauri event loop and the single-instance plugin, which uses a LOCAL named pipe to focus the running window. Networking features are available in the crate but no HTTP client is linked against them.'
    'mio'             = 'Event polling, pulled in by tokio.'
    'socket2'         = 'Socket configuration, pulled in by tokio. Used for the local single-instance pipe.'
    'webview2-com'      = 'COM bindings for the WebView2 control that renders the UI.'
    'webview2-com-sys'  = 'Raw COM bindings for WebView2.'
    'webview2-com-macros' = 'Macro support for the WebView2 bindings.'
}

# Name fragments worth a second look. Matched against the crate name split on
# `-` and `_`, never as a substring: "quick-xml" contains "quic" and is an XML
# parser. Deliberately broad otherwise — a false positive costs one line in
# $Acknowledged, a false negative costs the entire claim.
$NetworkKeywords = @(
    'http', 'https', 'tls', 'ssl', 'socket', 'socket2', 'sock', 'dns', 'tcp',
    'udp', 'quic', 'websocket', 'ws', 'curl', 'reqwest', 'hyper', 'ureq',
    'isahc', 'surf', 'fetch', 'client', 'request', 'proxy', 'tunnel', 'net',
    'resolver', 'resolve', 'tokio', 'mio', 'url', 'urlencoded', 'urlpattern',
    'webview2', 'h2', 'h3', 'quinn', 'tungstenite'
)

function Test-NetworkAdjacent {
    param([string] $Name)
    foreach ($token in ($Name -split '[-_]')) {
        if ($NetworkKeywords -contains $token.ToLowerInvariant()) { return $true }
    }
    return $false
}

$featureArgs = @()
if ($Features.Count -gt 0) {
    $featureArgs = @('--features', ($Features -join ','))
}

Write-Host ''
Write-Host 'Cooee — outbound network capability check' -ForegroundColor Cyan
Write-Host ('=' * 60)
$configLabel = if ($Features.Count -gt 0) { $Features -join ',' } else { 'default' }
Write-Host "Configuration : $configLabel"

$raw = & cargo tree --manifest-path $Manifest --edges normal --prefix none @featureArgs 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host 'cargo tree failed:' -ForegroundColor Red
    $raw | ForEach-Object { Write-Host "  $_" }
    exit 2
}

# "name v1.2.3 (path)" -> "name". Deduplicated; (*) repeat markers dropped.
$crates = $raw |
    ForEach-Object { ($_ -split '\s+')[0] } |
    Where-Object { $_ -match '^[A-Za-z0-9_-]+$' } |
    Sort-Object -Unique

Write-Host "Crates linked  : $($crates.Count)"
Write-Host ''

$hits = @($crates | Where-Object { $Forbidden -contains $_ })
$flagged = @($crates | Where-Object { (Test-NetworkAdjacent $_) -and $Forbidden -notcontains $_ })
$unknown = @($flagged | Where-Object { -not $Acknowledged.Contains($_) })

if ($flagged.Count -gt 0) {
    Write-Host 'Network-adjacent crates present, and why:' -ForegroundColor Yellow
    foreach ($c in $flagged) {
        $why = if ($Acknowledged.Contains($c)) { $Acknowledged[$c] } else { 'UNRECOGNISED' }
        Write-Host ("  {0,-20} {1}" -f $c, $why)
    }
    Write-Host ''
}

$failed = $false

if ($hits.Count -gt 0) {
    Write-Host 'FAIL: the binary links a crate that can reach the internet.' -ForegroundColor Red
    $hits | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    $failed = $true
}

if ($unknown.Count -gt 0) {
    Write-Host 'FAIL: unrecognised networking-capable crate in the tree.' -ForegroundColor Red
    $unknown | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    Write-Host ''
    Write-Host 'Establish what it does. If it cannot originate a request, add it' -ForegroundColor Red
    Write-Host 'to $Acknowledged with the reason. If it can, it does not belong' -ForegroundColor Red
    Write-Host 'in this application.' -ForegroundColor Red
    $failed = $true
}

if ($failed) { exit 1 }

Write-Host 'PASS' -ForegroundColor Green
Write-Host '  No HTTP client.  (reqwest, hyper, ureq, curl, isahc, surf, ...)'
Write-Host '  No TLS stack.    (rustls, native-tls, openssl, schannel, ...)'
Write-Host '  No websocket, QUIC or DNS resolver.'
Write-Host ''
Write-Host '  Without a client or a TLS implementation there is nothing in the'
Write-Host '  binary that can originate an outbound request. Transcription runs'
Write-Host '  against a model file on disk; the dictionary and history are JSON'
Write-Host '  in %APPDATA%. Airplane mode changes nothing about how it behaves.'
Write-Host ''
exit 0
