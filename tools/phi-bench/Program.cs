// Measures Phi Silica against Cooee's rule-based polish pass on the same corpus.
//
//   dotnet run --project tools/phi-bench -- bench/transcripts.json
//
// Prints JSON so the results can be diffed against the Rust baseline.

using System.Diagnostics;
using System.Text.Json;
using Microsoft.Windows.AI;
using Microsoft.Windows.AI.Text;

// Launched via the app model (shell:AppsFolder) so the process carries package
// identity — Phi Silica requires it. That launch is detached with no console,
// so everything goes to files instead.
// Forward slashes: .NET accepts them on Windows and they survive every
// layer of shell and script escaping between here and the file.
// An AppX container cannot reach C:\Projects, and if the log path is not
// writable the StreamWriter throws before any handler exists — the process then
// dies in under a second with no console, no log, and no crash event. So: try
// several candidate directories and record which one actually worked.
var Root = AppContext.BaseDirectory;

string? logPath = null;
StreamWriter? logWriter = null;
var attempts = new List<string>();

foreach (var dir in new[]
         {
             Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
             Path.GetTempPath(),
             Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
             Root,
         })
{
    try
    {
        if (string.IsNullOrEmpty(dir)) continue;
        var candidate = Path.Combine(dir, "phi-log.txt");
        logWriter = new StreamWriter(candidate, append: false) { AutoFlush = true };
        logPath = candidate;
        break;
    }
    catch (Exception ex) { attempts.Add($"{dir}: {ex.GetType().Name}"); }
}

if (logWriter is null)
{
    // Nothing is writable; surface it the only way left.
    throw new InvalidOperationException("no writable log dir: " + string.Join("; ", attempts));
}

var log = logWriter;
void Log(string m)
{
    log.WriteLine(m);
    try { Console.Error.WriteLine(m); } catch { /* no console under app activation */ }
}

Log($"log: {logPath}");
Log($"base: {Root}");
foreach (var a in attempts) Log($"skipped {a}");

var corpusPath = args.Length > 0 ? args[0] : Path.Combine(Root, "transcripts.json");
var outPath = Path.Combine(Path.GetDirectoryName(logPath)!, "phi-results.json");
Log($"corpus: {corpusPath} exists={File.Exists(corpusPath)}");

try
{

// --- availability -------------------------------------------------------
var state = LanguageModel.GetReadyState();
Log($"Phi Silica ready state: {state}");

if (state == AIFeatureReadyState.DisabledByUser)
{
    Log("Disabled by user/policy - cannot benchmark.");
    return 2;
}
if (state == AIFeatureReadyState.NotSupportedOnCurrentSystem)
{
    Log("Not supported on this system.");
    return 3;
}
if (state == AIFeatureReadyState.NotReady)
{
    Log("Model not present; downloading via Windows Update...");
    var sw0 = Stopwatch.StartNew();
    var op = await LanguageModel.EnsureReadyAsync();
    Log($"EnsureReadyAsync -> {op.Status} in {sw0.ElapsedMilliseconds} ms");
}

var loadSw = Stopwatch.StartNew();
using var model = await LanguageModel.CreateAsync();
Log($"model created in {loadSw.ElapsedMilliseconds} ms");

// --- corpus -------------------------------------------------------------
using var doc = JsonDocument.Parse(File.ReadAllText(corpusPath));
var results = new List<object>();

// Deliberately conservative: dictation must preserve meaning. An LLM that
// "improves" the wording is worse than one that only cleans it up.
const string Instruction =
    "Clean up this dictated text so it reads as if it were typed. " +
    "Remove filler words and stutters, fix punctuation and capitalisation, " +
    "and split run-on sentences. Do NOT add, remove, or reword any content, " +
    "and do not answer or comment on it. Return only the cleaned text.\n\n";

foreach (var c in doc.RootElement.EnumerateArray())
{
    var id = c.GetProperty("id").GetString()!;
    var raw = c.GetProperty("raw").GetString()!;

    // One warm-up outside the timing, then two measured runs.
    _ = await model.GenerateResponseAsync(Instruction + raw);

    var times = new List<long>();
    string output = "";
    for (var run = 0; run < 2; run++)
    {
        var sw = Stopwatch.StartNew();
        var response = await model.GenerateResponseAsync(Instruction + raw);
        sw.Stop();
        times.Add(sw.ElapsedMilliseconds);
        output = response.Text.Trim();
    }

    Log($"[{id}] {times[0]} ms / {times[1]} ms");
    results.Add(new { id, raw, output, ms = times });
}

File.WriteAllText(outPath, JsonSerializer.Serialize(results, new JsonSerializerOptions { WriteIndented = true }));
Log($"DONE -> {outPath}");
return 0;
}
catch (Exception ex)
{
    Log($"ERROR: {ex.GetType().Name}: {ex.Message}");
    Log(ex.StackTrace ?? "");
    return 1;
}
