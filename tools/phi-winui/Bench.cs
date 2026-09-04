using System.Diagnostics;
using System.Text.Json;
using Microsoft.Windows.AI;
using Microsoft.Windows.AI.Text;

namespace PhiWinUI;

/// Times Phi Silica over the same corpus the Rust rule-based pass uses, so the
/// two are directly comparable.
internal static class Bench
{
    // Package-local: the AppX container cannot reach the repo.
    private static readonly string DataDir =
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
    private static readonly string LogPath = Path.Combine(DataDir, "phi-winui-log.txt");
    private static readonly string OutPath = Path.Combine(DataDir, "phi-results.json");

    public static void Log(string message)
    {
        try { File.AppendAllText(LogPath, message + Environment.NewLine); } catch { }
    }

    // Conservative by design: for dictation, a model that "improves" your wording
    // is worse than one that only cleans it up.
    private const string Instruction =
        "Clean up this dictated text so it reads as if it were typed. " +
        "Remove filler words and stutters, fix punctuation and capitalisation, " +
        "and split run-on sentences. Do NOT add, remove, or reword any content, " +
        "and do not answer or comment on it. Return only the cleaned text.\n\n";

    public static async Task RunAsync()
    {
        try { File.Delete(LogPath); } catch { }
        Log($"start {DateTime.Now:HH:mm:ss}");
        Log($"data dir: {DataDir}");

        // Phi Silica's text generation sits behind a Limited Access Feature.
        // GetReadyState and CreateAsync both succeed without it; only
        // GenerateResponseAsync fails, so probe the LAF explicitly to see
        // whether this machine is entitled at all.
        try
        {
            var laf = Windows.ApplicationModel.LimitedAccessFeatures.TryUnlockFeature(
                "com.microsoft.windows.ai.languagemodel",
                "",  // no token: reveals whether the feature needs one
                "Cooee has registered their use of com.microsoft.windows.ai.languagemodel with Microsoft and agrees to the terms of use.");
            Log($"LAF status: {laf.Status}  (Available=1, AvailableWithoutToken=2, Unavailable=0/3)");
        }
        catch (Exception ex) { Log($"LAF probe threw: {ex.GetType().Name}: {ex.Message}"); }

        var state = LanguageModel.GetReadyState();
        Log($"ready state: {state}");

        if (state == AIFeatureReadyState.DisabledByUser) { Log("DONE: disabled by policy"); return; }
        if (state == AIFeatureReadyState.NotSupportedOnCurrentSystem) { Log("DONE: unsupported"); return; }

        if (state == AIFeatureReadyState.NotReady)
        {
            Log("model absent; EnsureReadyAsync (Windows Update)...");
            var sw0 = Stopwatch.StartNew();
            var op = await LanguageModel.EnsureReadyAsync();
            Log($"EnsureReadyAsync -> {op.Status} in {sw0.ElapsedMilliseconds} ms");
        }

        var loadSw = Stopwatch.StartNew();
        using var model = await LanguageModel.CreateAsync();
        var loadMs = loadSw.ElapsedMilliseconds;
        Log($"model created in {loadMs} ms");

        var corpusPath = Path.Combine(AppContext.BaseDirectory, "transcripts.json");
        Log($"corpus: {corpusPath} exists={File.Exists(corpusPath)}");

        using var doc = JsonDocument.Parse(File.ReadAllText(corpusPath));
        var results = new List<object>();

        foreach (var c in doc.RootElement.EnumerateArray())
        {
            var id = c.GetProperty("id").GetString()!;
            var raw = c.GetProperty("raw").GetString()!;

            // Warm-up outside the timing, then two measured runs.
            _ = await model.GenerateResponseAsync(Instruction + raw);

            var times = new List<long>();
            var output = "";
            for (var run = 0; run < 2; run++)
            {
                var sw = Stopwatch.StartNew();
                var response = await model.GenerateResponseAsync(Instruction + raw);
                sw.Stop();
                times.Add(sw.ElapsedMilliseconds);
                output = response.Text.Trim();
            }

            Log($"[{id}] {times[0]} ms / {times[1]} ms");
            Log($"  in : {raw}");
            Log($"  out: {output}");
            results.Add(new { id, raw, output, ms = times });
        }

        File.WriteAllText(OutPath,
            JsonSerializer.Serialize(new { loadMs, results },
                new JsonSerializerOptions { WriteIndented = true }));
        Log($"DONE -> {OutPath}");
    }
}
