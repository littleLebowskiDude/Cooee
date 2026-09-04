using Microsoft.UI.Xaml;

namespace PhiWinUI;

/// <summary>
/// Headless WinUI 3 host for the Phi Silica benchmark.
///
/// WinUI rather than a console app deliberately: a console
/// <c>Windows.FullTrustApplication</c> activated inside an AppX container died
/// before reaching <c>Main</c>. Every Microsoft Phi Silica sample is WinUI 3,
/// so this follows the documented, actually-exercised path.
/// </summary>
public partial class App : Application
{
    public App() => InitializeComponent();

    protected override async void OnLaunched(LaunchActivatedEventArgs args)
    {
        try
        {
            await Bench.RunAsync();
        }
        catch (Exception ex)
        {
            Bench.Log($"FATAL: {ex.GetType().Name}: {ex.Message}");
            Bench.Log(ex.StackTrace ?? "");
        }
        finally
        {
            Exit();
        }
    }
}
