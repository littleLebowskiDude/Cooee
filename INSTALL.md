# Installing Cooee

Cooee is push-to-talk dictation. You hold a key, say something, let go, and your
words appear in whatever app you were typing in. It runs entirely on your own
laptop. Nothing you say is sent anywhere.

This page assumes you are not a developer. There is no command line and nothing
to build.

---

## First: will it run on your laptop?

**Cooee only works on a Snapdragon laptop** (these are sometimes called
Copilot+ PCs, or Windows on ARM). It uses the NPU chip those machines have, and
that chip is the whole point of the app. On a normal Intel or AMD laptop it
will not run at all.

To check, press the Windows key and type **About your PC**, open it, and look
at **System type**. You need it to say **ARM-based processor**.

If it says x64, stop here. There is no version that will work on your machine,
and that is not something you can fix by downloading a different file.

You will also need about **1.5 GB of free space** and a few minutes on a
decent internet connection.

---

## Step 1: Install the app

Download **`Cooee_1.1.0_arm64-setup.exe`** and run it.

It is a normal installer. Click through it and Cooee will appear in your Start
menu.

> If you would rather use an MSI (some managed machines prefer it), download
> `Cooee_1.1.0_arm64_en-US.msi` instead. Either one is fine, you only need one.

---

## Step 2: Read this bit before you click past the warning

**Windows will almost certainly warn you about this app, and may quarantine it
as a trojan.** You have not done anything wrong and the download is not
corrupt. Here is exactly what is going on, so you can make your own decision.

Cooee does three things that, taken together, look precisely like a keylogger
to an antivirus classifier:

- it watches the keyboard globally, so it can notice you holding the hotkey
- it types text into other applications for you
- it touches the clipboard while doing so

That is also a complete and accurate description of spyware. The scanner cannot
tell the difference, so it guesses, and it guesses badly. Installed builds have
been flagged as `Trojan:Win32/Bearfoos.A!ml`. The `!ml` on the end means a
machine-learning hunch rather than a match against a known piece of malware.

**The real fix is a code signing certificate, and this project does not have
one.** It is a personal hackathon project, not a product, and certificates cost
money and require a registered company.

So the honest position is this: you are being asked to install an unsigned app
that hooks your keyboard, from a colleague. That is a reasonable thing to be
cautious about. All of the source code is public at
<https://github.com/littleLebowskiDude/Cooee> and you are very welcome to have
someone you trust look at it first, or to decide it is not worth it. Either is
a sensible answer.

If you are not comfortable, please do not install it. Nobody will mind.

---

## Step 3: Give Cooee its voice

Cooee has no speech model inside it. You need to download one, and there is a
file that does the whole job for you.

Download **`Setup-Cooee-Model.bat`** and double-click it.

A black window opens and tells you what it is about to download and roughly how
big it is. Press a key and leave it to run. It takes a few minutes and you can
go and do something else.

When it finishes it will say **Done**, and it will have pointed Cooee at the
model for you. There is nothing to configure.

### Why is this a separate download?

Because Cooee genuinely has no ability to reach the internet. There is no
networking code compiled into it at all, and there is a test in the repository
that fails the build if anyone ever adds any. That is what makes "nothing you
say leaves your laptop" a real claim rather than a promise.

If the app could fetch its own model, it would need that networking code, and
the claim would quietly stop being true. So the download lives out here in a
small file you can open in Notepad and read before you run it.

---

## Step 4: Say something

Open **Cooee** from the Start menu. It lives in the system tray, down by the
clock. You may need to click the little arrow to see it.

The first time it starts it spends about **20 seconds** getting the model ready
for the NPU. That happens once, not every time.

Now click into any app you can type in. Then:

**Hold `Ctrl` + `Windows`, say a sentence, and let go.**

Your words appear where the cursor is.

There is a small tone when it starts listening and another when it stops. If
you hear a third, lower tone, it means nothing was inserted, usually because it
did not hear any speech.

---

## Prefer not to hold a key down?

Holding two keys steady for a whole sentence is awkward, and for anyone dealing
with RSI, a tremor, limited dexterity, or using one hand, it rules the app out
entirely.

Open the tray icon, click the cog, and switch to **Latch** mode. Then you tap
the hotkey once to start, talk for as long as you like, and tap again to stop.
You can click and scroll around mid-sentence.

---

## If something goes wrong

**The black window closed and said the download did not finish.**
Run `Setup-Cooee-Model.bat` again. It keeps whatever already downloaded and
carries on from there rather than starting over.

**Cooee is open but nothing happens when I hold the keys.**
If Cooee was already running when you ran the model setup, it has not noticed
the model yet. Right-click the tray icon, quit, and open it again.

**My words go into the wrong place, or nothing appears.**
Some apps refuse pasted text and some refuse simulated typing. Open the cog,
go to the per-app settings, and use the **Detect** button to name the app you
are having trouble with, then try the other insertion method.

**It disappeared after I installed it.**
Antivirus has almost certainly quarantined it. See Step 2. This is a decision
for you and your own judgement, not something to work around casually.

Anything else, come and find me.

---

## What Cooee writes to your laptop

Open the cog and go to **Your data**. It lists the engine, the model, and every
file the app has written, with buttons to export it, delete all of it, or open
the folder and look.

Nothing is sent anywhere, so there is nothing to delete at the other end.
