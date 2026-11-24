# VD screen solver

`./start-solver.sh` captures the selected monitor, watches a centered 320×240
skill-check area, and sends Space while LMB is held. By default it captures the
monitor and uses the `(800, 420)` area for 1920×1080. You can instead capture
just the game window with `--capture-source window`; that mode centers the same
area inside the selected window.
The predicted aim stays at the center of white, with up to 1 degree of margin
from each boundary. If the center is already passed, it presses immediately
instead of waiting for the trailing edge. The response delay is a provisional
setting inferred from recent CV telemetry, not a confirmed game constant.

It always targets the white GREAT arc on every confirmed check, including
successive checks on a continuous Frenzy ring. GOOD is only a possible observed
result, never an aiming target. If the GREAT deadline is already missed, tracking
fails through the intended press time, or the ring begins to disappear, it makes
one immediate LAST_CHANCE attempt instead of waiting for GOOD or withholding
Space. Without usable movement from startup it allows 300 ms for acquisition
before that attempt.

A frame gap or phase jump still discards the invalid speed fit. Ordinary duplicate
frames are tolerated using recent capture and motion cadences. A newer native
frame from the same source does not cancel a pending timer. A due timer is checked
before running CV on the next frame. LMB release, stale capture and a changed
source still block input. Each confirmed generation permits one press. LAST_CHANCE marks an uncertain or
late attempt at GREAT; its actual outcome can be GREAT, GOOD or MISS.

## Start on Windows

The [Windows x64 pre-release](https://github.com/jakuza0ae/vd-auto-skill-check/releases/tag/v0.1.0-beta.4)
includes Python, OpenCV and the DXGI worker. On Windows 10 or 11, download
`vd-windows-x64.zip`, extract it and open PowerShell in the extracted folder.
Run `./vd.cmd preflight` to check the installation.

You can also paste this block into PowerShell to download, verify, extract and
check the same release in your user folder. It installs on a per-user basis.

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $tag = 'v0.1.0-beta.4'
    $base = "https://github.com/jakuza0ae/vd-auto-skill-check/releases/download/$tag"
    $dir = Join-Path $env:LOCALAPPDATA "VD-Solver\$tag"
    $zip = Join-Path $env:TEMP ("vd-{0}.zip" -f [guid]::NewGuid())
    try {
        Invoke-WebRequest -UseBasicParsing "$base/vd-windows-x64.zip" -OutFile $zip
        Invoke-WebRequest -UseBasicParsing "$base/vd-windows-x64.zip.sha256" -OutFile "$zip.sha256"
        $hash = (Get-Content -LiteralPath "$zip.sha256" -TotalCount 1).Split(' ')[0]
        if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash -ne $hash) { throw 'VD download checksum mismatch.' }
        Expand-Archive -LiteralPath $zip -DestinationPath $dir -Force
        & "$dir\vd.cmd" preflight
        if ($LASTEXITCODE -ne 0) { throw 'VD preflight failed.' }
    } finally {
        Remove-Item -LiteralPath $zip,"$zip.sha256" -ErrorAction SilentlyContinue
    }
}
```

Put Roblox fullscreen on the display you want to capture. The primary display
is selected by default. The first two commands below only observe the display.
Keep moving content visible during the probe so Windows supplies new frames.
Use `run` to enable Space output while LMB is held and Roblox is in the
foreground on the captured display. Ctrl+C stops the solver.

```powershell
Set-Location "$env:LOCALAPPDATA\VD-Solver\v0.1.0-beta.4" # for the PowerShell installation
./vd.cmd capture-probe --seconds 10
./vd.cmd dry-run --seconds 10 --no-recording
./vd.cmd run
```

Live testing at 1366×768 confirmed prompt detection and the beta 3 Space input.
Beta 4 reserves more time for the existing center timer when recent image
processing is slow. The timing changes were tested with a simulated clock.
In-game accuracy on beta 4 remains unmeasured.

The worker uses [DXGI Desktop Duplication](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/desktop-duplication-api)
with the standard Roblox launcher or Bloxstrap and supports the ordinary Direct3D 11
renderer. It copies only the centered skill-check region to CPU memory. The solver
receives fresh desktop timestamps and uses the same detector and GREAT timing as
on Linux. Use an unrotated SDR display. When switching display modes, restart the
solver if capture reports a lost source.

Start at the default 60 FPS with event-only recording. At a higher display refresh
rate you can try `--fps 120` and compare received FPS and the p95 desktop frame
age from `capture-probe`, along with Roblox FPS in the same scene with the solver
stopped and running. A faster capture rate can improve timing only when the
display provides fresh frames. FFmpeg is optional and needed for `--video`.

To choose a second display, list the available DXGI adapter and display indices,
then pass the displayed `--device-index` and `--output-index` to `run.py`.

```powershell
./build/windows-capture/vd-windows-capture.exe --list-outputs
./vd.cmd capture-probe --device-index 0 --output-index 1 --seconds 10
```

### Build from source on Windows

In an x64 Visual Studio Developer Command Prompt with C++ tools, a Windows
SDK and 64-bit Python 3.14 installed, run these from the checkout.

```bat
native\build_windows_capture.cmd
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run.py preflight
```

Use `.venv\Scripts\python.exe run.py` in place of `vd.cmd` for a source build.

## Start on Linux

```bash
./setup.sh
python run.py preflight
./start-solver.sh
```

Select the game monitor in KDE's screen-sharing dialog. With
`--capture-source window`, select the game window instead. Ctrl+C stops the
solver. Decision and timing logs are saved under `recordings/` unless an
explicit `--recording DIR` is supplied. Frame video is off by default; add
`--video` only when you need replay footage. The solver neither reads nor writes
the game process. It analyzes every captured frame while LMB is held and skips
CV while released. The capture worker runs at lower CPU priority than the game.
The provisional game response delay is 4 ms (`--response-delay-ms`, alias
`--lead-ms`), with 20 ms uncertainty (`--response-uncertainty-ms`, alias
`--lead-uncertainty-ms`). Dispatch delay is measured and learned separately;
learning never replaces the configured game response delay. Active check
telemetry is saved even without video; CV outcome labels are diagnostic and do
not establish real game accuracy.

The 4 ms response setting replaces the previous 12 ms default after repeated
early stops in the fast-check recording and the user's confirmation that misses
usually stop before white. It remains provisional. An explicit older delay flag
still overrides this default. The target stays at the GREAT center; phase jitter
tolerance scales with observed frame cadence, while real jumps still reset the
fit. Performance output includes separate active-check CV, processing and source
age summaries so idle frames cannot hide the cost of fast checks.

Before an imminent timer, the runtime reserves the recent CV p95 plus 1 ms
(4–12 ms on Linux, 4–60 ms on Windows) and checks a busy writer without another
pixel read or CV pass. The final millisecond uses bounded polling. Source and
mouse validation still apply. Freeze diagnostics stop at prompt disappearance;
red scenery cannot overwrite a same-ring landing. An atomic incomplete manifest
is saved at session start, so a forced exit retains configuration and provenance.

For focused Codex navigation see `AGENTS.md`, `docs/PROJECT_MAP.md` and
`docs/CURRENT_TASK.md`. Refresh the metadata-only script index with
`python3 tools/generate_script_index.py`; `--check` verifies freshness.

Vulkan capture uses SHM contract v3, three staging slots and a publication
thread. Busy slots skip capture without waiting for the GPU in the present
thread. A consumer renews its lease every 250 ms; capture stops after two seconds
without a consumer. After updating the native layer, manually close Sober and
start `./start-vulkan-sober.sh`, then `./start-vulkan-solver.sh`. The updater does
not restart the game. Older recordings remain readable.

`./start-vulkan-sober.sh` also arms a display-only brightness pass in the same
layer (`VD_BRIGHT=0` disables it). It turns on only while the Sober log shows a
Violence District join and runs after the capture copy, so the solver still sees
original pixels. Tune it live in
`~/.var/app/org.vinegarhq.Sober/data/vulkan/vd_bright.conf`
(`mode`, `strength`, `lift`, `detail`, `universes`).

## Compare capture profiles

The default is `quiet`: 60 FPS, capture worker nice 10, and an event-only
recording. The saved `baseline` remains available at 60 FPS and nice 5. To
compare load and responsiveness, stop the current solver before starting
another profile:

```bash
./start-solver.sh --variant baseline
./start-solver.sh --variant responsive
./start-solver.sh --variant quiet
./start-solver.sh --variant fps50
./start-solver.sh --variant fps45
./start-solver.sh --variant fps30
./start-solver.sh --variant deep-quiet
```

`baseline` requests 60 FPS and nice 5 for the capture worker. `responsive`
keeps 60 FPS and requests nice 0; `quiet` keeps 60 FPS and requests nice 10.
`fps50`, `fps45`, and `fps30` lower the capture rate while requesting nice 5;
`deep-quiet` uses 30 FPS and requests nice 10. A higher nice value lets the game
get CPU sooner but can delay frames. Lower FPS reduces capture work but can miss
a fast check between frames. All profiles use the same solver and continue
trying on each check it detects. The chosen profile, requested FPS, and worker
priority are recorded in each session's `manifest.json`.

You can also mix settings for an extra test, for example
`./start-solver.sh --variant quiet --fps 50 --capture-priority 5`.
Explicit `--fps` and `--capture-priority` values override the selected profile.
To compare monitor capture with a smaller game-window stream while keeping the
same solver profile, use `./start-solver.sh --variant quiet --capture-source window`.
Window capture can reduce PipeWire traffic when the game window is smaller than
the monitor; a fullscreen window may have the same cost as monitor capture.
For an opt-in lower-resolution test, run the game windowed at 16:9 and use
`--capture-source window --normalize-window-scale`. The worker crops a region
scaled to the window and resizes it to the solver's 320×240 input. This
normalization happens after PipeWire supplies the window image; it reduces
capture traffic only when the selected game window itself is smaller than the
monitor. A fullscreen 1920×1080 window still arrives as 1920×1080. The source
must be at least 960×540; upscaling does not restore detail, so verify actual
skill checks before relying on this mode. The stop summary reports both the
selected source dimensions and the 320×240 frame dimensions passed to the
detector.

On KDE Wayland, `--capture-source region` is a separate experimental mode that
asks KWin to crop a 180×250 strip around the two observed check positions before
it enters PipeWire:

```bash
./start-solver.sh --variant quiet --capture-source region
```

This avoids sending the full monitor image to the solver and cuts another 41%
from the 320×240 solver input. The detector offsets its two known search points
into the cropped strip. It builds a small
native helper on first use and registers its hidden KDE desktop entry. It needs
Qt 6, KPipeWire, PipeWire, Wayland client headers, and the Plasma screencast
protocol development file. It uses KWin's private unstable region protocol, so
it is limited to KDE Wayland. In multi-monitor layouts it captures the display
at workspace position `(0, 0)` when available. The requested ROI is scaled from
the 1920×1080 reference layout. Use `--capture-source window` if that does not
match the display running the game. Keep the solver output from a real run: a
faster capture cadence does not by itself prove that a skill check was hit.
`capture_source_size` is the full display mode; `capture_frame_size` is the
image sent to the detector (320×240 for monitor/window capture, 180×250 for
region capture).

Frame recording is off by default to avoid the frame copies and FFmpeg encoder
during play. Decision events and performance statistics are still recorded.
Use `--video` when you need a video for replay or inspection, for example:

```bash
./start-solver.sh --variant quiet
./start-solver.sh --variant baseline --video
```

`--no-video` is accepted as an explicit way to disable frame recording. Without
`--video`, sessions cannot be replayed or inspected frame by frame.

## Browser test bench

Run `./start-practice.sh` (or `./start-practice.sh --solver-test`). Open the
browser at a 1920×1080 viewport; F11 may be needed. The synthetic red pointer
and white/black zones appear at the exact
screen coordinates the solver captures. Hold LMB to activate the solver; press
F to start or stop a Frenzy sequence. The bench is for testing screen detection
and input timing, not proof of an in-game hit. Its Space response delay defaults
to 35 ms; set it to 4 ms to match the solver's current default, or to the
same value as your explicit `--response-delay-ms` setting.

## Replay and diagnostics

```bash
python run.py replay --recording recordings/SESSION
python run.py summary --recording recordings/SESSION
python tools/audit_solver_coverage.py recordings/SESSION
node --test simulator/*.test.cjs
.venv/bin/python -m pytest -q
```

Replay reprocesses recorded video and timestamps, but virtual presses change
what would happen next, so later video is counterfactual. A CV landing label
only describes where the visible needle appeared to stop; it is not a
confirmed game result. The coverage audit separates predicted attempts from
`BLIND_NO_NEEDLE` and `BLIND_NO_MOTION` attempts. Performance output includes
frame age, delivery gaps, processing time, CPU shares for the main process and
PipeWire capture worker, capture skips, and recording drops. It excludes the
optional video encoder. Percentiles use the last 600 frames; `lifetime_max`
keeps the largest spike from the full run. A real-game run with visible results
is required to establish GREAT rate and game lag.

## Check game results and presentation pauses

Native present interval counters are recorded once per second in `events.jsonl`
when using Vulkan capture, including sessions without video. The first native
snapshot is used as a baseline; historical counters before recording are excluded.
Percentiles use 1 ms buckets (the last bucket is 255 ms or longer). These measure
intervals between the game's Present calls, not GPU execution or display latency.

Compare two sessions:

```bash
.venv/bin/python tools/compare_solver_sessions.py recordings/BEFORE recordings/AFTER
```

The report separates intended targets, CV predictions and manually confirmed
verdicts. Without verdict annotations, game results remain unknown. Export a
CSV template, fill `outcome` with GREAT/GOOD/MISS only when the game's verdict is
visible, and put the recording timestamp or other supporting observation in
`evidence`. Leave unresolved checks as UNKNOWN.

```bash
.venv/bin/python tools/compare_solver_sessions.py recordings/AFTER --write-outcome-template after-outcomes.csv
.venv/bin/python tools/compare_solver_sessions.py recordings/BEFORE recordings/AFTER --outcomes-before before-outcomes.csv --outcomes-after after-outcomes.csv
```

For a presentation timing baseline with the solver stopped, this sampler reads
the native status without renewing a consumer lease or enabling capture:

```bash
.venv/bin/python tools/sample_present_timing.py baseline-present.jsonl --seconds 60
```

Pass that file as `--native-before baseline-present.jsonl` to the comparison.
Use the same scene and settings. Older layer versions lack the histogram; the
report leaves presentation timing unavailable instead of estimating it from
capture FPS. Manual verdict annotations are not automatically verified against
video by the comparison tool.
