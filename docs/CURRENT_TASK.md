# Current task — Windows solver, 2026-11-23

- `run.py` selects DXGI on Windows with a default 60 FPS limit and explicit
  adapter/display selection. `capture-probe` reports received FPS, p95 desktop
  frame age and median age at ROI copy. `dry-run` runs the detector without
  opening input. Linux launchers still select their existing backends.
- `native/windows_capture.cpp`, `native/build_windows_capture.cmd` and
  `vd/windows_capture.py` implement center ROI capture over a pipe, QPC
  timestamps for desktop updates and capture invalidation on geometry changes.
  `vd/windows_input.py` and the final dispatch gate bind a held LMB to the
  foreground Roblox player on the captured monitor. The same GREAT planner is
  shared by both platforms.
- `vd/preflight.py`, `requirements.txt`, `vd/image_io.py`, `vd/vision.py` and
  `vd/recording.py` cover the Windows setup, Unicode image paths and platform
  differences in input dependencies and process priority. README has setup and
  passive checks. The script index includes Windows batch files.
- Verified with a strict MinGW x64 build and on Windows CPython 3.14 under Wine.
  59 Wine tests passed and one Windows FFmpeg CLI case was deselected. The
  native helper listed the test display. DXGI capture correctly reported the
  Wine `E_NOTIMPL` error and cleaned up the helper process.
- Linux full suite passed with 338 tests and one Windows-only skip. The
  synthetic portal probe produced frames and stopped its worker and reader.
- Windows capture now reserves the final millisecond before a scheduled press
  for bounded polling between frames. The reader can still publish a frame in
  that interval. Ordinary capture waits continue to block on the condition.
- A user's Windows tester confirmed live desktop frames at 1366×768 in beta 1.
  The session ended with `DXGI_ERROR_ACCESS_LOST`. The manifest recorded zero
  input events. The cause of an accompanying Roblox crash is unverified.
- The beta 2 default crop keeps at least 320×240 physical pixels on smaller
  displays that can fit it. The detector tries the viewport scale during
  acquisition if the native pixel scale fails, and keeps the accepted scale
  through tracking. At 1366×768 the original crop missed the fixed-size prompt
  in the screenshot fixture. Both scaled and fixed-pixel fixtures now detect it.
- Live beta 2 archived capture at 1366×768 has 55 positive detector frames out
  of 221 samples. In a separate run Windows accepted one virtual-key Space
  invocation, but the tester saw no game response. That attempt also logged
  `GREAT_PASSED`. Neither the cause of the missing response nor the live grade
  was established.
- Beta 3 uses Win32 scan code 0x39 for both Space edges and holds it for 80 ms
  to cover longer input polling intervals. An overlapping attempt is disarmed
  and the scheduled release stays in place. Roblox acceptance and weak-PC
  performance remain open.
- Windows distribution uses the `v0.1.0-beta.3` release and its portable ZIP.
  `tools/build_windows_release.py` bundles pinned Windows wheels, embedded
  CPython and the built DXGI worker. `vd.cmd` runs the CLI with bundled Python.
  README includes a PowerShell download with SHA-256 verification.

## Earlier task — updated prompt, 2026-10-24

Request: adapt the external solver to the updated skill-check appearance in the
user's screenshot; investigate whether the arrow obscures the target.

## Findings and decisions

- Reproduced on the original screenshot: the old detector returns NO_PROMPT
  (score 0.134). With the ring center supplied manually it already measures the
  white zone (~153–164 degrees), dark zone and new red arrow (~152 degrees).
- Added the new jagged Space prompt alongside the legacy icon. Kept the .80
  acceptance threshold, ring/needle measurements, timing, dispatch checks and
  GREAT-only targeting. Custom `Detector(template=...)` remains exclusive.
- New template: grayscale crop x=87..156, y=89..123 from the screenshot, with
  its 70x35 canvas centered on the ring at (122, 106.5), not on the glyph box.
- Existing `retained_target` handles tested new-arrow occlusion; no speculative
  geometry or timing changes. Response remains 4 ms, provisional; dispatch-delay
  learning stays separate. No game restart or physical input was performed.

## Changed files

- `vd/vision.py`: match either default prompt in local/full searches.
- `assets/space_template_20261024.png`: new prompt template.
- `tests/fixtures/skillcheck_20261024.png`: screenshot crop with original pixels and display metadata removed.
- `tests/test_updated_skillcheck.py`: screenshot, ROI/scale, compression,
  absent-prompt, occlusion/relocation and fake-motion GREAT regressions.
- PROJECT_MAP, CURRENT_TASK and metadata-only SCRIPT_INDEX refreshed.
- Preserve all pre-existing dirty files and recordings; the starting checkout
  already contained substantial uncommitted timing/capture work.

## Validation and remaining work

- Before the fix, seven new screenshot/ROI/engine cases failed with NO_PROMPT;
  the absent-prompt negative control passed.
- After the fix, all 13 new checks pass; full Python suite: **295 passed in
  49.66 s**. The original screenshot now returns OK with score ~1.0.
- Before/after comparison: 480 sampled frames from two legacy videos, 151
  accepted prompts (150 with a visible white target), no visibility or accepted
  measurement differences. Recent sessions have no video to validate the update.
- Offline detector timing: new prompt with center hint median 2.76 ms, p95
  3.61 ms; legacy hinted median 2.72 ms vs 2.70 ms before. Host-only timings,
  not a capture/FPS or input-latency measurement. Report:
  `reports/skillcheck-update-2026-10-24.json`.
- Live GREAT/GOOD/MISS results for the updated game remain unmeasured. Restart
  only the solver to load the change; do not automatically restart Sober.
- Prior timing evidence: `reports/rare-miss-audit-2026-10-19.json` and related
  reports. CV labels and synthetic claims are not authoritative game verdicts.

## 2026-11-14 — Sober 1.8.0 + brightness

- Sober 1.8.0 (GNOME 51 → VulkanLayer 26.08): installed capture layer v3 loads,
  `capture=YES`; swapchain now sometimes VK_FORMAT_R8G8B8A8 (37), already handled
  and tested. Installed `.so` == current build. No solver-side break found.
- Join failures after the update are `TlsVerificationFail` (network/route, not this
  repo): one 1.8.0 session joined and ran 24 min. Not fixed here.
- Brightness (user: VD only, no cheats): Roblox FFlag allowlist rejects lighting flags,
  so the effect is a screen post-process inside our capture layer (vkBasalt attempt
  dropped). `VD_BRIGHT_ENABLE=1` (set by `start-vulkan-sober.sh`, `VD_BRIGHT=0` off)
  arms it; a watcher thread reads `$XDG_DATA_HOME/sober/sober_logs/latest.log`
  (`GameJoinLoadTime ... universeid:N` on, `setStage: (stage:LuaApp)` off) and
  `vulkan/vd_bright.conf` (mode auto/on/off, strength, lift, detail, universes;
  live, 0.5 s). VD = universe 6739698191 (place 93978595733734); 10774406718 is a
  different game. Chain: app waits -> capture copy -> effect (copy + fullscreen
  shader `shaders/bright.*`, embedded `bright_spv.h`) -> present, so the solver
  reads original pixels. UNORM 8/10-bit swapchains only; any failure fails open.
- Checks: native contract covers chain order, effect off, no-consumer, submit
  failure, teardown, config/log parsing. vkcube (RADV) runs the effect on formats
  44/64 without errors; FPS cost not measured (hidden window throttled).
- After checker: effect requires a graphics present family, resets on log
  truncation, keeps last settings if the config reads empty, no longer counts as a
  capture drop. Left: shared intermediate image is unordered if presents alternate
  between two queues of one family (not seen in Sober).
- Unverified: capture -> effect semaphore chain on a real GPU (vkcube gives format
  64 = no capture under Wayland; XWayland run presented 0 frames). Only one short
  VD session in Sober logs, so lobby -> match teleport behaviour of the LuaApp
  "off" rule is unknown (a new VD join line turns it back on).
- Not verified live: no Sober run with the effect, no recording, no GREAT/GOOD/MISS
  data on 1.8.0. Python suite: 296 passed.
