# Current task — Windows solver, 2026-11-26

- Last available tester run reports 24.8 ms median and 69.4 ms p95 frame
  delivery gaps, 65.5 ms active CV wall p95 and 117.7 ms active source-age
  p95. The report named beta.5 lacks beta.5-specific timing fields, so its
  exact revision is unverified. Neither old logs nor local CV imply live grades.
- On 221 archived Windows ROI images, acquisition CPU p95 fell from 3.22 to
  2.49 ms on Linux using a centered Windows prompt search before full search.
  All 55 accepted centers, arcs and needle measurements matched. Correlation
  scores can change slightly due to cropped versus full OpenCV matching.
  Historical replay attribution put engine and planner CPU work below 0.08 ms
  p95. [fast-anything](https://github.com/EL4CTEO/fast-anything) and
  [rust-anything](https://github.com/erkamyaman/rust-anything) were reviewed.
  Rust replacement of the planner has little headroom at the measured cost.
- Windows runs one CV call in parallel with an outstanding GREAT timer.
  Capture waits use high-resolution sleeps near a deadline. During initial
  geometry confirmation, unambiguous needles in two fresh frames can establish
  motion. The final source/freshness/focus/LMB checks remain in the dispatch
  path. The default game response stays provisional at 4 ms.
- The DXGI worker reports present, acquire and copy timestamps and accumulated
  frames. Manifests also report pipe and publication delays, CV wall/thread
  CPU, dispatch wake lag, final key gate and SendInput. Scheduled errors are
  model estimates. Actual center accuracy still needs the Windows tester.
- Beta 7 rechecks freshness and the press deadline after the final Win32
  window and mouse validation. A simulated 15 ms stall at that gate now
  records one skipped attempt and produces no keydown. An older Windows log
  placed one delay at 8.25 ms before the Win32 request and a separate sample
  at 10.47 ms inside SendInput. The simulated case does not identify the
  cause of any actual grade.
- Windows PowerShell downloaded and verified public beta.7 on a clean runner
  (run 45829094905). A single-run report from `tools/compare_solver_sessions.py`
  reads DXGI capture stages, signed modeled offsets and deadline misses. Live
  acceptance still requires visible grades and the matching logs from beta.7.

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
