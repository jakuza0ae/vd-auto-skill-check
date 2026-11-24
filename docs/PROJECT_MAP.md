# Project map

Scope verified 2026-10-19 from README, entry points and the timing/capture modules;
prompt matching rechecked 2026-10-24 against the user's updated-game screenshot.
This repository is an external CV skill-check solver plus a browser practice app.
It contains no Roblox place tree, Luau scripts, RemoteEvents, RemoteFunctions,
ModuleScripts or game datastore implementation. Those game internals are unknown.
Rojo exists on this machine but has no project configuration/sourcemap here.
Roblox Studio MCP tools are exposed; their connection and project ownership are
unverified and are not needed to inspect this solver. Nothing was installed.

## Entry points and hot path

- `start-vulkan-sober.sh`: manually launch the game with the installed capture
  layer. `start-vulkan-solver.sh` validates the already running layer and starts
  `run-vulkan.py`; it never automatically restarts Sober.
- `run-vulkan.py` / `run.py` -> `vd/runtime.py`: CLI configuration and the
  single-owner capture, CV, lifecycle and dispatch loop. Portal capture is an
  alternative exposed by `start-solver.sh`.
- On Windows `run.py` starts `vd/windows_capture.py` -> `native/windows_capture.cpp`.
  DXGI duplicates the chosen monitor, copies only the scaled center ROI and
  publishes it over a pipe with separate desktop present and copy timestamps
  from QPC. The default is the primary monitor with a 60 FPS capture limit.
  `vd/windows_input.py` uses Win32 mouse state and Space output. Dispatch also
  verifies that the foreground Roblox window contains the captured region and
  that focus and the button still match just before keydown.
- `native/vulkan_layer/vd_capture_layer.cpp/.h` -> `vd/vulkan_capture.py`:
  three asynchronous GPU staging slots, worker publication and SHM v3. Source
  time, GPU-copy completion, publication, epoch and swapchain generation stay
  separate. Consumer lease: 250 ms heartbeat, 2 s expiry. Pixel seqlock protects
  reads; a busy writer never authorizes input from unchecked source metadata.
  Optional display-only brightness pass (`VD_BRIGHT_ENABLE`, `vd_bright.conf`,
  Sober-log game detection) runs after the capture copy in the same present hook.
- `vd/vision.py` -> `vd/motion.py` -> `vd/planning.py`: independent image
  measurements, robust speed/latest accepted phase, immutable GREAT occurrence
  and white-center timer. Cadence-scaled jitter tolerance supports 1800 degrees/s.
  Default prompt matching accepts both the legacy icon and the jagged Space key
  seen in the user's 2026-10-24 screenshot (`assets/space_template*.png`). Both
  template canvases are aligned to the ring center. Explicit custom templates
  remain exclusive. The screenshot and synthetic new-arrow occlusion/motion
  checks live in `tests/fixtures/skillcheck_20261024.png` and
  `tests/test_updated_skillcheck.py`; live updated-game accuracy is unmeasured.
- `vd/engine.py`: confirmed geometry, generations, Frenzy transitions and one
  bounded GREAT attempt. `vd/dispatch.py` performs the final capture freshness,
  source identity and LMB checks before `vd/input.py` (Linux) or
  `vd/windows_input.py` sends Space.
- The runtime reserves a measured CV-sized slice before a pending deadline
  (4–12 ms) for dispatch. Busy publication gets bounded status retries. Completed
  CV still updates the forecast before the next dispatch; input has no second
  competing worker thread.

## Evidence, persistence and decisions

- `vd/calibration.py`: diagnostic same-ring freeze observer; stop accepting
  scenery after prompt loss. The configured response remains 4 ms, provisional;
  only scheduler/output dispatch delay is learned. GOOD is never targeted.
- `vd/recording.py`: asynchronous JSONL decisions/events, optional video and
  atomic start/final manifests. A start manifest remains incomplete after a
  forced process exit. Recordings are evidence, not authoritative game verdicts.
- `tools/audit_solver_coverage.py`: enumerate recordings including missing
  manifests; optional same-ring CV rescoring. `tools/compare_solver_sessions.py`
  compares timing and separately supplied confirmed verdicts.
- `tests/`: fake-input runtime, motion/planner/lifecycle, cached real-frame
  measurements and native capture contracts. `simulator/skillcheck.html` is an
  empirical practice model; `simulator/bench.test.cjs` checks that model.
- `reports/` and `recordings/` are ignored local evidence. Source changes belong
  in versioned files. There is no remote publication or persistent game data API.

## Navigation and checks

Use relevant rows of SCRIPT_INDEX, then targeted `rg`. The metadata-only index
excludes recordings, old recordings, generated reports and dependency folders.
Regenerate with `python3 tools/generate_script_index.py`; `--check` rejects a
stale inventory. No watcher is needed. Full Python validation:
`.venv/bin/python -m pytest -q`; browser model: `node simulator/bench.test.cjs`.
Existing historical design documents are under `docs/superpowers/`; consult only
when the current architecture or a specific decision needs more context.
