# Working in this checkout

- Start with `docs/CURRENT_TASK.md`, the relevant section of `docs/PROJECT_MAP.md`
  and matching rows in `docs/SCRIPT_INDEX.md`. Run
  `python3 tools/generate_script_index.py --check` before trusting the index.
- Use targeted `rg` searches and read the affected implementation and necessary
  dependencies. Avoid repeating a whole-project scan without a reason. Expand
  investigation and repeat reads whenever correctness requires them.
- Update the index with `python3 tools/generate_script_index.py` after source
  additions, deletions, renames or edits. It reads metadata, uses no daemon and
  installs nothing. Update PROJECT_MAP only for architectural changes and keep
  CURRENT_TASK compact with decisions, checks, changed files and open work.
- This is an external solver, not a Roblox place source tree. No Rojo project is
  configured here. Do not invent Explorer paths or game server/client ownership.
  Roblox Studio MCP tools are available but belong to a separate Studio surface;
  use them only for operations requiring the intended Studio project. Do not
  change Rojo/Studio synchronization or install dependencies without need.
- Preserve unrelated local changes and existing recordings. Never restart or
  close Sober automatically. Offline tests must use fake input; do not send
  physical Space or drive the game without the user's authorization.
- Always aim at GREAT, around the white center. GOOD is a possible actual
  overshoot, never a planned target. Keep one bounded attempt per generation;
  source changes, stale capture and LMB release retain final dispatch checks.
- Configured game response and learned dispatch delay are separate. Never learn
  game response automatically from CV freeze labels. Do not add frame age twice.
- Enumerate relevant recording directories, including ones without a manifest.
  CV labels, replay predictions and passed tests do not prove actual game
  GREAT/GOOD/MISS results. Keep that distinction in conclusions.
- Run meaningful targeted regressions and the full Python suite for timing,
  capture or lifecycle changes. Run affected native checks for native changes.
  Do not replace necessary checks with token-saving shortcuts. Keep user-facing
  reports concise; do not dump recordings, source files, process arguments or
  environment variables into the conversation.
