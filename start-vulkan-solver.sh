#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
app_id="org.vinegarhq.Sober"
vulkan_dir="$HOME/.var/app/$app_id/data/vulkan"
shm_path="$vulkan_dir/vd_layer_shm.dat"
python_bin="$project_dir/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin="$(command -v python3)"
fi

if [[ ! -s "$shm_path" ]]; then
  printf '%s\n' 'Vulkan capture не найден. Sober не будет перезапущен автоматически.' >&2
  printf '%s\n' 'Закрой обычный Sober вручную и один раз запусти ./start-vulkan-sober.sh.' >&2
  exit 2
fi

if ! "$python_bin" - "$project_dir" "$shm_path" <<'PY'
import sys
import time
from pathlib import Path

project_dir = Path(sys.argv[1])
shm_path = Path(sys.argv[2])
sys.path.insert(0, str(project_dir))
from tools.vulkan_layer_status import read_shm_header

first = read_shm_header(shm_path)
if not first or "error" in first:
    raise SystemExit(1)
if first.get("version") != 3:
    print("Нужен обновлённый Vulkan layer v3. Перезапусти Sober вручную через ./start-vulkan-sober.sh.",file=sys.stderr)
    raise SystemExit(1)
first_present = first.get("last_present_ns",0)
time.sleep(1.1)
second = read_shm_header(shm_path)
if not second or "error" in second:
    raise SystemExit(1)
raise SystemExit(0 if second.get("last_present_ns",0)>first_present else 1)
PY
then
  printf '%s\n' 'Vulkan layer устарел или не показывает новые кадры. Sober не трогаю.' >&2
  printf '%s\n' 'Если Sober был открыт обычным способом, закрой его вручную и запусти ./start-vulkan-sober.sh один раз.' >&2
  exit 2
fi

printf '%s\n' 'Подключаю solver к уже открытому Sober. Ctrl+C остановит только solver.'
exec "$python_bin" "$project_dir/run-vulkan.py" "$@"
