# Live tests — Sign Tools against a running ComfyUI

Not part of the pytest suite: these need ComfyUI on :8189 with Krea 2 and
LM Studio on :1234. Run them from this directory with the ComfyUI venv.

| Script | What it does |
|---|---|
| `make_real.py` | Generates three photographic test scenes with Krea 2 (street, wine shelf, noticeboard). Real material — synthetic rectangles hide the failures that matter. |
| `krea_run.py` | One full pass over an image. `--set sel.max_regions=24 --set detailer.glyph_denoise=0.45` overrides any widget. |
| `suite.py` | The acceptance test: text A -> B -> A across four scenes, each result read back by the vision model. Prints a PASS/FAIL table. |
| `roundtrip.py` | Helper module used by the other two (graph building, upload, judging). |

## Restarting ComfyUI 8189

There is no manager reboot route (404). What works:

```powershell
$conn = Get-NetTCPConnection -LocalPort 8189 -State Listen
$owner = ($conn.OwningProcess | Select-Object -First 1)
$parent = (Get-CimInstance Win32_Process -Filter "ProcessId=$owner").ParentProcessId
Stop-Process -Id $parent -Force; Stop-Process -Id $owner -Force
Start-Process cmd.exe -ArgumentList "/c","D:\AI\ComfyUI\LaunchScripts\GPU1_P8189_cu130.bat" -WindowStyle Minimized
```

Do NOT name the variable `$pid` — that is PowerShell's own process id and
silently returns the wrong process.

`GET /object_info/<Node>` answers **HTTP 200 with an empty body** when a node is
absent, so the status code alone proves nothing. Check for the node key.
