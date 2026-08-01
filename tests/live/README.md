# Live tests — Sign Tools against a running ComfyUI

Not part of the pytest suite: these need ComfyUI on :8189 with Krea 2 and
LM Studio on :1234. Run them from this directory with the ComfyUI venv.

| Script | What it does |
|---|---|
| `make_real.py` | Generates three photographic test scenes with Krea 2 (street, wine shelf, noticeboard). Real material — synthetic rectangles hide the failures that matter. |
| `krea_run.py` | One full pass over an image. `--set sel.max_regions=24 --set detailer.glyph_denoise=0.45` overrides any widget. |
| `suite.py` | The acceptance test: text A -> B -> A across four scenes, each result read back by the vision model. Prints a PASS/FAIL table. `--set glyph_surface_restyle=1.0` overrides any detailer widget, which is how one change gets separated from another. `--fast` skips the census, `--max-slop N` tolerates N invented strings. |
| `slop_census.py` | Transcribes EVERY piece of text in a picture, tile by tile, and marks each one a real word or gibberish. Point it at any image. |
| `selector_probe.py` | Selector only, no sampling, so it is cheap. Saves the numbered preview **and every mask**, which is what makes the glyph step replayable offline — the only way to tell a detection problem from a rendering problem. |
| `roundtrip.py` | Helper module used by the others (graph building, upload, judging). |

The scenes are looked up in this directory first and in `ComfyUI/input` second, so
a fresh clone does not have to regenerate them. A run that executes **nothing**
exits 2 — reporting 0/0 as a pass is how a broken harness gets mistaken for a
working pipeline.

## What counts as passing

"The target text arrived, and no second copy shows through" is far too weak a
bar. It passed a noticeboard that still carried five lines of pseudo-writing
beside the one word that had been replaced — the exact failure these tools exist
to remove. Under that bar the suite read 12/12; a census that judges every
string in the picture read **4/12** on the same images.

So a pass now needs all three:

1. the target wording is present,
2. no second set of letters shows through (ghosting <= 0.2),
3. **no invented strings anywhere in the picture** — whether left over from the
   original, or written by the pipeline itself.

Point 3 catches both halves of the problem. Text on a surface nobody selected
survives untouched, and a surface swept clean at full denoise gets filled back
in with plausible-looking writing. Both look identical to a reader.

## Change one thing at a time

Two changes went in together once and the score fell from 9/12 to 7/12. Splitting
them showed one was worth +1 and the other -3; the second, fixed, was worth +1
more. Whenever a run gets worse, re-run with the parts separated before touching
anything else:

```
suite.py --set glyph_surface_restyle=1.0    # band replacement only
suite.py                                    # everything at its default
```

Measured on Krea 2 Turbo, seed 11, four scenes x three passes:

| Stand | Ergebnis |
|---|---|
| vor dem Umbau | 9/12 |
| Band + volle Bandmaske im Sampler | 7/12 |
| nur Band-Ersetzung | 10/12 |
| Band + Denoise nur auf der neuen Schrift | **11/12** |

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
