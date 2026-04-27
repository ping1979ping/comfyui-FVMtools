# FVMtools — Project Context

## What this is
ComfyUI custom-node suite for fashion / portrait generation pipelines. Existing capabilities: outfit + color + palette generators (string-based), SAM3-based person detection, per-slot LoRA detailers.

## Current milestone: v1-smp — StructPromptMaker (SMP)

Add a typed, dict-based structural prompt pipeline alongside the existing string-based generators. Bridge it into the existing SAM3 + PersonDetailer stack via a `StructuredPromptAssembler`. New sub-suite namespace: `FVM Tools/SMP/...`.

### Source specs
- `INBOX/ComfyUI-PromptForge-SPEC.md` — generic JSON-prompt pipeline (Pydantic, Builder→Aggregator→Serialize, BatchVariator, sidecar JSON).
- `INBOX/ComfyUI-PromptForge-Generators-SPEC.md` — Generator/Combiner layer + StructuredPromptAssembler.

### Locked decisions
- **Single repo:** sub-suite lives inside FVMtools (not a separate `ComfyUI-PromptForge` repo).
- **Naming:** `StructPromptMaker` / `SMP`. Class prefix `FVM_SMP_*`. Category `FVM Tools/SMP/...`.
- **V1 nodes untouched:** new SMP nodes are additive parallel nodes; existing `FVM_OutfitGenerator`, `FVM_ColorPaletteGenerator`, etc. stay.
- **Reuse engines:** wrap `core/outfit_engine.py`, `core/palette_engine.py`, `core/role_assignment.py` under thin dict adapters; do not re-implement.
- **Reuse data format:** existing `outfit_lists/` (`name | probability | formality | fabric`) is the file format; new `location_lists/` mirrors it.

### Approved plan
`C:/Users/vmett/.claude/plans/in-der-inbox-sind-pure-stream.md`

## Stack
Python 3.x via ComfyUI venv at `D:/AI/ComfyUI/ComfyUI/venv/Scripts/python.exe`. pydantic 2.11.10 confirmed available. PyTorch via ComfyUI runtime.

## Verification rules (from CLAUDE.md)
- Concrete artifact required per phase acceptance (image / pytest output / valid JSON).
- No agent self-reported scores.
- Restart ComfyUI between phases that change `__init__.py`.
