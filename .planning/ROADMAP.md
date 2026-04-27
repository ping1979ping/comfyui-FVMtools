# Roadmap — Milestone v1-smp

| Phase | Title | Status | Acceptance |
|---|---|---|---|
| P0 | GSD bootstrap | in_progress | `.planning/` initialized, pydantic confirmed, plan + roadmap committed |
| P1 | Schema & types | pending | `pytest tests/smp/test_schema.py` green; PromptDict importable |
| P2 | Dict generators V2 (Outfit + Color + Combiner) | pending | Same seed → identical OUTFIT_DICT; combiner resolves all `#tokens#` |
| P3 | LocationGenerator + 3 sets | pending | All 3 sets parse clean; same seed → byte-identical LOCATION_DICT |
| P4 | StructuredPromptAssembler + SAMClassRouter | pending | Synthetic PROMPT_DICT → 4 strings + region map (golden test); E2E render OK |
| P5 | PROMPT_DICT plumbing (3 builders + Aggregator + Serialize) | pending | Full pipeline from generators → KSampler runs; semantic-match sidecar |
| P6 | SidecarSaver + reference workflow | pending | `examples/workflows/smp_v1_minimal.json` runs → valid PromptDict sidecar |
| P7 | Backlog | parked | BatchVariator, JS widgets, qwen_chatml, presets, LLMEnhance |

## Cross-phase invariants
- V1 nodes (`FVM_OutfitGenerator`, `FVM_ColorPaletteGenerator`, `FVM_PromptColorReplace`, all `Person*`) must remain functional after every phase.
- All SMP code under `core/smp/` and `nodes/smp/`.
- Conventional commit prefixes: `feat(smp):`, `fix(smp):`, `test(smp):`, `docs(smp):`.

## Source plan
`C:/Users/vmett/.claude/plans/in-der-inbox-sind-pure-stream.md`
