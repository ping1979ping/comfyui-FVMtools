# Roadmap — Milestone v1-smp

| Phase | Title | Status | Acceptance |
|---|---|---|---|
| P0 | GSD bootstrap | done | `.planning/` initialized, pydantic confirmed, plan + roadmap committed |
| P1 | Schema & types | done | 20 schema tests green; PromptDict + defaults + types module shipped |
| P2 | Dict generators V2 (Outfit + Color + Combiner) | done | 26 tests; same seed → identical OUTFIT_DICT; combiner resolves all `#tokens#` |
| P3 | LocationGenerator + 3 sets | done | 26 tests; all 3 sets parse clean; 10 runs same seed = byte-identical LOCATION_DICT |
| P4 | StructuredPromptAssembler + SAMClassRouter | done | 29 tests including end-to-end pipeline producing 4 token-free regional prompts |
| P5 | PROMPT_DICT plumbing (3 builders + Aggregator + Serialize) | done | 26 tests including full Generators→Combiners→Builders→Aggregator→Serialize integration |
| P6 | SidecarSaver + reference workflow | done | 5 sidecar tests green; `examples/workflows/smp_v1_minimal.md` documents the wire-up |
| P7 | Backlog | parked | BatchVariator, JS widgets, qwen_chatml, presets, LLMEnhance |

## Final state — v1-smp shipped

- **13 new SMP nodes** under `FVM Tools/SMP/...` category. V1 nodes untouched.
- **3 location sets** (urban_brutalist, beach_mediterranean, studio_minimal) with 7 element files each, mirroring the outfit_lists format.
- **Test suite: 516 passing** (404 V1 + 112 SMP).
- **Atomic commits** P0..P6, each green at the unit-suite level.
- **No new external dependencies** — pydantic 2.11 was already in the venv.

## Cross-phase invariants
- V1 nodes (`FVM_OutfitGenerator`, `FVM_ColorPaletteGenerator`, `FVM_PromptColorReplace`, all `Person*`) must remain functional after every phase.
- All SMP code under `core/smp/` and `nodes/smp/`.
- Conventional commit prefixes: `feat(smp):`, `fix(smp):`, `test(smp):`, `docs(smp):`.

## Source plan
`C:/Users/vmett/.claude/plans/in-der-inbox-sind-pure-stream.md`
