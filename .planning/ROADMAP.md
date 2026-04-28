# Roadmap

## Milestone v2-jb — JSON Builder suite (current, shipping)

| Phase | Title | Status | Acceptance |
|---|---|---|---|
| P0 | Tag SMP nodes as `(legacy)` | done | All 13 FVM_SMP_* display names suffixed; full suite still green |
| P1 | core/jb serialize + catalog + Builder Python | done | 50 tests; JSON ↔ rows round-trip; Builder textarea fallback shipped |
| P2 | Stitcher + Extractor | done | 20 tests; deep-merge per spec; dot-path extraction; dynamic-slot JS |
| P3 | OutfitBlock + LocationBlock combo nodes | done | 17 tests incl. user hosiery #color# regression; Edit List for both |
| P4 | JB Builder custom row widget JS | done | Canvas + DOM widget with key/value rows, indent bars, ⋮ menu (move up/down/indent/duplicate/delete), + Add Row, Insert From Catalog dropdown, Edit Catalog modal; 4 catalog HTTP routes |
| P5 | Catalog seed content | done | 11 starter snippets across 5 categories; user hosiery shipped verbatim; 23 contract tests verify each round-trips through dict_to_rows |
| P6 | Reference workflow + README | done | examples/workflows/jb_v1_minimal.md ships; README updated with JB suite + SMP marked legacy |
| P7 | Backlog | parked | Drag-to-reorder rows, multi-select, undo/redo, keyboard shortcuts, dynamic textarea-row height, more catalog seeds |

### v2-jb final state

- **5 new JB nodes** under `FVM Tools/JB/...`: Builder, Stitcher, Extractor, Outfit Block, Location Block.
- **11 catalog seed snippets** across faces / garments / locations / scenes / props.
- **Test suite: 632 passing** (404 V1 + 112 SMP + 116 JB).
- **Atomic commits** P0..P6, each green at the unit-suite level.
- **No new external dependencies.**

---

## Milestone v1-smp — StructPromptMaker (legacy, shipped)

| Phase | Title | Status | Acceptance |
|---|---|---|---|
| P0 | GSD bootstrap | done | `.planning/` initialized, pydantic confirmed, plan + roadmap committed |
| P1 | Schema & types | done | 20 schema tests green; PromptDict + defaults + types module shipped |
| P2 | Dict generators V2 (Outfit + Color + Combiner) | done | 26 tests; same seed → identical OUTFIT_DICT; combiner resolves all `#tokens#` |
| P3 | LocationGenerator + 3 sets | done | 26 tests; all 3 sets parse clean; 10 runs same seed = byte-identical LOCATION_DICT |
| P4 | StructuredPromptAssembler + SAMClassRouter | done | 29 tests including end-to-end pipeline producing 4 token-free regional prompts |
| P5 | PROMPT_DICT plumbing (3 builders + Aggregator + Serialize) | done | 26 tests including full Generators→Combiners→Builders→Aggregator→Serialize integration |
| P6 | SidecarSaver + reference workflow | done | 5 sidecar tests green; `examples/workflows/smp_v1_minimal.md` documents the wire-up |

13 SMP nodes still registered under `FVM Tools/SMP/...` with `(legacy)` suffix in the display names. Superseded by the v2-jb suite for new work; existing SMP workflows continue to function.

---

## Cross-milestone invariants

- V1 nodes (`FVM_OutfitGenerator`, `FVM_ColorPaletteGenerator`, `FVM_PromptColorReplace`, all `Person*`) must remain functional after every phase.
- JB code lives under `core/jb/` and `nodes/jb/`. SMP code lives under `core/smp/` and `nodes/smp/`. They are independent — JB does not import from SMP except for `core/smp/merge.deep_merge` (the deep-merge utility).
- Conventional commit prefixes: `feat(jb):`, `feat(smp):`, `fix(...)`, `test(...)`, `docs(...)`, `chore(...)`.

## Source plan

`C:/Users/vmett/.claude/plans/in-der-inbox-sind-pure-stream.md`
