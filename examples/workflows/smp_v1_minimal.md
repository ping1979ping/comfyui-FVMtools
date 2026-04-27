# SMP v1 — Minimal Reference Workflow

End-to-end wire-up for the StructPromptMaker (SMP) sub-suite. The pipeline
flows from generators → combiners → builders → aggregator → assembler →
encoder → KSampler → SidecarSaver. Same image branch can also feed
SAM3 + PersonDetailer for region-specific detailing.

---

## Pipeline graph

```
                ┌─────────────────────┐
seed (INT) ─────│ SMP Color Generator │── COLOR_PALETTE_DICT ┐
                └─────────────────────┘                       │
                ┌─────────────────────┐                       │
seed (INT) ─────│ SMP Outfit Generator│── OUTFIT_DICT_RAW ┐   │
                └─────────────────────┘                   │   │
                                                          ▼   ▼
                                              ┌──────────────────────┐
                                              │ SMP Outfit Combiner  │── OUTFIT_DICT
                                              └──────────────────────┘    │
                                                                          ▼
                ┌────────────────────────┐                ┌──────────────────────┐
seed (INT) ─────│ SMP Location Generator │── LOC_RAW ─────│ SMP Location Combiner│── LOCATION_DICT
                └────────────────────────┘                └──────────────────────┘    │
                                                                          (palette ───┘ shared)

  OUTFIT_DICT  ──► SMP Clothing Builder ──► PROMPT_DICT ──┐
  LOCATION_DICT ──► SMP Environment Builder ──► PROMPT_DICT ─┤
  (subject widgets) ──► SMP Subject Builder ──► PROMPT_DICT ─┤
                                                              ▼
                                                       SMP Aggregator ─► PROMPT_DICT
                                                              ├─► SMP Prompt Serialize
                                                              │       ├─► positive_prompt ─► CLIPTextEncode ─► KSampler …
                                                              │       └─► raw_json
                                                              │
                                                              └─► SMP Structured Prompt Assembler
                                                                      ├─► face_prompt    ─► (FaceDetailer)
                                                                      ├─► body_prompt    ─► (Person Detailer body slot)
                                                                      ├─► outfit_prompt  ─► (Person Detailer upper-body slot)
                                                                      ├─► location_prompt
                                                                      ├─► region_map     ─► (sidecar)
                                                                      └─► structured ────► SMP SAM3 Class Router
                                                                                              (handles per-mask routing)

  KSampler ──► VAEDecode ──► SMP Sidecar Saver ──► .png + .prompt.json
```

---

## Node-by-node setup

1. **SMP · Color Generator (dict)**
   - seed: `42`
   - num_colors: `5`
   - harmony_type: `auto`
   - style_preset: `general`
   - vibrancy / contrast / warmth: `0.5 / 0.5 / 0.6`
   - Outputs: `color_palette` → both Combiners.

2. **SMP · Outfit Generator (dict)**
   - outfit_set: any from `outfit_lists/` (e.g. `general_female`)
   - seed: `42` (or different to vary outfit independently)
   - style_preset: `general`
   - formality / coverage: `0.5 / 0.6`
   - Toggle slots: `top`, `bottom`, `footwear` ON; others OFF for a basic look.
   - Output: `outfit_raw` → SMP Outfit Combiner.

3. **SMP · Outfit Combiner**
   - Inputs: `outfit_raw` from generator, `color_palette` from color generator.
   - Output: `outfit_dict` → SMP Clothing Builder.

4. **SMP · Location Generator (dict)**
   - location_set: `urban_brutalist` | `beach_mediterranean` | `studio_minimal`
   - seed: `42`
   - Toggle elements: `background` + `foreground_element` + `time_of_day` + `weather` ON.
   - Output: `location_raw` → SMP Location Combiner.

5. **SMP · Location Combiner**
   - Inputs: `location_raw`, `color_palette`.
   - Output: `location_dict` → SMP Environment Builder.

6. **SMP · Subject Builder**
   - subject_id: `subject_1`
   - age_desc: `young`
   - gender: `woman`
   - expression: `confident, subtle smile`
   - hair_color_length: `dark auburn hair`
   - pose_hint: `seated on wide steps, slight S-curve`
   - extra_json:
     ```json
     {
       "skin_tags": ["smooth skin", "subtle freckles"],
       "eye_desc": "bright green almond eyes",
       "brow_desc": "arched brows",
       "lip_desc": "soft full lips",
       "body_build": "slim build"
     }
     ```
   - Output: `prompt_dict` → SMP Aggregator (input A).

7. **SMP · Clothing Builder**
   - Inputs: `outfit_dict` from combiner, `subject_id`: `subject_1`.
   - Output → SMP Aggregator (input B).

8. **SMP · Environment Builder**
   - Input: `location_dict` from combiner.
   - Output → SMP Aggregator (input C).

9. **SMP · Aggregator**
   - Inputs A, B, C from the three builders.
   - Output: merged `prompt_dict` → SMP Prompt Serialize AND SMP Structured Prompt Assembler.

10. **SMP · Prompt Serialize**
    - format: `natural_language` (or `raw_json` for Z-Image / Flux 2 power use).
    - Outputs:
      - `positive_prompt` → `CLIPTextEncode` (positive)
      - `negative_prompt` → `CLIPTextEncode` (negative)
      - `raw_json` → optionally a `ShowText` node for inspection.

11. **CLIPTextEncode + KSampler + VAEDecode** — standard ComfyUI diffusion stack.

12. **SMP · Structured Prompt Assembler**
    - Inputs: same `prompt_dict` from Aggregator (or wire the combiner outputs directly).
    - face_eye_boost: `1.1`
    - include_quality: `True`
    - Outputs:
      - `face_prompt` → an existing `FaceDetailer` (or feed back into a second pass for the face crop).
      - `outfit_prompt` → `PersonDetailer` upper-body slot prompt.
      - `body_prompt`, `location_prompt` → optional auxiliary passes.
      - `structured` → SMP SAM3 Class Router for per-mask routing.

13. **SMP · SAM3 Class Router** (one per Detailer slot you wire)
    - Input: `structured` from the Assembler.
    - class_name: `upper_clothes`, `face`, `skirt`, `shoes`, `background`, …
    - Output: prompt fragment → that slot's prompt input on `PersonDetailer`.

14. **SMP · Sidecar Saver**
    - Inputs:
      - `images` from `VAEDecode`
      - `prompt_dict` from Aggregator
      - `structured` from Assembler (optional)
    - filename_prefix: `SMP/v1`
    - Side effect: writes `<output>/SMP/v1_NNNNN_.png` and `<output>/SMP/v1_NNNNN_.prompt.json`.

---

## Verification checklist

After the workflow runs:
1. The generated PNG sits in `ComfyUI/output/SMP/v1_NNNNN_.png`.
2. Next to it, `v1_NNNNN_.prompt.json` exists and contains:
   - `prompt_dict.subjects[0].age_desc == "young"`
   - `prompt_dict.outfits.subject_1.garments` populated with resolved color names (no `#token#`).
   - `prompt_dict.location.elements` populated.
   - If the Assembler was wired into the saver: `structured_prompts.face`, `.outfit`, `.location`, `.region_map[]`.
3. The encoded positive prompt (visible in the workflow's saved `prompt` metadata in the PNG) reflects all three sources: subject, outfit, location.
4. Re-running with the same seeds produces a byte-identical sidecar JSON for the structural-prompt fields.

---

## Tip: parallel branches with the Aggregator

The Aggregator deep-merges per spec §6.2:
- Lists concatenate → two `SubjectBuilder` outputs into different aggregator inputs produce a multi-person scene.
- Scalars overwrite (later wins) → useful when you want an `EnvironmentBuilder` to override a default location set elsewhere in the graph.
- `null` deletes a field — useful for the planned BatchVariator (P7).

## What's not yet in v1

Deferred to P7 and tracked in `.planning/ROADMAP.md`:
- BatchVariator (lock/vary semantics for batch runs).
- Preset YAML loader with `extends:` inheritance.
- Camera / Lighting / Pose / Composition / SceneStyle / NegativePrompt builders.
- `qwen_chatml` and `comma_tags` serializer formats.
- JS frontend widgets (silhouette preview, JSON CodeMirror editor, lock/vary toggles).
- LLMEnhance via Ollama.
