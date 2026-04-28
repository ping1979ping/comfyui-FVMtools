# JB v1 — Minimal Reference Workflow

End-to-end wire-up for the **JB (JSON Builder)** suite. Five nodes assemble a
character + scene as JSON, hand it to a CLIPTextEncode, and render through
your usual diffusion stack.

---

## Five-node pipeline

```
              ┌──────────────────────┐
              │ JB · Outfit Block    │── outfit_string ──┐
              │ (outfit + color)     │                    │
              └──────────────────────┘                    ▼
                                          ┌────────────────────────────┐
              ┌──────────────────────┐    │ JB · Stitcher              │── string ──► CLIPTextEncode
              │ JB · Builder         │── string ─────► input_2 ─►│ title = "character_1"     │
              │ (face — from catalog)│                            │                            │── raw_json ──► (sidecar)
              └──────────────────────┘                            └────────────────────────────┘
                                                                                ▲
              ┌──────────────────────┐                                          │
              │ JB · Location Block  │── location_string ─────────► input_3 ───┘
              │ (location + atmos.)  │
              └──────────────────────┘
```

Optional fifth node — **`JB · Extractor`** — pulls a sub-tree out of the
Stitcher output for a regional detailer prompt
(`category = character_1.outfit.garments.upper_body`).

---

## Step-by-step build

### 1. JB · Outfit Block
- `outfit_set` = `general_female` (or any other set; click **Edit List** to tweak).
- `seed` = `42`.
- Slot toggles: top + bottom + footwear ON; rest OFF.
- `formality` = `0.5`, `coverage` = `0.6`.
- Color: `num_colors` = `5`, `harmony_type` = `auto`, `palette_style` = `general`,
  `vibrancy` = `0.5`, `contrast` = `0.5`, `warmth` = `0.6`.
- `output_format` = `loose_keys`.
- Output: `outfit_string` → wire to JB Stitcher's `input_1`.

### 2. JB · Builder (face)
- Click **Insert ▾** → `faces` → `east_asian_young_woman`. Four+ rows pop into the editor.
- Tweak any row's value if you like (each row is `[key] [value] [⋮]`).
- `output_format` = `loose_keys`.
- Output: `string` → JB Stitcher's `input_2`.

### 3. JB · Location Block
- `location_set` = `urban_brutalist` (or any).
- `seed` = `42`. Element toggles: background + foreground + time + weather ON.
- Color section identical to step 1 (or different — drives atmosphere phrasing).
- Output: `location_string` → JB Stitcher's `input_3`.

### 4. JB · Stitcher
- `title` = `character_1`.
- Three inputs already connected from steps 1–3. A fourth empty `input_4` slot is shown
  but has no connection — completely fine, it gets ignored at runtime.
- `output_format` = `loose_keys`.
- Output: `string` → CLIPTextEncode positive.
- Output: `raw_json` → optional `Save Text` / sidecar / debug ShowText node.

### 5. CLIPTextEncode + KSampler + VAEDecode + SaveImage
Standard ComfyUI diffusion stack. The Stitcher's `string` output is a
loose-keys JSON blob — most CLIP/SD encoders handle it fine; if your model
prefers strict JSON or comma-tag prompts, switch the Stitcher's
`output_format` accordingly.

---

## What the encoder sees (loose-keys, `seed=42`)

```
character_1: {
  outfit: {
    set_name: "general_female",
    seed: 42,
    formality: "smart_casual",
    coverage_target: 0.6,
    color_tone: "neutral",
    garments: {
      upper_body: { name: "blouse", fabric: "silk", color_role: "primary",
                    color_resolved: "burgundy",
                    prompt_fragment: "burgundy silk blouse", ... },
      lower_body: { ... },
      footwear:   { ... }
    }
  },
  face: {
    age: "early twenties", ethnicity: "east asian",
    skin: "smooth dewy skin with subtle freckles",
    eyes: "bright green almond-shaped eyes, long dark lashes",
    ...
  },
  location: {
    set_name: "urban_brutalist", seed: 42, color_tone: "neutral",
    elements: {
      background: { name: "monolithic concrete facade",
                    prompt_fragment: "monolithic concrete facade ... illuminated by warm amber afternoon", ... },
      ...
    }
  }
}
```

---

## Verification checklist

After Queue Prompt:
1. The render reflects all three sources (face from the Builder, outfit from the
   OutfitBlock, location from the LocationBlock).
2. No `#token#` placeholders show up anywhere in the encoded prompt — all
   color/atmosphere tokens resolved by the combo nodes.
3. Re-running with the same seed across Builder / OutfitBlock / LocationBlock
   produces an identical JSON (deterministic).
4. Inserting a different catalog snippet via Insert ▾ on the Builder swaps the
   face / character without touching the outfit / location.

---

## Mapping to the user's hosiery example

The user's worked example from the brief:
```json
"hosiery": {
  "type": "sheer black stockings, dark nylon with bold seams ...",
  "opacity": "semi-sheer (20-30 denier), glossy finish ...",
  "details": "smooth texture, visible skin tone underneath"
}
```

is shipped verbatim as `prompt_catalog/garments/hosiery_sheer_seamed.json`.
On the Builder:
- Insert ▾ → `garments` → `hosiery_sheer_seamed` lands as four rows:
  ```
  hosiery   :  (branch)
   type     : sheer black stockings, dark nylon with bold seams ...
   opacity  : semi-sheer (20-30 denier), glossy finish ...
   details  : smooth texture, visible skin tone underneath
  ```
- The ⋮ menu lets you reorder / indent / duplicate / delete any row.

---

## What's not in v1 (parking lot)

These are real backlog items, not bugs:

- **No drag-to-reorder.** Use the ⋮ menu Move Up / Move Down for now.
- **No multi-select / multi-row operations.** Each ⋮ menu acts on one row.
- **No undo/redo on the row stack.** Workflow save/load is the safety net.
- **No keyboard shortcuts** for indent/move (Ctrl+] / Ctrl+[ would be nice).
- **JS row widget**: the host element height is a fixed function of row count
  with a 50px floor — doesn't dynamically grow when single rows wrap text.

Everything above is an additive future PR; the v1 wire-up works without it.
