# /krea2-set-test — render-check location / outfit sets on Krea 2

Render a location or outfit set through the real JB pipeline on a running ComfyUI, then judge the contact sheet visually and fix the entries that failed. Use it after `/build-location-set` or `/build-outfit-set`, or whenever a set "looks off" in Krea 2.

## Input

$ARGUMENTS

Expected form: one or more set names (globs allowed) plus optional seeds/port, e.g.
- `indoor/hospital_campus/*`
- `outdoor/reading_pennsylvania_downtown_spot/penn_square_street_corner seeds 1 2 3`
- `outfit female/business/*_with_heels on outdoor/suburban_office_park/landscaped_front_entrance`

## Step 1: Pick the instance

- **8189** is the test instance (GPU 1). **8188** is the user's working instance; use it only when the user said so, e.g. "8188 hat Pause".
- Check that it is idle first: `curl -s http://127.0.0.1:<port>/queue`. If another client has jobs running, wait or ask.
- New set files are picked up without a restart, because the engine scans disk on every call. Python changes in `core/` or `nodes/` need a restart, or a local text check.

## Step 2: Render

```bash
cd D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools && \
  D:/AI/ComfyUI/ComfyUI/venv/Scripts/python.exe scripts/krea2_set_check.py \
    --location "indoor/hospital_campus/*" \
    --outfit female/business/office_day_business_casual_with_heels \
    --seeds 11 22 --accessories --bag --port 8189 --out "<scratchpad>/hospital.jpg"
```

- The script's `--port` default is 8188, so always pass the port you checked in Step 1.
- `--preset cielbleu_native` (default) is the user's everyday stack with the turbo LoRA muted (the checkpoint already contains it): `cielbleuKrea2_v1` + inline-skin 0.35, 10 steps, cfg 1.2, euler/beta. `--preset cielbleu` additionally loads the turbo LoRA 0.95 (double turbo; only for comparison). `--preset turbo` is stock Krea 2 Turbo: 8 steps, cfg 1, er_sde/simple. Run the final check on both.
- Use 2-3 seeds per set. One seed proves nothing; the goal is that the result is *reproducible*.
- `--style` defaults to `business_executive` (always draws the hairstyle with its face cue); `--accessories --bag` switch those slots on, which the defaults leave off. The default person is a 45-year-old woman; for `male/*` outfits pass `--person "A 45-year-old man, …"`.
- The JSONL next to the sheet holds the exact prompt and character count of every tile. The script warns above 2400 chars, where Krea 2 blacks out.
- To isolate a location problem, pair it with a fixed, simple outfit. To isolate an outfit problem, pair it with a calm location such as `outdoor/suburban_office_park/landscaped_front_entrance`.
- For many sets, render one sheet per wave or category, not one per set. `--help` lists the rest (`--no-props`, `--no-detail`, moods, size).

## Step 3: Judge the sheet yourself (Read the .jpg)

Per tile, check:

| Check | Fail looks like | Usual cause → fix |
|---|---|---|
| one person | waiters, passers-by, faces in frames | people word / `busy` / `cafe patio` crowd cue → remove it, add `empty` |
| head + shoes in frame | head cut at the top, close-up on shoes | shoe-centric outfit text, missing framing sentence → keep shoes one clause; headwear entry that names the face |
| props small and far | mug / lamp / plant huge beside the subject | prop without a size + far placement, object in `foreground_element` → rewrite per Krea 2 rules |
| background reads as the place | flat wall, generic studio | background too short or an object list → one area sentence across the frame |
| clean surfaces | smeared blotches, mosaic patches on the floor | busy floor pattern (carpet tile, mosaic) → plain floor wording |
| no gibberish text | fake letters on signs, screens, menus | implied writing → blank / pictogram / exact quoted word |
| realism | CGI look, glossy stock photo, oversaturated | luxury adjectives, fantasy words → mundane, concrete materials and light |
| outfit as written | wrong garment, colour on hair, missing heels | engine output (check the JSONL text first), doublettes, colour tag on a hairstyle |

Write a short per-set verdict (pass / fix: which entry, why). Then edit the files, re-run the lint (`tests/unit/test_location_krea2_rules.py` for locations) and render the failing sets again with the same seeds. Repeat until every seed passes.

## Known model effects (not caused by the set text)

- **cielbleu smudges.** With the turbo LoRA stacked on the checkpoint (`--preset cielbleu`) about 1 in 5 renders of bright, evenly lit interiors got a dark smeared patch on the floor, the same seeds every time. The checkpoint already contains the turbo distillation, so the LoRA applies it twice. `cielbleu_native` (LoRA muted) rendered 12 of 12 clean on the sets that smeared worst (2026-10-10). Stock Krea 2 Turbo never showed it either.
- **Server code is loaded at start.** List files are re-read on every call, but changes in `core/` or `nodes/` (hairstyle colour fix, new `style_preset` values) reach a running ComfyUI only after a restart. Until then, `business_executive` fails validation with `value not in list`. The script builds prompts locally, so it already uses the new code.
- **Batch variation without new nodes.** Use Impact `ImpactWildcardProcessor` with `mode = fixed`, `{set|set|…}` in `populated_text` and its seed on randomize, then rgthree `Any Switch` into the combo; see `examples/krea2_business_batch_reading.json`. A Primitive node with `control_filter_list` does **not** filter in frontend 1.53, because the filter widget is not linked after a workflow load.

## Rules

- **Evidence is the sheet, not a score.** Never accept a sub-agent's self-report. Open the image.
- Build sets with sub-agents if you like, but keep the judging and the final verdict here, on the rendered sheet.
- Do not delete user outputs. The script writes under `output/fvm_krea2_check/`, which is safe to clear.
