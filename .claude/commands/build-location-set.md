# /build-location-set — FVMtools Location Set Builder & Verifier

Build a new location set under `location_lists/<indoor|outdoor>/<category>/<leaf>/` with automatic plausibility verification.

## Input

$ARGUMENTS

Expected form: a hierarchical slug (`indoor/<category>/<leaf>` or `outdoor/<category>/<leaf>`) plus a short theme description, e.g.
- `indoor/office/corporate_open_plan glass walls, monitors, ergonomic chairs`
- `outdoor/winter_vacation/alpine_lodge snow-packed mountain village, wooden chalets, golden hour`
- `outdoor/sports/soccer_pitch amateur soccer pitch in late summer, faded line paint`

## Background — what this builds

`core/location_engine.py` scans `location_lists/` dynamically. Each subdirectory becomes a UI dropdown choice in the SMP LocationGenerator and JB LocationBlock nodes. Format per element file is pipe-delimited:

```
name | probability | coverage_range | texture
```

Atmosphere files (`time_of_day.txt`, `weather.txt`) use `-` for both coverage_range and texture.

## Workflow — follow strictly in order

### Step 1: Validate slug — and read it aloud

Slug MUST be a 3-segment POSIX path: `<indoor|outdoor>/<category>/<leaf>`. Lowercase, snake_case for each segment, no spaces or hyphens.

**The slug is prompt text.** The JB `sentences` output format turns it into the first sentence of the prompt, mechanically (`_` → space, no lookup table):

```
The scene takes place indoors, it is a <category>, namely the <leaf>.
```

Write that sentence out with the proposed slug before creating anything. If it does not read as plain English, fix the slug. Rules (enforced by `tests/unit/test_set_naming.py`):

- **category** — a singular noun phrase that fits *"it is a …"*: `family_event`, `household_scene`, `luxury_hotel`, `pennsylvania_suburb`, `beach_or_pool_spot`. Not `family_events`, not `household`, not `prom`. Prefer an existing category (`ls location_lists/indoor location_lists/outdoor`) over inventing one.
- **leaf** — a noun phrase of at least two words that fits *"namely the …"*: `graduation_party_at_home`, `hotel_lobby`, `busy_sandy_beach`. Not `lobby`, not `graduation_party_home`.
- **no country / state codes** as tokens (`_us`, `_de`, `_pa`) — spell them out: `american_office`, `german_bus_stop`, `pennsylvania_nature_spot`.
- **qualifiers go in front**, never at the end: `busy_sandy_beach` not `sandy_beach_busy`, `plain_conference_room` not `conference_room_plain`. Trailing `_inside`, `_interior`, `_plain`, `_busy`, `_private`, `_public` are rejected.

Examples:
- valid: `indoor/american_office/plain_conference_room`, `outdoor/family_event/backyard_birthday_party`, `indoor/household_scene/walk_in_pantry`
- invalid: `yoga/private` (no scope), `Indoor/Yoga/X` (case), `outdoor-hiking/...` (dash), `indoor/office_us/conference_room_plain` (code + trailing qualifier), `indoor/hotel_luxury/lobby` (category is not a noun phrase, one-word leaf)

Reject the build with a one-line error if invalid — do not silently rewrite the slug.

### Step 2: Read a reference set

Pick the closest existing set as a stylistic anchor and read every file in it (ignoring its content for distinctness — only the format/density matters):

- For indoor themes: `location_lists/indoor/household_scene/laundry_room/` or `location_lists/indoor/family_event/graduation_party_at_home/`
- For outdoor themes: `location_lists/outdoor/beach_or_pool_spot/busy_sandy_beach/`, `location_lists/outdoor/pennsylvania_nature_spot/state_park_trailhead/`, or `location_lists/outdoor/private_space/loading_dock_after_hours/`

Each set has 7 files:
- `background.txt` — large surface behind subject (wall, sky, distant scenery)
- `midground.txt` — secondary mass at distance (furniture group, treeline, kiosk)
- `foreground_element.txt` — floor/ground plane (mat, sand, trail, dirt)
- `architecture_detail.txt` — small features (window light, blaze paint, beam edge)
- `props.txt` — small objects in scene (cushion, water bottle, lamp)
- `time_of_day.txt` — atmospheric mood (golden hour, overcast, sodium dusk)
- `weather.txt` — atmospheric condition (drizzle, haze, calm interior air)

### Step 2b: Research first (one agent per category)

Before drafting a new category, run one web-research agent per category (efficient model, e.g. Sonnet). The agent collects real anchors (real buildings, materials, views, what is closed or demolished), proposes 5-6 leaves and writes short visual phrases per element plus an avoid list with sources. It does not write set files. The agent briefs from 2026-10-09 live in the session scratchpad. They ask for no people words, no implied writing, a position for every phrase, and props that are small only.

### Step 2c: Krea 2 composition rules — mandatory for every new set

These rules come from fixed-seed A/B renders on Krea 2 Turbo, cielbleu + turbo LoRA (2026-10-09, `/krea2-set-test`), plus the Krea 2 docs. `tests/unit/test_location_krea2_rules.py` enforces them for every category listed in its `KREA2_STRICT_CATEGORIES`. **Add the new category there.**

What the renders showed: Krea 2 paints every noun it reads, and it paints it large and next to the subject. In the old object-list style, a mug in `props` sat beside her hand, a lamppost in `architecture_detail` stood at her shoulder, and anything in `foreground_element` filled the lower third. Changing the lead-in sentence ("Small props far in the background include …") made no difference, so the entry text itself has to carry the placement.

1. **Background carries the scene (90-260 chars).** Name the far mass, then describe what spans the frame, with a position in the image:
   `far side of Penn Street | … | a continuous row of three- and four-storey brick and limestone commercial buildings across the whole width of the image, a wooded ridge rising above the rooftops on the left`.
   One area sentence beats a list of objects.
2. **Midground gets a distance (60-210 chars):** `rows of empty white desks a few meters back on the left | … | black mesh task chairs`. Use `empty` for furniture that otherwise invites people.
3. **Architecture detail and props are small, few and far (≤120 / ≤115 chars).** Every prop has a size word (`small`, `tiny`, `slim`, `compact`) and a far placement (`far down the block`, `on a distant cabinet`, `at the back`). Props that are big by nature (bike rack, planter, trash can) are not props; describe them as part of the midground at a distance, or leave them out.
3b. **Size words belong to the object, never float free.** `small trash can far down the sidewalk, tiny in the distance` painted a child-sized figure on the sidewalk in 4 of 4 seeds; `small dark green trash can at the curb far down the block` did not. Banned: `tiny in the distance`, `tiny in the frame`, `small in the distance`.
4. **Foreground is the ground plane only (≤100 chars)**: floor, pavement, lawn, deck. Never an object: the engine writes "In the foreground is …" and Krea makes it huge. Avoid busy micro-patterns (`carpet tile`, `mosaic`, `checkered`), which the cielbleu finetune smears into blotches. Plain low-pile carpet, polished concrete, wide oak planks and brick pavers render clean.
5. **Relative positions are subject-neutral:** `far behind`, `on the left of the frame`, `a few meters back`, `across the street`. Never `her` or `his`, because the sets are reused for every subject.
5b. **Seating says it is empty.** A seating group without `empty` gets occupied: `low cognac leather chairs around a plain wood table` put two seated guests into a coffee-roaster render, and the same entry with `, all empty` did not. Every background or midground entry that names chairs, booths, banquettes, stools, benches, sofas or tables carries `empty` (or `pushed in`, `set for`, `stacked`).
6. **No people words**, including `busy`, `lively`, `crowded`, roles such as `resident`, `exhibitor` or `diner`, and pronouns. Krea draws them. Krea's own blog: interiors get people unless the empty frame is stated. Use `empty`, `set for`, `quiet`.
7. **No implied writing.** Signs, menus, screens, donor walls, whiteboards, brochures and directory boards turn into gibberish. Either leave them out, or make them `blank` / `unlit` / `abstract` / `pictogram`, or give an exact short quoted word (`{|reading "EXIT"}`).
8. **No parentheses and no `#`.** `CLIPTextEncode` parses `( )` as weights, and `#…#` is reserved for the engine's atmosphere tokens.
9. **Budget:** the longest possible draw over all seven elements must stay ≤ 950 chars. With lead-ins, outfit and person, the whole prompt then stays under ~2400 chars. Beyond about 640 tokens, Krea 2 in ComfyUI produces black or noisy images (ComfyUI issue #14782).
10. **Time of day / weather ≤ 60 chars**, plain photographic light (`soft overcast late-morning light`), and realistic for Pennsylvania across the seasons. Indoor "weather" means the light and air through the windows, not a joke (`humid air the AC can't keep up with` is out).

### Step 3: Draft all 7 files

Target item counts:
- `background`, `midground`, `foreground_element`, `props`: **12-18 items**
- `architecture_detail`: **10-15 items**
- `time_of_day`, `weather`: **10-12 items**

**Plausibility rules — apply WHILE drafting, not after:**

1. **Probability range:** every entry uses a `probability` ∈ [0.3, 1.0]. Highest-fit items get 0.85-1.0; rarer flavor items get 0.4-0.7. Do not put filler at 0.1 — leave it out instead.
2. **Coverage range:** non-atmosphere entries use `min-max` with `0.0 ≤ min < max ≤ 1.0`. Backgrounds dominate (`0.6-1.0`); midground (`0.2-0.5`); foreground (`0.1-0.4`); architecture details and props (`0.0-0.2`). Atmosphere files use `-`.
3. **Name has ≥2 words for non-atmosphere:** descriptive phrases, not single nouns. `bench` → `low minimalist bench`. `tree` → `oak and maple mixed canopy`.
4. **No duplicates within a file:** case-insensitive, trimmed.
5. **Indoor/outdoor banlist (non-atmosphere only):**
   - Indoor sets must NOT contain: `trail`, `ridge`, `mountain`, `forest canopy`, `ocean`, `beach`, `appalachian`, `summit`, `wilderness`, `switchback`
   - Outdoor sets must NOT contain: `hardwood floor`, `sofa`, `office desk`, `kitchen counter`, `duvet`, `shower stall`, `bath tub`, `indoor pool`
6. **Theme coherence:** every entry belongs to the same scene archetype. An `indoor/office/corporate_open_plan` set should not have a `singing bowl` (yoga vibe leak). An `outdoor/hiking/spring_pennsylvania` set should not have `palm tree` (climate leak).

### Step 4: Create the directory + files

```bash
# Slug "indoor/office/corporate_open_plan" maps to indoor/office/corporate_open_plan/.
mkdir -p D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools/location_lists/<slug>
```

Write all 7 files using the `Write` tool. Each file MUST start with the two header lines:

```
# element: <element_id>
# format: name | probability | coverage_range | texture
```

followed by a blank line and then entries.

### Step 5: Run extended validator (mandatory, do not skip)

```bash
cd D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools && \
  D:/AI/ComfyUI/ComfyUI/venv/Scripts/python.exe -m pytest tests/unit/test_location_lists_extended.py tests/unit/test_set_naming.py -v 2>&1 | tail -40
```

Must end with all tests `PASSED`. If any fail:

- **`test_set_naming.py::*`**: the slug does not read as a sentence (code token, trailing qualifier, plural category, one-word leaf) — rename the directory, do not touch the test.

- **`test_file_meets_min_entries`**: file has fewer than 10 entries — add more.
- **`test_no_duplicate_names_within_file`**: duplicate name — rename one.
- **`test_probability_in_curated_range`**: a probability is < 0.3 or > 1.0 — bump it up to 0.4 or remove the line.
- **`test_non_atmosphere_name_has_at_least_two_words`**: a name is one word — expand it descriptively.
- **`test_no_banlist_violations`**: an outdoor token leaked into an indoor set or vice versa — replace the entry.
- **`test_set_has_indoor_or_outdoor_prefix`**: slug doesn't start with `indoor/` or `outdoor/` — rename the directory.

Fix the offending file directly, do NOT skip the test.

### Step 5b: Krea 2 lint + render check (mandatory for new categories)

```bash
cd D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools && \
  D:/AI/ComfyUI/ComfyUI/venv/Scripts/python.exe -m pytest tests/unit/test_location_krea2_rules.py -q 2>&1 | tail -30
```

Then render the set with `/krea2-set-test` (2-3 seeds × every leaf, one business outfit) and **look at the sheet yourself**. A set passes when, on every seed:
- exactly one person, head and shoes in frame;
- props and details read small and far, nothing oversized beside the subject;
- the background reads as the named place (deep, not a flat wall);
- no gibberish text, no smeared floor patches.

Fix the entries that caused a failure (the JSONL log next to the sheet has the exact prompt per tile) and render again. A sub-agent's "looks good" is not evidence; the sheet is.

### Step 6: Run upstream regression tests

```bash
cd D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools && \
  D:/AI/ComfyUI/ComfyUI/venv/Scripts/python.exe -m pytest tests/unit/test_smp_location.py -q
```

The new set is auto-picked up by `get_available_location_sets()` and exercised by the parametrized determinism + token tests. Must end fully green.

### Step 7: Loader smoke check

```bash
D:/AI/ComfyUI/ComfyUI/venv/Scripts/python.exe -c "from core.location_engine import get_available_location_sets, generate_location_records; from core.jb.sentences import location_intro; sets = get_available_location_sets(); print(sets); rec = generate_location_records(seed=42, location_set='<slug>'); print(rec['elements']['background']['name']); print(location_intro('<slug>'))"
```

Confirms the set is on disk, that the engine can pick a background entry from it, and shows the intro sentence the slug produces — read it once more.

### Step 8: Commit

```bash
cd D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools && \
  git add location_lists/<slug>/ && \
  git commit -m "feat(location): add <slug> location set

<one-line theme description>. 7 element files, all extended-validator tests green."
```

## Definition of Done

- [ ] Directory `location_lists/<scope>/<category>/<leaf>/` exists with all 7 files
- [ ] Slug is a 3-segment POSIX path starting with `indoor/` or `outdoor/`
- [ ] Each file has ≥10 entries; non-atmosphere files target 12-18
- [ ] `test_location_lists_extended.py` fully green
- [ ] Category listed in `KREA2_STRICT_CATEGORIES`, `test_location_krea2_rules.py` green
- [ ] Render sheet from `/krea2-set-test` inspected: one person, small far props, clean floor, no gibberish
- [ ] `test_smp_location.py` fully green
- [ ] Loader smoke-test prints the new slug AND samples a background entry
- [ ] Committed with descriptive message

## Orchestrated mode (Phase 3 — bulk category builds)

When called as a worker by an orchestrator agent (Phase 3 of the indoor_/outdoor_ rollout):

- Worker: build exactly one set following Steps 1-5 (skip 6-8 — orchestrator runs aggregate tests + commit at the end)
- Worker reports back: full list of every `name` field per file, plus pass/fail per Step 5 sub-check
- If orchestrator returns a delta request ("set <slug> overlaps >30% with <other-slug> on props — rebuild props with alternative theme variant"), rewrite ONLY the requested file and re-run Step 5

## Gotchas

- **Loader cache:** none. The engine scans disk every call to `get_available_location_sets()`. New sets appear in the UI on the next ComfyUI restart or after an SMP node refresh.
- **Atmosphere files use `-`:** time_of_day and weather entries set `coverage_range` and `texture` to `-`. Do NOT write a real range there — `_parse_coverage` returns `(0.0, 0.0)` for non-numeric, but the convention is `-`.
- **Workflow JSONs reference set names:** a saved workflow with an old slug fails ComfyUI's combo validation ("value not in list") until the set is re-picked in the UI. Prefer creating new sets over renaming shipped ones. When a rename is unavoidable, add a rule to `scripts/rename_sets_map.txt` and run `scripts/rename_sets.py --dry-run`, then without the flag — it `git mv`s the directory and rewrites every reference in code, tests, docs and generator scripts.
- **Probability semantics:** the engine weights entries by probability when sampling. Set highest-fit entries near 1.0 so the dominant scene reads consistently across seeds.
