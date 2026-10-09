# /build-location-set — FVMtools Location Set Builder & Verifier

Build a new location set under `location_lists/<indoor|outdoor>/<category>/<leaf>/`, or rewrite an existing one in place to the Krea 2 rules, with automatic plausibility verification.

## Input

$ARGUMENTS

Expected form: a hierarchical slug (`indoor/<category>/<leaf>` or `outdoor/<category>/<leaf>`) plus a short theme description, e.g.
- `indoor/office/corporate_open_plan glass walls, monitors, ergonomic chairs`
- `outdoor/winter_vacation/alpine_lodge snow-packed mountain village, wooden chalets, golden hour`
- `outdoor/sports/soccer_pitch amateur soccer pitch in late summer, faded line paint`

Or `rewrite <slug> [<slug> …]` for sets that already exist: then follow **Rewrite mode** at the end; Step 1 is skipped because the slug stays.

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

Pick the closest set that is already covered by the Krea 2 lint (`KREA2_STRICT_CATEGORIES` or `KREA2_STRICT_EXTRA_SETS` in `tests/unit/test_location_krea2_rules.py`) and read every file in it. Copy its style, not its content. Sets outside the lint show the file format only: most of them are still object lists, which Krea 2 paints wrong.

- Indoor: `location_lists/indoor/luxury_hotel/hotel_bar/`, `location_lists/indoor/reading_pennsylvania_corporate_office/open_plan_marketing_department/`
- Outdoor: `location_lists/outdoor/american_everyday_scene/sidewalk_cafe_patio/`, `location_lists/outdoor/suburban_office_park/landscaped_front_entrance/`

Each set has 7 files:
- `background.txt` — large surface behind subject (wall, sky, distant scenery)
- `midground.txt` — secondary mass at distance (furniture group, treeline, kiosk)
- `foreground_element.txt` — floor/ground plane (mat, sand, trail, dirt)
- `architecture_detail.txt` — small features (window light, blaze paint, beam edge)
- `props.txt` — small objects in scene (cushion, water bottle, lamp)
- `time_of_day.txt` — atmospheric mood (golden hour, overcast, sodium dusk)
- `weather.txt` — atmospheric condition (drizzle, haze, calm interior air)

### Step 2b: Research first (one agent per category)

Before drafting a new category, run one web-research agent per category (efficient model, e.g. Sonnet). The agent collects real anchors (real buildings, materials, views, what is closed or demolished), proposes 5-6 leaves and writes short visual phrases per element plus an avoid list with sources. It does not write set files. Brief it with the place and "realistic today", and with the Step 2c rules that shape the phrases: no people words, no implied writing (but real brand wordmarks are fine, see rule 7), a position for every phrase, props small only, no season outside `weather`.

### Step 2c: Krea 2 composition rules — mandatory for every new or rewritten set

These rules come from fixed-seed A/B renders on Krea 2 (`/krea2-set-test`, see Provenance at the end), plus the Krea 2 docs. `tests/unit/test_location_krea2_rules.py` enforces them, but only for the sets it knows. **Register the set there before drafting,** so the lint guides the draft:
- a new category → `KREA2_STRICT_CATEGORIES`;
- a single set in a category that still holds sets nobody rewrote → `KREA2_STRICT_EXTRA_SETS`. Never put such a category into `KREA2_STRICT_CATEGORIES`, its old sets would all turn red.

What the renders showed: Krea 2 paints every noun it reads, and it paints it large and next to the subject. In the old object-list style, a mug in `props` sat beside her hand, a lamppost in `architecture_detail` stood at her shoulder, and anything in `foreground_element` filled the lower third. Changing the lead-in sentence ("Small props far in the background include …") made no difference, so the entry text itself has to carry the placement.

1. **Background carries the scene (90-260 chars).** Name the far mass, then describe what spans the frame, with a position in the image:
   `far side of Penn Street | … | a continuous row of three- and four-storey brick and limestone commercial buildings across the whole width of the image, a wooded ridge rising above the rooftops on the left`.
   One area sentence beats a list of objects. **Lead with the named place itself**: in `dollar_general_storefront` a field and a pickup in the background covered the shop until the yellow-and-black storefront came first.
2. **Midground gets a distance (60-210 chars):** `rows of empty white desks a few meters back on the left | … | black mesh task chairs`. Use `empty` for furniture that otherwise invites people.
3. **Architecture detail and props are small, few and far (≤120 / ≤115 chars).** Every prop has a size word (`small`, `tiny`, `slim`, `compact`) and a far placement (`far down the block`, `on a distant cabinet`, `at the back`). Props that are big by nature (bike rack, planter, trash can) are not props; describe them as part of the midground at a distance, or leave them out.
3b. **Size words belong to the object, never float free.** `small trash can far down the sidewalk, tiny in the distance` painted a child-sized figure on the sidewalk in 4 of 4 seeds; `small dark green trash can at the curb far down the block` did not. Banned: `tiny in the distance`, `tiny in the frame`, `small in the distance`.
4. **Foreground is the ground plane only (≤100 chars)**: floor, pavement, lawn, deck. Never an object: the engine writes "In the foreground is …" and Krea makes it huge. Plain matte surfaces render clean: low-pile carpet, polished concrete, wide oak planks, brick pavers. The lint bans busy micro-patterns (`carpet tile`, `mosaic`, `checkered`); the rewrite also dropped dust, flour, rugs, painted lines, gloss and reflections. Those smears came mostly from the doubled turbo LoRA (see `/krea2-set-test`, Known model effects), so a smear on `--preset cielbleu` says nothing about the text. Lawns: `plain mown lawn` renders as flat synthetic green; write `short natural grass` with a blade texture.
5. **Relative positions are subject-neutral:** `far behind`, `on the left of the frame`, `a few meters back`, `across the street`. Never `her` or `his`, because the sets are reused for every subject.
5b. **Seating says it is empty.** A seating group without `empty` gets occupied: `low cognac leather chairs around a plain wood table` put two seated guests into a coffee-roaster render, and the same entry with `, all empty` did not. Every background or midground entry that names chairs, booths, banquettes, stools, benches, sofas or tables carries `empty` (or `pushed in`, `set for`, `stacked`).
6. **No people words**, including `busy`, `lively`, `crowded`, roles such as `resident`, `exhibitor` or `diner`, and pronouns. The lint matches whole words, so `diner` and `pedestrian` are out even as buildings or places: write `restaurant`, and leave `pedestrian` out of `pedestrian_zone` (the slug may keep it). Krea draws them. Krea's own blog: interiors get people unless the empty frame is stated. Use `empty`, `set for`, `quiet`.
7. **No implied writing.** Signs, menus, screens, donor walls, whiteboards, brochures and directory boards turn into gibberish. Either leave them out, or make them `blank` / `unlit` / `abstract` / `pictogram`, or give the exact short wording in quotes, with options: `exit sign reading "{EXIT|WAY OUT}"`, `banner over the doorway{| reading "{CONGRATS|GRAD}"}` (the empty option leaves the banner blank).
   - Real brands are fine (`Wawa`, `Sheetz`, `Target`). Quote only the lettering the real logo carries (`"WAWA"`). A logo without lettering is described, not quoted (`red bullseye logo`).
   - Quotes always come in pairs and hold short words only. The JB `sentences` output passes them through as content, and `test_jb_sentences.py` rejects an unpaired `"` or a `":` that looks like a JSON key.
8. **No parentheses and no `#`.** `CLIPTextEncode` parses `( )` as weights, and `#…#` is reserved for the engine's atmosphere tokens.
9. **Budget:** the longest possible draw over all seven elements must stay ≤ 950 chars. With lead-ins, outfit and person, the whole prompt then stays under ~2400 chars. Beyond about 640 tokens, Krea 2 in ComfyUI produces black or noisy images (ComfyUI issue #14782).
10. **Time of day / weather ≤ 60 chars**, plain photographic light (`soft overcast late-morning light`), and realistic for the place across the seasons (most sets are in Pennsylvania). Indoor "weather" means the light and air through the windows, not a joke (`humid air the AC can't keep up with` is out).
11. **Seasons live in `weather.txt` only.** Background, midground, details and props stay season-neutral: no snow, autumn leaves, Christmas lights or summer haze there. Every element is drawn independently per seed, so a snowy background lands next to a midsummer weather line.

### Step 2d: `{a|b|c}` variants

Entry names and textures may carry variants: `{EXIT|WAY OUT}`, weights `{%3%oak|maple}`, an empty option `{|…}`, nesting. Pipes inside braces do not split fields (`core/list_braces.py`). The engine resolves them per seed with its own salt, so the shared RNG stream and determinism stay unchanged. The lint counts the longest option against the length caps. Use variants for small wording changes (material, colour, the quoted word), not to pack two different scenes into one line.

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

### Step 5b: Krea 2 lint + render check (mandatory for every new or rewritten set)

```bash
cd D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools && \
  D:/AI/ComfyUI/ComfyUI/venv/Scripts/python.exe -m pytest tests/unit/test_location_krea2_rules.py -q 2>&1 | tail -30
```

Then render the set with `/krea2-set-test` (2-3 seeds × every leaf, one fixed business outfit) and **look at the sheet yourself**:

```bash
D:/AI/ComfyUI/ComfyUI/venv/Scripts/python.exe scripts/krea2_set_check.py \
  --location "<scope>/<category>/*" \
  --outfit female/business/office_day_business_casual_with_heels \
  --seeds 11 22 --accessories --bag --port 8189 --out "<scratchpad>/<category>.jpg"
```

A set passes when, on every seed:
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

### Step 8: Full suite + commit

The unit suite is fully green, so any red test after your change is yours:

```bash
cd D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools && \
  git status --short && \
  D:/AI/ComfyUI/ComfyUI/venv/Scripts/python.exe -m pytest tests/unit -q -p no:cacheprovider 2>&1 | tail -3
```

Other sessions may be working in the same repo. If `git status` shows changes you did not make, leave them alone, add only your own paths (never `git add -A`), and if their work-in-progress turns the suite red, check your own commit on a clean export (`mkdir -p <scratchpad>/clean && git archive HEAD | tar -x -C <scratchpad>/clean`, then run pytest there).

```bash
git add location_lists/<slug>/ tests/unit/test_location_krea2_rules.py && \
  git commit -m "feat(location): add <slug> location set

<one-line theme description>. 7 element files, Krea 2 lint green, render-checked on <seeds>: <one-line verdict>."
```

## Definition of Done

- [ ] Directory `location_lists/<scope>/<category>/<leaf>/` exists with all 7 files
- [ ] Slug is a 3-segment POSIX path starting with `indoor/` or `outdoor/`
- [ ] Each file has ≥10 entries; non-atmosphere files target 12-18
- [ ] `test_location_lists_extended.py` fully green
- [ ] Set registered in the Krea 2 lint (new category → `KREA2_STRICT_CATEGORIES`, single set → `KREA2_STRICT_EXTRA_SETS`), `test_location_krea2_rules.py` green
- [ ] Render sheet from `/krea2-set-test` inspected: one person, small far props, clean floor, no gibberish
- [ ] `test_smp_location.py` and the full `tests/unit` suite green
- [ ] Loader smoke-test prints the new slug AND samples a background entry
- [ ] Committed with explicit paths and a descriptive message

## Rewrite mode — existing sets in place

For older sets in the object-list style. Typical faults of the old versions: object lists instead of an area sentence, objects in `foreground_element`, people and role words, joke "weather", implied writing, seasonal details outside `weather`.

1. **Keep the slug and the file names.** Saved workflows reference the slug; a rename breaks them (see Gotchas). Keep the theme, make it realistic today.
2. **Register each set in `KREA2_STRICT_EXTRA_SETS`** before rewriting it. Its category stays out of `KREA2_STRICT_CATEGORIES` as long as it holds sets nobody rewrote.
3. **Rewrite all 7 files** to Step 2c/2d and the counts in Step 3, then run Steps 5 and 5b.
4. **Disarm the generator scripts.** `scripts/gen_*.py` (`gen_us_scenario_locations.py`, `gen_everyday_locations.py`, `gen_everyday_sets.py`, `gen_vacation_water_locations.py`, `gen_private_spaces_locations.py`) may still hold the set's old data, and a run with `--force` would write it back. Remove the set's block and leave a comment in its place, like the one for `indoor/american_office` in `gen_us_scenario_locations.py`.
5. **Commit per group:** `refactor(lists): rewrite <group> sets to the Krea 2 rules`, listing the sets and the render verdict.

### Agent waves (many sets at once)

- At most 6 sub-agents (Sonnet) in parallel, each with a group of 2-6 sets.
- Before a wave, the orchestrator registers the wave's sets in the lint itself.
- Each agent gets: its set paths, "rewrite in place, keep the theme, realistic today", this file (Step 2c/2d) and `/krea2-set-test`, the old faults listed above, the rule "write only in your own set folders, do not commit", and its validation command:
  `pytest tests/unit/test_location_krea2_rules.py tests/unit/test_location_lists_extended.py tests/unit/test_set_naming.py tests/unit/test_smp_location.py -q -p no:cacheprovider -k "<set1> or <set2>"`
- Generator blocks, the render check, the verdict and the commit stay with the orchestrator. An agent's "lint green" says nothing about the render.

## Gotchas

- **Loader cache:** none. The engine scans disk every call to `get_available_location_sets()`. New sets appear in the UI on the next ComfyUI restart or after an SMP node refresh.
- **Atmosphere files use `-`:** time_of_day and weather entries set `coverage_range` and `texture` to `-`. Do NOT write a real range there — `_parse_coverage` returns `(0.0, 0.0)` for non-numeric, but the convention is `-`.
- **Workflow JSONs reference set names:** a saved workflow with an old slug fails ComfyUI's combo validation ("value not in list") until the set is re-picked in the UI. Prefer creating new sets over renaming shipped ones. When a rename is unavoidable, add a rule to `scripts/rename_sets_map.txt` and run `scripts/rename_sets.py --dry-run`, then without the flag — it `git mv`s the directory and rewrites every reference in code, tests, docs and generator scripts.
- **Probability semantics:** the engine weights entries by probability when sampling. Set highest-fit entries near 1.0 so the dominant scene reads consistently across seeds.
- **Tests must not pin set data.** A test that expects one exact entry or one flat slug breaks with the next rewrite. Check behaviour over many draws instead (every draw has a position, no draw has a people word).
- **Backslash-b in a file written through a tool call** becomes a backspace character. Build a regex word boundary with `chr(92)` and check with `grep -c $'\x08'`.

## Provenance (dated)

- 2026-10-07: implied-writing pass over all location entries (8e668bb, quoted wording with `{A|B}` options, real brands kept, user rule: quote only lettering a logo really has) and the brace-aware loader (01b4e4e).
- 2026-10-09: six fixed-seed A/B tests on Krea 2 Turbo with cielbleu + turbo LoRA and on stock Turbo gave rules 1-10; 42 Reading, PA business sets built to them (1879d83), 9 older office, cafe and hotel sets rewritten (3b3223a).
- 2026-10-10: 56 older everyday sets rewritten in two agent waves and render-checked (4df1176 indoor, 4ee68f1 outdoor); source of rule 11, the wave procedure, the lawn and lead-with-the-place notes. The floor smears of that run were traced to the turbo LoRA applied twice on the cielbleu checkpoint, hence the render default `cielbleu_native`.
