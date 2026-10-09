# /build-outfit-set — FVMtools Outfit Set Builder & Verifier

Build a new outfit set in `outfit_lists/<female|male>/<category>/<leaf>/` with automatic doublette verification.

## Input

$ARGUMENTS

Expected form: a hierarchical slug + theme description, e.g.
- `female/elegant_leggings/sequin_glam sexy but not casual, high heels only`
- `male/streetwear/y2k_baggy Y2K vibes, baggy pants, chunky sneakers`

A plain theme + gender (`elegant leggings female, …`) is accepted too — derive the slug in Step 1 and confirm it before writing files.

## Background — why verification matters

The outfit engine's `_build_description()` in `core/outfit_engine.py:199` concatenates
`color_tag + fabric_name + garment_name + decoration` with spaces. If a garment name
contains a word that also appears in its fabric list, the output contains doublettes:

- `leather leggings` + fabric `leather` → `black leather leather leggings`
- `silk blouse` + fabric `silk` → `navy silk silk blouse`

The only engine-side suppression is `INVISIBLE_FABRICS = {"metal", "plastic", "rubber"}`.
All other fabrics are prepended verbatim. The validator must catch this empirically —
static inspection is not enough because `faux_fur` + `fur coat` → visual `fur fur`
even though token-match doesn't trigger on the underscore.

## Workflow — follow strictly in order

### Step 1: Parse theme — and read the slug aloud

From `$ARGUMENTS` extract:
- **slug**: `<female|male>/<category>/<leaf>`, lowercase snake_case per segment
- **formality_target**: rough range this set lives in (e.g. 0.4-0.7 for night_out-ish)
- **key garments**: what MUST appear in `bottom.txt` / `top.txt` / `footwear.txt`

**The slug is prompt text.** The JB `sentences` output format turns it into the first sentence of the outfit prompt, mechanically (`_` → space, no lookup table):

```
The outfit is a <category> look, in the <leaf> style.
```

The gender segment never appears (the outfit string describes clothes, not a person), and a leaf called `general` is omitted: `female/business/general` → *The outfit is a business look.* Write the sentence out with the proposed slug; if it does not read as plain English, fix the slug. Rules (enforced by `tests/unit/test_set_naming.py`):

- **category** — a noun phrase that fits *"a … look"*: `business`, `date_night`, `formal_event`, `boudoir_lingerie`, `dress_with_heels`, `teen`. Not `events`, not `dresses_heels`, not `lingerie_1940`. Prefer an existing category (`ls outfit_lists/female outfit_lists/male`) over inventing one.
- **leaf** — a noun phrase that fits *"in the … style"*: `grocery_run`, `first_date_heels`, `two_piece_set_with_heels`, `competition_meet_leotard`. Use `general` for the category's default set.
- **no abbreviations, no country codes**: `two_piece_set` not `twoset`, `leotard` not `leo`, `t_shirt` not `tshirt`, `german_` not `_de`.

### Step 2: Inspect a reference set

Read `outfit_lists/female/night_out/general/` (or `outfit_lists/male/casual/general/` etc.) to understand format.

Each set has 10 files:
- `top.txt` / `bottom.txt` / `footwear.txt` / `outerwear.txt` / `accessories.txt` / `bag.txt` / `headwear.txt` — format: `garment_name | probability | formality_min-formality_max | fabric1,fabric2,...`
- `fabrics.txt` — format: `fabric_name | formality | family | weight`
- `prints.txt` — format: `print_name | probability | compatible_slots | formality_min-formality_max`
- `texts.txt` — format: `text_value | probability | compatible_slots | font_or_descriptor`

`texts.txt` is the one file that used to get forgotten. A set without it silently
produces no lettering at all, no matter how the node is configured — `load_texts()`
returns `[]` and `_pick_decoration()` falls back to prints. Always write the file,
even when the answer is "no lettering here" (see Step 3b).

Valid fabric names come from the set's own `fabrics.txt`. Families: `natural`, `synthetic`, `animal`, `knit`, `embellished`, `accessories`.

### Step 3: Draft all 10 files

Target item counts per slot:
- top, bottom, footwear, outerwear, accessories: **10-20 items**
- bag: 8-15
- prints: 10-15
- fabrics: 8-14 (cover what the garments reference)
- headwear: 3-6 (usually low-probability, optional)
- texts: 4-12 (or a deliberately empty file — see Step 3b)

**Doublette prevention rule** — when drafting each row:
> For any `garment_name | ... | fab1,fab2,...`, none of `fab1, fab2, ...` may appear as a
> whole word (or part of a hyphenated/underscored compound that reads like a word) in
> `garment_name`.

Practical patterns:
- Prefer bare names (`leggings`, `blouse`, `pumps`) and let fabric add material.
- When a material is structurally part of the design, drop that fabric from the item's
  fabric list: `lace bodysuit | ... | jersey,satin` (no `lace`).
- Avoid `faux_fur` on items named `fur X` — rename to `plush X`.
- `patent leather X` is always a doublette with `leather` fabric — use `patent X`.

### Step 3b: texts.txt — the lettering rules

Two decisions, in this order.

**Decision 1 — does this set carry lettering at all?**

Ask whether a real person in this setting wears a garment with words on it. Campus,
sports, camp, beach-town souvenir, band merch, bachelorette, workwear: yes. Black-tie,
power suit, evening gowns, boudoir, sheer eveningwear: no. When the answer is no, still
write the file — with a reason specific to this set, not a copied boilerplate line:

```
# texts/slogans
# format: text_value | probability | compatible_slots | font_or_descriptor

# intentionally empty: black-tie eveningwear carries no printed lettering
```

**Decision 2 — real names first. Generic only where nothing real can be found.**

Real, verifiable names always take priority: real brands, real towns and beaches, real
schools and campuses, real clubs, resorts, lodges, marinas, regattas, tournaments,
venues and events. Research them and use them — that is what makes the rendered garment
look like a photograph instead of a mockup.

Invent a name only as a fallback, when research genuinely turns up nothing usable for
that context. Never invent one where a real name exists. A fallback should read like an
ordinary print, not like a counterfeit brand.

Generic wording without brand character needs no research and is always fine: `VARSITY`,
`CREW`, `TEAM BRIDE`, `EST. 1892`, `LIFEGUARD`. Those are ordinary prints, not marks.

Make the difference visible in the header, so a later reader can tell what is sourced
and what is filler: real names get a sourcing line, fallbacks get a short note saying
research found nothing verifiable for that slot.

Carry the evidence in the file header — one short line per wordmark or group, naming
company seat, association tie, typical merch, or campus/tournament affiliation. The
reference for this style is `female/gymnastics/competition_meet_leotard/texts.txt`:

```
# Sourcing notes — every wordmark below is a real, verifiable mark:
#  * GK Elite (legally Elite Sportswear, L.P.) is headquartered in READING, PA and is
#    the USAG Official Provider for the Development and Xcel programs through 2029.
"GK ELITE" | 0.30 | outerwear,bag | small embroidered lettering
```

**Format rules that bite silently:**
- Exactly 4 pipe fields. Fewer → `load_texts()` drops the line without warning.
- Field 1 keeps its quotes — they are rendered into the prompt as literal content.
- Field 3: only `headwear, top, bottom, footwear, outerwear, accessories, bag`, and only
  slots this set actually fills. Lettering on `footwear` is essentially always wrong;
  `bottom` is wrong for outerwear sets but correct for underwear (waistband text).
- Unlike `prints.txt`, text entries are **not** filtered by formality — an entry can
  surface at any formality value. Weight it via field 2 instead.

### Step 3c: Krea 2 rules for headwear, heels and realism (render-verified 2026-10-09)

These come from fixed-seed A/B renders on Krea 2 (`/krea2-set-test`, 4 seeds per variant):

- **Headwear is the hairstyle slot, and it keeps the face in the picture.** Write the style plus, at most, a small *colourless* hair accessory. End every entry with a face cue:
  `smooth low chignon at the nape, the face turned toward the camera | 0.9 | 0.6-1.0 | -`
  `soft side part with the hair tucked behind the ears, the face open toward the camera | 0.8 | 0.5-0.9 | -`
  `small claw clip holding a neat half-up twist, the face turned toward the camera | 0.7 | 0.4-0.8 | -`
  Measured: a bare style (`sleek low chignon`) turned 3 of 4 heads into profile to show the bun. The same style with the face cue gave 8 of 8 frontal faces, with the head fully in frame.
- **The hairstyle has to be drawn.** `enable_headwear` only allows the slot; the `style_preset` rolls it (`general` gives 15 %). For the `female/business/*_with_heels` sets use `style_preset = business_executive`, which always draws headwear and rolls outerwear 0.55, accessories 0.70 and bag 0.35.
- **Fabric `-` (or `none`) marks a hairstyle.** The engine then prints no palette colour and no fabric (`is_hairstyle()` in `core/outfit_engine.py`). Before this fix, the output read `cornflower none polished low chignon`, and Krea painted the colour into the hair.
- **No hair colour, no hair length** (`long`, `short`, `shoulder-length`, `bob`, `pixie`, `blonde`, `dark`…). The person prompt or LoRA decides those.
- **No headbands, and no coloured hair accessories.** `slim black silk headband` turned the hair grey in 4 of 4 seeds, and a colourless `slim headband` did it in 3 of 4. `tortoiseshell claw clip` gave grey-blonde streaks. Plain `small claw clip`, `slim barrette`, `thin hair pin` and `pearl hair pin` are fine.
- **Heels say "heel".** Write every footwear entry so the heel is explicit, as one clause: `pointed-toe pumps with a slim three-inch heel`, `slingback pumps with a block heel`. Never write a shoe close-up phrase (`focus on the shoes`), because Krea zooms in on it.
- **Realistic business means** knee-length or midi hems, no corset or bustier tops, nothing sheer or sequined outside an evening set, no pleated or plaid minis. The old `elegant_office_heels` produced exactly those, and they read as costume, not workwear.
- **No person words** in any garment string. Outfit strings describe clothes, and Krea paints an extra person otherwise.
- **ID badges and lanyards** count as implied writing. Write them as `blank white ID card` at low probability, or leave them out.

### Step 4: Create the directory + files

```bash
mkdir -p D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools/outfit_lists/<slug>
```

Write all 10 files. Include the `# slot: X` and `# format: ...` header comments that
match the reference set — `test_outfit_lists.py` checks these.

### Step 5: Run validator (mandatory, do not skip)

```python
# Paste this into a Bash Python call — adjust SET name
PYTHONIOENCODING=utf-8 D:/AI/ComfyUI/ComfyUI/venv/Scripts/python.exe << 'EOF'
import sys, re
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, "D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools")
from core.outfit_engine import generate_outfit

SET = "<female|male>/<category>/<leaf>"

def check_slot(slot_text):
    clean = re.sub(r"#[^#]+#", "", slot_text).strip().replace("_", " ")
    words = re.findall(r"[a-zA-Z][a-zA-Z-]*", clean)
    return [(words[i].lower(), " ".join(words[max(0,i-2):i+3]))
            for i in range(len(words)-1) if words[i].lower() == words[i+1].lower()]

issues = []
for preset in ("general", "night_out", "date_night"):
    for formality in (0.4, 0.5, 0.6, 0.7):
        for seed in range(1, 51):
            try:
                r = generate_outfit(seed=seed, outfit_set=SET, style_preset=preset,
                                    formality=formality, coverage=0.5)
            except Exception:
                continue
            body = re.sub(r"^wearing\s*", "", r["outfit_prompt"])
            slots = [s.strip() for s in body.split(",") if s.strip()]
            merged = []
            for s in slots:
                if s.startswith("with ") and merged:
                    merged[-1] += ", " + s
                else:
                    merged.append(s)
            for s in merged:
                for w, ctx in check_slot(s):
                    issues.append((preset, formality, seed, w, ctx, s))

print(f"600 outfits tested, {len(issues)} doublette issues")
seen = set()
for it in issues:
    k = (it[3], it[4])
    if k not in seen:
        seen.add(k)
        print(f"  '{it[3]}' in '{it[4]}' (preset={it[0]} f={it[1]} seed={it[2]})")
        print(f"    slot: {it[5]}")
EOF
```

### Step 5b: Verify texts.txt actually parses (mandatory)

A malformed line is dropped silently, so "I wrote the file" proves nothing. Check that
the parser sees what you wrote, and that lettering really reaches a prompt:

```python
PYTHONIOENCODING=utf-8 D:/AI/ComfyUI/ComfyUI/venv/Scripts/python.exe << 'EOF'
import sys
from unittest.mock import MagicMock
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, "D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools")
for m in ("comfy", "folder_paths", "server", "execution"):
    sys.modules.setdefault(m, MagicMock())
from core.outfit_lists import load_texts
from core.outfit_engine import generate_outfit_records

SET = "<female|male>/<category>/<leaf>"
entries = load_texts(SET)
print(f"{SET}: {len(entries)} entries parsed")

slots = {s: True for s in ("headwear", "top", "bottom", "footwear",
                           "outerwear", "accessories", "bag")}
hits = sum(
    1
    for seed in range(40)
    for g in generate_outfit_records(seed=seed, outfit_set=SET, slot_enables=slots,
                                     print_probability=0.5, text_probability=0.5,
                                     text_mode="quoted")["garments"].values()
    if "text in" in (g.get("decoration") or "")
)
print(f"text hits over 40 seeds: {hits}")
EOF
```

Expected: entry count matches the non-comment lines you wrote, and `hits > 0`. If you
wrote 9 lines and the parser reports 6, three lines have the wrong field count — fix
them, don't explain them away. For a deliberately empty set, `0 entries / 0 hits` is
the correct result.

### Step 6: Fix any collisions found

For each reported collision, edit the offending garment row:
- **Option A (preferred)**: rename garment to drop the material word
  (`leather leggings` → `leggings`)
- **Option B**: remove the colliding fabric from the item's fabric list
  (`lace bodysuit | ... | lace,jersey` → `lace bodysuit | ... | jersey`)
- **Option C**: rename the fabric (only if globally safe — usually don't).

Re-run Step 5 until validator reports **0 issues**.

### Step 7: Unit tests + commit

```bash
cd D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools && \
  D:/AI/ComfyUI/ComfyUI/venv/Scripts/python.exe -m pytest tests/unit/test_outfit_lists.py tests/unit/test_set_naming.py -q
```

Must end fully green (a `test_set_naming.py` failure means: rename the directory, not the test). Then:

```bash
cd D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools && \
  git add outfit_lists/<slug>/ && \
  git commit -m "feat: add <slug> outfit set

<one-line theme description>. Verified doublette-free across 600 outfit
permutations (3 presets x 4 formalities x 50 seeds)."
```

## Definition of Done

- [ ] Directory `outfit_lists/<female|male>/<category>/<leaf>/` has all 10 files — `texts.txt` included
- [ ] Slug reads as a sentence (*The outfit is a <category> look, in the <leaf> style.*) — `test_set_naming.py` green
- [ ] Each slot file has 10-20 items (headwear may have 3-6)
- [ ] `texts.txt` either has 4-12 entries that `load_texts()` actually parses, or is
      deliberately empty with a set-specific reason in the header
- [ ] Names in `texts.txt` are real wherever research could find them, and sourced in
      the header; any invented fallback is marked as such
- [ ] Validator reports **0 doublettes** over 600 permutations
- [ ] `test_outfit_lists.py` green (`50 passed`)
- [ ] Committed with descriptive message

## Gotchas

- **Validator false positives**: do NOT use a validator that strips punctuation tokens —
  it will flag phantom `leather leather` between adjacent slots separated by commas.
  The correct validator splits prompt by `,`, merges trailing `with X` decorations back
  into the preceding slot, then checks each slot in isolation.
- **`faux_fur` fabric**: engine renders this as `faux_fur` with underscore. Readers see
  `faux fur` + next word. If a garment is named `fur coat`, visual result is `faux fur fur coat`.
  Rename such garments (e.g. `plush coat`).
- **Shared fabrics.txt**: fabric names here must match what your garment rows reference.
  Missing fabrics silently fall through the harmony lookup.
- **Renaming a shipped set** breaks saved workflows (ComfyUI rejects the stale combo
  value until it is re-picked). Prefer new sets. If a rename is unavoidable, add a rule
  to `scripts/rename_sets_map.txt` and run `scripts/rename_sets.py --dry-run`, then
  without the flag — it `git mv`s the directory and rewrites every reference.
- **User sees prompt at inference time**: doublettes don't crash anything; they just
  read awkward. But they can confuse SD/SDXL tokenizers — `silk silk blouse` may bias
  toward texture over silhouette. Worth fixing.
