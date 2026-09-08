# JB for Krea 2 — natural prompts, colour moods, everyday sets

The JB suite was built for Ideogram-4-style JSON prompting, where keys and
structure are part of the instruction. Krea 2 (and any Qwen3-VL-based text
encoder) works the other way round: it *reads* the JSON. Braces, key names,
seeds and coverage numbers all end up described in the image or diluting the
tokens that matter.

This page covers the three additions that make the same nodes usable for
Krea 2, and the everyday sets that go with them.

---

## 1. `output_format: natural`

`JB · Outfit Block` and `JB · Location Block` both have an `output_format`
widget. The structured formats are unchanged; `natural` is new.

| format | output |
|---|---|
| `loose_keys` | `outfit: {upper_body: {name: crew tee, fabric: jersey, color_role: primary, …}}` |
| `pretty_json` / `compact_json` | strict JSON |
| **`natural`** | `wearing sand jersey crew tee, black denim straight-leg jeans` |

`natural` keeps only the phrases a text encoder can use — the resolved
`prompt_fragment` of every element — and drops everything else: `seed`,
`set_name`, `coverage`, `color_role`, `formality`, `layer`, all keys, all
braces. Duplicate phrases collapse, so two garments that resolved to the same
fragment appear once.

The `location_json` / `outfit_json` outputs are unaffected — they always carry
the full structure, so a Stitcher downstream still sees everything.

Rule of thumb: **`natural` for Krea 2 / Flux / SDXL, `loose_keys` for
Ideogram 4.**

### `output_format: sentences`

`natural` throws the keys away. `sentences` keeps what they *mean* and writes
one sentence per fact, so a language-reading encoder knows which phrase is
the background and which one is the shoes:

```
The outfit is an everyday look, in the grocery run style. The top is grey
cotton striped long sleeve top. The bottom is navy denim straight-leg jeans.
The footwear is charcoal grey canvas simple slip-on trainers. The bag is
taupe canvas reusable shopping bag.

The scene takes place indoors, it is a family event, namely the graduation
party at home. The background is living room wall with graduation banner,
painted wall behind hung sign, illuminated by even overcast daylight. The
middle ground shows framed school photos on side table, assorted picture
frame cluster. The props include small congratulations balloon weight,
ribbon-tied foil accent. In the foreground is plate of snacks close up,
small appetizer selection on plate, in soft neutral grey shadows. The time
of day is early evening lit interior. The weather is bright daylight through
windows.
```

Everything is mechanical and deterministic (`core/jb/sentences.py`):

- **Intro from the set path.** `indoor/family_event/graduation_party_at_home`
  becomes *indoors · a family event · the graduation party at home*;
  `female/business/dress` becomes *a business look, in the dress style*.
  Underscores turn into spaces, the article follows the vowel rule, the
  gender segment is dropped — the outfit string describes clothes, never a
  person (Krea 2 paints an extra one otherwise, see [K2 Lab](K2_LAB.md) §4).
  A leaf called `general` is omitted. Because the wording *is* the directory
  name, set slugs follow naming rules — see `/build-location-set` and
  `/build-outfit-set`.
- **One lead-in per key.** `background → The background is …`,
  `midground → The middle ground shows …`, `props → The props include …`,
  `foreground_element → In the foreground is …`, `upper_body → The top is …`,
  `lower_body → The bottom is …`, `accessories → The accessories are …`.
  Unknown keys read `The <key> is …`.
- **One-piece garments.** Dress sets keep the dress in the bottom slot. A
  garment whose head noun is a dress, gown, jumpsuit, romper, leotard,
  swimsuit, chemise, kaftan … is written as `The one-piece garment is …`
  instead of `The bottom is …`. Head noun only, so *dress shirt* and
  *bikini top* stay two-piece.
- **Metadata is dropped**, as in `natural` — plus `formality` and
  `color_tone`, which say nothing the garment phrases don't already say.
- **Anything else** (Builder rows, hand-written JSON) gets one sentence per
  branch: `The face has age twenties and eyes amber. The hair has colour blonde.`
- **Stitcher** puts the title in front instead of turning it into a sentence:
  `character_1: The outfit is … The scene takes place …`. Bare prose inputs
  pass through as their own sentence.

### Fragment cleanup

Independent of the format, the outfit engine no longer emits contradictory
fragments. Previously possible, now suppressed:

- `charcoal-gray canvas white tennis shoes` — colour prepended to a garment
  whose name already names a colour
- `pistachio-green denim basic denim jacket` — fabric repeated
- `crew tee with solid color` — a decoration that says nothing
- `navy - bare feet` / `grey bare feet` — colour and fabric on items that
  have neither
- `warm gold knit simple scarf` — textile accessories inheriting the
  `accessories` slot's metallic colour role

---

## 2. `color_mood` instead of five sliders

The harmony engine (`harmony_type`, `palette_style`, `vibrancy`, `contrast`,
`warmth`, `num_colors`) is still there, but it is now optional and hidden
behind `color_mood: auto`. The default is a single dropdown:

| mood | what it does |
|---|---|
| `auto` | the old harmony engine, driven by the sliders |
| `everyday_muted` | washed everyday colours — the default |
| `neutral_basics` | only neutrals: sand, stone, off-white, taupe, charcoal |
| `warm_earth` | terracotta, olive, ochre, rust |
| `cool_muted` | slate, dusty blue, sage, grey-green |
| `denim_casual` | neutrals with a denim tone on the secondary role |
| `one_accent` | one accent on the top, everything else neutral |
| `monochrome` | shades of a single base |
| `soft_pastel` | pale, low-saturation |
| `bold` | presets the sliders for high vibrancy/contrast |

Two kinds of mood: **pool moods** pick concrete colour names per role
(deterministic per seed), **engine moods** (`bold`, `auto`) just preset the
sliders. Either way `warmth` still applies, because it drives the ambient
light and shadow phrasing rather than the garment colours.

The `denim` pool deliberately contains no entry with the word "denim" in it —
otherwise `denim_casual` produced `dark denim blue denim jeans`.

---

## 3. Everyday sets

The shipped sets skew towards studio and editorial looks; the `everyday_us`
locations are good but US-specific (Walmart, CVS, Little League). These are
the plain-clothes counterpart.

**`outfit_lists/female/everyday/`** — 8 sets:

```
grocery_run      home_lounge      school_run       rainy_errands
gym_commute      weekend_market   office_casual    evening_walk
```

**`location_lists/indoor/everyday_de/`** and
**`location_lists/outdoor/everyday_de/`** — 8 sets:

```
indoor:   kitchen_cooking  living_room_sofa  bathroom_mirror  supermarket_aisle_de
outdoor:  apartment_balcony  bus_stop_de  pedestrian_zone  playground_de
```

Sample output with `output_format: natural`, `color_mood: everyday_muted`:

```
grocery_run       wearing white cotton plain long sleeve tee, beige zip-up fleece
                  jacket, beige cotton comfortable jogger trousers, navy leather
                  flat ankle boots
home_lounge       wearing sand fleece soft hoodie, beige fleece soft jogging
                  bottoms, dusty rose knit thick house socks
rainy_errands     wearing grey thin knit jumper, navy denim jeans with damp hems,
                  charcoal grey softshell waterproof trainers, taupe polyester
                  shoulder bag held close

bus_stop_de       bare street trees along the kerb, thin trunks, push bike leaning
                  on the shelter, mud on the tyres, stickers on the shelter glass,
                  coat collar pulled up, grey commuter morning, light drizzle
apartment_balcony distant tram wires, drying rack on the balcony, folding aluminium
                  frame, ashtray on the railing, cardigan pulled around the
                  shoulders, grey commuter morning, light drizzle in the air
```

### Regenerating / extending

Both sets are generated from checked-in scripts, so they stay editable as
data rather than by hand:

```bash
python scripts/gen_everyday_sets.py           # outfits
python scripts/gen_everyday_locations.py      # locations
# add --force to overwrite existing files
```

Location files must satisfy the curation rules in
`tests/unit/test_location_lists_extended.py` — at least 10 entries per file,
probability in `[0.3, 1.0]`, names of two words or more, no duplicates, and
no indoor/outdoor token bleed. The generator checks the entry count itself and
refuses to write an under-filled set.

One trap worth knowing: the indoor banlist matches on substrings, and
`fridge` contains `ridge`. Use `refrigerator`.

### Archived sets

The editorial categories (skyscraper lobbies, ice hotels, Star Trek uniforms,
paragliding launches) moved to `location_lists/_archive/` and
`outfit_lists/female/_archive/`. Discovery skips underscore directories, so
they disappear from the dropdowns but stay on disk — move a folder back out
of `_archive/` to reactivate it. Their US/PA everyday replacements live in
`indoor/{office_us, fitness_us, private_us, vacation_us}` and
`outdoor/{suburb_pa, town_pa, nature_pa, fitness_us}`
(generator: `scripts/gen_us_scenario_locations.py`), plus the outfit
categories `female/{dresses_heels, dresses_flats, underwear}`
(generator: `scripts/gen_dresses_underwear_sets.py`).

`dresses_heels` is the leg-showing-with-heels group, `dresses_flats` the
covered/flat counterpart — both run from everyday office to grocery runs
rather than only fancy occasions. `underwear` is deliberately non-fancy
(cotton basics, t-shirt bras, sleep sets, laundry-day mismatch) with a single
slightly nicer `simple_lace_touch`.

### Private spaces (signage-free)

`indoor/private_spaces_us` and `outdoor/private_spaces_us`
(generator: `scripts/gen_private_spaces_locations.py`) are quiet spots with
no public traffic — parking garage level and roof deck, mall lot far corner,
home garage, spare room, office stairwell and supply room, store stockroom,
school gym storage and back field, side yard, behind the shed, office
courtyard, loading dock after hours.

Their defining rule: **nothing with lettering on it**. Text encoders render
every mentioned sign, poster or label as (usually broken) readable text in
the image, so these sets avoid such objects at the data level. The generator
enforces a signage banlist on top of the standard curation rules — substring
matched, which is stricter than it looks: "assigned" contains "sign",
"cartridge" contains "ridge", "trailer" contains "trail".

---

## 4. Overrides and forcing palette colours

`JB · Outfit Block` has an **Edit Overrides** button that opens a structured
editor for the `overrides` widget: per-slot rows (auto / custom / exclude,
fabric, garment, colour role, decoration) and a palette section. Everything
serialises to the plain-text grammar, so hand-written text round-trips:

```
top: silk blouse | accent | floral print
bag: exclude
palette: primary=navy blue, secondary=cream, accent=burnt orange
```

The `palette:` line is new — it forces the actual colour behind a role for
this node. Garments keep their roles (top=primary, bottom=secondary,
footwear=neutral, accessories=metallic, headwear/bag=accent); you swap what
the role resolves to. Unlisted roles keep the mood/harmony colour, unknown
role names are ignored, and the `palette_summary` output notes what was
overridden. `ambient_light` and `shadow_tone` are also accepted.

Dress sets keep a `none` stub in `top.txt` (the dress lives in the bottom
slot). The engine drops placeholder garments (`none`, `-`) instead of
emitting `#primary# none`.

`JB · Location Block` has the same **Edit Overrides** button. Element lines
force a phrase verbatim (it may contain palette tokens), `exclude` drops an
element even when its enable toggle is on, and a forced element is emitted
even when its toggle is off:

```
background: red brick wall with ivy
props: exclude
time_of_day: golden hour before sunset
palette: ambient_light=dim tungsten evening, shadow_tone=inky shadows
```

For locations the palette line is mostly about `ambient_light` and
`shadow_tone`; the garment roles apply when a list entry embeds a colour
token. Forced and excluded elements consume no RNG (same rule as disabled
elements), so sibling elements may re-roll when an override is added.

### Beach variants and winter vacation spots

`outdoor/beach_variants` (generator: `scripts/gen_vacation_water_locations.py`)
covers the shoreline spectrum — sandy ocean beach in a **busy** and a
**private** variant, rocky cove (private) and tidepool point (busy), public
lake beach and private lake cove, river bend beach, resort pool (busy),
private backyard pool, and `hotel_pool_after_dark` as an explicit night set.
`outdoor/vacation_winter_us` adds the private winter counterparts:
cabin_hot_tub_deck, frozen_lake_shore, snowed_in_cottage_yard,
empty_ski_slope_edge.

**Busy vs. private** are separate sets (different content). **Day vs. night**
lives in each set's `time_of_day` pool — force it with an override
(`time_of_day: moonlight silvering the water`). Night objects are phrased
time-neutrally ("fire ring of blackened stones" works day and night), and
both categories follow the signage-free discipline.

When a night phrase is drawn (or forced), the Location Block now swaps the
daylight atmosphere phrases for night ones automatically — without this,
a natural night draw still rendered "illuminated by balanced natural
daylight". Explicit `palette: ambient_light=…` overrides win over the sync,
and the palette summary notes `[night atmosphere]` when it fired.

### Why weather says nothing about light

`time_of_day` and `weather` are drawn independently, so any light-level claim
in a weather entry can contradict the time entry:

```
evening under the ceiling light, bright sun through the window
blue hour after sunset, hazy warm afternoon
```

In these sets, weather entries describe precipitation, air and temperature
only — `rain streaks on the glass`, `damp air after rain`, `cold air with
breath visible`. Light belongs to `time_of_day` alone.
