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
