# FVM Outfit Generator -- Complete User & Creator Guide

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Node Inputs -- Detailed Reference](#node-inputs----detailed-reference)
- [Override String Format](#override-string-format)
- [Creating Custom Outfit Sets](#creating-custom-outfit-sets)
- [Fabric System](#fabric-system)
- [Available Outfit Sets](#available-outfit-sets)
- [Slot System](#slot-system)
- [Tips and Tricks](#tips-and-tricks)

---

## Overview

The **Outfit Generator** is a ComfyUI node that produces seed-controlled, randomized outfit descriptions for use in image generation prompts. Instead of manually writing clothing descriptions, you configure a few parameters and let the engine assemble a complete outfit from curated garment lists.

### How it works

The node outputs a prompt string containing color placeholder tags (like `#primary#`, `#secondary#`) that are designed to be resolved by the **Prompt Color Replace** node using colors from the **Color Palette Generator**.

**The standard pipeline:**

```
Color Palette Generator  -->  Prompt Color Replace  -->  CLIP Text Encode
                                      ^
                              Outfit Generator
                          (outfit_prompt output)
```

1. **Outfit Generator** picks garments, fabrics, and assigns color tags per slot.
2. **Prompt Color Replace** swaps `#primary#`, `#secondary#`, etc. with actual color names from a palette.
3. **CLIP Text Encode** receives a fully resolved prompt like `"wearing navy blue silk blouse, charcoal wool trousers, cream leather heels"`.

### The concept

Each outfit is deterministic for a given seed: the same seed + settings always produces the same outfit. Change the seed to explore variations. The color tags remain abstract until resolved by Prompt Color Replace, so you can pair any outfit with any color palette.

---

## Installation

The Outfit Generator is part of the **comfyui-FVMtools** custom node pack. No separate installation is needed.

### Requirements

- ComfyUI (any recent version)
- Python 3.10 or later
- No extra pip dependencies -- the outfit system is pure Python using only the standard library

### Steps

1. Clone or download `comfyui-FVMtools` into your ComfyUI `custom_nodes/` directory:
   ```bash
   cd /path/to/ComfyUI/custom_nodes/
   git clone https://github.com/your-repo/comfyui-FVMtools.git
   ```
2. Restart ComfyUI.
3. The node appears as **"Outfit Generator"** under the **FVM Tools/Fashion** category in the node menu.

---

## Quick Start

1. **Add the "Outfit Generator" node** (right-click canvas > FVM Tools > Fashion > Outfit Generator).
2. **Select an outfit set** from the dropdown (e.g., `business_female_skirt`, `streetwear_male`).
3. **Set a seed** -- any integer. The same seed always produces the same outfit.
4. **Leave slot toggles at defaults** -- `top`, `bottom`, and `footwear` are enabled by default.
5. **Connect the `outfit_prompt` output** to the `prompt` input of a **Prompt Color Replace** node.
6. **Connect a Color Palette Generator** to the `palette_string` input of Prompt Color Replace.
7. **Connect Prompt Color Replace's output** to a **CLIP Text Encode** node.
8. Queue the prompt. The outfit is generated, colors are filled in, and the result is a natural-language clothing description.

### Minimal 3-node chain

```
[Color Palette Generator]
        |
        | palette_string
        v
[Prompt Color Replace] <--- outfit_prompt --- [Outfit Generator]
        |
        | prompt
        v
[CLIP Text Encode]
```

---

## Node Inputs -- Detailed Reference

### Required Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `outfit_set` | Dropdown | (first set) | Which garment list collection to use. Auto-detected from subdirectories in `outfit_lists/`. |
| `seed` | INT | 0 | Seed for deterministic random generation. Range: 0 to 4,294,967,295. |
| `style_preset` | Dropdown | (first preset) | Influences slot probabilities, formality range clamping, and preferred fabric families. |
| `formality` | FLOAT | 0.5 | 0.0 = casual, 1.0 = formal. Filters which garments are eligible. Step: 0.05. |
| `coverage` | FLOAT | 0.5 | 0.0 = minimal, 1.0 = maximal. Adjusts the probability of optional slots appearing. Step: 0.05. |
| `enable_headwear` | BOOLEAN | False | Whether the headwear slot can produce output. |
| `enable_top` | BOOLEAN | True | Whether the top slot can produce output. |
| `enable_outerwear` | BOOLEAN | False | Whether the outerwear slot can produce output. |
| `enable_bottom` | BOOLEAN | True | Whether the bottom slot can produce output. |
| `enable_footwear` | BOOLEAN | True | Whether the footwear slot can produce output. |
| `enable_accessories` | BOOLEAN | False | Whether the accessories slot can produce output. |
| `enable_bag` | BOOLEAN | False | Whether the bag slot can produce output. |

### Optional Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `override_string` | STRING (multiline) | `""` | Power-user override for specific slots. See [Override String Format](#override-string-format). |
| `prefix` | STRING | `"wearing "` | Text prepended to the outfit prompt. |
| `separator` | STRING | `", "` | Text between individual garment descriptions. |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `outfit_prompt` | STRING | The assembled outfit description with color tags, e.g., `"wearing #primary# silk blouse, #secondary# wool trousers, #neutral# leather heels"`. |
| `outfit_details` | STRING | Structured data string. Format: `slot:garment:fabric:color_tag` separated by `\|`. Useful for debugging or downstream processing. |
| `outfit_info` | STRING | Human-readable summary: seed, style, formality, coverage, active slots, and chosen pieces. |

### How formality works

The `formality` slider filters the garment pool. Each garment in the list files has a formality range (e.g., `0.0-0.3` for a t-shirt, `0.5-1.0` for a dress shirt). Only garments whose range includes the current formality value are eligible. If the style preset constrains the formality range, the effective formality is clamped to that range.

### How coverage works

The `coverage` slider modifies the probability of optional slots (headwear, outerwear, accessories, bag) appearing. Each style preset defines base probabilities for these slots. The effective probability is:

```
effective_prob = base_prob * (0.5 + coverage)
```

- Coverage 0.0: optional slots appear at 50% of their base probability.
- Coverage 0.5: optional slots appear at 100% of their base probability (no change).
- Coverage 1.0: optional slots appear at 150% of their base probability (capped at 1.0).

The core slots (`top`, `bottom`, `footwear`) are always active when enabled -- they are not subject to probability rolls.

---

## Override String Format

The `override_string` input lets you manually control individual slots without changing the outfit set files. Each line controls one slot.

### Syntax

```
slot_name: garment_spec | color_tag
```

### Special values

- `exclude` -- Prevents the slot from producing any output, even if enabled.
- `auto` -- Explicitly tells the engine to auto-generate this slot (the default behavior).

### Garment spec format

The garment spec consists of an optional fabric name followed by the garment name:

```
slot: fabric garment_name | #color_tag#
```

If only one word is given, it is treated as the garment name (no fabric).

### Wildcards

Use `{option1|option2|option3}` to let the engine randomly pick one option (controlled by seed):

```
bottom: {pencil skirt|midi skirt|a-line skirt} | #secondary#
```

### Comments

Lines starting with `#` are ignored.

### Full example

```
# Force a silk blouse for the top slot
top: silk blouse | #primary#

# Randomly pick between two skirt types
bottom: {pencil skirt|midi skirt} | #secondary#

# Force stilettos with a specific color tag
footwear: stilettos | #neutral#

# Prevent any bag from appearing
bag: exclude

# Let headwear be auto-generated normally
headwear: auto

# Force leather jacket for outerwear, using the accent color
outerwear: leather jacket | #accent#
```

### Color tag reference

Any tag recognized by Prompt Color Replace can be used:

| Tag | Short form | Default palette position |
|-----|-----------|------------------------|
| `#primary#` | `#pri#` | Color 1 |
| `#secondary#` | `#sec#` | Color 2 |
| `#accent#` | `#acc#` | Color 3 |
| `#neutral#` | `#neu#` | Color 4 |
| `#metallic#` | `#met#` | Color 5 |
| `#color1#` - `#color8#` | `#c1#` - `#c8#` | Colors 1-8 |

---

## Creating Custom Outfit Sets

This section explains how to create your own outfit sets from scratch. This is the primary way to customize the Outfit Generator for specific themes, eras, subcultures, or project needs.

### Step 1: Create the directory

Create a new subdirectory inside `outfit_lists/`:

```
outfit_lists/
  my_custom_set/       <-- your new set
    top.txt
    bottom.txt
    footwear.txt
    headwear.txt
    outerwear.txt
    accessories.txt
    bag.txt
    fabrics.txt
```

The directory name becomes the value shown in the `outfit_set` dropdown. Use lowercase with underscores. Common convention: `theme_gender` (e.g., `cyberpunk_female`, `western_male`).

### Step 2: Create the garment files (7 slot files)

Each slot has its own `.txt` file. All 7 files follow the same format. You need at least `top.txt`, `bottom.txt`, and `footwear.txt` (the always-active slots). The other 4 files can be empty or omitted if you do not want that slot to produce output.

#### Garment file format

```
# Comments start with #
# format: garment_name | probability | formality_min-formality_max | fabric1,fabric2,...

garment name | 0.75 | 0.0-0.5 | cotton,jersey,polyester
another garment | 0.40 | 0.3-0.8 | silk,satin,velvet
```

Each non-comment, non-empty line defines one garment entry with 4 pipe-separated fields.

#### Field reference

| Field | Format | Description |
|-------|--------|-------------|
| `garment_name` | Free text | The name that appears in the prompt. Use the exact wording you want in the final prompt (e.g., "pencil skirt", "combat boots"). |
| `probability` | Float 0.0-1.0 | Relative selection weight. Higher values make this garment more likely to be chosen. A garment with probability 0.80 is roughly twice as likely as one with 0.40. |
| `formality_range` | `min-max` | The formality range where this garment is eligible. Only garments whose range includes the current formality value will be considered. Use `0.0-1.0` to make it always eligible. |
| `fabrics` | Comma-separated | List of fabric names that are valid for this garment. Must match entries in your `fabrics.txt` file. |

#### Annotated example: `top.txt`

```
# slot: top
# format: garment_name | probability | formality_min-formality_max | fabric1,fabric2,...

# Very casual tops (only appear at low formality)
tank top | 0.50 | 0.0-0.2 | cotton,jersey,mesh
crop top | 0.40 | 0.0-0.2 | cotton,jersey,knit
t-shirt | 0.85 | 0.0-0.3 | cotton,jersey,polyester
#  ^          ^       ^              ^
#  |          |       |              |
#  |          |       |              +-- valid fabrics (must exist in fabrics.txt)
#  |          |       +-- only eligible when formality is between 0.0 and 0.3
#  |          +-- high probability = appears frequently
#  +-- garment name as it will appear in the prompt

# Mid-range tops
blouse | 0.65 | 0.3-0.7 | silk,chiffon,cotton,satin,lace
turtleneck | 0.45 | 0.3-0.7 | wool,cashmere,knit,cotton

# Formal tops
dress shirt | 0.60 | 0.5-1.0 | cotton,silk,linen
```

#### How probability works

Probability is a relative weight, not an absolute percentage. When the engine selects a garment for a slot, it:

1. Filters the garment list by formality (only garments whose range includes the current formality).
2. Uses the `probability` values as weights in a weighted random choice.
3. A garment with probability `0.85` is not chosen 85% of the time -- it is chosen proportionally to its weight relative to all other eligible garments.

For example, if three garments are eligible with probabilities `0.85`, `0.50`, and `0.40`, their actual selection chances are approximately 49%, 29%, and 23%.

#### How formality ranges work

The formality range defines the "window" where a garment is eligible:

```
tank top   | 0.50 | 0.0-0.2 | ...   # only at very casual settings
blouse     | 0.65 | 0.3-0.7 | ...   # mid-range
dress shirt| 0.60 | 0.5-1.0 | ...   # formal settings
```

- If the user sets formality to `0.1`, only `tank top` and any other garments with ranges including `0.1` are eligible.
- If formality is `0.5`, both `blouse` and `dress shirt` are eligible.
- Overlapping ranges are fine and encouraged -- they create natural transitions between formality levels.
- If no garments match the current formality (all filtered out), the engine falls back to the full unfiltered list.

#### How fabric references work

The fabric names in the garment file's fourth field must match entries in your `fabrics.txt` file exactly (case-sensitive). The engine uses these references to:

1. Pick a fabric for the garment from the listed options.
2. Weight the choice by how close the fabric's own formality is to the current formality setting.
3. Respect fabric family preferences from the style preset (e.g., "evening_gala" prefers "luxury" family fabrics).

### Step 3: Create `fabrics.txt`

The fabric database defines the properties of each fabric available to garments in this set.

#### Format

```
# format: fabric_name | formality | family | weight
cotton | 0.2 | natural | medium
silk | 0.85 | luxury | light
leather | 0.45 | tough | heavy
```

| Field | Description |
|-------|-------------|
| `fabric_name` | Unique name. Must match references in garment files exactly. |
| `formality` | Float 0.0-1.0. The fabric's inherent formality level. The engine prefers fabrics close to the current formality. |
| `family` | Category name. Used for fabric harmony pairing. Standard families: `natural`, `casual`, `sporty`, `tough`, `luxury`. |
| `weight` | Descriptive weight: `light`, `medium`, or `heavy`. Currently informational. |

#### "Invisible" fabrics

Some fabrics should not appear in the final prompt text because they would sound unnatural (e.g., "metal earrings" reads better as just "earrings" with the color tag). The engine automatically suppresses fabric names from the prompt for these materials:

- `metal`
- `plastic`
- `rubber`

These fabrics still function normally for selection logic -- they just do not appear in the output text. If you add a garment like `earrings | 0.55 | 0.2-1.0 | metal`, the output will be `"#metallic# earrings"` rather than `"#metallic# metal earrings"`.

### Step 4: Verify

1. Save all your `.txt` files.
2. In ComfyUI, the new set appears in the `outfit_set` dropdown after a browser refresh or re-adding the node (no ComfyUI restart needed -- the dropdown is scanned live).
3. Set a seed, queue the prompt, and check the `outfit_info` output for the chosen pieces.

### Tip: Start from an existing set

The fastest way to create a custom set is to copy an existing one and modify it:

```bash
cp -r outfit_lists/general_female/ outfit_lists/my_theme_female/
```

Then edit the `.txt` files to add, remove, or adjust garments for your theme.

---

## Fabric System

### How fabrics are selected

When the engine picks a garment, it also picks a fabric for it. The selection process:

1. Start with the garment's fabric list (from the garment file).
2. If the style preset specifies `preferred_fabric_families`, filter to only fabrics in those families. If none match, keep the full list.
3. Among remaining candidates, weight by inverse distance to the target formality. Fabrics closer to the current formality are preferred.
4. Use weighted random choice (controlled by seed) to pick one.

### `fabrics.txt` -- Per-set fabric database

Each outfit set has its own `fabrics.txt` defining all fabrics available to garments in that set. This allows themed sets to use different fabric pools. For example, a "gothic" set might include `pvc` and `fishnet` but not `linen`.

### `fabric_harmony.txt` -- Global harmony rules

Located at `outfit_lists/fabric_harmony.txt` (not inside a set directory), this file defines which fabric families pair well together. It is shared across all outfit sets.

#### Format

```
# format: family | compatible_families (comma-separated)
luxury | luxury,natural
natural | natural,luxury,casual
casual | casual,natural,sporty
tough | tough,casual,sporty
sporty | sporty,casual,tough
```

This means:
- `luxury` fabrics pair well with other `luxury` and `natural` fabrics.
- `tough` fabrics pair well with `tough`, `casual`, and `sporty` fabrics.
- `luxury` and `tough` do not harmonize (neither lists the other as compatible).

### Standard fabric families

| Family | Formality tendency | Example fabrics |
|--------|-------------------|-----------------|
| `casual` | Low (0.1-0.3) | jersey, fleece, knit, corduroy |
| `sporty` | Low (0.1-0.2) | mesh, nylon, polyester |
| `natural` | Mid (0.2-0.5) | cotton, linen, wool, tweed |
| `tough` | Mid (0.2-0.45) | denim, canvas, leather, suede |
| `luxury` | High (0.6-0.85) | silk, satin, velvet, cashmere, chiffon, lace |

---

## Available Outfit Sets

The following 40 outfit sets are included. Each set contains curated garment lists tailored to its theme.

| Set | Theme | Target | Typical formality |
|-----|-------|--------|-------------------|
| `general_female` | All-purpose, full range | Female | 0.0-1.0 |
| `general_male` | All-purpose, full range | Male | 0.0-1.0 |
| `business_female_skirt` | Business with skirts only (legs visible) | Female | 0.5-1.0 |
| `business_female_dress` | Business dresses | Female | 0.5-1.0 |
| `business_female2` | Office, corporate, professional | Female | 0.5-1.0 |
| `business_male` | Office, corporate, professional | Male | 0.5-1.0 |
| `casual_female` | Everyday relaxed wear | Female | 0.0-0.4 |
| `casual_male` | Everyday relaxed wear | Male | 0.0-0.4 |
| `streetwear_female` | Urban street fashion | Female | 0.0-0.4 |
| `streetwear_male` | Urban street fashion | Male | 0.0-0.4 |
| `athleisure_female` | Sporty, athletic-casual | Female | 0.0-0.3 |
| `athleisure_male` | Sporty, athletic-casual | Male | 0.0-0.3 |
| `sporty_female` | Active sport/fitness | Female | 0.0-0.3 |
| `sporty_male` | Active sport/fitness | Male | 0.0-0.3 |
| `bohemian_female` | Boho, free-spirited | Female | 0.1-0.5 |
| `bohemian_male` | Boho, free-spirited | Male | 0.1-0.5 |
| `gothic_female` | Dark, gothic aesthetic | Female | 0.1-0.6 |
| `gothic_male` | Dark, gothic aesthetic | Male | 0.1-0.6 |
| `preppy_female` | Preppy, collegiate | Female | 0.3-0.7 |
| `preppy_male` | Preppy, collegiate | Male | 0.3-0.7 |
| `vintage_retro_female` | Retro/vintage inspired | Female | 0.2-0.7 |
| `vintage_retro_male` | Retro/vintage inspired | Male | 0.2-0.7 |
| `date_night_female` | Romantic evening out | Female | 0.4-0.8 |
| `date_night_male` | Romantic evening out | Male | 0.4-0.8 |
| `night_out_female` | Clubbing, nightlife | Female | 0.3-0.7 |
| `night_out_male` | Clubbing, nightlife | Male | 0.3-0.7 |
| `party_female` | Party, celebration | Female | 0.3-0.8 |
| `party_male` | Party, celebration | Male | 0.3-0.8 |
| `festival_female` | Music festival, outdoor | Female | 0.0-0.4 |
| `festival_male` | Music festival, outdoor | Male | 0.0-0.4 |
| `wedding_guest_female` | Wedding guest attire | Female | 0.6-1.0 |
| `wedding_guest_male` | Wedding guest attire | Male | 0.6-1.0 |
| `beach_holiday_female` | Beach, resort, vacation | Female | 0.0-0.2 |
| `beach_holiday_male` | Beach, resort, vacation | Male | 0.0-0.2 |
| `winter_wonderland_female` | Cold weather, winter | Female | 0.1-0.6 |
| `winter_wonderland_male` | Cold weather, winter | Male | 0.1-0.6 |
| `sheer_business_female` | Business with sheer/visible bra tops | Female | 0.5-0.9 |
| `sheer_casual_female` | Casual with sheer/visible bra tops | Female | 0.0-0.4 |
| `sheer_evening_female` | Evening with sheer/visible bra tops | Female | 0.3-0.8 |
| `female_lingerie` | Lingerie, bras, bodysuits, robes, stockings | Female | 0.0-0.6 |

---

## Slot System

The outfit is assembled from 7 slots, processed in head-to-toe order. Each slot has a default color tag that determines which palette color is used.

| Slot | Default color tag | Always active? | Default enabled? | Description |
|------|------------------|----------------|------------------|-------------|
| `headwear` | `#accent#` | No | No | Hats, caps, headbands, beanies |
| `top` | `#primary#` | Yes | Yes | Shirts, blouses, tops, sweaters |
| `outerwear` | `#secondary#` | No | No | Jackets, coats, blazers, cardigans |
| `bottom` | `#secondary#` | Yes | Yes | Pants, skirts, shorts, jeans |
| `footwear` | `#neutral#` | Yes | Yes | Shoes, boots, sandals, heels |
| `accessories` | `#metallic#` | No | No | Jewelry, belts, scarves, sunglasses |
| `bag` | `#accent#` | No | No | Bags, clutches, backpacks |

### "Always active" vs "optional" slots

- **Always active** slots (`top`, `bottom`, `footwear`): When enabled, they always produce a garment. They are not subject to probability rolls.
- **Optional** slots (`headwear`, `outerwear`, `accessories`, `bag`): When enabled, they still might not produce output, depending on the probability roll (influenced by the `coverage` slider and the style preset's `slot_probabilities`).

### Disabling a slot

Setting an `enable_*` checkbox to False guarantees no output from that slot, regardless of overrides or style preset.

---

## Tips and Tricks

### Exploration

- **Use different seeds** to explore outfit variations while keeping all other settings fixed. This is the fastest way to find combinations you like.
- **Try different style presets** with the same seed. The preset changes fabric preferences and slot probabilities, producing noticeably different results from the same garment lists.

### Batch generation

- **Connect multiple Outfit Generators** with different seeds to generate diverse outfits in a single workflow. Each can feed into its own Prompt Color Replace and CLIP chain.

### Custom sets

- **Copy an existing set** as a starting point. The `general_female` and `general_male` sets have the broadest garment range and make good templates.
- **Edit .txt files while ComfyUI is running.** The outfit lists are loaded fresh on every execution (no caching). Changes take effect immediately on the next queue -- no restart needed.
- **Start with fewer garments** and add more as you test. A slot with just 3-5 well-tuned garments works better than 30 untested ones.

### Formality tuning

- **Use formality 0.0** for beachwear, loungewear, gym looks.
- **Use formality 0.5** for everyday outfits, smart-casual.
- **Use formality 1.0** for black-tie, gala, formal business.
- **Overlap formality ranges** in your garment files to create smooth transitions. Garments that span a wide formality range (e.g., `0.2-0.8`) act as versatile staples.

### Color control

- The default color tag assignments (primary for top, secondary for bottom, etc.) produce balanced results. Override them only when you want specific creative control.
- **Use the override string** to force a specific color tag on a specific slot without changing the outfit set files.
- **Combine with Palette From Image** instead of Color Palette Generator to match outfits to reference image colors.

### Layering and sheer clothing

- The `sheer_business_female`, `sheer_casual_female`, and `sheer_evening_female` outfit sets contain tops designed to show visible undergarments (sheer fabric, visible bra straps, low necklines).
- Garment names like `sheer blouse` are paired with fabrics like `chiffon`, `organza`, or `mesh`. The engine builds prompts like `#primary# chiffon sheer blouse` -- no fabric duplication since "sheer" is part of the garment name, not the fabric.
- For even more explicit layering, use the override string to add detail: `top:thin white shirt with visible dark bra underneath:silk`
- **Z-Image Turbo tip:** Dark bra + light sheer top creates the strongest contrast. Add `backlit` to your environment prompt for enhanced see-through effect.

### Debugging

- Check the **`outfit_info`** output for a readable summary of what was generated and why.
- Check the **`outfit_details`** output for the structured `slot:garment:fabric:tag` breakdown.
- If a slot produces unexpected garments, verify the formality slider is within the garment's formality range.

### Advanced: `outfit_config.ini`

The file `outfit_config.ini` in the project root lets you:
- **Set a custom path** for outfit lists (useful if you want to keep your sets outside the node directory).
- **Set default probabilities** for optional slots.

```ini
[paths]
custom_lists_path = D:/my_outfits/lists

[defaults]
default_outfit_set = general_female
headwear_probability = 0.15
outerwear_probability = 0.25
accessories_probability = 0.50
bag_probability = 0.20
```
