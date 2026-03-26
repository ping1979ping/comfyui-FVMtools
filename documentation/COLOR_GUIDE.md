# FVM Color Nodes -- Complete User Guide

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Color Palette Generator -- Detailed Reference](#color-palette-generator----detailed-reference)
- [Prompt Color Replace -- Detailed Reference](#prompt-color-replace----detailed-reference)
- [Palette From Image -- Detailed Reference](#palette-from-image----detailed-reference)
- [Color Database Reference](#color-database-reference)
- [Workflow Examples](#workflow-examples)
- [Example Prompts](#example-prompts)
- [Tips and Tricks](#tips-and-tricks)

---

## Overview

The FVM Color Nodes are a system of three ComfyUI nodes that work together to provide intelligent, seed-controlled color management for fashion and image generation prompts. Instead of hard-coding color names into your prompts, you write abstract color tags and let the system fill them in from a generated or extracted palette.

### The three nodes

| Node | Purpose |
|------|---------|
| **Color Palette Generator** | Generates a harmonious named color palette from a seed, using color theory and style presets |
| **Palette From Image** | Extracts a named color palette from a reference image using K-Means clustering |
| **Prompt Color Replace** | Replaces color placeholder tags in a prompt with actual color names from a palette |

### The concept

Separate color choice from prompt writing. Your prompt describes *what* to generate using abstract color roles (`#primary#`, `#accent#`, etc.), and the palette provides *which* colors fill those roles. Change the seed to get a completely different color scheme without touching the prompt. Change the prompt without touching the colors. Mix and match freely.

### The standard pipeline

```
Color Palette Generator ──> palette_string ──> Prompt Color Replace ──> prompt ──> CLIP Text Encode
       (or)                                            ^
Palette From Image ────────> palette_string ────────────┘
```

---

## Quick Start

The fastest way to get color-controlled prompts running in 5 steps:

1. **Add a Color Palette Generator node.** Set a seed value and pick a style preset (or leave it on `general`).

2. **Add a Prompt Color Replace node.** In the prompt field, write your prompt using color tags:
   ```
   wearing #primary# silk blouse with #neutral# pencil skirt, #accent# heels, #metallic# earrings
   ```

3. **Connect** the `palette_string` output from the generator to the `palette_string` input on the replace node.

4. **Connect** the `prompt` output from Prompt Color Replace to a **CLIP Text Encode** node.

5. **Done.** Queue a prompt. The colors change every time you change the seed. The `palette_preview` output shows you a swatch image of the selected colors.

---

## Color Palette Generator -- Detailed Reference

**Node name:** `FVM_ColorPaletteGenerator`
**Category:** `FVM Tools/Color`

Generates a harmonious named color palette using color theory relationships and fashion-aware style presets. Every palette is deterministic for a given seed.

### Inputs

| Input | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `seed` | INT | 0 | 0 -- 4294967295 | Random seed. Same seed = same palette. |
| `num_colors` | INT | 5 | 2 -- 8 | Number of colors to generate. |
| `harmony_type` | ENUM | auto | see below | Color theory relationship between hues. |
| `style_preset` | ENUM | general | see below | Biases hue ranges, vibrancy, and contrast toward a visual theme. |
| `vibrancy` | FLOAT | 0.5 | 0.0 -- 1.0 | Controls color saturation intensity. Higher = more vivid colors. |
| `contrast` | FLOAT | 0.5 | 0.0 -- 1.0 | Controls lightness spread between colors. Higher = more light/dark variation. |
| `warmth` | FLOAT | 0.5 | 0.0 -- 1.0 | Shifts hue bias. Lower = cooler blues/greens, higher = warmer reds/oranges. |
| `neutral_ratio` | FLOAT | 0.4 | 0.0 -- 1.0 | Fraction of palette slots allocated to neutral colors (black, cream, gray, etc.). |
| `include_metallics` | BOOLEAN | True | -- | Whether to include metallic colors (gold, silver, bronze, etc.) in the palette. |
| `palette_source` | ENUM | generate | generate, from_file | Switch between algorithmic generation and loading from a text file. |
| `wildcard_file` | STRING | "" | multiline | Palette definitions, one per line. Only used when `palette_source` is `from_file`. |
| `palette_index` | INT | -1 | -1 -- 9999 | Which line to pick from wildcard_file. -1 = random (based on seed). |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `palette_string` | STRING | Comma-separated color names (e.g., `"navy-blue, coral, charcoal-gray, gold, cream"`). Connect this to Prompt Color Replace. |
| `color_1` -- `color_8` | STRING | Individual color name at each palette position. Empty string if fewer colors were generated. |
| `primary` | STRING | The color assigned the "primary" role (most vivid, dominant hue). |
| `secondary` | STRING | The color assigned the "secondary" role. |
| `accent` | STRING | The color assigned the "accent" role (pop color). |
| `neutral` | STRING | The color assigned the "neutral" role (base/background color). |
| `metallic` | STRING | The color assigned the "metallic" role (if metallics are included). |
| `palette_preview` | IMAGE | A 512x128 swatch image showing all palette colors with their names labeled below. |
| `palette_info` | STRING | Summary text describing the generation parameters used. |

### Harmony Types

The harmony type determines the angular relationship between hues on the color wheel. When set to `auto`, the style preset picks an appropriate harmony.

| Harmony Type | Hue Relationship | Character |
|-------------|-----------------|-----------|
| `analogous` | Base +/- 30 degrees | Harmonious and calm. Colors that sit next to each other on the wheel. |
| `complementary` | Base + 180 degrees | High contrast. Two colors on opposite sides of the wheel. |
| `split_complementary` | Base + 150 and + 210 degrees | Bold but less harsh than complementary. A V-shape on the wheel. |
| `triadic` | Base + 120 and + 240 degrees | Vibrant and balanced. Three evenly spaced colors. |
| `tetradic` | Base + 90, + 180, + 270 degrees | Rich and varied. Four evenly spaced colors (rectangle on the wheel). |
| `monochromatic` | Same hue, varying saturation/lightness | Elegant and cohesive. Different shades and tints of one color. |

When more colors are requested than the harmony naturally provides, the generator subdivides the largest hue gap to create additional distinct hues.

### Style Presets

Each preset biases the generation toward a specific visual theme by restricting hue ranges, adjusting vibrancy/contrast/warmth, preferring certain harmonies, and excluding inappropriate color names.

| Preset | Hue Focus | Character |
|--------|-----------|-----------|
| `general` | All hues | No restrictions. Neutral starting point for any style. |
| `beach` | Aqua, sandy, coral | Warm and breezy. Forbids dark grays/blacks. Neutral bias toward sand, cream, ivory. |
| `urban_streetwear` | All hues | High contrast and vibrancy. Forbids pastels. Neutral bias toward black and gray. |
| `evening_gala` | Reds, purples, deep blues | Elegant and dramatic. Forbids lime/chartreuse/neons. Neutral bias toward black, ivory, champagne. |
| `casual_daywear` | All hues | Softened vibrancy and contrast. Forbids neons and electrics. Neutral bias toward cream, beige, oatmeal. |
| `vintage_retro` | Warm reds, teals, magentas | Desaturated and warm. Forbids neons. Neutral bias toward cream, ivory, taupe. |
| `cyberpunk_neon` | Cyan, purple, yellow-green | High vibrancy and contrast, cool. Forbids beige/oatmeal/cream. Neutral bias toward black and dark grays. |
| `pastel_dream` | All hues | Low vibrancy and contrast. Forbids black and dark colors. Neutral bias toward soft-white, ivory. |
| `earthy_natural` | Warm browns, greens, terracotta | Low vibrancy, warm. Forbids neons, electrics, hot pink. Neutral bias toward taupe, mushroom, sand. |
| `monochrome_chic` | All hues (desaturated) | Very low vibrancy, high contrast. Black/white/gray neutral bias. Best with monochromatic harmony. |
| `tropical` | Reds, greens, magentas | High vibrancy and warmth. Forbids dark grays. Neutral bias toward cream, sand. |
| `winter_cozy` | Warm reds, blues, berry tones | Slightly muted, warm. Forbids neons and bright greens. Neutral bias toward cream, oatmeal, espresso. |
| `festival` | All hues | High vibrancy and contrast. No color restrictions. Neutral bias toward black and white. |
| `office_professional` | Blues, deep reds, burgundy | Low vibrancy, controlled. Forbids neons, electrics, hot pink, lime. Neutral bias toward charcoal, white, slate. |

### How the Sliders Work

**Vibrancy** (0.0 -- 1.0): Controls color saturation. At 0.0 you get muted, desaturated colors. At 1.0 you get fully saturated, vivid colors. Above 0.8, neon colors become available in the name matching pool.

**Contrast** (0.0 -- 1.0): Controls the lightness spread across the palette. Low contrast means colors cluster around mid-lightness. High contrast means the palette includes both very light and very dark colors.

**Warmth** (0.0 -- 1.0): Shifts the hue bias. At 0.0, the generator favors cool hues (blues, greens, purples). At 1.0, it favors warm hues (reds, oranges, yellows). At 0.5, no bias is applied.

Each style preset applies a modifier on top of these sliders. For example, `cyberpunk_neon` adds +0.35 to vibrancy and -0.2 to warmth, so even at slider defaults you get vivid, cool colors.

### Wildcard File Mode (from_file)

When `palette_source` is set to `from_file`, the node reads palettes from the `wildcard_file` text input instead of generating them algorithmically.

**Format:** One palette per line, colors separated by commas.

```
navy-blue, coral, cream, gold, charcoal-gray
forest-green, burgundy, ivory, bronze, taupe
electric-blue, hot-pink, black, silver, white
```

**Selection:** If `palette_index` is set to a value >= 0, that line is selected (wraps around if index exceeds the number of lines). If `palette_index` is -1, the `seed` determines which line is picked randomly.

Color names must match entries in the color database. Unrecognized names default to black.

---

## Prompt Color Replace -- Detailed Reference

**Node name:** `FVM_PromptColorReplace`
**Category:** `FVM Tools/Color`

Replaces color placeholder tags in a prompt string with actual color names from a palette. This is the bridge between abstract color roles and concrete color names.

### Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | STRING | "" | Multiline prompt text containing color tags like `#primary#`, `#color1#`, etc. |
| `palette_string` | STRING | "" | Comma-separated color names (typically connected from a palette generator). **Force input** -- must be wired. |
| `primary` | STRING | "" | Override for `#primary#` / `#pri#` tags. Takes precedence over palette position. **Force input.** |
| `secondary` | STRING | "" | Override for `#secondary#` / `#sec#` tags. **Force input.** |
| `accent` | STRING | "" | Override for `#accent#` / `#acc#` tags. **Force input.** |
| `neutral` | STRING | "" | Override for `#neutral#` / `#neu#` tags. **Force input.** |
| `metallic` | STRING | "" | Override for `#metallic#` / `#met#` tags. **Force input.** |
| `fallback_color` | STRING | "black" | Color name used when a tag references a palette position that does not exist. |
| `strip_hyphens` | BOOLEAN | True | Converts hyphenated names like `navy-blue` to `navy blue`. Recommended for Stable Diffusion prompts. |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `prompt` | STRING | The prompt with all color tags replaced by actual color names. |
| `replacements_log` | STRING | A log of all replacements made (e.g., `"#primary# -> navy blue, #accent# -> coral"`). |

### Tag Reference

All tags are case-insensitive. `#Primary#`, `#PRIMARY#`, and `#primary#` all work identically.

| Tag | Short Alias | Maps To |
|-----|-------------|---------|
| `#color1#` | `#c1#` | Palette position 1 |
| `#color2#` | `#c2#` | Palette position 2 |
| `#color3#` | `#c3#` | Palette position 3 |
| `#color4#` | `#c4#` | Palette position 4 |
| `#color5#` | `#c5#` | Palette position 5 |
| `#color6#` | `#c6#` | Palette position 6 |
| `#color7#` | `#c7#` | Palette position 7 |
| `#color8#` | `#c8#` | Palette position 8 |
| `#primary#` | `#pri#` | Palette position 1 (most vivid color) |
| `#secondary#` | `#sec#` | Palette position 2 |
| `#accent#` | `#acc#` | Palette position 3 (pop color) |
| `#neutral#` | `#neu#` | Palette position 4 (base neutral) |
| `#metallic#` | `#met#` | Palette position 5 (metallic or secondary neutral) |

Semantic tags (`#primary#`, etc.) and numbered tags (`#color1#`, etc.) both reference the same palette positions by default. The difference is readability: semantic tags make your prompt's intent clearer.

### Override Behavior

The optional override inputs (`primary`, `secondary`, `accent`, `neutral`, `metallic`) take precedence over palette positions. If you wire a string value to the `primary` override, every `#primary#` and `#pri#` tag will use that value regardless of what is in `palette_string`.

This is useful for locking a specific color role while letting other roles vary with the seed.

### strip_hyphens

When `strip_hyphens` is True (the default), hyphenated color names like `navy-blue` are converted to `navy blue` in the output prompt. This is recommended because Stable Diffusion tokenizers handle space-separated words better than hyphenated compound words.

### Example: Before and After

**Palette string:** `navy-blue, dusty-rose, gold, charcoal-gray, silver`

**Input prompt:**
```
a woman wearing #primary# silk blouse with #neutral# wool trousers, #accent# leather belt, #metallic# hoop earrings
```

**Output prompt** (with strip_hyphens=True):
```
a woman wearing navy blue silk blouse with charcoal gray wool trousers, gold leather belt, silver hoop earrings
```

**Replacements log:**
```
#primary# -> navy blue, #neutral# -> charcoal gray, #accent# -> gold, #metallic# -> silver
```

---

## Palette From Image -- Detailed Reference

**Node name:** `FVM_PaletteFromImage`
**Category:** `FVM Tools/Color`

Extracts a color palette from a reference image using K-Means clustering, then maps cluster centers to named colors from the color database. Supports skin-tone and background filtering for cleaner fashion-focused palettes.

### Inputs

| Input | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `image` | IMAGE | -- | -- | Input image tensor to extract colors from. |
| `num_colors` | INT | 5 | 2 -- 8 | Number of colors to extract. |
| `extraction_mode` | ENUM | dominant | dominant, vibrant, fashion_aware | How extracted colors are ranked and selected (see below). |
| `ignore_background` | BOOLEAN | True | -- | Filter out large low-variance background regions (clusters covering >35% of pixels with low variance). |
| `ignore_skin` | BOOLEAN | True | -- | Filter out skin-tone colors. Detects skin as hue 5-50, moderate saturation, mid-range lightness. |
| `sample_region` | ENUM | full | full, center_crop, upper_half, lower_half | Which part of the image to analyze. |
| `saturation_threshold` | FLOAT | 0.1 | 0.0 -- 1.0 | Colors below this saturation level (mapped to 0-100 internally) are classified as neutral rather than chromatic. |
| `include_neutrals` | BOOLEAN | True | -- | Include neutral colors (grays, whites, blacks, beiges) in the extracted palette. |
| `include_metallics` | BOOLEAN | True | -- | Include metallic colors (gold, silver, bronze) in the extracted palette. |
| `seed` | INT | 0 | 0 -- 4294967295 | Random seed for K-Means initialization. Affects cluster placement. |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `palette_string` | STRING | Comma-separated color names extracted from the image. |
| `color_1` -- `color_8` | STRING | Individual color names at each palette position. |
| `primary` | STRING | Color assigned the "primary" role. |
| `secondary` | STRING | Color assigned the "secondary" role. |
| `accent` | STRING | Color assigned the "accent" role. |
| `neutral` | STRING | Color assigned the "neutral" role. |
| `metallic` | STRING | Color assigned the "metallic" role. |
| `palette_preview` | IMAGE | A 512x128 swatch image showing extracted colors with labels. |
| `source_annotated` | IMAGE | The original image with small color swatches overlaid in the bottom-left corner. |
| `palette_info` | STRING | Summary text with extraction parameters (mode, region, cluster count, active filters). |

### Extraction Modes

| Mode | Selection Strategy | Best For |
|------|-------------------|----------|
| `dominant` | Sorts clusters by pixel count (most common colors first). | Capturing the overall color impression of an image. Backgrounds and large areas dominate. |
| `vibrant` | Sorts clusters by saturation (most vivid colors first). | Pulling out the most colorful elements regardless of how much area they cover. |
| `fashion_aware` | Greedy max-hue-distance selection. Starts with the most saturated cluster, then iteratively picks the cluster whose hue is farthest from all already-selected colors. | Getting maximally diverse palettes. Best for fashion use where you want distinct colors for different garment pieces. |

### Sample Region

| Region | Area Analyzed | Use Case |
|--------|--------------|----------|
| `full` | Entire image | General purpose. |
| `center_crop` | Center 50% (removes outer 25% on each side) | Focuses on the subject, ignoring edge backgrounds. |
| `upper_half` | Top half of image | Extracting colors from face/hair/upper body in portraits. |
| `lower_half` | Bottom half of image | Extracting colors from clothing/outfit in full-body portraits. |

The `lower_half` region is particularly useful for outfit-based workflows: if your reference image shows a full-body shot, `lower_half` focuses on the clothing and ignores the face, hair, and sky.

### Skin and Background Filtering

**Skin-tone filter** (`ignore_skin`): Detects colors in the typical skin-tone HSL range (hue 5-50, moderate saturation, mid lightness) and removes them from the candidate pool. This prevents skin colors from appearing in your clothing palette when processing photos of people.

**Background filter** (`ignore_background`): Removes clusters that occupy more than 35% of all pixels while having low color variance. This catches plain studio backgrounds, solid walls, and sky regions that would otherwise dominate a `dominant` mode extraction.

If both filters are active and they remove all candidates, the node falls back to the unfiltered candidate list to ensure you always get a result.

### How Extraction Works Internally

1. The first frame of the image tensor is converted to a numpy array.
2. The selected `sample_region` crop is applied.
3. The image is downsampled to max 256px on the longest side (for speed).
4. K-Means clustering runs with 3x oversampling (up to 24 clusters for better color resolution).
5. Skin-tone and background clusters are filtered out.
6. Remaining clusters are separated into chromatic vs. neutral based on `saturation_threshold`.
7. Chromatic clusters are sorted according to the selected `extraction_mode`.
8. The top clusters are matched to the nearest named color from the database (with deduplication).
9. Roles (primary, secondary, accent, neutral, metallic) are assigned automatically.

---

## Color Database Reference

The color database contains approximately 161 named, fashion-relevant colors organized into 10 categories.

### Categories

| Category | Count | Examples |
|----------|-------|---------|
| Reds | ~20 | red, crimson, scarlet, burgundy, wine, coral, salmon, rose, cherry, ruby, brick-red, maroon |
| Oranges | ~14 | orange, tangerine, burnt-orange, peach, apricot, rust, amber, terracotta, papaya |
| Yellows | ~13 | yellow, mustard, gold, lemon, sunflower, saffron, canary, maize, buttercup |
| Greens | ~25 | green, emerald, sage, olive, forest-green, mint, jade, teal, seafoam, lime, chartreuse, hunter-green |
| Blues | ~24 | blue, navy-blue, sky-blue, cobalt, royal-blue, powder-blue, steel-blue, ocean-blue, denim, periwinkle |
| Purples | ~19 | purple, lavender, plum, violet, mauve, amethyst, eggplant, grape, lilac, orchid |
| Pinks | ~14 | pink, hot-pink, blush-pink, fuchsia, magenta, bubblegum, flamingo, rose-pink, pastel-pink |
| Neutrals | ~20 | black, white, cream, ivory, beige, taupe, charcoal-gray, slate-gray, oatmeal, mushroom, linen, khaki |
| Metallics | 8 | silver, gold, rose-gold, bronze, copper, champagne-gold, pewter, platinum |
| Neons | 7 | neon-green, neon-yellow, neon-orange, neon-pink, neon-blue, electric-blue, electric-purple |

### How Color Name Matching Works

When a generated or extracted HSL color needs to be mapped to a name, the engine uses a weighted Euclidean distance in HSL space:

- **Hue distance** is weighted 2x (most important for perceived color identity)
- **Saturation distance** is weighted 1x
- **Lightness distance** is weighted 1x

The nearest match wins. Already-used names are excluded to prevent duplicates within a palette.

### Neon Availability

Neon colors (`neon-green`, `neon-pink`, `electric-blue`, etc.) are only considered as matches when the vibrancy parameter is set to 0.8 or higher. At lower vibrancy levels, they are excluded from the candidate pool entirely. This prevents neon names from appearing in palettes where they would be unexpected.

---

## Workflow Examples

### 1. Simple Batch with Generated Palettes

Generate multiple images with different color schemes by incrementing the seed.

```
[Color Palette Generator]          [Prompt Color Replace]          [CLIP Text Encode]
  seed: 42                            prompt: "..."      ------>   text input
  num_colors: 5           ------>   palette_string
  style_preset: general
  harmony_type: triadic
                                                                   [KSampler]
                                                                     ...
```

Change the seed on the palette generator from 42 to 43, 44, 45... Each seed produces a completely different color scheme while the prompt structure stays identical.

### 2. Reference Image Driven

Extract colors from a mood board or fashion photo, then apply them to a new prompt.

```
[Load Image]                    [Palette From Image]          [Prompt Color Replace]
  image: reference.png ------>    image                          prompt: "..."
                                  extraction_mode: fashion_aware   palette_string  ------>  [CLIP Text Encode]
                                  sample_region: lower_half
                                  ignore_skin: True
                                  ignore_background: True
```

Use `lower_half` + `ignore_skin` when your reference is a full-body photo and you want to capture only the clothing colors.

### 3. Hybrid: Image Palette with Style Variation

Extract a base palette from an image, but override specific roles with generated colors.

```
[Palette From Image]                    [Color Palette Generator]
  palette_string  ------>  [Prompt Color Replace]
                              primary (override)  <------  primary output
                              accent (override)   <------  accent output
                              prompt: "..."
                              palette_string: (from image)
```

This keeps the neutral and metallic tones from the reference image while letting the generator supply fresh primary and accent colors on each seed change.

### 4. Integration with Outfit Generator

The Color Nodes are designed to pair with the FVM Outfit Generator. The Outfit Generator outputs prompts containing color tags that Prompt Color Replace resolves.

```
[Outfit Generator]                                    [Prompt Color Replace]
  outfit_prompt  ---------------------------------------->  prompt
  seed: 100                                                 palette_string  <------  [Color Palette Generator]
                                                                                       seed: 200
                                                                                       style_preset: evening_gala

                                                      prompt output  ------>  [CLIP Text Encode]
```

The outfit seed and the palette seed are independent. Seed 100 always produces the same garment combination, while seed 200 always produces the same color palette. Change one without affecting the other.

---

## Example Prompts

### Beach / Resort

```
a woman at the beach wearing #primary# flowing maxi dress,
#neutral# straw sun hat, #accent# sandals,
#metallic# shell bracelet, #secondary# sarong wrap
```
Recommended preset: `beach`

### Evening / Formal

```
a woman at a gala wearing #primary# velvet evening gown,
#metallic# clutch purse, #neutral# satin heels,
#accent# gemstone necklace, #secondary# silk shawl
```
Recommended preset: `evening_gala`

### Urban / Street Style

```
a woman in the city wearing #primary# oversized hoodie,
#neutral# cargo pants, #accent# high-top sneakers,
#secondary# crossbody bag, #metallic# chain necklace
```
Recommended preset: `urban_streetwear`

### Office / Professional

```
a woman in an office wearing #primary# fitted blazer,
#neutral# tailored trousers, #secondary# silk blouse,
#metallic# watch, #accent# leather pumps
```
Recommended preset: `office_professional`

### Cyberpunk / Futuristic

```
a woman in a neon-lit alley wearing #primary# PVC bodysuit,
#secondary# mesh crop top, #neutral# tactical pants,
#accent# platform boots, #metallic# chrome visor
```
Recommended preset: `cyberpunk_neon`, vibrancy: 0.9

### Pastel / Soft Aesthetic

```
a woman in a garden wearing #primary# knit cardigan,
#secondary# pleated midi skirt, #neutral# cotton tee,
#accent# ballet flats, #metallic# delicate pendant
```
Recommended preset: `pastel_dream`

### Sheer / Visible Bra Layering

```
a woman wearing #primary# sheer chiffon blouse with visible dark bra underneath,
#secondary# pencil skirt, #neutral# stiletto heels, #metallic# pendant necklace
```
Recommended outfit sets: `sheer_business_female`, `sheer_casual_female`, `sheer_evening_female`

**Tip:** For visible undergarments, name both layers explicitly in the prompt. Use specific fabric names like chiffon, organza, or mesh instead of generic "see-through". A dark bra under a light sheer top creates the strongest visual contrast. Adding `backlit` or `rim lighting` to the environment description enhances the see-through effect. For visible bra straps, describe them directly: `with visible bra strap`, `with peeking bra straps`.

### Lingerie / Boudoir

```
a woman in a bedroom wearing #primary# lace bralette,
#secondary# silk robe, #secondary# satin panties,
#neutral# stiletto heels, #metallic# pearl necklace
```
Recommended outfit set: `female_lingerie`

**Tip:** The `female_lingerie` set has ~20 tops (bralettes, corsets, bodysuits, bras) and ~16 bottoms, mixing sheer and opaque pieces. Pair with `coverage: 0.7` to include robes, stockings, and accessories. Disable the `bag` slot for cleaner results.

---

## Tips and Tricks

### Seed Control

Same seed always produces the same colors. This is powerful for consistency:
- Use one palette seed across an entire batch for a unified color story.
- Increment the seed by 1 to explore nearby variations.
- The palette seed and any image generation seed are independent -- you can lock colors while varying composition, or vice versa.

### Preview Before Generating

Connect the `palette_preview` output to a Preview Image node. This gives you a quick visual check of the palette before spending time on a full image generation. The preview shows color swatches with their names labeled underneath.

### strip_hyphens Should Usually Be True

Stable Diffusion's CLIP tokenizer handles `navy blue` (two tokens) better than `navy-blue` (which may tokenize unpredictably). Keep `strip_hyphens` enabled unless you have a specific reason to preserve hyphens.

### Locking Specific Colors with Overrides

Use the override inputs on Prompt Color Replace to lock specific color roles:
- Wire a fixed string ("red") to the `primary` override to keep the primary color constant.
- Leave other overrides disconnected so they continue to vary with the palette seed.
- You can also wire individual color outputs (`primary`, `accent`, etc.) from one palette node into the overrides of another Prompt Color Replace node for cross-palette mixing.

### Combining With a Mood Board

Feed a mood board image into Palette From Image with `fashion_aware` mode, then use the extracted palette across multiple prompts. This gives you a consistent color scheme derived from your visual reference without manually picking color names.

### Region Selection for Portraits

When extracting colors from full-body portrait photos:
- Use `lower_half` to focus on clothing (ignores face, hair, sky).
- Use `upper_half` to focus on hair color, accessories, and upper garments.
- Use `center_crop` to ignore background edges while keeping the full figure.

### Neutral Ratio for Realism

A `neutral_ratio` of 0.3-0.5 produces natural-looking palettes where not every piece is a bold color. Real outfits typically have 1-2 statement colors supported by neutrals. Set it to 0.0 only if you want every color to be chromatic.

### Style Preset + Manual Slider Adjustments

Style presets apply modifiers on top of your slider values. You can fine-tune after selecting a preset:
- Start with `evening_gala`, then push vibrancy to 0.7 for bolder jewel tones.
- Start with `earthy_natural`, then lower warmth to 0.3 for cooler sage/moss tones.
- Start with `cyberpunk_neon`, then raise vibrancy to 0.9+ to unlock neon color names.

### Using Multiple Palette Generators

You can wire multiple palette generators into a single workflow. For example:
- One generator for main outfit colors (style preset: `casual_daywear`).
- Another generator for accessory accent colors (style preset: `festival`).
- Connect each generator's individual color outputs to different override slots on Prompt Color Replace.

### Live Updates -- No Restart Needed

All dropdown lists (style presets, outfit sets) are scanned dynamically each time a node is placed or the page is refreshed. If you add new style presets or outfit sets to the code or file system, a browser refresh is enough to see them -- no ComfyUI restart required.
