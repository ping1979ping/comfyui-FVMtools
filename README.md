# comfyui-FVMtools

A comprehensive ComfyUI custom node pack for **face-aware detailing**, **color palette generation**, and **outfit prompt building** -- designed for character-consistent image generation workflows.

**Face Tools** -- Detect, match, and detail faces using InsightFace embeddings, BiSeNet parsing, and per-person LoRA inpainting.
**Color Tools** -- Generate harmonious named color palettes from color theory or extract them from reference images.
**Fashion Tools** -- Build seed-controlled outfit descriptions with color tag placeholders that integrate with the color pipeline.

---

## Table of Contents

- [Installation](#installation)
- [Required Models](#required-models)
- [Workflow Overview](#workflow-overview)
- [Node Reference -- Face Tools](#node-reference----face-tools)
  - [Person Selector (Match)](#1-person-selector-match)
  - [Person Selector Multi](#2-person-selector-multi)
  - [Person Detailer](#3-person-detailer)
  - [Detail Daemon Options](#4-detail-daemon-options)
  - [Inpaint Options](#5-inpaint-options)
- [Node Reference -- Color Tools](#node-reference----color-tools)
  - [Color Palette Generator](#6-color-palette-generator)
  - [Palette From Image](#7-palette-from-image)
  - [Prompt Color Replace](#8-prompt-color-replace)
- [Node Reference -- Fashion Tools](#node-reference----fashion-tools)
  - [Outfit Generator](#9-outfit-generator)
- [Color Tag Reference](#color-tag-reference)
- [Outfit Set Customization](#outfit-set-customization)
- [Z-Image Turbo Compatibility](#z-image-turbo-compatibility)
- [Detailed Guides](#detailed-guides)
- [Credits and References](#credits-and-references)
- [License](#license)

---

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/comfyui-FVMtools.git
```

Install Python dependencies (required for Face Tools):

```bash
pip install insightface>=0.7.3 onnxruntime-gpu>=1.17.0 opencv-python>=4.8.0 numpy>=1.24.0
```

**Color Tools and Fashion Tools have no extra dependencies** -- they use only Python standard library and numpy (already required by ComfyUI).

Restart ComfyUI after installation.

---

## Required Models

### InsightFace buffalo_l (Face Detection and Embedding)

Five ONNX models for face detection, landmark alignment, recognition, and gender/age estimation. **Auto-downloaded on first use** if `onnxruntime` is installed.

**Stored in:**

```
ComfyUI/models/insightface/models/buffalo_l/
├── det_10g.onnx        (16.9 MB — face detection)
├── w600k_r50.onnx      (174 MB — face recognition)
├── 1k3d68.onnx         (144 MB — 3D face alignment)
├── 2d106det.onnx       (5.0 MB — 2D landmark detection)
└── genderage.onnx      (1.3 MB — gender/age estimation)
```

**Download links (manual install):**

| Source | Link |
|--------|------|
| HuggingFace (individual files) | [public-data/insightface/models/buffalo_l](https://huggingface.co/public-data/insightface/tree/main/models/buffalo_l) |
| GitHub (ZIP bundle, ~326 MB) | [buffalo_l.zip](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip) |

### BiSeNet (Face/Head Segmentation)

**File:** `parsing_bisenet.pth` (53 MB)

A BiSeNet face parsing model with 19 semantic classes, used for generating face, head, hair, and other segmentation masks. **Must be downloaded manually.**

Place in one of these directories:

```
ComfyUI/models/gfpgan/parsing_bisenet.pth           (recommended)
ComfyUI/models/facedetection/parsing_bisenet.pth
ComfyUI/models/facerestore_models/parsing_bisenet.pth
```

**Download links:**

| Source | Link |
|--------|------|
| GitHub (official) | [parsing_bisenet.pth](https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth) |
| HuggingFace (mirror) | [leonelhs/facexlib](https://huggingface.co/leonelhs/facexlib/resolve/main/parsing_bisenet.pth) |

### SAM2.1 (Body Segmentation) -- Optional

Required only for **body** mask mode. Uses the SAM model loaded via [ComfyUI Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)'s **SAMLoader** node.

**Recommended model:** `sam2.1_hiera_large.pt` (898 MB)

Place in:

```
ComfyUI/models/sams/sam2.1_hiera_large.pt
```

**Download links:**

| Source | Link |
|--------|------|
| HuggingFace (official) | [facebook/sam2.1-hiera-large](https://huggingface.co/facebook/sam2.1-hiera-large) |
| Meta CDN (direct) | [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) |

Connect the SAMLoader output to Person Selector Multi's `sam_model` input.

### Custom Model Paths

If your models are stored in a non-standard location, you can configure fallback paths in `outfit_config.ini`:

```ini
[models]
insightface_path = D:\AI\models\insightface
bisenet_path = D:\AI\models\facerestore_models\parsing_bisenet.pth
```

These paths are only used as a **last resort**. The automatic search order is:

1. ComfyUI standard model directories (`models/insightface/`, `models/gfpgan/`, etc.)
2. Paths from `extra_model_paths.yaml`
3. Paths from `outfit_config.ini`

Changes to `outfit_config.ini` take effect on the next node execution -- no restart needed.

---

## Workflow Overview

### Pipeline 1: Face Detailing

```
Image Batch --> Person Selector Multi --> Person Detailer --> Output Images
                       ^                        ^
                 Reference Images          Model / CLIP / VAE
                 + SAM Model              + LoRAs & Prompts
```

1. **Person Selector Multi** detects faces, matches them to references, generates masks, and outputs `PERSON_DATA`.
2. **Person Detailer** iterates over each enabled reference slot, applies the slot's LoRA and prompt, inpaints the masked face region, and stitches it back.

### Pipeline 2: Color Palette

```
Color Palette Generator --> palette_string --> Prompt Color Replace --> prompt --> CLIP Text Encode
```

Generate a harmonious color palette from color theory rules, then inject named colors into prompt templates.

### Pipeline 3: Fashion Outfit

```
Outfit Generator --> outfit_prompt --> Prompt Color Replace <-- Color Palette Generator
                                              |
                                              v
                                       CLIP Text Encode
```

Generate seed-controlled outfit descriptions containing `#color#` placeholders, then replace them with palette colors for consistent, varied character styling.

### Combined Pipeline

All three systems work together:

```
Color Palette Generator ----> Prompt Color Replace <---- Outfit Generator
                                      |
                                      v
                              CLIP Text Encode --> Person Detailer <-- Person Selector Multi
                                                        |
                                                        v
                                                  Output Images
```

Use color palettes to colorize outfit prompts, then feed the resulting prompt into Person Detailer's per-slot prompt inputs for character-consistent, color-coordinated face detailing.

---

## Node Reference -- Face Tools

All Face Tool nodes appear under **FVM Tools/Face** in the ComfyUI menu.

### 1. Person Selector (Match)

**Display name:** Person Selector (Match)

Performs single-reference face matching using InsightFace ArcFace embeddings. Compares each detected face in the current image against one or more reference images and returns the best match similarity, a boolean match result, and an optional segmentation mask.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `current_image` | IMAGE | -- | The image to analyze for faces |
| `reference_images` | IMAGE | -- | One or more reference face images |
| `threshold` | FLOAT | 0.65 | Similarity threshold for a positive match (0.0--1.0) |
| `aggregation` | COMBO | max | How to aggregate multi-reference scores: `max`, `mean`, `min` |
| `mask_mode` | COMBO | none | Mask generation mode: `none`, `face`, `head`, `body` |
| `mask_fill_holes` | BOOLEAN | True | Fill holes in generated masks |
| `mask_blur` | INT | 0 | Gaussian blur radius for mask edges (0--100) |
| `det_size` | COMBO | 640 | Face detection resolution: `320`, `480`, `640`, `768` |

**Optional inputs:**

| Name | Type | Description |
|------|------|-------------|
| `sam_model` | SAM_MODEL | Required for `body` mask mode |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `similarity` | FLOAT | Best match similarity score (0.0--1.0) |
| `match` | BOOLEAN | True if similarity exceeds threshold |
| `mask` | MASK | Segmentation mask for the matched face |
| `best_reference` | IMAGE | The reference image that matched best |
| `face_count` | INT | Number of faces detected in current image |
| `matched_face_index` | INT | Index of the best-matching face |
| `report` | STRING | Human-readable match report |

---

### 2. Person Selector Multi

**Display name:** Person Selector Multi

Multi-reference batch face matching node that outputs `PERSON_DATA` for use with Person Detailer. Supports up to 10 reference slots with exclusive face assignment -- each detected face matches at most one reference, preventing duplicate assignments. Generates face, head, body, and auxiliary masks for all detected persons.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `sam_model` | SAM_MODEL | -- | SAM model for body mask generation |
| `current_image` | IMAGE | -- | The image(s) to process |
| `auto_threshold` | BOOLEAN | True | Automatically determine match thresholds |
| `threshold` | FLOAT | 0.40 | Manual similarity threshold when auto is off (0.0--1.0) |
| `aggregation` | COMBO | max | Multi-reference score aggregation: `max`, `mean`, `min` |
| `mask_fill_holes` | BOOLEAN | True | Fill holes in generated masks |
| `mask_blur` | INT | 0 | Gaussian blur radius for mask edges (0--100) |
| `det_size` | COMBO | 640 | Face detection resolution: `320`, `480`, `640`, `768` |
| `aux_mask_type` | COMBO | none | Auxiliary mask type: `none`, `hair`, `facial_skin`, `eyes`, `mouth`, `neck`, `accessories` |
| `detect_threshold` | FLOAT | 0.30 | Detection confidence threshold |
| `detect_dilation` | INT | 10 | Mask dilation in pixels |
| `detect_crop_factor` | FLOAT | 3.0 | Crop factor for face region extraction |
| `reference_1` | IMAGE | -- | First reference image (required) |
| `reference_2` -- `reference_10` | IMAGE | -- | Additional reference images (optional) |

**Optional inputs:**

| Name | Type | Description |
|------|------|-------------|
| `segs` | SEGS | Pre-computed segments from external detector |
| `bbox_detector` | BBOX_DETECTOR | Bounding box detector for face detection |
| `segm_detector` | SEGM_DETECTOR | Segmentation detector |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `person_data` | PERSON_DATA | Structured data for Person Detailer |
| `face_masks` | MASK | Per-reference face masks |
| `head_masks` | MASK | Per-reference head masks |
| `body_masks` | MASK | Per-reference body masks |
| `combined_face` | MASK | Combined face mask (all references) |
| `combined_head` | MASK | Combined head mask (all references) |
| `combined_body` | MASK | Combined body mask (all references) |
| `aux_masks` | MASK | Auxiliary masks (hair, skin, etc.) |
| `preview` | IMAGE | Annotated preview image showing matches |
| `similarities` | FLOAT | Per-reference similarity scores |
| `matches` | BOOLEAN | Per-reference match results |
| `matched_count` | INT | Number of successfully matched faces |
| `face_count` | INT | Total faces detected in the image |
| `report` | STRING | Detailed match report |

---

### 3. Person Detailer

**Display name:** Person Detailer

Per-person LoRA-based inpainting pipeline with 5 reference slots plus a generic catch-all slot. Each slot can have its own LoRA, prompt, and mask type. Processing is sequential: each slot inpaints into the result of the previous slot, building up the final image progressively.

The generic slot can optionally catch unprocessed faces (those not matched by any reference slot), applying a general-purpose LoRA and prompt to all remaining faces.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `images` | IMAGE | -- | Input images to detail |
| `person_data` | PERSON_DATA | -- | From Person Selector Multi |
| `model` | MODEL | -- | Base diffusion model |
| `clip` | CLIP | -- | CLIP model for text encoding |
| `vae` | VAE | -- | VAE for encode/decode |
| `seed` | INT | 0 | Random seed |
| `steps` | INT | 4 | Sampling steps |
| `denoise` | FLOAT | 0.52 | Denoise strength (0.0--1.0) |
| `sampler_name` | COMBO | -- | Sampler algorithm |
| `scheduler` | COMBO | -- | Noise scheduler |
| `detail_daemon_enabled` | BOOLEAN | True | Enable Detail Daemon sigma manipulation |
| `detail_amount` | FLOAT | 0.20 | Detail Daemon strength |
| `dd_smooth` | BOOLEAN | True | Smooth Detail Daemon curve |
| `mask_blend_pixels` | INT | 32 | Mask feathering width in pixels |
| `mask_expand_pixels` | INT | 0 | Mask expansion in pixels |
| `target_width` | INT | 800 | Target inpaint crop width |
| `target_height` | INT | 1200 | Target inpaint crop height |

**Per-reference slot (ref_1 through ref_5):**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `ref_N_enabled` | BOOLEAN | True/False | Enable this reference slot |
| `ref_N_lora` | COMBO | -- | LoRA file for this person |
| `ref_N_lora_strength` | FLOAT | 1.0 | LoRA application strength |
| `ref_N_prompt` | STRING | -- | Positive prompt for this person |

**Generic slot:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `generic_enabled` | BOOLEAN | False | Enable the generic slot |
| `generic_catch_unprocessed` | BOOLEAN | True | Process faces not matched by any reference |
| `generic_lora` | COMBO | -- | LoRA file for generic faces |
| `generic_lora_strength` | FLOAT | 1.0 | Generic LoRA strength |
| `generic_prompt` | STRING | -- | Prompt for generic faces |

**Optional inputs:**

| Name | Type | Description |
|------|------|-------------|
| `positive_base` | CONDITIONING | Base positive conditioning (combined with slot prompt) |
| `negative` | CONDITIONING | Negative conditioning |
| `dd_options` | DD_OPTIONS | Fine-tuned Detail Daemon parameters |
| `inpaint_options` | INPAINT_OPTIONS | Per-slot mask type and repeat settings |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `images` | IMAGE | Final detailed images |
| `refined` | IMAGE | All inpainted crop regions |
| `refined_references` | IMAGE | Crops from reference slots only |
| `refined_generic` | IMAGE | Crops from the generic slot only |

**Note:** Empty outputs return a 64x64 black placeholder image.

---

### 4. Detail Daemon Options

**Display name:** Detail Daemon Options

Fine-tunes the sigma manipulation curve used by Detail Daemon within Person Detailer. Controls when and how strongly detail enhancement is applied during the denoising process.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `dd_start` | FLOAT | 0.0 | Start point of the sigma curve (0.0--1.0) |
| `dd_end` | FLOAT | 0.5 | End point of the sigma curve (0.0--1.0) |
| `dd_bias` | FLOAT | 1.0 | Bias of the manipulation curve |
| `dd_exponent` | FLOAT | 1.0 | Exponent shaping the curve |
| `dd_start_offset` | FLOAT | 0.0 | Offset at curve start |
| `dd_end_offset` | FLOAT | 0.0 | Offset at curve end |
| `dd_fade` | FLOAT | 0.0 | Fade amount at curve boundaries |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `dd_options` | DD_OPTIONS | Connect to Person Detailer's `dd_options` input |

---

### 5. Inpaint Options

**Display name:** Inpaint Options

Configures per-slot mask types, repeat counts, and Detail Daemon toggles for Person Detailer. Allows different inpainting strategies for each reference slot (e.g., face mask for one person, head mask for another).

#### Inputs

**Global settings:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `mask_fill_holes` | BOOLEAN | True | Fill holes in inpaint masks |
| `context_expand_factor` | FLOAT | 1.20 | Context area expansion factor |
| `output_padding` | INT | 32 | Output padding in pixels |

**Per-slot settings (ref_1 through ref_5 + generic):**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `slot_mask_type` | COMBO | face | Mask type: `face`, `head`, `body`, `hair`, `facial_skin`, `eyes`, `mouth`, `neck`, `accessories`, `aux` |
| `slot_repeat` | INT | 1 | Number of inpaint passes for this slot |
| `slot_detail_daemon` | BOOLEAN | True | Enable Detail Daemon for this slot |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `inpaint_options` | INPAINT_OPTIONS | Connect to Person Detailer's `inpaint_options` input |

---

## Node Reference -- Color Tools

All Color Tool nodes appear under **FVM Tools/Color** in the ComfyUI menu.

### 6. Color Palette Generator

**Display name:** Color Palette Generator

Generates harmonious named color palettes using color theory principles. Includes 161 fashion-relevant color names, 7 harmony types, and 14 style presets. The output palette string contains named colors that can be injected into prompts via Prompt Color Replace.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `seed` | INT | 0 | Random seed for reproducible palettes |
| `num_colors` | INT | 5 | Number of colors to generate (2--8) |
| `harmony_type` | COMBO | auto | Color harmony rule: `auto`, `analogous`, `complementary`, `split_complementary`, `triadic`, `tetradic`, `monochromatic` |
| `style_preset` | COMBO | general | Style preset (see list below) |
| `vibrancy` | FLOAT | 0.5 | Color vibrancy/saturation (0.0--1.0) |
| `contrast` | FLOAT | 0.5 | Contrast between palette colors (0.0--1.0) |
| `warmth` | FLOAT | 0.5 | Color temperature bias (0.0=cool, 1.0=warm) |
| `neutral_ratio` | FLOAT | 0.4 | Ratio of neutral tones in palette (0.0--1.0) |
| `include_metallics` | BOOLEAN | True | Include metallic color names |

**Optional inputs:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `palette_source` | COMBO | generate | Source mode: `generate` or `from_file` |
| `wildcard_file` | STRING | -- | Path to external palette file |
| `palette_index` | INT | -1 | Index within file (-1 = random) |

**Available style presets:** `general`, `beach`, `urban_streetwear`, `evening_gala`, `casual_daywear`, `vintage_retro`, `cyberpunk_neon`, `pastel_dream`, `earthy_natural`, `monochrome_chic`, `tropical`, `winter_cozy`, `festival`, `office_professional`

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `palette_string` | STRING | Serialized palette for Prompt Color Replace |
| `color_1` -- `color_8` | STRING | Individual named colors |
| `primary` | STRING | Primary color (most vivid) |
| `secondary` | STRING | Secondary color |
| `accent` | STRING | Accent color (most distinct hue) |
| `neutral` | STRING | Neutral tone |
| `metallic` | STRING | Metallic or second neutral |
| `palette_preview` | IMAGE | Visual swatch preview |
| `palette_info` | STRING | Human-readable palette details |

---

### 7. Palette From Image

**Display name:** Palette From Image

Extracts a color palette from a reference image using K-Means clustering. Implements a pure numpy K-Means algorithm with no scikit-learn dependency. Supports multiple extraction modes including fashion-aware analysis that prioritizes clothing colors over skin and background.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | IMAGE | -- | Reference image to analyze |
| `num_colors` | INT | 5 | Number of colors to extract (2--8) |
| `extraction_mode` | COMBO | dominant | Extraction strategy: `dominant`, `vibrant`, `fashion_aware` |
| `ignore_background` | BOOLEAN | False | Attempt to exclude background pixels |
| `ignore_skin` | BOOLEAN | False | Attempt to exclude skin-tone pixels |
| `sample_region` | COMBO | full | Image region to sample: `full`, `center_crop`, `upper_half`, `lower_half` |
| `saturation_threshold` | FLOAT | -- | Minimum saturation for included colors |
| `include_neutrals` | BOOLEAN | True | Include neutral/desaturated colors |
| `include_metallics` | BOOLEAN | True | Include metallic color names |
| `seed` | INT | 0 | Random seed for K-Means initialization |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `palette_string` | STRING | Serialized palette for Prompt Color Replace |
| `color_1` -- `color_8` | STRING | Individual named colors |
| `primary` | STRING | Primary color (most vivid) |
| `secondary` | STRING | Secondary color |
| `accent` | STRING | Accent color |
| `neutral` | STRING | Neutral tone |
| `metallic` | STRING | Metallic or second neutral |
| `palette_preview` | IMAGE | Visual swatch preview |
| `source_annotated` | IMAGE | Source image with color region annotations |
| `palette_info` | STRING | Extraction details and color information |

---

### 8. Prompt Color Replace

**Display name:** Prompt Color Replace

Replaces color tag placeholders in prompts with actual color names from a palette. Tags use the `#tagname#` format (e.g., `#color1#`, `#primary#`). Supports both positional tags (`#color1#` through `#color8#`) and semantic role tags (`#primary#`, `#accent#`, etc.). See the [Color Tag Reference](#color-tag-reference) for the full list of supported tags.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `prompt` | STRING | -- | Prompt text containing `#color#` tags (multiline) |
| `palette_string` | STRING | -- | Palette string from Color Palette Generator or Palette From Image (force input) |
| `primary` | STRING | -- | Override for `#primary#` tag (force input, optional) |
| `secondary` | STRING | -- | Override for `#secondary#` tag (force input, optional) |
| `accent` | STRING | -- | Override for `#accent#` tag (force input, optional) |
| `neutral` | STRING | -- | Override for `#neutral#` tag (force input, optional) |
| `metallic` | STRING | -- | Override for `#metallic#` tag (force input, optional) |
| `fallback_color` | STRING | black | Color used when a tag cannot be resolved |
| `strip_hyphens` | BOOLEAN | True | Remove hyphens from color names (e.g., "blue-gray" becomes "blue gray") |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `prompt` | STRING | Prompt with all color tags replaced |
| `replacements_log` | STRING | Log of all tag replacements performed |

---

## Node Reference -- Fashion Tools

All Fashion Tool nodes appear under **FVM Tools/Fashion** in the ComfyUI menu.

### 9. Outfit Generator

**Display name:** Outfit Generator

Generates seed-controlled outfit descriptions with color tag placeholders, ready for colorization via Prompt Color Replace. Ships with 40 built-in outfit sets covering business, casual, evening, sheer/layered, lingerie, and many more themes. Outfit list files are plain text and can be edited or extended without restarting ComfyUI.

Each generated outfit assembles garment pieces from the selected outfit set, filtered by the `formality` and `coverage` sliders, and inserts `#color#` tags for later replacement.

#### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `outfit_set` | COMBO | general_female | Outfit set to draw from (dynamically scanned) |
| `seed` | INT | 0 | Random seed for reproducible outfits |
| `style_preset` | COMBO | general | Style preset (shared with Color Palette Generator) |
| `formality` | FLOAT | 0.5 | Formality level (0.0=casual, 1.0=formal) |
| `coverage` | FLOAT | 0.5 | Coverage level (0.0=minimal, 1.0=full coverage) |
| `enable_headwear` | BOOLEAN | True | Include headwear in outfit |
| `enable_top` | BOOLEAN | True | Include top garment |
| `enable_outerwear` | BOOLEAN | True | Include outerwear/jacket |
| `enable_bottom` | BOOLEAN | True | Include bottom garment |
| `enable_footwear` | BOOLEAN | True | Include footwear |
| `enable_accessories` | BOOLEAN | True | Include accessories |
| `enable_bag` | BOOLEAN | True | Include bag/purse |
| `print_probability` | FLOAT | 0.3 | Probability (0.0--1.0) of adding prints/patterns/logos/text to garments |
| `text_mode` | COMBO | auto | How text on clothing is rendered: `auto`/`quoted` = exact text in quotes (ZImage Turbo/Flux2), `descriptive` = generic description (SD/SDXL safe), `off` = no text decorations |
| `prefix` | STRING | -- | Text prepended to the outfit prompt |
| `separator` | STRING | , | Separator between garment items |

**Optional inputs:**

| Name | Type | Description |
|------|------|-------------|
| `override_string` | STRING | Manual override for outfit (bypasses generation) |

#### Outputs

| Name | Type | Description |
|------|------|-------------|
| `outfit_prompt` | STRING | Generated outfit description with `#color#` tags |
| `outfit_details` | STRING | Breakdown of selected garments |
| `outfit_info` | STRING | Full generation metadata |

**Available outfit themes:** `general`, `business_female_skirt`, `business_female_dress`, `business_female2`, `business_male`, `casual`, `night_out`, `party`, `beach_holiday`, `winter_wonderland`, `festival`, `sporty`, `date_night`, `wedding_guest`, `streetwear`, `bohemian`, `preppy`, `gothic`, `vintage_retro`, `athleisure`, `sheer_business_female`, `sheer_casual_female`, `sheer_evening_female`, `female_lingerie` -- most themes have `_female` and `_male` variants. See the [Outfit Guide](documentation/OUTFIT_GUIDE.md#available-outfit-sets) for the full list of all 40 sets.

#### Usage Example

Outfit Generator produces prompts like:

```
#color1# silk blouse, #color2# wool pencil skirt, #accent# leather heels, #neutral# cashmere cardigan
```

When connected through Prompt Color Replace with a palette, the tags become actual colors:

```
ivory silk blouse, navy wool pencil skirt, burgundy leather heels, charcoal cashmere cardigan
```

---

## Color Tag Reference

All tags use the `#tagname#` format and are replaced by Prompt Color Replace.

| Tag | Alias | Maps To |
|-----|-------|---------|
| `#color1#` | `#c1#` | Palette position 1 |
| `#color2#` | `#c2#` | Palette position 2 |
| `#color3#` | `#c3#` | Palette position 3 |
| `#color4#` | `#c4#` | Palette position 4 |
| `#color5#` | `#c5#` | Palette position 5 |
| `#color6#` | `#c6#` | Palette position 6 |
| `#color7#` | `#c7#` | Palette position 7 |
| `#color8#` | `#c8#` | Palette position 8 |
| `#primary#` | `#pri#` | Position 1 (most vivid color) |
| `#secondary#` | `#sec#` | Position 2 |
| `#accent#` | `#acc#` | Position 3 (most distinct hue) |
| `#neutral#` | `#neu#` | Position 4 (neutral tone) |
| `#metallic#` | `#met#` | Position 5 (metallic or second neutral) |

Tags are case-insensitive. If a tag cannot be resolved (e.g., `#color7#` when only 5 colors were generated), the `fallback_color` is used.

---

## Outfit Set Customization

### Directory Structure

Each outfit set is a directory inside `outfit_lists/` containing up to 10 text files:

```
outfit_lists/
  my_custom_set/
    top.txt
    bottom.txt
    footwear.txt
    headwear.txt
    outerwear.txt
    accessories.txt
    bag.txt
    fabrics.txt
    prints.txt          # optional: patterns, logos, graphics
    texts.txt           # optional: text/slogans on clothing
```

The `prints.txt` and `texts.txt` files are optional. When present, the Outfit Generator can add decorations to garments based on the `print_probability` setting. See the [Outfit Guide](documentation/OUTFIT_GUIDE.md#prints-text-and-logos) for full format documentation and model compatibility notes.

### File Format

Each line defines one garment option:

```
garment_name | probability | formality_min-formality_max | fabric1,fabric2,...
```

Example (`top.txt`):

```
# Lines starting with # are comments
tank top | 0.50 | 0.0-0.2 | cotton,jersey,mesh
blouse | 0.65 | 0.3-0.7 | silk,chiffon,cotton,satin,lace
dress shirt | 0.70 | 0.6-1.0 | cotton,poplin,silk
```

- **probability** -- Base chance of selection (0.0--1.0)
- **formality range** -- The garment is only eligible when the node's `formality` slider falls within this range
- **fabrics** -- Comma-separated fabric options; one is randomly selected

### Creating Custom Sets

1. Create a new directory under `outfit_lists/` (e.g., `outfit_lists/my_style_female/`)
2. Copy the 8 `.txt` files from an existing set as a template
3. Edit the files to define your garments
4. The new set appears automatically in the node's dropdown -- no restart needed

### Custom Path via outfit_config.ini

To store outfit lists outside the node pack directory, edit `outfit_config.ini`:

```ini
[paths]
custom_lists_path = D:\MyOutfitLists
```

When a custom path is set, only sets in that directory are available. Copy any built-in sets you want to keep into your custom directory.

### Default Slot Probabilities

The `[defaults]` section in `outfit_config.ini` controls baseline probabilities for optional slots:

```ini
[defaults]
default_outfit_set = general_female
headwear_probability = 0.15
outerwear_probability = 0.25
accessories_probability = 0.50
bag_probability = 0.20
```

Changes to `outfit_config.ini` and outfit list files take effect on the next node execution -- no restart needed.

---

## Z-Image Turbo Compatibility

Person Detailer automatically detects Z-Image Turbo (Lumina2) models and converts LoRAs trained with diffusers-style trainers on the fly. Z-Image Turbo uses fused QKV attention layers, while standard LoRAs have separate `to_q`/`to_k`/`to_v` projections -- these are automatically concatenated into the fused format when needed.

No user action required. The conversion is logged in the console:

```
[FVMTools] Auto-converting LoRA 'my_lora.safetensors' for Z-Image Turbo QKV format
```

---

## Detailed Guides

For in-depth documentation beyond this README, see the guides in the `documentation/` folder:

| Guide | Description |
|-------|-------------|
| [Color Guide](documentation/COLOR_GUIDE.md) | Complete reference for Color Palette Generator, Palette From Image, and Prompt Color Replace -- includes style presets, example prompts, and tips for sheer/layered clothing |
| [Outfit Guide](documentation/OUTFIT_GUIDE.md) | Full guide for the Outfit Generator -- creating custom outfit sets, fabric system, override strings, formality tuning, and layering tips |
| [Outfit Installation](documentation/OUTFIT_INSTALLATION.md) | Setup and configuration for the outfit system, including custom paths via `outfit_config.ini` |
| [Person Detailer Spec](documentation/comfyui_person_detailer_spec.md) | Technical specification for the face detection, matching, and per-person LoRA inpainting pipeline |

---

## Credits and References

This project builds on the work of several open-source projects:

| Component | Author / Project | License |
|-----------|-----------------|---------|
| Face detection and embeddings | [InsightFace](https://github.com/deepinsight/insightface) (Jia Guo et al.) | MIT |
| BiSeNet face parsing | [facexlib](https://github.com/xinntao/facexlib) (Xintao Wang) | BSD |
| Detail Daemon concept | [ComfyUI-Detail-Daemon](https://github.com/Jonseed/ComfyUI-Detail-Daemon) (Jonseed) | MIT |
| Z-Image Turbo QKV conversion | [Comfyui-ZiT-Lora-loader](https://github.com/capitan01R/Comfyui-ZiT-Lora-loader) (capitan01R) | MIT |
| SAM2 segmentation | [Segment Anything 2](https://github.com/facebookresearch/sam2) (Meta AI) | Apache 2.0 |
| SAM integration | [ComfyUI Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) (ltdrdata) | GPL-3.0 |
| Color theory harmonies | Classical color wheel theory (analogous, complementary, triadic, tetradic, split-complementary) | -- |
| K-Means clustering | Pure numpy implementation (no scikit-learn dependency) | -- |

---

## License

MIT
