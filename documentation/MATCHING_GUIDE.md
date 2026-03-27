# Appearance-Enhanced Face Matching

> How Person Selector Multi uses hair color and head appearance to improve face matching accuracy — especially when two people have similar facial features.

---

## The Problem

ArcFace face embeddings are excellent at identity matching, but they focus on **facial geometry** — the shape of eyes, nose, jaw. When two people have similar bone structure (e.g. siblings, or different people who just look alike), the face similarity scores land in a narrow range (0.55–0.70), making it hard to assign the correct reference.

Meanwhile, other visual cues are obvious to the human eye:

- **Alice** has blonde hair and a round face
- **Bob** has dark hair and a sharp jawline
- But ArcFace scores them at 0.62 vs 0.58 — too close for reliable assignment

## The Solution: Multi-Signal Matching

Person Selector Multi now blends **three signals** into one combined score:

```
┌─────────────────────────────────────────────────────┐
│               Combined Similarity Score              │
│                                                     │
│   ┌──────────┐  ┌───────────┐  ┌────────────────┐  │
│   │  ArcFace │  │ Hair Color│  │ Head Histogram │  │
│   │ (face)   │  │  (hair)   │  │ (head crop)    │  │
│   │  60%     │  │   20%     │  │    20%         │  │
│   └──────────┘  └───────────┘  └────────────────┘  │
│                                                     │
│   face identity   blonde vs    overall look:        │
│   from ArcFace    brunette     hair + skin + jaw    │
│   embeddings      via BiSeNet  via HSV histogram    │
└─────────────────────────────────────────────────────┘
```

### Signal 1: Face Embedding (ArcFace)

The existing matching system. InsightFace ArcFace produces a 512-dimensional embedding that encodes facial identity. Cosine similarity between embeddings tells you how likely two faces belong to the same person.

- **Strength:** Best signal for identity matching
- **Weakness:** Struggles with similar-looking faces, sensitive to expression changes
- **Cost:** Already loaded and running (InsightFace buffalo_l)

### Signal 2: Hair Color (BiSeNet Label 17)

BiSeNet already runs for mask generation and segments the image into 19 classes. Class 17 is **hair**. We extract the pixels in the hair region and compute their median color in HSV space.

```
Reference Image              Detected Face
┌──────────────┐            ┌──────────────┐
│   ┌──────┐   │            │   ┌──────┐   │
│   │ HAIR │   │            │   │ HAIR │   │
│   │blonde│   │            │   │blonde│   │
│   │H:25  │   │  compare   │   │H:28  │   │
│   │S:120 │   │ ────────►  │   │S:115 │   │
│   │V:200 │   │  similar!  │   │V:195 │   │
│   └──────┘   │            │   └──────┘   │
│   ┌──────┐   │            │   ┌──────┐   │
│   │ FACE │   │            │   │ FACE │   │
│   └──────┘   │            │   └──────┘   │
└──────────────┘            └──────────────┘
```

**How the comparison works:**

- Hue (H): The actual color — most important (blonde=25, brunette=12, red=8, black=0)
- Saturation (S): Color intensity — distinguishes grey/silver from vivid colors
- Value (V): Brightness — separates dark from light shades

The distance formula weights Hue at 50%, Saturation at 30%, Value at 20%.

- **Strength:** Instantly tells blonde from brunette. Zero extra VRAM.
- **Weakness:** Fails when hair is hidden (hat), very short, or same color on both people.
- **Cost:** Near zero — BiSeNet label map is already computed for masks.

### Signal 3: Head Histogram (HSV)

A color histogram of the **head crop** (face bbox expanded by ~40%). This captures not just hair color, but also skin tone, glasses, and upper clothing in one compact descriptor.

```
Reference Crop              Detected Crop
┌──────────────┐            ┌──────────────┐
│  dark hair   │            │  dark hair   │
│  light skin  │  compare   │  light skin  │
│  glasses     │ ────────►  │  glasses     │
│  blue top    │ histogram  │  blue jacket │
│              │  match!    │              │
└──────────────┘            └──────────────┘
```

The histogram is computed in HSV space (30 hue bins × 32 saturation bins) and compared using OpenCV's correlation metric.

- **Strength:** Holistic appearance comparison. Catches things hair color alone misses (skin tone, accessories).
- **Weakness:** Sensitive to lighting changes and clothing changes between reference and target.
- **Cost:** Pure CPU operation with numpy/OpenCV. Zero VRAM.

---

## How to Use

### The match_weights Field

The `match_weights` input on Person Selector Multi controls the blend:

```
Format: face/hair/head
```

| Setting | When to use |
|---------|-------------|
| `60/20/20` | **Default.** Balanced blend — good starting point for most workflows. |
| `100/0/0` | Pure face matching. Use when appearance should not influence matching (e.g. same person in different wigs/costumes). |
| `50/30/20` | Strong hair weight. Use when faces are similar but hair colors differ clearly. |
| `40/40/20` | Heavy hair focus. Best for blonde-vs-brunette disambiguation. |
| `70/15/15` | Mostly face with a subtle appearance boost. Conservative option. |
| `50/25/25` | Equal appearance weight. Good when both hair and overall look help. |

Values are **auto-normalized** — `3/1/1` is the same as `60/20/20`.

### Example Scenario

You have a group photo with 4 people. Two women have very similar faces (score 0.63 vs 0.60 with pure ArcFace), but one is blonde and the other brunette.

**With `100/0/0` (face only):**
```
ref_1 (blonde) → face #2 (sim 0.63) ← might be wrong person
ref_2 (brunette) → face #0 (sim 0.60) ← might be wrong person
```
The 0.03 difference is within noise — assignment could flip between runs.

**With `60/20/20` (appearance-enhanced):**
```
ref_1 (blonde) → face #2 (combined 0.72) ← hair match boosts score
ref_2 (brunette) → face #0 (combined 0.68) ← correct assignment, clear gap
```
The hair color signal widens the gap from 0.03 to 0.04+, and the histogram adds further confirmation. Assignment becomes stable.

---

## Technical Details

### Hair Color Extraction

1. BiSeNet segments the face crop into 19 classes
2. Pixels where `label_map == 17` (hair) are extracted
3. Minimum 100 hair pixels required (otherwise hair signal is skipped)
4. RGB pixels are converted to HSV color space
5. **Median** (not mean) is used — robust against highlights and shadows
6. Distance is computed as weighted Euclidean in HSV

### Head Histogram

1. Face bounding box is expanded by 40% in all directions (slightly less downward)
2. Crop is converted to HSV
3. 2D histogram (30 hue bins × 32 saturation bins) is computed
4. Histogram is L2-normalized
5. Comparison uses OpenCV `HISTCMP_CORREL` (Pearson correlation)
6. Result is mapped from [-1, 1] to [0, 1]

### Graceful Degradation

When appearance data is unavailable (e.g. no hair visible, face too small):

- Hair signal returns `None` → treated as similarity 0.0 for that pair
- Head histogram returns `None` → treated as similarity 0.0 for that pair
- The face embedding signal always works, so the system falls back gracefully
- With `100/0/0` weights, appearance is completely disabled — identical to previous behavior

### Performance Impact

| Signal | Extra VRAM | Extra Time | Notes |
|--------|-----------|-----------|-------|
| Face (ArcFace) | 0 MB | 0 ms | Already running |
| Hair Color | 0 MB | <1 ms | Reuses BiSeNet label map |
| Head Histogram | 0 MB | <1 ms | Pure numpy/OpenCV |

**Total overhead: negligible.** The BiSeNet label map is already computed for mask generation. Hair color extraction is just a masked array mean. The histogram is a single OpenCV call.

### Console Output

When appearance matching is active, the console shows additional information:

```
[PersonSelectorMulti] Appearance matching: weights=60%/20%/20%
  ref 1: hair=HSV(25,120,200), histogram=yes
  ref 2: hair=HSV(12,80,60), histogram=yes
[PersonSelectorMulti] face_sim:
[[0.6300 0.6050]
 [0.5800 0.6000]]
[PersonSelectorMulti] combined_sim (weights 60%/20%/20%):
[[0.7180 0.5420]
 [0.4920 0.6830]]
```

This helps you see exactly how appearance signals shift the scores.

---

## Tips

- **Start with defaults** (`60/20/20`). Only adjust if you see mismatches.
- **Check the console** for `face_sim` vs `combined_sim` to see how weights affect scoring.
- **Use `100/0/0`** when the same person appears in different costumes/wigs — appearance matching would hurt in that case.
- **Increase hair weight** (`40/40/20`) when you have clearly different hair colors and similar faces.
- **Reference image quality matters** — use a clear, well-lit reference with visible hair for best hair color extraction.
- **Multiple references** improve both face and appearance matching — the system aggregates across all reference images per slot.
