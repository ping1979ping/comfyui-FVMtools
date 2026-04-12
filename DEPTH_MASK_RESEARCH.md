# Depth + BiRefNet Mask Research Log

> What we tried, what worked, what didn't — so we don't start over.
> Session: 2026-04-08 to 2026-04-12

---

## The Problem

When a `depth_map` is connected to PersonSelectorMulti, body masks of **occluded people** (person behind another person) get fragmented into horizontal stripes. The back person's mask is carved at depth discontinuities and fragments at "wrong depth" are dropped.

### Root cause chain
1. `carve_mask_at_depth_edges()` in `depth_refine.py` cuts masks along depth edges
2. After cutting, connected components whose median depth falls outside the person's depth band are **dropped**
3. For a back person, the front person creates depth edges WITHIN the back person's mask
4. Fragments behind the front person have the front person's depth → wrong band → dropped → stripes

---

## What Works (shipped, in production)

### BiSeNet normalization fix
- **File:** `masker.py:_run_bisenet()` — added `(crop_t - 0.5) / 0.5` normalization
- **Result:** Face masks now correctly shaped instead of filling entire bbox
- **Commit:** `e4c2e8b`

### Depth growth guards
- **File:** `depth_refine.py:grow_mask_between_edges()`
- **Fix:** Bail out when `edges_binary` is all-zero (prevents whole-image fill), cap growth at 3× starting area
- **Result:** Masks no longer explode to fill entire image on flat depth maps
- **Commit:** `7fcfda0`

### Fused edges (bilateral + LAB Canny + depth morph gradient)
- **File:** `depth_refine.py:compute_fused_edges()`
- **Pipeline:** bilateral filter (d=9, sigmaColor=75) → Canny on LAB L-channel (30/90) → morph gradient on depth → AND with 5px-dilated depth edges
- **Result:** Only cuts where BOTH color AND depth agree. Suppresses within-person depth changes (arm forward)
- **Status:** Active when NO BiRefNet connected. Skipped when BiRefNet is connected.
- **Commit:** `dce35e9`

### Pre-carved component seed
- **File:** `depth_refine.py:carve_mask_at_depth_edges(face_bbox=...)`
- **Logic:** Before carving, find the connected component containing the face center. After carving, keep any fragment overlapping this pre-carved body.
- **Problem:** Face bbox seed only covers face+neck — lower body fragments (legs, hips) are too far below and still get dropped
- **Improvement:** Changed to use the full pre-carved connected component (not just face bbox rect). Better but still doesn't fully solve heavy occlusion.
- **Status:** Active when NO BiRefNet connected.
- **Commits:** `a2f1549`, `39483e3`

### BiRefNet as hard clip (CURRENT APPROACH)
- **File:** `masker.py:generate_all_masks_for_face()` + `person_selector_multi.py`
- **Logic:** person_mask (BiRefNet/RMBG foreground) passed as full-scene mask to every face. SAM runs normally for per-person separation. All masks clipped to BiRefNet at the end. Depth carving SKIPPED entirely when envelope connected.
- **Result:** Clean masks. SAM quality + BiRefNet edge sharpness. No stripes.
- **Status:** Active, shipped.
- **Commit:** `cc2bab9`

### Deconfliction removal
- **File:** `person_selector_multi.py` and `person_data_refiner.py`
- **Logic:** Removed the cross-reference `deconflict_masks()` pass that carved overlap pixels from the farther person
- **Reason:** Caused stripe artifacts. PersonDetailer handles occlusion via back-to-front render order.
- **Status:** Permanent removal.
- **Commit:** `208ba76`

---

## What We Tried and DIDN'T Work

### Approach 1: Split BiRefNet by SAM-seed distance transform
- **File:** `masker.py:split_person_mask_by_seeds()`
- **Idea:** Generate per-face SAM body masks (with negative prompts), use them as seeds for `cv2.distanceTransform` to split BiRefNet foreground per person
- **Problem:** SAM's negative prompts carve out the front person from the back person's mask, creating HOLES in the seed. Distance transform from holey seeds assigns those hole-pixels to the wrong person → stripes propagate from SAM into the BiRefNet split
- **Fix attempts:**
  - Contour fill + morph close (15×15) on SAM seeds before split → not enough, gaps wider than kernel
  - Exclusive cores (each seed minus overlap with others) → fixed speckle/dithering at boundaries but didn't fix the fundamental SAM hole problem
  - Connected-component cleanup (drop blobs not touching seed) → helped with small artifacts
- **Status:** Code still in masker.py but unused. Could be useful for other purposes.

### Approach 2: Watershed on bilateral-filtered image
- **File:** `masker.py:split_person_mask_by_watershed()`
- **Idea:** Use face centers as watershed markers on bilateral-filtered image. Bilateral preserves person-boundary edges while smoothing texture → watershed follows natural silhouettes
- **Problem:** TOO RESTRICTIVE. Watershed follows EVERY edge in the bilateral image — skin-to-clothing boundaries, shadow edges within a person. Each person's region was confined to a tiny area around the face.
- **Tried:** sigmaColor=75 was too low. Even higher sigma would smooth out real person boundaries too.
- **Status:** Code replaced by body-column Voronoi, then both replaced by current approach.

### Approach 3: Body-column Voronoi (weighted face-center distance)
- **File:** `masker.py:split_person_mask_by_watershed()` (reused function name)
- **Idea:** Weighted distance where horizontal distance is full weight, vertical distance below face is 0.3× weight. Produces vertical body columns.
- **Problem:** Still produced wrong-shaped columns. The 0.3× weight wasn't enough — people at similar X positions but different depths got merged. The woman (ref 1) only got 27926px (narrow strip between adjacent faces).
- **Status:** Code replaced by current approach.

### Approach 4: Face-center Voronoi (original, depth-weighted)
- **File:** `masker.py:split_person_mask_by_anchors()`
- **Idea:** Euclidean distance to face center, optionally weighted by depth similarity
- **Problem:** Diagonal cuts between adjacent people. Face centers are up high, bodies are down low → hip-level pixels are equidistant between adjacent face centers → straight diagonal bisectors through the body
- **Status:** Code still in masker.py, unused.

### Approach 5: Skip depth entirely in multi-person
- **Idea:** Just disable `refine_mask_with_depth` and `carve_mask_at_depth_edges` when `other_faces` exist
- **Problem:** User rejected — valid use case where one person's arm reaches in front of another. Depth should help distinguish "this arm at depth 0.3 belongs to person A, not person B at depth 0.6"
- **Status:** Reverted. Fused edges approach used instead (for no-BiRefNet path).

---

## Key Insights

1. **SAM with negative prompts is the best per-person segmenter we have.** Every attempt to replace it (BiRefNet splits) produced worse results. BiRefNet should CLIP SAM, not REPLACE it.

2. **Depth carving is fundamentally at odds with occlusion.** Depth edges exist both between-person AND within-person (arm forward). No edge detection strategy reliably distinguishes them. The fused-edges approach (bilateral + LAB + depth agreement) is the best compromise but still not perfect.

3. **The bilateral filter (d=9, sigmaColor=75) is useful** for edge detection preprocessing. It removes 60-80% of false internal edges (clothing texture, hair detail) while keeping person silhouette edges. But it's not enough to make watershed work for splitting.

4. **Fill-mask-holes on SAM seeds doesn't help when gaps are large.** Morph close kernel needs to be wider than the front person's body to bridge the gap — that's too destructive.

5. **PersonDetailer's back-to-front render order is the correct occlusion solution.** Mask deconfliction (carving overlap from back person) was always wrong — it destroyed the back person's mask. Render order handles it by painting back first, front last.

---

## Unused Code (still in codebase, may be useful later)

| Function | File | Purpose | Why unused |
|----------|------|---------|------------|
| `split_person_mask_by_seeds()` | masker.py | Distance-transform split from SAM seeds | SAM holes cause stripe propagation |
| `split_person_mask_by_watershed()` | masker.py | Currently contains body-column Voronoi (misleading name) | Too narrow columns |
| `split_person_mask_by_anchors()` | masker.py | Face-center Voronoi with depth weight | Diagonal cuts |
| `compute_fused_edges()` | depth_refine.py | Bilateral+LAB+depth edge fusion | Active for no-BiRefNet, skipped with BiRefNet |
| Pre-carved seed in `carve_mask_at_depth_edges()` | depth_refine.py | Keep fragments overlapping pre-carved body | Active for no-BiRefNet |

---

## Important Observation: Deconfliction Was a Double-Edged Sword

The "best" result the user saw during this session was **early on, before deconfliction
was removed** (around commit `071eea1`). At that point, SAM + depth + deconfliction
was active. Deconfliction caused STRIPES on the back person (bad), but it ALSO cleaned
up SAM's cross-person leakage by removing overlap pixels from the "loser" (good).

After removing deconfliction (`208ba76`), the stripes went away but SAM's leak from
the back person over the boy in front was no longer cleaned up. The current result
(image 19) shows ref 2 (green) extending over the boy — that's raw SAM leak with no
cleanup.

**The trade-off:** deconfliction fixes SAM leak but causes stripes. No deconfliction
fixes stripes but allows SAM leak. No solution in this session solved both simultaneously.

**Best current workaround:** For this specific scene, don't connect person_mask or
depth_map. Pure SAM with negative prompts (no depth, no BiRefNet, no deconfliction)
was actually the cleanest result — image 4 in the session.

---

## Future Ideas (not tried yet)

- **Depth only for render order + selective carving:** Instead of carving all depth edges, only carve at edges where the mask EXITS the person (touching background), not where another person occludes. Would need to classify each depth edge as "person-background" vs "person-person".
- **Guided filter with depth as guide:** Smooth the mask using depth as guidance — pixels at similar depth stay together. Would sharpen boundaries between people (different depth) while keeping within-person regions smooth.
- **Use BiRefNet per-person (multiple runs):** Instead of one BiRefNet for the whole scene, run BiRefNet with each person's face as a prompt/crop. Some BiRefNet variants support this. Would give per-person foreground directly.
- **Toggle for users:** `depth_carve_mode` combo: "off" / "fused" / "aggressive". Let users pick based on their specific scene.
