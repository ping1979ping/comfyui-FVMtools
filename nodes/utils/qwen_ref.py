"""Helpers for the Qwen Image 2.1 reference head detailer.

Pure functions, kept apart from the node so they can be tested without a GPU:
reference preparation (split, face crop, resize to the Qwen pixel budget, grid),
prompt assembly and PERSON_DATA mask lookup.
"""

import math

import cv2
import numpy as np
import torch

import comfy.utils

# TextEncodeQwenImage21 accepts image_1..image_16, but the official edit
# templates stay at ten; image_1 is the canvas, so nine references at most.
MAX_TOTAL_IMAGES = 10
MAX_REFS = MAX_TOTAL_IMAGES - 1

MASK_TYPE_PARTS = {
    "head": ("head",),
    "head+neck": ("head", "neck"),
    "face": ("face",),
    "face+hair": ("face", "hair"),
}

# Expression and pose are bound to <image1> explicitly and, with
# expression_hint=auto, described in words ({expression}): measured on
# IMG_0014, the plain "keep its facial expression" lost every open-mouth smile
# whose reference photos were neutral (the model copies the portraits' closed
# mouths); the worded hint brought all of them back.
#
# Skin tone, light and colour come from <image1>, only the identity (and,
# with take_hairstyle=yes, the hair) from the references: the round-1 prompt
# took "skin tone" from the portraits, and in warm evening light (IMG_0001)
# the swapped faces came out pale and cool against their own necks.
DEFAULT_PROMPT_TEMPLATE = (
    "Replace the face of the person in the center of {canvas} with the face of the person shown in {refs}, "
    "so that it is unmistakably this person. Keep the facial expression, mouth shape, open or closed mouth, "
    "visible teeth, gaze and head pose of the person in {canvas} exactly as they are; only the identity "
    "changes. {refs} show the same person ({ref_count}); take the identity and face shape{hair} from them, "
    "but not their facial expression, skin tone, lighting or colors. {canvas} is the canvas: keep its "
    "lighting, skin tone, color grading, exposure and camera perspective, and keep everything else in "
    "{canvas} unchanged. {expression} {extra}"
)

# {hair} of the template, by take_hairstyle.
HAIR_TEXTS = {
    "yes": " as well as the hairstyle and hair color",
    "no": " (keep the hairstyle and hair color of the person in {canvas})",
}

# Candidate scoring: a candidate whose mouth state (open/closed) differs
# from the original face loses this much of its identity cosine.
MOUTH_MISMATCH_PENALTY = 0.25
# ... and this much when the head turned: max(|d yaw|, |d pitch|) above
# POSE_TOLERANCE degrees (InsightFace pose). IMG_0014, K=4: the best-scoring
# Jerri candidate had turned her head by 15 degrees.
POSE_TOLERANCE = 10.0
POSE_MISMATCH_PENALTY = 0.15
# ... and SMILE_WEIGHT per unit of corner-lift change beyond SMILE_TOLERANCE
# (mouth_metrics smile, eye-distance units): K=4 robustness runs kept the
# mouth state but often grinned wider than the original (+0.04..0.07).
SMILE_TOLERANCE = 0.03
SMILE_WEIGHT = 2.0
# ... and MAR_SHRINK_PENALTY when an open mouth closed to less than MAR_KEEP of
# its opening (still "open", but IMG_0010 Jerri went mar 0.21 -> 0.10).
MAR_KEEP = 0.6
MAR_SHRINK_PENALTY = 0.15
NO_FACE_SCORE = -10.0

# Worded expression of the canvas face, picked from mouth landmarks (no LLM).
EXPRESSION_TEXTS = {
    # "laugh" is kept mild on purpose: "opened just as wide" doubled the
    # measured mouth opening (Jerri, IMG_0014: mar 0.28 -> 0.56).
    "laugh": (
        "In {canvas} the person is laughing, smiling broadly with the mouth open and the teeth showing, "
        "cheeks raised. The new face must have exactly this open-mouth laughing smile."
    ),
    "toothy_smile": (
        "In {canvas} the person is smiling broadly with parted lips, showing the upper teeth, cheeks "
        "raised. The new face must have exactly this broad toothy smile."
    ),
    "closed_smile": (
        "In {canvas} the person has a gentle closed-mouth smile. The new face must have exactly this "
        "closed-mouth smile, lips together."
    ),
    "neutral": (
        "In {canvas} the person has a relaxed, neutral expression with closed lips. The new face must "
        "keep this neutral expression with closed lips."
    ),
}

_COUNT_WORDS = {
    1: "this photo",
    2: "these 2 photos",
    3: "these 3 photos",
    4: "these 4 photos",
}


# ── Geometry ──────────────────────────────────────────────────────────────


def qwen_dims(width, height, resolution):
    """Target size exactly as TextEncodeQwenImage21 computes it.

    resolution > 0: area ~ resolution², aspect kept, both sides multiples of 32.
    resolution == 0: each side rounded to a multiple of 32.
    """
    if resolution > 0:
        ratio = width / height
        w = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
        h = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
    else:
        w, h = round(width / 32) * 32, round(height / 32) * 32
    return max(32, w), max(32, h)


def face_square_box(bbox, img_w, img_h, factor):
    """Square crop around a face bbox (x0, y0, x1, y1).

    Side = factor · max(face w, face h), centre shifted up by 0.1 · face h so
    hair and forehead fit, then pushed back inside the image. The side is
    capped at the shorter image side.
    """
    x0, y0, x1, y1 = [float(v) for v in bbox]
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    side = int(round(min(factor * max(bw, bh), img_w, img_h)))
    side = max(1, side)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0 - 0.1 * bh
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    left = max(0, min(left, img_w - side))
    top = max(0, min(top, img_h - side))
    return left, top, left + side, top + side


def mask_bbox(mask_2d, threshold=0.5):
    """(x0, y0, x1, y1) inclusive bbox of mask > threshold, or None."""
    coords = torch.nonzero(mask_2d > threshold)
    if coords.numel() == 0:
        return None
    y0, x0 = coords.min(dim=0).values.tolist()
    y1, x1 = coords.max(dim=0).values.tolist()
    return int(x0), int(y0), int(x1), int(y1)


# ── Image helpers ─────────────────────────────────────────────────────────


def resize_image(image, width, height):
    """[1,H,W,C] → [1,height,width,C] (lanczos); no-op when already that size."""
    if image.shape[2] == width and image.shape[1] == height:
        return image
    out = comfy.utils.common_upscale(
        image.movedim(-1, 1), width, height, "lanczos", "disabled"
    )
    return out.movedim(1, -1).clamp(0.0, 1.0)


def _to_bgr_uint8(image):
    """[1,H,W,3] float → HxWx3 BGR uint8 (InsightFace input)."""
    rgb = (
        (image[0, :, :, :3].detach().cpu().numpy() * 255.0)
        .clip(0, 255)
        .astype(np.uint8)
    )
    return np.ascontiguousarray(rgb[:, :, ::-1])


def crop_to_face(image, analyzer, factor):
    """Crop [1,H,W,3] to a square around its largest face.

    Returns (cropped, found). Without an analyzer or a face the image comes
    back untouched.
    """
    if analyzer is None:
        return image, False
    try:
        faces = analyzer.detect_faces(_to_bgr_uint8(image))
    except Exception as e:  # detector trouble must not kill the render
        print(f"[FVM QwenRef] face detection failed on reference: {e}")
        return image, False
    if not faces:
        return image, False
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    h, w = image.shape[1], image.shape[2]
    x0, y0, x1, y1 = face_square_box(face.bbox, w, h, factor)
    return image[:, y0:y1, x0:x1, :], True


def prepare_refs(
    reference_images,
    max_refs=4,
    ref_mode="separate",
    ref_crop="face",
    ref_crop_factor=2.2,
    ref_resolution=512,
    analyzer=None,
):
    """Split a reference batch into Qwen-ready single images.

    Returns (refs, notes): refs is a list of [1,h,w,3] tensors (h, w multiples
    of 32); in grid mode it holds one grid image. notes is a list of strings
    for the info output.
    """
    notes = []
    if reference_images is None or reference_images.shape[0] == 0:
        return [], ["no reference images"]
    imgs = reference_images[..., :3].float()
    count = imgs.shape[0]
    cap = max(1, min(int(max_refs), MAX_REFS))
    if ref_mode == "first_only":
        cap = 1
    if count > cap:
        notes.append(f"{count} references given, using the first {cap}")
    refs = []
    for i in range(min(count, cap)):
        img = imgs[i : i + 1]
        found = False
        if ref_crop == "face":
            img, found = crop_to_face(img, analyzer, ref_crop_factor)
            if not found:
                notes.append(f"ref {i + 1}: no face found, using whole image")
        w, h = qwen_dims(img.shape[2], img.shape[1], ref_resolution)
        refs.append(resize_image(img, w, h))
    if ref_mode == "grid" and len(refs) > 1:
        refs = [make_grid(refs, ref_resolution)]
    return refs, notes


def make_grid(refs, cell=512):
    """Tile references on a white grid, cols = ceil(sqrt(N)).

    Every reference is fitted (aspect kept) into a square cell of `cell` px
    (rounded to 32), so the grid spends about N·cell² pixels and both sides
    stay multiples of 32.
    """
    n = len(refs)
    if n == 0:
        raise ValueError("make_grid needs at least one image")
    cell = max(32, int(round(cell / 32)) * 32)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    grid = torch.ones(1, rows * cell, cols * cell, 3, dtype=torch.float32)
    for i, ref in enumerate(refs):
        r, c = divmod(i, cols)
        h, w = ref.shape[1], ref.shape[2]
        scale = cell / max(h, w)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        tile = resize_image(ref[..., :3].float(), nw, nh)
        oy = r * cell + (cell - nh) // 2
        ox = c * cell + (cell - nw) // 2
        grid[:, oy : oy + nh, ox : ox + nw, :] = tile
    return grid


OPEN_MOUTH_KINDS = ("laugh", "toothy_smile")

# ── Expression ────────────────────────────────────────────────────────────


def mouth_metrics(landmarks_68, bgr):
    """Mouth shape from 68-point iBUG landmarks (InsightFace landmark_3d_68, x/y used).

    Points are rotated so the eye line is level and scaled by the eye distance
    (iod). mar = inner-lip opening / inner mouth width; smile = corner lift
    relative to the lip centre / iod (> 0: corners up); teeth = bright,
    unsaturated pixels inside the inner lip contour / iod².
    """
    L = np.asarray(landmarks_68, dtype=np.float64)[:, :2]
    le, re = L[36:42].mean(0), L[42:48].mean(0)
    iod = float(np.linalg.norm(re - le)) + 1e-6
    ang = math.atan2(re[1] - le[1], re[0] - le[0])
    c, s = math.cos(-ang), math.sin(-ang)
    A = (L - (le + re) / 2.0) @ np.array([[c, -s], [s, c]]).T
    inner_w = abs(A[64, 0] - A[60, 0]) + 1e-6
    inner_h = np.mean([A[67, 1] - A[61, 1], A[66, 1] - A[62, 1], A[65, 1] - A[63, 1]])
    mar = float(max(0.0, inner_h) / inner_w)
    smile = float(((A[51, 1] + A[57, 1]) / 2.0 - (A[48, 1] + A[54, 1]) / 2.0) / iod)
    teeth = 0.0
    if bgr is not None:
        mask = np.zeros(bgr.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(L[60:68]).astype(np.int32)], 1)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        bright = (hsv[:, :, 2] > 140) & (hsv[:, :, 1] < 90) & (mask > 0)
        teeth = float(bright.sum() / (iod * iod))
    return {"mar": mar, "smile": smile, "teeth": teeth}


def classify_expression(metrics):
    """laugh / toothy_smile / closed_smile / neutral from mouth_metrics().

    An "open mouth, slight smile" class with its own wording was tried for
    open mouths with barely lifted corners (Brad): all 4 candidates came back
    with closed lips (IMG_0010), against 1 of 4 open with the toothy wording.
    Over-wide grins are handled by the candidate score instead.
    """
    mar, smile, teeth = metrics["mar"], metrics["smile"], metrics["teeth"]
    if teeth > 0.015 or mar > 0.12:
        return "laugh" if mar >= 0.30 else "toothy_smile"
    if smile > 0.03:
        return "closed_smile"
    return "neutral"


def pick_face_for_mask(faces, mask_2d):
    """The detected face whose bbox is covered most by the person's mask, or None."""
    m = (
        mask_2d.detach().cpu().numpy()
        if hasattr(mask_2d, "detach")
        else np.asarray(mask_2d)
    )
    h, w = m.shape[:2]
    best, best_cov = None, 0.3
    for f in faces or []:
        x0, y0, x1, y1 = [int(round(float(v))) for v in f.bbox]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        cov = float((m[y0:y1, x0:x1] > 0.5).mean())
        if cov > best_cov:
            best, best_cov = f, cov
    return best


# ── Prompt ────────────────────────────────────────────────────────────────


def _ref_tags(n):
    tags = [f"<image{i}>" for i in range(2, n + 2)]
    if len(tags) == 1:
        return tags[0]
    return ", ".join(tags[:-1]) + " and " + tags[-1]


def build_prompt(
    template, n_refs, ref_mode="separate", extra="", expression="", hair="yes"
):
    """Fill {hair}, {canvas}, {refs}, {ref_count}, {expression}, {extra} by plain replacement.

    hair: "yes"/"no" picks HAIR_TEXTS (any other string is inserted as is).

    str.replace, not str.format: extra text may contain braces. A template
    without {expression} gets the expression text appended before {extra}'s
    text would go (i.e. at the end).
    """
    if ref_mode == "grid" and n_refs > 1:
        refs, count = "<image2>", "this photo grid"
    elif ref_mode == "first_only" or n_refs <= 1:
        refs, count = "<image2>", "this photo"
    else:
        refs, count = (
            _ref_tags(n_refs),
            _COUNT_WORDS.get(n_refs, f"these {n_refs} photos"),
        )
    text = template or DEFAULT_PROMPT_TEMPLATE
    expression = (expression or "").strip()
    if expression and "{expression}" not in text:
        text = text + " " + expression
    text = (
        text.replace("{hair}", HAIR_TEXTS.get(hair, hair or ""))
        .replace("{expression}", expression)
        .replace("{canvas}", "<image1>")
        .replace("{refs}", refs)
        .replace("{ref_count}", count)
        .replace("{extra}", (extra or "").strip())
    )
    return " ".join(text.split())


# ── PERSON_DATA ───────────────────────────────────────────────────────────


def get_person_mask(person_data, ri, b, mask_type, height, width):
    """Mask of reference `ri` (0-based) in batch image `b` as [H, W].

    Returns (mask, reason): mask is None when nothing usable exists and
    reason says why. Unions for head+neck / face+hair; a mask of another
    size is resized (with a note in reason).
    """
    num_refs = int(person_data.get("num_references", 0))
    if ri < 0 or ri >= num_refs:
        return None, f"reference {ri + 1} not in PERSON_DATA ({num_refs} references)"
    matches = person_data.get("matches") or []
    if matches:
        mb = matches[min(b, len(matches) - 1)]
        if ri < len(mb) and not mb[ri]:
            return None, f"reference {ri + 1} not matched in image {b + 1}"
    parts = MASK_TYPE_PARTS.get(mask_type, (mask_type,))
    union = None
    for part in parts:
        per_ref = person_data.get(f"{part}_masks")
        if per_ref is None or ri >= len(per_ref):
            continue
        m = per_ref[ri]
        if m is None or m.numel() == 0:
            continue
        if m.ndim == 2:
            m = m.unsqueeze(0)
        m = m[min(b, m.shape[0] - 1)].float()
        union = m if union is None else torch.maximum(union, m)
    if union is None or union.sum().item() < 1.0:
        return None, f"empty {mask_type} mask for reference {ri + 1} in image {b + 1}"
    reason = ""
    if union.shape != (height, width):
        reason = f"mask {tuple(union.shape)} resized to {(height, width)}"
        union = torch.nn.functional.interpolate(
            union[None, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return union.cpu(), reason


# ── Candidates ────────────────────────────────────────────────────────────


def candidate_seed(seed, b, k, candidates):
    """Seed of candidate k (0-based) for batch image b.

    One candidate keeps the old seed + b; K candidates of image b use
    seed + b·K + k, so batch image 0 renders seed, seed+1, …, seed+K-1.
    """
    return int(seed) + int(b) * max(1, int(candidates)) + int(k)


def mouth_state(kind):
    """"open" / "closed" for a classify_expression() kind, None for None."""
    if kind is None:
        return None
    return "open" if kind in OPEN_MOUTH_KINDS else "closed"


def pose_delta(orig_pose, cand_pose):
    """max(|d pitch|, |d yaw|) in degrees of two InsightFace poses (pitch, yaw, roll), or None."""
    if orig_pose is None or cand_pose is None:
        return None
    return float(max(abs(cand_pose[0] - orig_pose[0]), abs(cand_pose[1] - orig_pose[1])))


def candidate_score(
    sim,
    orig_kind,
    cand_kind,
    penalty=MOUTH_MISMATCH_PENALTY,
    d_pose=None,
    pose_tolerance=POSE_TOLERANCE,
    pose_penalty=POSE_MISMATCH_PENALTY,
    d_smile=None,
    mar_ratio=None,
):
    """Identity cosine minus `penalty` when the mouth state left the original's,
    minus `pose_penalty` when the head turned more than `pose_tolerance` degrees,
    minus SMILE_WEIGHT·(|d_smile| - SMILE_TOLERANCE) for a smile that changed,
    minus MAR_SHRINK_PENALTY when an open mouth kept less than MAR_KEEP of its
    opening (mar_ratio = candidate mar / original mar, only for open originals).

    sim None (no face found in the candidate) → NO_FACE_SCORE, the lowest
    priority. Unknown mouth states / poses are not penalised.
    """
    if sim is None:
        return NO_FACE_SCORE
    score = float(sim)
    o, c = mouth_state(orig_kind), mouth_state(cand_kind)
    if o is not None and c is not None and o != c:
        score -= penalty
    if d_pose is not None and d_pose > pose_tolerance:
        score -= pose_penalty
    if d_smile is not None and abs(d_smile) > SMILE_TOLERANCE:
        score -= SMILE_WEIGHT * (abs(d_smile) - SMILE_TOLERANCE)
    if (
        mar_ratio is not None
        and mouth_state(orig_kind) == "open"
        and mouth_state(cand_kind) == "open"
        and mar_ratio < MAR_KEEP
    ):
        score -= MAR_SHRINK_PENALTY
    return score


def pick_candidate(scores):
    """Index of the best score; ties go to the earlier (lower-seed) candidate."""
    if not scores:
        raise ValueError("pick_candidate needs at least one score")
    best = 0
    for i, s in enumerate(scores):
        if s > scores[best]:
            best = i
    return best


def format_candidates(rows, chosen):
    """One info line: 'candidates: seed 5 sim 0.712 open ok = 0.712 | … -> seed 6'.

    rows: dicts with seed, sim (None = no face), kind, mismatch, score.
    """
    parts = []
    for i, r in enumerate(rows):
        if r["sim"] is None:
            txt = f"seed {r['seed']} no face"
        else:
            mouth = r["kind"] or "mouth ?"
            flag = " MOUTH CHANGED" if r["mismatch"] else ""
            if r.get("d_smile") is not None:
                flag += f" dsmile {r['d_smile']:+.3f}"
            if r.get("mar_ratio") is not None:
                flag += f" mar x{r['mar_ratio']:.2f}"
            if r.get("d_pose") is not None:
                flag += f" pose {r['d_pose']:.0f}deg" + (
                    " TURNED" if r["d_pose"] > POSE_TOLERANCE else ""
                )
            txt = f"seed {r['seed']} sim {r['sim']:.3f} {mouth}{flag} = {r['score']:.3f}"
        parts.append(("*" if i == chosen else "") + txt)
    return (
        "candidates: " + " | ".join(parts) + f" -> seed {rows[chosen]['seed']}"
    )


# ── Skin tone match ───────────────────────────────────────────────────────


def _rgb_to_lab(rgb):
    """HxWx3 float RGB 0..1 → Lab (L 0..100), OpenCV float convention."""
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2LAB)


def _lab_to_rgb(lab):
    return cv2.cvtColor(np.ascontiguousarray(lab, dtype=np.float32), cv2.COLOR_LAB2RGB)


def robust_skin_lab(rgb, mask, lo=15, hi=85, min_px=30):
    """Median Lab of the masked pixels, luminance outliers (highlights, deep
    shadow) outside the lo..hi percentiles dropped. None if too few pixels."""
    sel = np.asarray(mask) > 0.5
    if int(sel.sum()) < min_px:
        return None
    px = _rgb_to_lab(rgb)[sel]
    a, b = np.percentile(px[:, 0], [lo, hi])
    px = px[(px[:, 0] >= a) & (px[:, 0] <= b)]
    if len(px) < max(5, min_px // 2):
        return None
    return np.median(px, axis=0)


# Soft skin key around the new face's skin median: sky, shirts and hair stay
# untouched. Measured on IMG_0001: shifting the whole head mask painted
# orange halos into the sky and tinted the neighbours' shirts.
SKIN_KEY_SIGMA_AB = 10.0
SKIN_KEY_SIGMA_L = 25.0


def cheek_mask(landmarks_68, shape):
    """HxW uint8 mask of both cheeks from 68-point iBUG landmarks (x/y used).

    Polygon jaw (1-4 / 12-15), mouth corner, nose wing and the lower eyelid
    (moved down by 0.12 eye distances), minus dilated eyes, nose and mouth.
    Cheeks carry no hair and no brows, so they compare the skin of two
    different faces fairly - unlike a mask cut out of one of them.
    """
    L = np.asarray(landmarks_68, dtype=np.float64)[:, :2]
    le, re = L[36:42].mean(0), L[42:48].mean(0)
    iod = float(np.linalg.norm(re - le)) + 1e-6
    ex = (re - le) / iod
    down = np.array([-ex[1], ex[0]])
    if down @ (L[8] - (le + re) / 2) < 0:
        down = -down
    h, w = shape[:2]
    cheek = np.zeros((h, w), np.uint8)
    for idx in ([1, 2, 3, 4, 48, 31, 40, 41], [15, 14, 13, 12, 54, 35, 47, 46]):
        pts = L[idx].copy()
        pts[-2:] += down * 0.12 * iod
        cv2.fillPoly(cheek, [np.round(pts).astype(np.int32)], 1)
    excl = np.zeros((h, w), np.uint8)
    for idx in (range(36, 42), range(42, 48), range(27, 36), range(48, 60)):
        hull = cv2.convexHull(np.round(L[list(idx)]).astype(np.int32))
        cv2.fillPoly(excl, [hull], 1)
    k = max(3, int(0.08 * iod)) | 1
    excl = cv2.dilate(excl, np.ones((k, k), np.uint8))
    return cv2.erode(cheek, np.ones((3, 3), np.uint8)) & (1 - excl)


def skin_tone_shift(orig_rgb, new_rgb, skin_mask, new_mask=None):
    """(Lab offset, new skin median) that moves the new face's skin onto the original's.

    The original's median is taken over `skin_mask`, the new one over
    `new_mask` (default: the same pixels). Returns (None, None) if either
    side has too few pixels.
    """
    o = robust_skin_lab(orig_rgb, skin_mask)
    n = robust_skin_lab(new_rgb, skin_mask if new_mask is None else new_mask)
    if o is None or n is None:
        return None, None
    return (o - n).astype(np.float32), n.astype(np.float32)


def skin_key(lab, key_lab, sigma_ab=SKIN_KEY_SIGMA_AB, sigma_l=SKIN_KEY_SIGMA_L):
    """0..1 similarity of each Lab pixel to the skin colour `key_lab`."""
    d_ab = (lab[..., 1] - key_lab[1]) ** 2 + (lab[..., 2] - key_lab[2]) ** 2
    d_l = (lab[..., 0] - key_lab[0]) ** 2
    return np.exp(-d_ab / (2 * sigma_ab**2) - d_l / (2 * sigma_l**2)).astype(np.float32)


# Change gate: pixels the render left (almost) as they were need no colour
# correction - on IMG_0001 the keyed shift still tinted the peach sky next to
# Brad's pale new face. ΔE76 ramp from GATE_LO (off) to GATE_HI (full).
GATE_LO = 3.0
GATE_HI = 10.0


def apply_lab_shift(rgb, shift, weight, strength=1.0, key_lab=None, ref_rgb=None):
    """Add strength·weight·shift in Lab; rgb HxWx3 float 0..1, weight HxW 0..1.

    key_lab: restrict the shift to pixels close to this skin colour
    (skin_key, lightly blurred). ref_rgb: the image before the render; the
    shift fades out where rgb barely differs from it (GATE_LO..GATE_HI).
    Pixels that end up with no weight are returned bit-identical.
    """
    out = np.array(rgb, dtype=np.float32, copy=True)
    w = np.asarray(weight, dtype=np.float32)
    ys, xs = np.nonzero(w > 1e-4)
    if shift is None or strength <= 0 or len(ys) == 0:
        return out
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    lab = _rgb_to_lab(out[y0:y1, x0:x1])
    ww = strength * w[y0:y1, x0:x1]
    k = max(3, int(0.01 * max(lab.shape[:2]))) | 1
    if key_lab is not None:
        key = skin_key(lab, np.asarray(key_lab, np.float32))
        ww = ww * cv2.GaussianBlur(key, (k, k), 0)
    if ref_rgb is not None:
        ref_lab = _rgb_to_lab(np.asarray(ref_rgb, np.float32)[y0:y1, x0:x1])
        de = np.linalg.norm(lab - ref_lab, axis=-1)
        gate = np.clip((de - GATE_LO) / (GATE_HI - GATE_LO), 0.0, 1.0).astype(np.float32)
        ww = ww * cv2.GaussianBlur(gate, (k, k), 0)
    lab += ww[..., None] * np.asarray(shift, np.float32)
    lab[..., 0] = np.clip(lab[..., 0], 0.0, 100.0)
    region = out[y0:y1, x0:x1]
    touched = ww > 1e-3
    region[touched] = np.clip(_lab_to_rgb(lab), 0.0, 1.0)[touched]
    return out
