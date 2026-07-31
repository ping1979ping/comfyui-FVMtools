"""Glyph guidance for sign / text repair.

An AI-generated image has a sign whose lettering is garbled. We know the desired
replacement text and the pixel mask of the sign, so instead of asking the diffusion
model to invent letterforms we render the text as clean typography, warp it onto the
sign's actual quadrilateral (usually rotated and perspective-skewed) and hand the
result back as an init image. The model then only has to restyle material and lighting.

That is the AnyText / GlyphControl trick without an extra model: numpy, cv2 and Pillow.

Pipeline
--------
    mask  -> mask_quad()          4 ordered corners of the sign
          -> quad_size()          the un-rotated pixel size of the text box
          -> render_text_block()  clean typography, auto-fitted into that box
          -> warp_to_quad()       perspective-mapped back onto the sign
          -> composite_glyph()    alpha-blended over the source image

`render_glyph_layer()` chains the middle four steps for callers that just want a layer.
"""

import difflib
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ──── Font discovery ────

FONT_EXTENSIONS = (".ttf", ".otf")
SYSTEM_DEFAULT_LABEL = "<system default>"
REPO_FONTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "fonts")
WINDOWS_FONTS_DIR = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "Fonts")

# ──── Font matching ────

# Substrings that mark a font file as belonging to a bucket. Matched against the
# display name with everything but [a-z0-9] stripped out, so "Arial Bold" -> "arialbold".
FONT_FAMILY_KEYWORDS = {
    "sans": (
        "sans", "grotesk", "grotesque", "gothic", "arial", "helvetica", "verdana",
        "tahoma", "segoeui", "calibri", "candara", "corbel", "roboto", "inter",
        "franklin", "futura", "aileron", "lato", "montserrat", "poppins", "nunito",
    ),
    "serif": (
        "serif", "times", "georgia", "garamond", "cambria", "constantia", "palatino",
        "book", "roman", "didot", "bodoni", "baskerville", "playfair", "sylfaen",
    ),
    "condensed": (
        "condensed", "cond", "narrow", "compressed", "impact", "haettenschweiler",
        "oswald", "bebas", "archivonarrow",
    ),
    "mono": (
        "mono", "consol", "courier", "cour", "lucon", "menlo", "inconsolata",
        "typewriter", "terminal",
    ),
    "script": (
        "script", "brush", "hand", "segoesc", "inkfree", "gabriola", "mistral",
        "pristina", "freestyle", "comic", "casual", "caveat", "dancing", "vladimir",
        "edwardian", "rage", "lucidahandwriting",
    ),
    "display": (
        "display", "poster", "showcard", "broadway", "stencil", "bauhaus", "cooper",
        "jokerman", "playbill", "impact", "bebas", "alfaslab", "chiller", "harrington",
    ),
    "bold": ("bold", "bd", "black", "heavy", "semibold", "extrabold", "blk"),
    "italic": ("italic", "oblique", "ital"),
}

# Free-text words a vision LLM is likely to use, mapped onto the buckets above.
FONT_HINT_SYNONYMS = {
    "sans": "sans", "sansserif": "sans", "grotesk": "sans", "grotesque": "sans",
    "gothic": "sans", "helvetica": "sans", "arial": "sans", "neutral": "sans",
    "geometric": "sans", "humanist": "sans", "clean": "sans", "modern": "sans",
    "serif": "serif", "roman": "serif", "times": "serif", "antiqua": "serif",
    "slab": "serif", "traditional": "serif", "classical": "serif", "bookish": "serif",
    "condensed": "condensed", "narrow": "condensed", "compressed": "condensed",
    "compact": "condensed", "tall": "condensed", "extended": "condensed",
    "mono": "mono", "monospace": "mono", "monospaced": "mono", "typewriter": "mono",
    "code": "mono", "terminal": "mono", "fixedwidth": "mono",
    "script": "script", "handwriting": "script", "handwritten": "script",
    "hand": "script", "cursive": "script", "brush": "script", "signature": "script",
    "calligraphy": "script", "calligraphic": "script", "chalk": "script",
    "display": "display", "poster": "display", "headline": "display",
    "decorative": "display", "signage": "display", "fascia": "display",
    "bold": "bold", "heavy": "bold", "black": "bold", "thick": "bold",
    "strong": "bold", "fat": "bold", "chunky": "bold",
    "italic": "italic", "oblique": "italic", "slanted": "italic", "italics": "italic",
}

BUCKET_WEIGHT = 2.0
STYLE_PENALTY = 0.6
FUZZY_WEIGHT = 1.5
MIN_MATCH_SCORE = 1.0

# ──── Rendering ────

MIN_FONT_SIZE = 4
LINE_GAP_RATIO = 0.15
DEFAULT_PLATE_RGB = (128, 128, 128)
DEFAULT_INK_RGB = (255, 255, 255)

# A fitted rectangle this close to square carries no usable rotation: a round
# sign has no preferred direction, so minAreaRect returns an arbitrary angle
# (usually 45 degrees). Above this tolerance the angle is trusted.
SQUARE_TOLERANCE = 0.12
MIN_QUAD_EDGE = 2.0

# Share of the region's rim a component must occupy before it counts as structure
# rather than lettering. A frame or painted border wraps most of the way round; a
# letter reaching the edge of its plate covers a few percent at most.
BORDER_RIM_SHARE = 0.18

# How far a pixel must sit from the plate colour to count as lettering. Measured
# on a re-rendered sign: at 46 the faint outer edges of the strokes fall below
# the threshold and survive into the next pass as ghosting; at 28 they are caught
# while the frame is still excluded by its rim share (0.0% frame pixels hit).
DEFAULT_INK_TOLERANCE = 28

# ──── Colour recovery ────

LUMA_WEIGHTS = (0.299, 0.587, 0.114)
COLOR_SPLIT_ITERATIONS = 12
COLOR_SPLIT_EPSILON = 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Font discovery
# ──────────────────────────────────────────────────────────────────────────────


def FONT_SEARCH_DIRS() -> list:
    """Return the existing directories that are scanned for fonts, repo dir first.

    Order matters: the repo-local ``fonts/`` directory wins over system fonts of the
    same name so a bundled face makes a workflow reproducible across machines.
    """
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    candidates = [REPO_FONTS_DIR, WINDOWS_FONTS_DIR]
    if local_appdata:
        candidates.append(os.path.join(local_appdata, "Microsoft", "Windows", "Fonts"))

    dirs = []
    for path in candidates:
        try:
            if path and os.path.isdir(path) and path not in dirs:
                dirs.append(path)
        except OSError:
            continue
    return dirs


def _font_index() -> dict:
    """Build a live ``display name -> absolute path`` map of every installed font.

    Scanned on every call — never cached at import time, so fonts dropped into
    ``fonts/`` or installed in Windows show up without a ComfyUI restart.
    """
    index = {}
    for directory in FONT_SEARCH_DIRS():
        found = []
        try:
            for root, _dirs, files in os.walk(directory):
                for filename in files:
                    if filename.lower().endswith(FONT_EXTENSIONS):
                        found.append(os.path.join(root, filename))
        except OSError:
            continue

        for path in sorted(found, key=lambda p: os.path.basename(p).lower()):
            stem, ext = os.path.splitext(os.path.basename(path))
            name = stem
            if name in index:
                name = stem + ext.lower()
            suffix = 2
            while name in index:
                name = "{} [{}]".format(stem, suffix)
                suffix += 1
            index[name] = path
    return index


def discover_fonts() -> list:
    """List selectable font names, scanned live on every call.

    The first entry is always :data:`SYSTEM_DEFAULT_LABEL`, which maps to Pillow's
    built-in face, so a COMBO built from this list always has a usable default even on
    a machine with no fonts at all. Repo fonts come next, then system fonts, each block
    sorted alphabetically. Never raises.
    """
    try:
        names = list(_font_index().keys())
    except Exception:
        names = []
    return [SYSTEM_DEFAULT_LABEL] + names


def _normalize_name(text: str) -> str:
    """Lowercase and strip everything but letters and digits."""
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def _hint_buckets(hint: str) -> set:
    """Map a free-text font hint onto the keyword buckets it implies."""
    buckets = set()
    words = [w for w in "".join(ch if ch.isalnum() else " " for ch in hint.lower()).split() if w]
    for word in words:
        bucket = FONT_HINT_SYNONYMS.get(word)
        if bucket:
            buckets.add(bucket)
            continue
        # Sub-word hits, e.g. "sans-serif" already split, but "boldish" or "condensedd".
        for synonym, mapped in FONT_HINT_SYNONYMS.items():
            if len(synonym) >= 4 and synonym in word:
                buckets.add(mapped)
    return buckets


def _is_font_file(path: str) -> bool:
    """True if the path points at an existing .ttf/.otf file."""
    try:
        return bool(path) and str(path).lower().endswith(FONT_EXTENSIONS) and os.path.isfile(path)
    except OSError:
        return False


def resolve_font(hint: str, available: list = None) -> str:
    """Resolve a free-text font hint onto a concrete font file path.

    Accepts anything a vision LLM is likely to emit — ``"bold condensed sans"``,
    ``"Helvetica"``, ``"handwritten"``, a display name from :func:`discover_fonts`, or a
    plain path. Returns an absolute path to an existing font file, or ``None`` when the
    hint is empty, means "system default", or matches nothing well enough (the caller
    then falls back to Pillow's built-in face).

    Args:
        hint: free-text description, family name, display name or path.
        available: restrict the search to these display names (or paths). ``None``
            scans everything.

    Returns:
        A font file path, or ``None``.
    """
    if hint is None:
        return None
    hint = str(hint).strip()
    if not hint or hint == SYSTEM_DEFAULT_LABEL:
        return None
    if _is_font_file(hint):
        return hint

    try:
        index = _font_index()
    except Exception:
        return None

    if available is not None:
        filtered = {}
        for entry in available:
            if not entry or entry == SYSTEM_DEFAULT_LABEL:
                continue
            if entry in index:
                filtered[entry] = index[entry]
            elif _is_font_file(entry):
                filtered[os.path.splitext(os.path.basename(entry))[0]] = entry
        index = filtered

    if not index:
        return None

    # Exact display-name hit (case-insensitive) short-circuits the scoring.
    lowered = hint.lower()
    for name, path in index.items():
        if name.lower() == lowered:
            return path

    buckets = _hint_buckets(hint)
    normalized_hint = _normalize_name(hint)
    best_score = 0.0
    best_path = None

    for name, path in index.items():
        normalized = _normalize_name(name)
        score = 0.0
        for bucket in buckets:
            if any(keyword in normalized for keyword in FONT_FAMILY_KEYWORDS[bucket]):
                score += BUCKET_WEIGHT
        # Do not hand back a bold or italic cut nobody asked for.
        if "bold" not in buckets and any(k in normalized for k in FONT_FAMILY_KEYWORDS["bold"]):
            score -= STYLE_PENALTY
        if "italic" not in buckets and any(k in normalized for k in FONT_FAMILY_KEYWORDS["italic"]):
            score -= STYLE_PENALTY
        score += difflib.SequenceMatcher(None, normalized_hint, normalized).ratio() * FUZZY_WEIGHT

        if score > best_score:
            best_score = score
            best_path = path

    if best_path is None or best_score < MIN_MATCH_SCORE:
        return None
    return best_path if _is_font_file(best_path) else None


# ──────────────────────────────────────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────────────────────────────────────


def _to_numpy_2d(mask_2d) -> np.ndarray:
    """Coerce a torch tensor or array-like into a 2D float32 numpy array, or None."""
    if mask_2d is None:
        return None
    data = mask_2d
    if hasattr(data, "detach"):  # torch tensor, without importing torch
        try:
            data = data.detach().cpu().numpy()
        except Exception:
            return None
    try:
        arr = np.asarray(data, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2 or arr.size == 0:
        return None
    return arr


def _order_quad(points: np.ndarray) -> np.ndarray:
    """Order 4 corners top-left, top-right, bottom-right, bottom-left.

    Sorts clockwise by the angle around the centroid (image coordinates, y down), then
    rolls the sequence so the corner closest to the image origin (smallest x + y) comes
    first. Stable for rotations up to about +-45 degrees; beyond that "top-left" is
    geometrically ambiguous for a rectangle and the edge roles swap, which is inherent
    to a purely geometric rule.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    clockwise = pts[np.argsort(angles)]
    start = int(np.argmin(clockwise[:, 0] + clockwise[:, 1]))
    return np.roll(clockwise, -start, axis=0).astype(np.float32)


def edge_profiles(mask_2d, smooth: float = 0.12):
    """Top and bottom edge of the mask for every occupied column.

    A four-corner fit can only describe a flat plane. A label wrapped round a
    bottle, or a print following a fold, has edges that bow — and a ragged poster
    has edges that wander. Sampling the real edges per column is what lets the
    text follow them.

    The profiles are smoothed on purpose: without it the text would chase every
    notch in a torn outline and tear itself apart. What we want is the shape's
    drift, not its noise.

    Returns:
        ``(x_min, x_max, top, bottom)`` with two float arrays of equal length,
        or ``None`` when the mask is empty or too narrow to describe.
    """
    arr = _to_numpy_2d(mask_2d)
    if arr is None:
        return None
    scale = 255.0 if float(arr.max()) <= 1.0 else 1.0
    binary = (np.clip(arr, 0.0, None) * scale > 127.0)
    if not binary.any():
        return None

    columns = np.where(binary.any(axis=0))[0]
    if len(columns) < 4:
        return None
    x_min, x_max = int(columns[0]), int(columns[-1])

    width = x_max - x_min + 1
    top = np.empty(width, dtype=np.float32)
    bottom = np.empty(width, dtype=np.float32)
    for i, x in enumerate(range(x_min, x_max + 1)):
        rows = np.where(binary[:, x])[0]
        if len(rows):
            top[i], bottom[i] = rows[0], rows[-1]
        else:                      # a gap inside the shape: bridge it
            top[i], bottom[i] = np.nan, np.nan

    for profile in (top, bottom):
        holes = np.isnan(profile)
        if holes.all():
            return None
        if holes.any():
            profile[holes] = np.interp(np.flatnonzero(holes),
                                       np.flatnonzero(~holes), profile[~holes])

    window = max(3, int(width * smooth) | 1)
    if window < width:
        kernel = np.ones(window, dtype=np.float32) / window
        pad = window // 2
        top = np.convolve(np.pad(top, pad, mode="edge"), kernel, mode="valid")
        bottom = np.convolve(np.pad(bottom, pad, mode="edge"), kernel, mode="valid")

    if float(np.min(bottom - top)) < 2.0:
        return None
    return x_min, x_max, top, bottom


def warp_to_contour(block_rgb, mask_2d, out_shape, cylinder: float = 0.0):
    """Fit a rendered text block between the mask's own top and bottom edges.

    Column by column rather than corner to corner, so the baseline bows with a
    curved label and drifts with a ragged one. ``cylinder`` (0..1) additionally
    compresses the horizontal sampling towards the sides, the way lettering
    wrapped around a bottle foreshortens away from the viewer.

    Returns ``(rgb float32 [H, W, 3] 0..1, alpha float32 [H, W])``.
    """
    out_h, out_w = int(out_shape[0]), int(out_shape[1])
    empty = (np.zeros((out_h, out_w, 3), np.float32), np.zeros((out_h, out_w), np.float32))

    profiles = edge_profiles(mask_2d)
    if profiles is None or block_rgb is None or block_rgb.size == 0:
        return empty
    x_min, x_max, top, bottom = profiles

    block = block_rgb.astype(np.float32)
    if block.max() > 1.0:
        block = block / 255.0
    bh, bw = block.shape[:2]

    xs = np.arange(x_min, x_max + 1)
    u = (xs - x_min) / max(1.0, float(x_max - x_min))
    if cylinder > 0:
        # arcsin re-spacing: even steps around a cylinder project to steps that
        # crowd together at the silhouette edges.
        centred = np.clip(2.0 * u - 1.0, -1.0, 1.0)
        wrapped = np.arcsin(centred) / (np.pi / 2.0) * 0.5 + 0.5
        u = (1.0 - cylinder) * u + cylinder * wrapped

    map_x = np.full((out_h, out_w), -1.0, dtype=np.float32)
    map_y = np.full((out_h, out_w), -1.0, dtype=np.float32)

    rows = np.arange(out_h, dtype=np.float32)
    for i, x in enumerate(xs):
        if x < 0 or x >= out_w:
            continue
        y0, y1 = float(top[i]), float(bottom[i])
        span = y1 - y0
        if span < 1.0:
            continue
        v = (rows - y0) / span
        inside = (v >= 0.0) & (v <= 1.0)
        map_x[inside, x] = u[i] * (bw - 1)
        map_y[inside, x] = v[inside] * (bh - 1)

    covered = map_x >= 0
    warped = cv2.remap(block, np.where(covered, map_x, 0).astype(np.float32),
                       np.where(covered, map_y, 0).astype(np.float32),
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    alpha = covered.astype(np.float32)
    return warped * alpha[..., None], alpha


def _corner_quad(binary):
    """The mask's own four corners, or None if it is not convincingly a quad.

    ``minAreaRect`` always returns a rectangle, so a sign angled away from the
    camera comes back with both vertical edges the same length — the text then
    sits perfectly parallel and reads as a sticker pasted on top. Recovering the
    real corners lets the perspective warp reproduce the foreshortening.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 16:
        return None

    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 0:
        return None

    # Walk epsilon up until the hull collapses to four corners.
    for step in range(1, 41):
        approx = cv2.approxPolyDP(hull, perimeter * 0.005 * step, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            # Reject a fit that lost a meaningful part of the shape — a torn or
            # rounded outline is better served by the bounding rectangle.
            if cv2.contourArea(quad) >= area * 0.90:
                return quad
            return None
        if len(approx) < 4:
            return None
    return None


def _is_meaningfully_skewed(quad, tolerance: float = 0.06) -> bool:
    """Do the opposite edges differ enough to be worth a perspective warp?

    Below the tolerance the quad is a rectangle in all but rounding, and using it
    would only add jitter to text that should sit straight.
    """
    def length(a, b):
        return float(np.hypot(*(quad[a] - quad[b])))

    top, bottom = length(0, 1), length(3, 2)
    left, right = length(0, 3), length(1, 2)
    for near, far in ((top, bottom), (left, right)):
        longest = max(near, far)
        if longest > 0 and abs(near - far) / longest > tolerance:
            return True
    return False


def mask_quad(mask_2d, square_tolerance: float = SQUARE_TOLERANCE,
              perspective: bool = True) -> np.ndarray:
    """Fit the tightest rotated rectangle around a mask and return its 4 corners.

    Args:
        mask_2d: [H, W] mask, values in [0,1] (or 0-255). numpy array or torch tensor.
        square_tolerance: when the fitted rectangle is this close to square, its
            rotation carries no information and is discarded in favour of an
            axis-aligned box. A circular mask has no preferred direction, so
            ``minAreaRect`` returns an arbitrary angle — typically 45 degrees,
            which would set the text diagonally across a round sign. Pass 0 to
            keep the raw angle.

    Returns:
        float32 [4, 2] array of (x, y) corners ordered top-left, top-right,
        bottom-right, bottom-left — or ``None`` for an empty or degenerate mask.
    """
    arr = _to_numpy_2d(mask_2d)
    if arr is None:
        return None

    scale = 255.0 if float(arr.max()) <= 1.0 else 1.0
    binary = (np.clip(arr, 0.0, None) * scale > 127.0).astype(np.uint8)
    if binary.max() == 0:
        return None

    points = cv2.findNonZero(binary)
    if points is None or len(points) < 3:
        return None

    # A genuinely skewed outline beats any fitted rectangle: it is what gives the
    # rendered text its vanishing line instead of leaving it flat and parallel.
    if perspective:
        corners = _corner_quad(binary)
        if corners is not None:
            ordered = _order_quad(corners)
            if _is_meaningfully_skewed(ordered):
                return ordered

    rect = cv2.minAreaRect(points)
    (rect_w, rect_h) = rect[1]
    if rect_w < 1.0 or rect_h < 1.0:
        return None

    if square_tolerance > 0:
        longest, shortest = max(rect_w, rect_h), min(rect_w, rect_h)
        if longest > 0 and (longest - shortest) / longest <= square_tolerance:
            x, y, w, h = cv2.boundingRect(points)
            return _order_quad(np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32))

    return _order_quad(cv2.boxPoints(rect))


def quad_angle(quad) -> float:
    """Rotation of the quad's top edge in degrees, normalised to (-90, 90].

    Positive means the top edge descends to the right (clockwise on screen, since image
    coordinates run y-down). Returns 0.0 for a degenerate quad.
    """
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    dx = float(pts[1][0] - pts[0][0])
    dy = float(pts[1][1] - pts[0][1])
    if dx == 0.0 and dy == 0.0:
        return 0.0
    angle = math.degrees(math.atan2(dy, dx))
    while angle > 90.0:
        angle -= 180.0
    while angle <= -90.0:
        angle += 180.0
    return float(angle)


def quad_size(quad) -> tuple:
    """Pixel size (width, height) of the un-rotated text box behind a quad.

    Takes the longer of each pair of opposing edges so a perspective-skewed quad keeps
    the resolution of its nearest edge instead of averaging detail away.
    """
    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    top = np.linalg.norm(pts[1] - pts[0])
    bottom = np.linalg.norm(pts[2] - pts[3])
    left = np.linalg.norm(pts[3] - pts[0])
    right = np.linalg.norm(pts[2] - pts[1])
    width = int(round(max(top, bottom)))
    height = int(round(max(left, right)))
    return (max(1, width), max(1, height))


# ──────────────────────────────────────────────────────────────────────────────
# Text rendering
# ──────────────────────────────────────────────────────────────────────────────


def _load_font(font_path, size):
    """Load a font at the given pixel size, degrading to Pillow's default face."""
    size = max(MIN_FONT_SIZE, int(size))
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, ValueError, TypeError):
            pass
    try:
        return ImageFont.load_default(size=size)  # scalable default, Pillow >= 10.1
    except (TypeError, AttributeError, OSError):
        pass
    for fallback in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(fallback, size)
        except (OSError, ValueError):
            continue
    return ImageFont.load_default()


def _line_bbox(font, line: str) -> tuple:
    """Ink bounding box (x0, y0, x1, y1) of a line, relative to the draw origin."""
    if not line:
        return (0, 0, 0, 0)
    try:
        return tuple(font.getbbox(line))
    except (AttributeError, TypeError, OSError):
        try:
            width, height = font.getsize(line)  # very old Pillow
            return (0, 0, int(width), int(height))
        except Exception:
            return (0, 0, len(line) * 6, 11)


def _line_width(font, line: str) -> float:
    """Ink width of a single line in pixels."""
    bbox = _line_bbox(font, line)
    return float(bbox[2] - bbox[0])


def _line_height(font) -> int:
    """Full em height (ascent + descent) of a font in pixels."""
    try:
        ascent, descent = font.getmetrics()
        return max(1, int(ascent + descent))
    except (AttributeError, TypeError, OSError):
        bbox = _line_bbox(font, "Ag")
        return max(1, int(bbox[3] - bbox[1]))


def _split_balanced(words: list, measure, n_lines: int) -> list:
    """Split words into exactly ``n_lines`` lines, minimising the widest line.

    Exact dynamic program over word counts — sign copy is short, so the O(n^2 * lines)
    cost is irrelevant and the result reads far better than greedy wrapping.
    """
    if not words:
        return [""]
    total = len(words)
    n_lines = max(1, min(int(n_lines), total))
    memo = {}

    def solve(start, remaining):
        key = (start, remaining)
        if key in memo:
            return memo[key]
        if remaining == 1:
            line = " ".join(words[start:])
            memo[key] = (measure(line), [line])
            return memo[key]
        best = (float("inf"), None)
        for cut in range(start + 1, total - remaining + 2):
            head = " ".join(words[start:cut])
            tail_width, tail_lines = solve(cut, remaining - 1)
            worst = max(measure(head), tail_width)
            if worst < best[0]:
                best = (worst, [head] + tail_lines)
        memo[key] = best
        return best

    return solve(0, n_lines)[1]


def _layout_at_size(font, words: list, inner_w: float, inner_h: float, max_lines: int):
    """Return the fewest lines that fit the inner box at this font size, else None."""
    line_h = _line_height(font)
    gap = max(1, int(round(line_h * LINE_GAP_RATIO)))

    def measure(line):
        return _line_width(font, line)

    for count in range(1, max(1, int(max_lines)) + 1):
        if count > len(words):
            break
        lines = _split_balanced(words, measure, count)
        widest = max(measure(line) for line in lines)
        block_h = len(lines) * line_h + (len(lines) - 1) * gap
        if widest <= inner_w and block_h <= inner_h:
            return lines
    return None


def _fit_text(text: str, inner_w: float, inner_h: float, font_path, max_lines: int,
              target_line_height=None):
    """Binary-search the largest font size whose wrapped layout fits the inner box.

    ``target_line_height`` caps the search at the size the surface already used.
    Filling the box is right for a headline sign and wrong for everything else: a
    wine label carries one large number over three lines of small print, and
    scaling replacement copy to the full box turns it into a poster — long words
    then no longer fit and get cut off at the edge. When the existing lettering
    can be measured, it is the better reference than the box.

    Returns ``(font, lines)``. Falls back to :data:`MIN_FONT_SIZE` and a best-effort
    wrap when nothing fits, so the caller never has to handle a failure.
    """
    words = text.split()
    low = MIN_FONT_SIZE
    high = max(MIN_FONT_SIZE, int(inner_h) + 2)
    if target_line_height:
        # Cap, not a fixed size: a long replacement still shrinks to fit.
        high = max(MIN_FONT_SIZE, min(high, int(round(target_line_height * 1.35))))
    best = None

    while low <= high:
        mid = (low + high) // 2
        font = _load_font(font_path, mid)
        lines = _layout_at_size(font, words, inner_w, inner_h, max_lines)
        if lines is not None:
            best = (font, lines)
            low = mid + 1
        else:
            high = mid - 1

    if best is None:
        font = _load_font(font_path, MIN_FONT_SIZE)
        lines = _split_balanced(words, lambda line: _line_width(font, line), max_lines)
        best = (font, lines)
    return best


def render_text_block(text, width, height, font_path=None, fill=DEFAULT_INK_RGB,
                      bg=(0, 0, 0), margin_ratio=0.08, align="center", max_lines=3,
                      uppercase=False, target_line_height=None) -> np.ndarray:
    """Render text as clean typography, auto-fitted into a ``width`` x ``height`` box.

    The point size is found by binary search so the wrapped text fills the box down to
    the margin. Wrapping balances line lengths (exact DP), never exceeding ``max_lines``.

    Args:
        text: the replacement copy. Empty or whitespace yields a flat ``bg`` plate.
        width: block width in pixels.
        height: block height in pixels.
        font_path: path to a .ttf/.otf, or ``None`` for Pillow's default face.
        fill: ink colour, RGB 0-255.
        bg: plate colour, RGB 0-255.
        margin_ratio: padding on each side as a fraction of the box dimension.
        align: "left", "center" or "right".
        max_lines: hard cap on the number of lines.
        uppercase: upper-case the text before layout (common on signage).

    Returns:
        uint8 RGB array of shape [height, width, 3]. Never raises.
    """
    width = max(1, int(width))
    height = max(1, int(height))
    fill = tuple(int(c) for c in fill)[:3]
    bg = tuple(int(c) for c in bg)[:3]

    image = Image.new("RGB", (width, height), bg)
    text = "" if text is None else str(text)
    if uppercase:
        text = text.upper()
    if not text.strip():
        return np.array(image, dtype=np.uint8)

    margin_ratio = float(np.clip(margin_ratio, 0.0, 0.45))
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio
    inner_w = max(1.0, width - 2.0 * margin_x)
    inner_h = max(1.0, height - 2.0 * margin_y)

    font, lines = _fit_text(text, inner_w, inner_h, font_path, max_lines,
                            target_line_height=target_line_height)
    line_h = _line_height(font)
    gap = max(1, int(round(line_h * LINE_GAP_RATIO)))
    block_h = len(lines) * line_h + (len(lines) - 1) * gap
    start_y = margin_y + (inner_h - block_h) / 2.0

    draw = ImageDraw.Draw(image)
    for row, line in enumerate(lines):
        bbox = _line_bbox(font, line)
        line_w = bbox[2] - bbox[0]
        if align == "left":
            x = margin_x
        elif align == "right":
            x = margin_x + inner_w - line_w
        else:
            x = margin_x + (inner_w - line_w) / 2.0
        y = start_y + row * (line_h + gap)
        try:
            draw.text((x - bbox[0], y), line, font=font, fill=fill)
        except (OSError, ValueError):
            continue

    return np.array(image, dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Warping and compositing
# ──────────────────────────────────────────────────────────────────────────────


def _to_float_rgb(image) -> np.ndarray:
    """Coerce an image to float32 [H, W, 3] in 0..1."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        if float(arr.max(initial=0.0)) > 1.0 + 1e-6:
            arr = arr / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]
    return np.ascontiguousarray(np.clip(arr, 0.0, 1.0), dtype=np.float32)


def warp_to_quad(block_rgb, quad, out_shape) -> tuple:
    """Perspective-map a rendered text block onto a quad inside an empty canvas.

    Args:
        block_rgb: the rendered block, [h, w, 3] uint8 or float.
        quad: [4, 2] corners in the order returned by :func:`mask_quad`.
        out_shape: (H, W) of the destination canvas.

    Returns:
        ``(warped_rgb, coverage_alpha)`` — float32 [H, W, 3] in 0..1 and float32
        [H, W] in 0..1. Degenerate input yields zero arrays rather than an exception.
    """
    out_h = max(1, int(out_shape[0]))
    out_w = max(1, int(out_shape[1]))
    empty_rgb = np.zeros((out_h, out_w, 3), dtype=np.float32)
    empty_alpha = np.zeros((out_h, out_w), dtype=np.float32)

    if quad is None or block_rgb is None:
        return empty_rgb, empty_alpha

    block = _to_float_rgb(block_rgb)
    block_h, block_w = block.shape[:2]
    if block_h < 1 or block_w < 1:
        return empty_rgb, empty_alpha

    dst = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    edges = [float(np.linalg.norm(dst[(i + 1) % 4] - dst[i])) for i in range(4)]
    if min(edges) < MIN_QUAD_EDGE:
        return empty_rgb, empty_alpha

    src = np.array(
        [[0.0, 0.0], [block_w, 0.0], [block_w, block_h], [0.0, block_h]],
        dtype=np.float32,
    )
    try:
        matrix = cv2.getPerspectiveTransform(src, dst)
    except cv2.error:
        return empty_rgb, empty_alpha

    warped = cv2.warpPerspective(
        block, matrix, (out_w, out_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0, 0.0, 0.0),
    )
    alpha = cv2.warpPerspective(
        np.ones((block_h, block_w), dtype=np.float32), matrix, (out_w, out_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )

    warped = np.clip(warped.reshape(out_h, out_w, 3), 0.0, 1.0).astype(np.float32)
    alpha = np.clip(alpha.reshape(out_h, out_w), 0.0, 1.0).astype(np.float32)
    return warped, alpha


def existing_ink_mask(image_rgb, mask_2d, plate_rgb, tolerance: int = DEFAULT_INK_TOLERANCE,
                      drop_border_touching: bool = True) -> np.ndarray:
    """Where the CURRENT lettering sits inside a region.

    Painting the whole masked area with the plate colour erases everything the
    surface had: an enamel sign loses its border, a shop window loses the glass
    and the shelves behind it, and the sampler then has to invent all of it from
    a flat fill. Seen on real photographs, that turned a bordered oval sign into
    a plain white disc and a curved window inscription into a pasted-on banner.

    Only the letters need replacing. They are the parts that differ from the
    plate colour AND do not touch the region's edge — a frame, a rim or a
    painted border runs into the boundary, glyphs do not.

    Returns a float32 [H, W] mask in 0..1.
    """
    arr = _to_numpy_2d(mask_2d)
    if arr is None or image_rgb is None:
        return None
    rgb = _to_float_rgb(image_rgb) * 255.0
    if rgb.shape[:2] != arr.shape[:2]:
        arr = cv2.resize(arr, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    scale = 255.0 if float(arr.max()) <= 1.0 else 1.0
    inside = ((np.clip(arr, 0.0, None) * scale) > 127.0).astype(np.uint8)
    if inside.sum() == 0:
        return None

    distance = np.abs(rgb - np.asarray(plate_rgb, np.float32)).max(axis=2)
    ink = ((distance > tolerance) & (inside > 0)).astype(np.uint8)
    if ink.sum() == 0:
        return np.zeros(arr.shape[:2], np.float32)

    if drop_border_touching:
        # A frame runs around the whole region; a letter that happens to reach the
        # edge only brushes it. Discarding everything that touches at all threw
        # away 60% of the lettering on a sign whose text nearly spans its width,
        # and the leftovers then showed through the re-render as ghosting.
        # So measure HOW MUCH of the rim a component occupies, not whether it
        # touches at all.
        eroded = cv2.erode(inside, np.ones((3, 3), np.uint8), iterations=1)
        rim = (inside > 0) & (eroded == 0)
        rim_total = float(rim.sum())
        count, labels = cv2.connectedComponents(ink, connectivity=8)
        if count > 1 and rim_total > 0:
            structural = []
            for label in range(1, count):
                component = labels == label
                share = float((component & rim).sum()) / rim_total
                if share >= BORDER_RIM_SHARE:
                    structural.append(label)
            if structural:
                keep = ~np.isin(labels, structural)
                ink = (ink * keep).astype(np.uint8)

    return ink.astype(np.float32)


def measure_ink_height(image_rgb, mask_2d, plate_rgb, tolerance: int = DEFAULT_INK_TOLERANCE):
    """Typical stroke height of the lettering already on the surface, in pixels.

    Filling the box is wrong for anything but a headline sign. A wine label
    carries one large number and three lines of small print; scaling replacement
    copy to the full box turns it into a poster and pushes long words past the
    edge. The size to match is the size that was there.

    Returns ``None`` when there is nothing measurable to go on.
    """
    ink = existing_ink_mask(image_rgb, mask_2d, plate_rgb, tolerance=tolerance)
    if ink is None or ink.sum() == 0:
        return None

    count, _labels, stats, _cent = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8)
    if count <= 1:
        return None

    heights, areas = [], []
    for i in range(1, count):
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        a = int(stats[i, cv2.CC_STAT_AREA])
        if h >= 2 and a >= 4:
            heights.append(h)
            areas.append(a)
    if not heights:
        return None

    # Weight by area so a stray speck cannot outvote the actual glyphs, and take
    # the median so one oversized initial or a rule does not drag the estimate.
    order = np.argsort(heights)
    heights = np.asarray(heights)[order]
    areas = np.asarray(areas, dtype=np.float64)[order]
    cumulative = np.cumsum(areas)
    if cumulative[-1] <= 0:
        return None
    midpoint = np.searchsorted(cumulative, cumulative[-1] / 2.0)
    return float(heights[min(midpoint, len(heights) - 1)])


def quad_fit_error(mask_2d, quad) -> float:
    """Fraction of the mask's area that the fitted quad claims but does not cover.

    A flat sign scores near zero. A label round a bottle or a torn poster scores
    high, because a four-corner plane cannot follow a curved or wandering edge.
    """
    arr = _to_numpy_2d(mask_2d)
    if arr is None or quad is None:
        return 0.0
    scale = 255.0 if float(arr.max()) <= 1.0 else 1.0
    binary = (np.clip(arr, 0.0, None) * scale > 127.0)
    area = float(binary.sum())
    if area <= 0:
        return 0.0
    filled = np.zeros(binary.shape, np.uint8)
    cv2.fillPoly(filled, [quad.astype(np.int32)], 1)
    outside = float(((filled > 0) & ~binary).sum())
    return outside / area


# Above this mismatch a four-corner warp is the wrong tool and the column-wise
# fit takes over. Measured: a flat sign scores ~0.02, a bowed bottle label 0.34,
# a torn poster 0.23.
CONTOUR_FIT_THRESHOLD = 0.12


def render_glyph_layer(text, mask_2d, font_path=None, fill=DEFAULT_INK_RGB, bg=None,
                       uppercase=False, margin_ratio=0.08, fit: str = "auto",
                       cylinder: float = 0.0, target_line_height=None) -> tuple:
    """Render replacement text onto the sign described by a mask, end to end.

    mask -> quad -> correctly sized text block -> perspective warp.

    Args:
        text: the replacement copy.
        mask_2d: [H, W] sign mask. numpy array or torch tensor.
        font_path: path to a .ttf/.otf, or ``None`` for Pillow's default face.
        fill: ink colour, RGB 0-255.
        bg: plate colour, RGB 0-255. ``None`` uses a neutral mid grey — this function
            has no access to the source image, so the caller composites the real plate
            colour (see :func:`estimate_text_colors`).
        uppercase: upper-case the text before layout.
        margin_ratio: padding inside the sign box as a fraction of its size.

    Returns:
        ``(rgb, alpha)`` — float32 [H, W, 3] in 0..1 and float32 [H, W] in 0..1.
        An empty mask returns zero arrays. Never raises.
    """
    arr = _to_numpy_2d(mask_2d)
    if arr is None:
        return np.zeros((1, 1, 3), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)

    out_h, out_w = arr.shape[:2]
    quad = mask_quad(arr)
    if quad is None:
        return (np.zeros((out_h, out_w, 3), dtype=np.float32),
                np.zeros((out_h, out_w), dtype=np.float32))

    block_w, block_h = quad_size(quad)
    plate = DEFAULT_PLATE_RGB if bg is None else bg
    block = render_text_block(
        text, block_w, block_h,
        font_path=font_path, fill=fill, bg=plate,
        margin_ratio=margin_ratio, uppercase=uppercase,
        target_line_height=target_line_height,
    )

    # A four-corner warp describes a flat plane. When the outline is curved or
    # ragged it cannot follow it, and clipping to the mask would just chop the
    # text off at the edges instead of bending it. Fall through to the
    # column-wise fit in that case.
    mode = fit
    if mode == "auto":
        mode = "contour" if quad_fit_error(arr, quad) > CONTOUR_FIT_THRESHOLD else "perspective"

    rgb = alpha = None
    if mode == "contour":
        rgb, alpha = warp_to_contour(block, arr, (out_h, out_w), cylinder=cylinder)
        if float(alpha.max()) <= 0.0:      # profiles unusable — fall back
            rgb = alpha = None
    if rgb is None:
        rgb, alpha = warp_to_quad(block, quad, (out_h, out_w))

    # Clip to the mask itself, not just its bounding quad. SAM3 returns real
    # silhouettes — a round sign, a curved bottle label, a torn poster — and the
    # quad always overshoots them (measured: 20-27% of the object's own area for
    # ellipses, circles and irregular blobs). Painting outside the silhouette puts
    # a rectangular plate into the init latent where a round object stands, so the
    # sampler is conditioned on geometry that is not in the picture.
    clip = arr.astype(np.float32)
    if float(clip.max()) > 1.0:          # 0-255 mask; normalise BEFORE clamping,
        clip = clip / 255.0              # otherwise soft edges collapse to 1.0
    alpha = alpha * np.clip(clip, 0.0, 1.0)
    return rgb, alpha


def surface_preserving_alpha(glyph_rgb, alpha, image_rgb, mask_2d, plate_rgb,
                             grow=None, tolerance: int = DEFAULT_INK_TOLERANCE) -> np.ndarray:
    """Restrict the glyph layer to the lettering, old and new.

    Everything else the surface had — border, frame, texture, whatever shows
    through it — is left for the sampler to keep rather than to reinvent.
    """
    if alpha is None:
        return alpha
    old = existing_ink_mask(image_rgb, mask_2d, plate_rgb, tolerance=tolerance)
    if old is None:
        return alpha

    if grow is None:
        # Scale with the lettering. A fixed radius smears fine print into blobs
        # and barely covers the anti-aliasing of large signage.
        measured = measure_ink_height(image_rgb, mask_2d, plate_rgb, tolerance=tolerance)
        grow = int(max(1, round((measured or 20.0) * 0.14)))

    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1))
        old = cv2.dilate(old, k, iterations=1)      # cover anti-aliased stroke edges

    # Where the freshly typeset block puts ink, measured against its own plate.
    new = np.zeros_like(old)
    if glyph_rgb is not None and glyph_rgb.size:
        block = glyph_rgb.astype(np.float32)
        if block.max() <= 1.0:
            block = block * 255.0
        diff = np.abs(block - np.asarray(plate_rgb, np.float32)).max(axis=2)
        new = ((diff > tolerance) & (alpha > 0.05)).astype(np.float32)
        if grow > 0:
            new = cv2.dilate(new, np.ones((3, 3), np.uint8), iterations=1)

    paint = np.clip(old + new, 0.0, 1.0)
    return alpha * paint


def composite_glyph(base_rgb: np.ndarray, glyph_rgb: np.ndarray, alpha: np.ndarray,
                    strength: float = 1.0) -> np.ndarray:
    """Alpha-blend a glyph layer over a float32 0..1 base image.

    Mismatched glyph/alpha shapes are resized to the base rather than raising, because
    a node graph should degrade instead of dying mid-render.

    Args:
        base_rgb: [H, W, 3] float32 in 0..1.
        glyph_rgb: [h, w, 3] glyph layer.
        alpha: [h, w] coverage in 0..1.
        strength: global multiplier on the alpha, 0 disables the layer entirely.

    Returns:
        float32 [H, W, 3] in 0..1.
    """
    base = _to_float_rgb(base_rgb)
    out_h, out_w = base.shape[:2]

    glyph = _to_float_rgb(glyph_rgb)
    if glyph.shape[:2] != (out_h, out_w):
        glyph = cv2.resize(glyph, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        glyph = np.ascontiguousarray(glyph.reshape(out_h, out_w, 3), dtype=np.float32)

    coverage = np.asarray(alpha, dtype=np.float32)
    while coverage.ndim > 2 and coverage.shape[0] == 1:
        coverage = coverage[0]
    if coverage.ndim == 3:
        coverage = coverage[..., 0]
    if coverage.ndim != 2 or coverage.size == 0:
        coverage = np.zeros((out_h, out_w), dtype=np.float32)
    if coverage.shape != (out_h, out_w):
        coverage = cv2.resize(coverage, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        coverage = coverage.reshape(out_h, out_w)

    coverage = np.clip(coverage * float(strength), 0.0, 1.0)[..., None].astype(np.float32)
    blended = base * (1.0 - coverage) + glyph * coverage
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def _luminance_split(luminance: np.ndarray):
    """Split a 1D luminance array into a dark and a light class, deterministically.

    Seeded from the luminance median — never a random k-means init — then refined with
    a handful of Lloyd steps. The refinement is what makes a letter stroke covering 15%
    of a sign separate from its plate; a bare median split would instead cut the plate
    itself in half and report two nearly identical colours.

    Returns:
        A boolean "is dark" selector, or ``None`` when the luminance is constant.
    """
    initial = None
    for seed in (float(np.median(luminance)),
                 float(luminance.min() + luminance.max()) / 2.0):
        selector = luminance <= seed
        if selector.any() and not selector.all():
            initial = seed
            break
    if initial is None:
        return None

    threshold = initial
    for _ in range(COLOR_SPLIT_ITERATIONS):
        selector = luminance <= threshold
        if not selector.any() or selector.all():
            threshold = initial
            break
        moved = (float(luminance[selector].mean()) + float(luminance[~selector].mean())) / 2.0
        converged = abs(moved - threshold) < COLOR_SPLIT_EPSILON
        threshold = moved
        if converged:
            break

    selector = luminance <= threshold
    if not selector.any() or selector.all():
        selector = luminance <= initial
    return selector


def estimate_text_colors(image_rgb: np.ndarray, mask_2d: np.ndarray) -> tuple:
    """Recover a sign's ink and plate colours from the pixels inside its mask.

    Splits the masked pixels into a dark and a light cluster on luminance
    (see :func:`_luminance_split` — median-seeded, deterministic, no random init) and
    calls the *smaller* cluster the ink, since letter strokes always cover less of a
    sign than its plate. That keeps white-on-black signage from being inverted into
    black-on-white. Cluster colours are per-channel medians, so a few blown-out
    highlights cannot drag the result.

    Args:
        image_rgb: [H, W, 3], uint8 0-255 or float 0..1.
        mask_2d: [H, W] sign mask.

    Returns:
        ``(ink_rgb, plate_rgb)`` as 0-255 int tuples. An empty mask returns
        ``((0, 0, 0), (255, 255, 255))``.
    """
    fallback = ((0, 0, 0), (255, 255, 255))

    arr = _to_numpy_2d(mask_2d)
    if arr is None or image_rgb is None:
        return fallback

    rgb = _to_float_rgb(image_rgb) * 255.0
    if rgb.shape[:2] != arr.shape[:2]:
        arr = cv2.resize(arr, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    scale = 255.0 if float(arr.max()) <= 1.0 else 1.0
    inside = (np.clip(arr, 0.0, None) * scale) > 127.0
    pixels = rgb[inside]
    if pixels.size == 0:
        return fallback

    luminance = (pixels[:, 0] * LUMA_WEIGHTS[0]
                 + pixels[:, 1] * LUMA_WEIGHTS[1]
                 + pixels[:, 2] * LUMA_WEIGHTS[2])

    dark_sel = _luminance_split(luminance)
    if dark_sel is None:  # constant luminance inside the mask
        flat = tuple(int(round(v)) for v in np.median(pixels, axis=0))
        return (flat, flat)

    dark = tuple(int(round(v)) for v in np.median(pixels[dark_sel], axis=0))
    light = tuple(int(round(v)) for v in np.median(pixels[~dark_sel], axis=0))

    dark_count = int(dark_sel.sum())
    light_count = int(pixels.shape[0] - dark_count)
    if light_count < dark_count:
        return (light, dark)  # light ink on a dark plate
    return (dark, light)
