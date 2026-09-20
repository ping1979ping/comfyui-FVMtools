"""Cut a region that holds several lines of lettering into one region per line.

SAM3 answers the question it was asked: *where is the sign*. On a shop window
carrying three lines of signwriting that is one object, so it returns one region.
Everything downstream then treats it as one piece of text — the proposer names it
once, the caption gets one element, and that single word is stretched over the
whole block. It comes out as letters the height of a door.

Shrinking the text box inside the region does NOT fix it, and that was measured
twice in this project before it was believed: the mask stays the size of the
region, so a smaller text box only enlarges the masked area that no element
describes, and the model fills it — with small print on a wine label, with an
invented enamel plaque across a shop window.

So the region has to be split, and it has to be split BEFORE the proposer runs,
because each line needs its own word. That is all this node does: look at where
the ink actually sits inside the mask, find the horizontal gaps between lines,
and hand back one region per line.

It refuses to split when it is not sure. A region that comes back as one band is
returned untouched, and a band thinner than a line of readable text is merged
back into its neighbour. A wrong split costs a word written across half a sign;
a missed split costs what we had before, which is no worse than not running.
"""

import numpy as np

# A gap between two lines has to be a real gap. Below this share of the region's
# own height it is the space between an ascender and the line above, not a line
# break — splitting there cuts the tops off letters.
MIN_LUECKE = 0.055

# Below this a band cannot carry readable text, so it is not a line of its own.
MIN_BAND_PX = 8

# Ink is anything above this share of the row's peak. Absolute thresholds fail
# on soft, defocused lettering, which is most of what this pipeline sees.
TINTE = 0.18


_DB = {}


def _dbnet():
    """Load the text detector once. Returns (module, session) or None."""
    if "sess" in _DB:
        return _DB["sess"] and (_DB["ob"], _DB["sess"])
    _DB["ob"] = _DB["sess"] = None
    try:
        import importlib.util
        import os

        pfad = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "utils",
            "ocr_backend.py",
        )
        spec = importlib.util.spec_from_file_location("_split_ocr", pfad)
        ob = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ob)
        # `resolve_ocr_dir` asks ComfyUI's `folder_paths`, which only exists
        # inside a running server. Walking up to `models/onnx/ocr` as well means
        # this node can be checked from a plain script — and everything in this
        # project that could only be tested after a restart cost a restart to
        # find its first bug.
        ordner = ob.resolve_ocr_dir()
        if not ordner:
            hier = os.path.abspath(__file__)
            for _ in range(8):
                hier = os.path.dirname(hier)
                kand = os.path.join(hier, "models", "onnx", "ocr")
                if os.path.isdir(kand):
                    ordner = kand
                    break
        if not ordner:
            return None
        det = os.path.join(ordner, ob.OCR_MODEL_FILES["det"])
        if not os.path.isfile(det):
            return None
        _DB["ob"], _DB["sess"] = ob, ob._get_session(det)
    except Exception as e:  # noqa: BLE001
        print(
            f"[SplitLines] Detektor nicht ladbar ({type(e).__name__}: {e}) "
            f"— es gilt die Lueckensuche"
        )
        return None
    return (_DB["ob"], _DB["sess"])


def _zeilen_dbnet(bild, bbox, skalen=(2.0, 3.0), thresh=0.3, box_thresh=0.5):
    """Text lines inside `bbox`, found by a detector trained on real writing.

    The gap-hunting version of this could not work here and the profile says so
    outright: across the three-line shopfront the row energy never drops below
    0.31 of its peak, because the glass, the frames and the shelves behind it
    put edges in every gap. There is nothing to find a gap in.
    A detector does not look for gaps, it looks for writing.
    """
    import cv2

    db = _dbnet()
    if not db:
        return None
    ob, sess = db
    x0, y0, x1, y1 = (int(v) for v in bbox)
    aus = np.ascontiguousarray(bild[y0 : y1 + 1, x0 : x1 + 1])
    if aus.size == 0:
        return None

    kaesten = []
    for sk in skalen:
        try:
            h, w = aus.shape[:2]
            work = cv2.resize(
                aus,
                (max(8, int(w * sk)), max(8, int(h * sk))),
                interpolation=cv2.INTER_CUBIC,
            )
            inp, rw, rh = ob._det_preprocess(
                work, limit_side_len=max(work.shape[:2]) + 64
            )
            prob = sess.run(None, {sess.get_inputs()[0].name: inp})[0]
            prob = np.asarray(prob, np.float32).squeeze()
            if prob.ndim != 2:
                continue
            for q in ob._db_postprocess(prob, rw, rh, thresh, box_thresh):
                q = np.asarray(q, np.float32) / float(sk)
                kaesten.append((float(q[:, 1].min()), float(q[:, 1].max())))
        except Exception:  # noqa: BLE001 — a scale that fails is not the end
            continue
    return kaesten or None


def _zu_baendern(spannen, hoehe, min_band, luecke_anteil=0.08):
    """Group line detections into row bands — by their CENTRES, not their spans.

    Merging spans was the obvious way and it is wrong on exactly the surfaces
    this node exists for. Shopfront lettering is arched, so the bounding box of
    one line reaches from the top of its first capital to the bottom of the
    lowest letter in the middle of the arc — and it overlaps the box of the line
    below. Three clearly separate lines came out as one band.

    Their centres do not overlap. On the measured example the centres fall in
    three tight groups (35-63, 178-189, 345-380) with gaps of 115 and 156 px
    between them, against gaps of 14 and 19 px inside a group.
    """
    if not spannen:
        return []
    nach_mitte = sorted(spannen, key=lambda t: (t[0] + t[1]) / 2.0)
    schwelle = max(float(min_band), hoehe * luecke_anteil)

    gruppen, aktuell = [], [nach_mitte[0]]
    for s in nach_mitte[1:]:
        vor = (aktuell[-1][0] + aktuell[-1][1]) / 2.0
        jetzt = (s[0] + s[1]) / 2.0
        if jetzt - vor <= schwelle:
            aktuell.append(s)
        else:
            gruppen.append(aktuell)
            aktuell = [s]
    gruppen.append(aktuell)

    baender = []
    for g in gruppen:
        a = max(0.0, min(x[0] for x in g))
        b = min(float(hoehe - 1), max(x[1] for x in g))
        if b - a + 1 >= min_band:
            baender.append([a, b])

    # An arc still reaches past its neighbour's centre. Cut where the bands meet
    # so no row belongs to two lines — otherwise one line's mask covers part of
    # the next and the words fight over the same pixels.
    for i in range(len(baender) - 1):
        if baender[i][1] >= baender[i + 1][0]:
            mitte = (baender[i][1] + baender[i + 1][0]) / 2.0
            baender[i][1], baender[i + 1][0] = mitte, mitte
    return [(int(round(a)), int(round(b))) for a, b in baender if b - a + 1 >= min_band]


def _baender(bild, maske, bbox, min_luecke, min_band):
    """Row bands that hold ink, inside `bbox`.

    Read from the PICTURE, not from the mask. That was the first version and it
    never split anything: SAM3 answers "where is the sign", so its mask is a
    solid blob over the whole lettered area. Projecting it gives one unbroken
    band by construction, and the node reported "nothing to split" on exactly
    the three-line shopfront it was built for.

    Ink is found by gradient magnitude rather than by brightness, because these
    surfaces run both ways — dark letters on white enamel, pale gold on dark
    timber — and any brightness threshold gets one of them backwards.
    """
    import cv2

    x0, y0, x1, y1 = (int(v) for v in bbox)
    aus = bild[y0 : y1 + 1, x0 : x1 + 1]
    if aus.size == 0:
        return []
    grau = cv2.cvtColor(aus, cv2.COLOR_RGB2GRAY) if aus.ndim == 3 else aus
    grau = cv2.createCLAHE(2.5, (8, 8)).apply(grau.astype(np.uint8))
    gx = cv2.Sobel(grau, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grau, cv2.CV_32F, 0, 1, ksize=3)
    kante = cv2.magnitude(gx, gy)

    # Only where the selector actually claimed. The bounding box of a tilted
    # sign takes in wall and window, and their edges would fill every gap
    # between the lines.
    if maske is not None:
        m = np.asarray(maske)[y0 : y1 + 1, x0 : x1 + 1]
        kante = kante * (m > 0.5)

    zeilen = kante.sum(1).astype(np.float32)
    if zeilen.max() <= 0:
        return []
    zeilen = cv2.GaussianBlur(zeilen.reshape(-1, 1), (1, 5), 0).ravel()
    voll = zeilen > (zeilen.max() * TINTE)

    hoehe = y1 - y0 + 1
    luecke = max(2, int(round(hoehe * min_luecke)))

    baender, start = [], None
    leer = 0
    for i, v in enumerate(voll):
        if v:
            if start is None:
                start = i
            leer = 0
        elif start is not None:
            leer += 1
            if leer >= luecke:
                baender.append((start, i - leer))
                start, leer = None, 0
    if start is not None:
        baender.append((start, len(voll) - 1))

    # Bands too thin to hold text are not lines. Fold them into the nearest
    # neighbour rather than dropping them — their ink still has to be covered.
    fest = []
    for a, b in baender:
        if b - a + 1 >= min_band or not fest:
            fest.append([a, b])
        else:
            fest[-1][1] = b
    return [(y0 + a, y0 + b) for a, b in fest]


class FVM_SignSplitLines:
    """Split multi-line sign regions into one region per line of lettering."""

    CATEGORY = "FVMtools/signs"
    FUNCTION = "split"
    RETURN_TYPES = ("SIGN_DATA", "INT", "STRING")
    RETURN_NAMES = ("sign_data", "region_count", "report")
    DESCRIPTION = (
        "One region per LINE instead of per object. SAM3 returns a whole "
        "shopfront legend as a single region; one region gets one word, and that "
        "word is then stretched over the whole block. Put this between the "
        "selector and the proposer so every line gets named separately."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sign_data": (
                    "SIGN_DATA",
                    {
                        "tooltip": "Straight from Sign Selector SAM3, BEFORE the "
                        "proposer — each new line needs its own word."
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "The picture the selector scanned. Lines are "
                        "found in the PICTURE, not in the mask: SAM3's mask is a "
                        "solid blob over the whole lettered area and has no gaps "
                        "to find."
                    },
                ),
                "enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Off: pass everything through untouched, so the "
                        "effect of splitting can be measured on its own.",
                    },
                ),
                "min_gap": (
                    "FLOAT",
                    {
                        "default": MIN_LUECKE,
                        "min": 0.01,
                        "max": 0.4,
                        "step": 0.005,
                        "tooltip": "How large a horizontal gap must be, as a share of "
                        "the region's height, before it counts as a line "
                        "break. Too small and an ascender splits a line in "
                        "two; too large and two lines stay welded.",
                    },
                ),
                "min_line_px": (
                    "INT",
                    {
                        "default": MIN_BAND_PX,
                        "min": 3,
                        "max": 200,
                        "tooltip": "A band thinner than this cannot hold readable "
                        "text and is merged into its neighbour instead of "
                        "becoming a region of its own.",
                    },
                ),
                "max_lines": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 12,
                        "tooltip": "Upper bound per region. A region that seems to "
                        "hold more lines than this is left whole — beyond "
                        "a few lines it is a paragraph, and a paragraph is "
                        "not signage.",
                    },
                ),
                "pad_px": (
                    "INT",
                    {
                        "default": 2,
                        "min": 0,
                        "max": 32,
                        "tooltip": "Rows added above and below each band. The mask "
                        "must cover every stroke: what it misses is "
                        "restored unchanged and stands there as leftover "
                        "scribble.",
                    },
                ),
            },
        }

    def split(self, sign_data, image, enabled, min_gap, min_line_px, max_lines, pad_px):
        regionen = list((sign_data or {}).get("regions") or [])
        if not enabled or not regionen:
            n = len(regionen)
            return (sign_data, n, f"[SplitLines] aus, {n} Regionen unveraendert")

        bild_np = (image[0].detach().cpu().numpy() * 255.0).clip(0, 255)
        bild_np = bild_np.astype(np.uint8)
        h, w = bild_np.shape[:2]
        neu, bericht, geteilt = [], [], 0
        for r in regionen:
            maske, bbox = r.get("mask"), r.get("bbox")
            if maske is None or not bbox:
                neu.append(r)
                continue
            # A region the selector flagged as a surface rather than a sign is on
            # its way to being wiped, not lettered. Splitting it into lines would
            # only invent three surfaces to wipe instead of one.
            if r.get("too_big"):
                neu.append(r)
                continue
            x0, y0, x1, y1 = (int(v) for v in bbox)
            spannen = _zeilen_dbnet(bild_np, bbox)
            if spannen:
                baender = [
                    (y0 + a, y0 + b)
                    for a, b in _zu_baendern(spannen, y1 - y0 + 1, min_line_px)
                ]
                quelle = "Detektor"
            else:
                baender = _baender(bild_np, maske, bbox, min_gap, min_line_px)
                quelle = "Lueckensuche"
            if len(baender) <= 1 or len(baender) > max_lines:
                neu.append(r)
                if len(baender) > max_lines:
                    bericht.append(
                        f"  {r.get('class')}: {len(baender)} Baender "
                        f"— mehr als {max_lines}, bleibt ganz"
                    )
                continue

            x0, _, x1, _ = bbox
            for a, b in baender:
                ya = max(0, a - pad_px)
                yb = min((h - 1) if h else b + pad_px, b + pad_px)
                teil = dict(r)
                m = np.zeros_like(np.asarray(maske))
                m[ya : yb + 1, :] = np.asarray(maske)[ya : yb + 1, :]
                teil["mask"] = m
                teil["bbox"] = [int(x0), int(ya), int(x1), int(yb)]
                teil["height_px"] = int(yb - ya + 1)
                # Every line is its own surface now, so it needs its own word —
                # and its own judgement about whether it can carry one.
                teil["proposal"] = None
                teil["cluster_id"] = -1
                teil["too_small"] = (
                    bool(r.get("too_small")) or (yb - ya + 1) < min_line_px
                )
                neu.append(teil)
            geteilt += 1
            bericht.append(
                f"  {r.get('class')} {bbox} -> {len(baender)} Zeilen: "
                + ", ".join(f"y{a}-{b}" for a, b in baender)
            )

        for i, r in enumerate(neu):
            r["index"] = i
        kopf = (
            f"[SplitLines] {len(regionen)} Regionen -> {len(neu)} "
            f"({geteilt} aufgeteilt)"
        )
        print(kopf)
        out = dict(sign_data)
        out["regions"] = neu
        return (out, len(neu), "\n".join([kopf] + bericht))


NODE_CLASS_MAPPINGS = {"FVM_SignSplitLines": FVM_SignSplitLines}
NODE_DISPLAY_NAME_MAPPINGS = {"FVM_SignSplitLines": "Sign Split Lines"}
