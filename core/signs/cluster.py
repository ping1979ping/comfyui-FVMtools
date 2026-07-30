"""Group visually near-identical text crops so one repair can serve many.

A shelf of twelve identical bottles produces twelve label crops that only
differ by perspective and noise. Repairing one and reusing its wording keeps
the render consistent and cheap. Features are a DCT perceptual hash plus a
coarse HSV histogram; grouping is single-linkage agglomerative.
"""
import math

import cv2
import numpy as np

PHASH_SIZE = 8
PHASH_SCALE = 4
HIST_BINS = (8, 4, 4)
HIST_SAMPLE_SIZE = 32
HIST_SMOOTH_KERNEL = (0.25, 0.5, 0.25)
PHASH_WEIGHT = 0.6
HIST_WEIGHT = 0.4
DEFAULT_CLUSTER_DISTANCE = 0.15

_HIST_LENGTH = HIST_BINS[0] * HIST_BINS[1] * HIST_BINS[2]


def _to_uint8_rgb(image_rgb):
    """Normalise any 2D/3D array into an HxWx3 uint8 RGB image, or None."""
    if image_rgb is None:
        return None
    arr = np.asarray(image_rgb)
    if arr.size == 0 or arr.ndim not in (2, 3):
        return None

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] >= 4:
        arr = arr[..., :3]
    elif arr.shape[-1] != 3:
        return None

    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)

    arr = arr.astype(np.float32)
    if float(np.nanmax(np.abs(arr))) <= 1.0:
        arr = arr * 255.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    return np.ascontiguousarray(np.clip(arr, 0.0, 255.0).astype(np.uint8))


def _to_gray(image_rgb):
    """Grayscale view of any accepted image, or None for unusable input."""
    rgb = _to_uint8_rgb(image_rgb)
    if rgb is None:
        return None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def phash(image_rgb, hash_size=8):
    """
    DCT perceptual hash as a flat boolean array of ``hash_size ** 2`` bits.

    Input size, dtype and channel count do not matter; unusable input yields
    an all-false hash instead of raising.
    """
    size = max(1, int(hash_size))
    bits = size * size

    gray = _to_gray(image_rgb)
    if gray is None:
        return np.zeros(bits, dtype=bool)

    side = size * PHASH_SCALE
    small = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(small.astype(np.float32))
    flat = coefficients[:size, :size].reshape(-1)

    # The DC term carries overall brightness; excluding it from the median
    # keeps the hash stable under exposure changes.
    reference = float(np.median(flat[1:])) if flat.size > 1 else 0.0
    return np.asarray(flat > reference, dtype=bool)


def _smooth_axis(hist, axis, wrap):
    """Blur one histogram axis with a 1-2-1 kernel."""
    low, mid, high = HIST_SMOOTH_KERNEL
    if wrap:
        before = np.roll(hist, 1, axis=axis)
        after = np.roll(hist, -1, axis=axis)
    else:
        pad = [(0, 0)] * hist.ndim
        pad[axis] = (1, 1)
        padded = np.pad(hist, pad, mode="edge")
        length = hist.shape[axis]
        before = np.take(padded, range(0, length), axis=axis)
        after = np.take(padded, range(2, length + 2), axis=axis)
    return low * before + mid * hist + high * after


def color_signature(image_rgb):
    """
    Small L1-normalised HSV histogram, robust to crop size.

    The bins are blurred afterwards (circularly along hue) so that sensor
    noise nudging a pixel across a bin edge cannot swing the signature.
    """
    rgb = _to_uint8_rgb(image_rgb)
    if rgb is None:
        return np.zeros(_HIST_LENGTH, dtype=np.float32)

    small = cv2.resize(
        rgb, (HIST_SAMPLE_SIZE, HIST_SAMPLE_SIZE), interpolation=cv2.INTER_AREA
    )
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, list(HIST_BINS), [0, 180, 0, 256, 0, 256]
    ).astype(np.float32)

    hist = _smooth_axis(hist, 0, wrap=True)
    hist = _smooth_axis(hist, 1, wrap=False)
    hist = _smooth_axis(hist, 2, wrap=False)

    hist = hist.reshape(-1)
    total = float(hist.sum())
    if total > 0.0:
        hist = hist / total
    return np.ascontiguousarray(hist, dtype=np.float32)


def extract_features(image_rgb):
    """Bundle the hash and the colour signature of one crop."""
    return {"phash": phash(image_rgb, PHASH_SIZE), "hist": color_signature(image_rgb)}


def crop_distance(a_feat, b_feat):
    """
    Distance between two feature dicts, 0 (identical) .. 1 (unrelated).

    Combines the normalised hamming distance of the perceptual hash with the
    histogram distance. Missing or mismatched features count as fully distant.
    """
    a_feat = a_feat if isinstance(a_feat, dict) else {}
    b_feat = b_feat if isinstance(b_feat, dict) else {}

    a_hash = np.asarray(a_feat.get("phash", []), dtype=bool).reshape(-1)
    b_hash = np.asarray(b_feat.get("phash", []), dtype=bool).reshape(-1)
    if a_hash.size == 0 or a_hash.shape != b_hash.shape:
        hash_distance = 1.0
    else:
        hash_distance = float(np.mean(a_hash != b_hash))

    a_hist = np.asarray(a_feat.get("hist", []), dtype=np.float32).reshape(-1)
    b_hist = np.asarray(b_feat.get("hist", []), dtype=np.float32).reshape(-1)
    if a_hist.size == 0 or a_hist.shape != b_hist.shape:
        hist_distance = 1.0
    else:
        hist_distance = 0.5 * float(np.abs(a_hist - b_hist).sum())

    combined = PHASH_WEIGHT * hash_distance + HIST_WEIGHT * hist_distance
    return float(min(1.0, max(0.0, combined)))


def cluster_crops(crops, distance=DEFAULT_CLUSTER_DISTANCE):
    """
    Single-linkage grouping of near-identical crops.

    Returns one cluster id per crop, numbered in order of first appearance and
    starting at 0. A crop without a partner still gets its own id.
    """
    if not crops:
        return []

    features = [extract_features(crop) for crop in crops]
    count = len(features)
    parent = list(range(count))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    threshold = float(distance)
    for i in range(count):
        for j in range(i + 1, count):
            if crop_distance(features[i], features[j]) <= threshold:
                union(i, j)

    mapping = {}
    labels = []
    for i in range(count):
        root = find(i)
        if root not in mapping:
            mapping[root] = len(mapping)
        labels.append(mapping[root])
    return labels


def pick_cluster_representative(crops, labels, cluster_id):
    """
    Index of the sharpest and largest member of a cluster.

    Scores every member by ``variance of Laplacian * sqrt(area)`` and returns
    the argmax. Returns -1 when the cluster is empty.
    """
    if crops is None or labels is None:
        return -1

    members = [
        i for i, label in enumerate(labels)
        if label == cluster_id and i < len(crops)
    ]
    if not members:
        return -1

    best_index = members[0]
    best_score = -1.0
    for index in members:
        gray = _to_gray(crops[index])
        if gray is None or gray.size == 0:
            score = -1.0
        else:
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            area = float(gray.shape[0] * gray.shape[1])
            score = sharpness * math.sqrt(area)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index
