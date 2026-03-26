"""Extract color palette from an image using numpy K-Means clustering."""

import numpy as np
from PIL import Image

from .color_utils import rgb_to_hsl, hue_distance, find_nearest_color_name
from .color_database import NEUTRAL_NAMES, METALLIC_NAMES
from .role_assignment import assign_roles


def _kmeans(pixels, k, seed=0, max_iter=50):
    """Simple K-Means implementation using numpy (no sklearn dependency).

    Args:
        pixels: (N, 3) float64 array of RGB values [0-255]
        k: number of clusters
        seed: random seed for reproducibility
        max_iter: maximum iterations

    Returns:
        (labels, centers) — labels is (N,) int array, centers is (k, 3) float64
    """
    rng = np.random.RandomState(seed)
    n = len(pixels)
    k = min(k, n)
    if k == 0:
        return np.zeros(0, dtype=int), np.zeros((0, 3), dtype=np.float64)

    # Deduplicate for init selection
    indices = rng.choice(n, k, replace=False)
    centers = pixels[indices].astype(np.float64)

    for _ in range(max_iter):
        # Assign each pixel to nearest center
        dists = np.linalg.norm(
            pixels[:, None].astype(np.float64) - centers[None, :], axis=2
        )
        labels = np.argmin(dists, axis=1)

        # Recompute centers
        new_centers = np.array([
            pixels[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
            for i in range(k)
        ])

        if np.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers

    return labels, centers


def _is_skin_tone(h, s, l):
    """Detect if an HSL color is likely a skin tone."""
    if 5 <= h <= 50 and 15 <= s <= 70 and 20 <= l <= 85:
        return True
    if 5 <= h <= 40 and 5 <= s <= 15 and 75 <= l <= 90:
        return True
    return False


def _apply_region(img_np, region):
    """Crop image array [H, W, 3] according to region mode.

    Returns cropped [H', W', 3] array.
    """
    h, w = img_np.shape[:2]
    if region == "center_crop":
        ch, cw = h // 4, w // 4
        return img_np[ch:h - ch, cw:w - cw]
    elif region == "upper_half":
        return img_np[:h // 2]
    elif region == "lower_half":
        return img_np[h // 2:]
    else:  # "full"
        return img_np


def _downsample(img_np, max_size=256):
    """Resize image so longest side <= max_size, using PIL for quality."""
    h, w = img_np.shape[:2]
    if max(h, w) <= max_size:
        return img_np
    scale = max_size / max(h, w)
    new_w = max(int(w * scale), 1)
    new_h = max(int(h * scale), 1)
    pil_img = Image.fromarray(img_np.astype(np.uint8))
    pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    return np.array(pil_img)


def extract_palette_from_image(image_tensor, num_colors, mode="dominant",
                                region="full", filter_skin=True,
                                filter_background=True,
                                saturation_threshold=0.1,
                                include_neutrals=True,
                                include_metallics=True,
                                seed=0):
    """Extract a color palette from a ComfyUI image tensor.

    Args:
        image_tensor: [B, H, W, C] float32 tensor, range [0, 1]
        num_colors: number of colors to extract (2-8)
        mode: "dominant", "vibrant", or "fashion_aware"
        region: "full", "center_crop", "upper_half", "lower_half"
        filter_skin: remove skin-tone clusters
        filter_background: remove large low-variance background clusters
        saturation_threshold: HSL saturation (0-1 scale, mapped to 0-100) below which
            a color is considered neutral
        include_neutrals: include neutral colors in result
        include_metallics: include metallic colors in result
        seed: random seed for K-Means

    Returns:
        dict with keys:
            "colors": list of {"name": str, "hsl": (h,s,l), "rgb": (r,g,b), "role": str|None}
            "palette_string": comma-separated color names
            "info": summary string
    """
    # 1. Convert tensor to numpy [H, W, 3] in 0-255
    img_np = image_tensor[0].cpu().numpy()  # [H, W, C]
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

    # 2. Apply region crop
    img_np = _apply_region(img_np, region)

    # 3. Downsample for speed
    img_np = _downsample(img_np, max_size=256)

    # 4. Reshape to pixel array (N, 3)
    pixels = img_np.reshape(-1, 3).astype(np.float64)

    if len(pixels) == 0:
        return _empty_result()

    # 5. K-Means with oversampling
    n_clusters = min(num_colors * 3, 24)
    # Guard: reduce k if fewer unique pixels
    unique_pixels = np.unique(pixels, axis=0)
    n_clusters = min(n_clusters, len(unique_pixels))
    if n_clusters == 0:
        return _empty_result()

    labels, centers = _kmeans(pixels, n_clusters, seed=seed)

    # 6. Count pixels per cluster
    cluster_counts = np.bincount(labels, minlength=n_clusters)
    total_pixels = len(pixels)

    # 7. Convert centers to HSL and build candidate list
    candidates = []
    for i in range(len(centers)):
        r, g, b = int(centers[i][0]), int(centers[i][1]), int(centers[i][2])
        h, s, l = rgb_to_hsl(r, g, b)
        count = int(cluster_counts[i])

        # Compute cluster variance for background detection
        cluster_pixels = pixels[labels == i]
        variance = float(cluster_pixels.var()) if len(cluster_pixels) > 0 else 0.0

        candidates.append({
            "rgb": (r, g, b),
            "hsl": (h, s, l),
            "count": count,
            "fraction": count / total_pixels if total_pixels > 0 else 0,
            "variance": variance,
        })

    # 8. Filter
    sat_threshold_100 = saturation_threshold * 100  # convert 0-1 to 0-100 scale

    filtered = []
    for c in candidates:
        h, s, l = c["hsl"]

        # Filter skin tones
        if filter_skin and _is_skin_tone(h, s, l):
            continue

        # Filter background: > 35% of pixels AND low variance
        if filter_background and c["fraction"] > 0.35 and c["variance"] < 200:
            continue

        filtered.append(c)

    # If filtering removed everything, fall back to unfiltered
    if not filtered:
        filtered = candidates

    # 9. Separate chromatic vs neutral
    chromatic = []
    neutrals_list = []
    for c in filtered:
        if c["hsl"][1] < sat_threshold_100:
            neutrals_list.append(c)
        else:
            chromatic.append(c)

    # 10. Apply mode sorting
    if mode == "vibrant":
        chromatic.sort(key=lambda c: c["hsl"][1], reverse=True)
    elif mode == "fashion_aware":
        chromatic = _greedy_hue_distance(chromatic)
    else:  # "dominant"
        chromatic.sort(key=lambda c: c["count"], reverse=True)

    neutrals_list.sort(key=lambda c: c["count"], reverse=True)

    # 11. Pick num_colors best with dedup
    selected = []
    exclude_names = set()

    # Pick chromatic first
    for c in chromatic:
        if len(selected) >= num_colors:
            break
        name = find_nearest_color_name(
            c["hsl"][0], c["hsl"][1], c["hsl"][2],
            exclude_names=exclude_names
        )
        # Skip if name maps to neutral/metallic and those are excluded
        if name in NEUTRAL_NAMES and not include_neutrals:
            continue
        if name in METALLIC_NAMES and not include_metallics:
            continue
        exclude_names.add(name)
        selected.append({
            "name": name,
            "hsl": c["hsl"],
            "rgb": c["rgb"],
            "role": None,
        })

    # Fill remaining with neutrals if allowed
    if include_neutrals:
        for c in neutrals_list:
            if len(selected) >= num_colors:
                break
            name = find_nearest_color_name(
                c["hsl"][0], c["hsl"][1], c["hsl"][2],
                exclude_names=exclude_names
            )
            exclude_names.add(name)
            selected.append({
                "name": name,
                "hsl": c["hsl"],
                "rgb": c["rgb"],
                "role": None,
            })

    # If still not enough, fill from any remaining candidates
    all_remaining = chromatic + neutrals_list
    all_remaining.sort(key=lambda c: c["count"], reverse=True)
    for c in all_remaining:
        if len(selected) >= num_colors:
            break
        name = find_nearest_color_name(
            c["hsl"][0], c["hsl"][1], c["hsl"][2],
            exclude_names=exclude_names
        )
        if name not in exclude_names:
            exclude_names.add(name)
            selected.append({
                "name": name,
                "hsl": c["hsl"],
                "rgb": c["rgb"],
                "role": None,
            })

    # 12. Assign roles
    if selected:
        assign_roles(selected)

    # 13. Build result
    palette_string = ", ".join(c["name"] for c in selected)
    info_parts = [
        f"Extracted {len(selected)} colors from image",
        f"Mode: {mode}",
        f"Region: {region}",
        f"K-Means clusters: {n_clusters}",
    ]
    if filter_skin:
        info_parts.append("Skin-tone filter: on")
    if filter_background:
        info_parts.append("Background filter: on")

    return {
        "colors": selected,
        "palette_string": palette_string,
        "info": " | ".join(info_parts),
    }


def _greedy_hue_distance(candidates):
    """Sort candidates by greedy max-hue-distance selection."""
    if not candidates:
        return []

    # Start with highest saturation
    candidates_copy = list(candidates)
    candidates_copy.sort(key=lambda c: c["hsl"][1], reverse=True)

    result = [candidates_copy.pop(0)]

    while candidates_copy:
        best_idx = 0
        best_min_dist = -1
        for i, c in enumerate(candidates_copy):
            min_dist = min(
                hue_distance(c["hsl"][0], r["hsl"][0]) for r in result
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i
        result.append(candidates_copy.pop(best_idx))

    return result


def _empty_result():
    """Return an empty palette result."""
    return {
        "colors": [],
        "palette_string": "",
        "info": "No colors extracted (empty image)",
    }
