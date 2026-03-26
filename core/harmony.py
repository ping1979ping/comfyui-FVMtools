"""Generate harmony hues from a base hue using color theory relationships."""


def generate_harmony_hues(base_hue, harmony_type, count):
    """
    Generate a list of hues based on a harmony type.

    Args:
        base_hue: Starting hue (0-360)
        harmony_type: One of "analogous", "complementary", "split_complementary",
                      "triadic", "tetradic", "monochromatic"
        count: Number of hues to generate (minimum 1)

    Returns:
        List of hues (floats, 0-360), always starting with base_hue.
    """
    if count < 1:
        return []

    base_hue = base_hue % 360

    # Get natural harmony hues (always includes base)
    if harmony_type == "analogous":
        natural = _analogous(base_hue)
    elif harmony_type == "complementary":
        natural = _complementary(base_hue)
    elif harmony_type == "split_complementary":
        natural = _split_complementary(base_hue)
    elif harmony_type == "triadic":
        natural = _triadic(base_hue)
    elif harmony_type == "tetradic":
        natural = _tetradic(base_hue)
    elif harmony_type == "monochromatic":
        natural = [base_hue] * count
        return natural[:count]
    else:
        raise ValueError(f"Unknown harmony type: {harmony_type}")

    if count <= len(natural):
        return natural[:count]

    # Fill extras by subdividing between existing hues
    result = list(natural)
    while len(result) < count:
        # Find the largest gap and bisect it
        best_gap = -1
        best_idx = 0
        n = len(result)
        for i in range(n):
            h1 = result[i]
            h2 = result[(i + 1) % n]
            gap = (h2 - h1) % 360
            if gap > best_gap:
                best_gap = gap
                best_idx = i
        h1 = result[best_idx]
        h2 = result[(best_idx + 1) % len(result)]
        mid = (h1 + ((h2 - h1) % 360) / 2) % 360
        result.insert(best_idx + 1, mid)

    return result[:count]


def _analogous(base):
    """Base ±30°."""
    return [base, (base + 30) % 360, (base - 30) % 360]


def _complementary(base):
    """Base + 180°."""
    return [base, (base + 180) % 360]


def _split_complementary(base):
    """Base + 150°, base + 210°."""
    return [base, (base + 150) % 360, (base + 210) % 360]


def _triadic(base):
    """Base + 120°, base + 240°."""
    return [base, (base + 120) % 360, (base + 240) % 360]


def _tetradic(base):
    """Base + 90°, +180°, +270°."""
    return [base, (base + 90) % 360, (base + 180) % 360, (base + 270) % 360]
