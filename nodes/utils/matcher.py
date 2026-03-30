import numpy as np
from typing import Tuple

from .appearance import (
    hair_color_similarity,
    head_histogram_similarity,
    outfit_color_similarity,
    outfit_ranking_similarity,
    combined_similarity,
    parse_match_weights,
)


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity via dot product (embeddings are L2-normalized)."""
    return float(np.dot(emb1, emb2))


def aggregate_similarities(sims: list, mode: str) -> float:
    """Aggregate a list of similarities using max/mean/min."""
    if not sims:
        return 0.0
    if mode == "max":
        return max(sims)
    elif mode == "mean":
        return sum(sims) / len(sims)
    elif mode == "min":
        return min(sims)
    return max(sims)


def find_best_match(
    face_embs: list,
    ref_embs: list,
    aggregation: str = "max",
) -> Tuple[int, float, int]:
    """Find the face that best matches the reference embeddings.

    Returns:
        (face_idx, similarity, best_ref_idx) or (-1, 0.0, -1) if no faces.
    """
    if not face_embs or not ref_embs:
        return (-1, 0.0, -1)

    best_face_idx = -1
    best_sim = -1.0
    best_ref_idx = -1

    for fi, face_emb in enumerate(face_embs):
        sims_per_ref = []
        best_ref_for_face = -1
        best_single_sim = -1.0

        for ri, ref_emb in enumerate(ref_embs):
            sim = compute_similarity(face_emb, ref_emb)
            sims_per_ref.append(sim)
            if sim > best_single_sim:
                best_single_sim = sim
                best_ref_for_face = ri

        agg_sim = aggregate_similarities(sims_per_ref, aggregation)
        if agg_sim > best_sim:
            best_sim = agg_sim
            best_face_idx = fi
            best_ref_idx = best_ref_for_face

    return (best_face_idx, best_sim, best_ref_idx)


def build_appearance_matrix(face_sims, ref_hair_colors, face_hair_colors,
                            ref_head_hists, face_head_hists, weights,
                            ref_outfit_hists=None, face_clothing_hists=None,
                            ref_palette_colors=None, face_clothing_colors=None):
    """Build a combined similarity matrix incorporating face, hair, head, and outfit signals.

    Args:
        face_sims: np.ndarray (num_refs, num_faces) — ArcFace cosine similarities.
        ref_hair_colors: list of HSV arrays (or None) per reference.
        face_hair_colors: list of HSV arrays (or None) per detected face.
        ref_head_hists: list of histograms (or None) per reference.
        face_head_hists: list of histograms (or None) per detected face.
        weights: (face_w, hair_w, head_w, outfit_w) tuple summing to 1.0.
        ref_outfit_hists: list of HSV histograms (or None) per reference (legacy, fallback).
        face_clothing_hists: list of HSV histograms (or None) per detected face (legacy, fallback).
        ref_palette_colors: list of [HSV, HSV, ...] per reference (smart matching).
        face_clothing_colors: list of (upper_hsv, lower_hsv) per detected face (smart matching).

    Returns:
        np.ndarray (num_refs, num_faces) — combined similarity scores.
    """
    num_refs, num_faces = face_sims.shape
    result = np.zeros_like(face_sims)

    w_face = weights[0]
    w_hair = weights[1]
    w_head = weights[2]
    w_outfit = weights[3] if len(weights) > 3 else 0.0

    # Fast path: face-only mode
    if w_hair == 0 and w_head == 0 and w_outfit == 0:
        return face_sims.copy()

    # Determine outfit matching mode: smart (dominant colors) or legacy (histograms)
    use_smart_outfit = (ref_palette_colors is not None and face_clothing_colors is not None)

    for ri in range(num_refs):
        for fi in range(num_faces):
            f_sim = face_sims[ri, fi]

            h_sim = hair_color_similarity(
                ref_hair_colors[ri] if ri < len(ref_hair_colors) else None,
                face_hair_colors[fi] if fi < len(face_hair_colors) else None,
            )

            hd_sim = head_histogram_similarity(
                ref_head_hists[ri] if ri < len(ref_head_hists) else None,
                face_head_hists[fi] if fi < len(face_head_hists) else None,
            )

            o_sim = 0.0
            if w_outfit > 0:
                if use_smart_outfit:
                    palette_cols = ref_palette_colors[ri] if ri < len(ref_palette_colors) else []
                    upper, lower = face_clothing_colors[fi] if fi < len(face_clothing_colors) else (None, None)
                    o_sim = outfit_ranking_similarity(palette_cols, upper, lower)
                elif ref_outfit_hists is not None and face_clothing_hists is not None:
                    o_sim = outfit_color_similarity(
                        ref_outfit_hists[ri] if ri < len(ref_outfit_hists) else None,
                        face_clothing_hists[fi] if fi < len(face_clothing_hists) else None,
                    )

            result[ri, fi] = combined_similarity(f_sim, h_sim, hd_sim, weights, outfit_sim=o_sim)

    return result
