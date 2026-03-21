import numpy as np
from typing import Tuple


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
