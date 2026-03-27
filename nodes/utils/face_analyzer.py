import logging
import os
import numpy as np
from insightface.app import FaceAnalysis

import folder_paths

try:
    from ...core.config import get_model_path
except ImportError:
    from core.config import get_model_path

logger = logging.getLogger("FVMTools")


class FaceAnalyzer:
    """InsightFace wrapper for face detection and embedding extraction."""

    def __init__(self, det_size: int = 640):
        self.det_size = det_size
        model_root = self._find_insightface_root()
        self.app = FaceAnalysis(
            name="buffalo_l",
            root=model_root,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))

    @staticmethod
    def _find_insightface_root():
        """Find InsightFace root directory.

        Search order:
        1. ComfyUI standard: {models_dir}/insightface/
        2. folder_paths registered 'insightface' category (extra_model_paths.yaml)
        3. outfit_config.ini [models] insightface_path
        Falls back to standard path if nothing found (InsightFace may auto-download).
        """
        # 1. Standard ComfyUI location
        standard = os.path.join(folder_paths.models_dir, "insightface")
        if os.path.isdir(os.path.join(standard, "models", "buffalo_l")):
            return standard

        # 2. folder_paths registered paths (extra_model_paths.yaml etc.)
        try:
            for p in folder_paths.get_folder_paths("insightface"):
                if os.path.isdir(os.path.join(p, "models", "buffalo_l")):
                    return p
        except Exception:
            pass

        # 3. INI fallback
        ini_path = get_model_path("insightface_path")
        if ini_path and os.path.isdir(os.path.join(ini_path, "models", "buffalo_l")):
            logger.info(f"[FVMTools] InsightFace found via outfit_config.ini: {ini_path}")
            return ini_path

        # Fall through: return standard path (InsightFace may auto-download)
        logger.warning(
            f"[FVMTools] InsightFace buffalo_l not found at {standard}/models/buffalo_l/ "
            f"— will attempt auto-download on first use."
        )
        return standard

    def detect_faces(self, bgr_image: np.ndarray) -> list:
        """Detect faces with adaptive fallback to smaller det_size."""
        faces = self.app.get(bgr_image)

        # Fallback: try smaller detection size if nothing found
        if len(faces) == 0 and self.det_size > 256:
            self.app.prepare(ctx_id=0, det_size=(256, 256))
            faces = self.app.get(bgr_image)
            # Restore original det_size
            self.app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size))

        # Sort by bbox area (largest first)
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return faces

    @staticmethod
    def get_embedding(face) -> np.ndarray:
        """Returns L2-normalized 512-dim embedding."""
        return face.normed_embedding
