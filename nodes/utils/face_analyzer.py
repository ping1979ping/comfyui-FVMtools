import os
import numpy as np
from insightface.app import FaceAnalysis

import folder_paths


class FaceAnalyzer:
    """InsightFace wrapper for face detection and embedding extraction."""

    def __init__(self, det_size: int = 640):
        self.det_size = det_size
        model_root = os.path.join(folder_paths.models_dir, "insightface")
        self.app = FaceAnalysis(
            name="buffalo_l",
            root=model_root,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))

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
