"""PersonSelectorSAM3Native — the SAM3 selector on ComfyUI's built-in SAM3.

Same masks, same PERSON_DATA, same preview as PersonSelectorSAM3 — but grounded
against the SAM3 that ships with ComfyUI (comfy/ldm/sam3) instead of the
comfyui-sam3 custom extension. No extension, no second model load, no separate
Python environment that can drift out of sync with ComfyUI's own.

Wiring: one CheckpointLoaderSimple on the SAM3 checkpoint gives both MODEL and
CLIP — the SAM3 text encoder lives in the same file. Feed both in. The node
builds its own prompts ("person", "face", "hair", ...), which is why it needs
the CLIP object rather than a pre-encoded CONDITIONING.

Deliberately slimmer than PersonSelectorSAM3:
  - 5 reference slots instead of 10
  - no outfit_palettes (and match_weights is face/hair/head, no outfit term)
  - no SAM2 fallback model
  - no YOLO aux pathway (that needs forward_segment with box_inputs — a
    different native call than text grounding; left out rather than half-done)
  - no guaranteed_refs, aggregation fixed to "max"

Everything else — BiSeNet facial subtypes, the aux text-prompt presets, depth
render ordering, the preview — is inherited unchanged.
"""

import numpy as np
import torch
import cv2

from .person_selector_sam3 import PersonSelectorSAM3, AUX_PRESETS
from .utils.masker import NativeSAM3, sam3_prepare, sam3_ground
from .utils.tensor_utils import tensor2np, np2tensor, empty_mask


class PersonSelectorSAM3Native(PersonSelectorSAM3):
    """PersonSelectorSAM3 driven by ComfyUI's native SAM3 instead of the extension."""

    MAX_REFERENCES = 5

    CATEGORY = "FVM Tools/Face"
    FUNCTION = "execute"
    RETURN_TYPES = PersonSelectorSAM3.RETURN_TYPES + ("PERSON_DATA",)
    RETURN_NAMES = PersonSelectorSAM3.RETURN_NAMES + ("aux_data",)

    # Colour of the aux overlay in the preview (RGB).
    AUX_COLOR = (255, 0, 220)

    DESCRIPTION = (
        "Person selector on ComfyUI's BUILT-IN SAM3 — no custom extension needed.\n\n"
        "Wire one CheckpointLoaderSimple on the SAM3 checkpoint:\n"
        "  MODEL -> sam3_model     CLIP -> sam3_clip\n"
        "(the SAM3 text encoder ships inside the same checkpoint)\n\n"
        "Produces the same PERSON_DATA as Person Selector SAM3: body, face, head,\n"
        "hair, the BiSeNet facial subtypes (facial_skin/eyes/mouth/neck/accessories)\n"
        "and an aux mask from a preset or free text prompt.\n\n"
        "Slimmer on purpose: 5 references, no outfit palettes, no SAM2, no YOLO aux.\n"
        "Use Person Selector SAM3 if you need those."
    )

    @classmethod
    def INPUT_TYPES(cls):
        aux_choices = list(AUX_PRESETS.keys())
        return {
            "required": {
                "sam3_model": ("MODEL", {"tooltip": "MODEL from CheckpointLoaderSimple on the SAM3 checkpoint"}),
                "sam3_clip": ("CLIP", {"tooltip": "CLIP from the SAME CheckpointLoaderSimple.\n"
                                                  "The SAM3 text encoder is part of the checkpoint.\n"
                                                  "Needed because this node builds its own prompts."}),
                "current_image": ("IMAGE", {"tooltip": "Image(s) to process. Supports batch input."}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Face-to-reference matching threshold.\n"
                               "0.0 = AUTO: optimal 1:1 assignment (recommended).\n"
                               "Above 0 = manual minimum similarity."}),
                "det_size": (["320", "480", "640", "768"], {"default": "640",
                    "tooltip": "Face detection resolution (InsightFace, not SAM3)."}),
                "match_weights": ("STRING", {"default": "70/15/15",
                    "tooltip": "Face/hair/head blend for reference matching, e.g. '70/15/15'.\n"
                               "No outfit term here — this node has no outfit_palettes input."}),
                "aux_preset": (aux_choices, {"default": "none",
                    "tooltip": "Preset aux mask type:\n"
                               "- upper_body / lower_body: body halves\n"
                               "- clothing: all clothing (no shoes/socks)\n"
                               "- hands / feet / arms / legs: body parts\n"
                               "- headless_body: body minus head (computed)\n"
                               "- custom: uses aux_custom_prompt below"}),
                "aux_custom_prompt": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Custom SAM3 text prompt for the aux mask (aux_preset='custom').\n"
                               "Any noun phrase: 'shoes', 'necklace', 'backpack'.\n"
                               "Note: SAM3 treats commas as separate categories."}),
                "aux_threshold": ("FLOAT", {"default": 0.30, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": "Confidence threshold for aux detection."}),
                "aux_prompt": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Free SAM3 text prompt for the SECOND aux channel, e.g.\n"
                               "'sunglasses', 'handbag', 'tattoo', 'shoes'.\n"
                               "Result comes out as `aux_data` — a full PERSON_DATA where\n"
                               "EVERY mask type carries the aux region, so a PersonDetailer\n"
                               "hits it whatever its mask_type is set to.\n"
                               "Also drawn into the preview. Empty = channel off."}),
                "aux_prompt_threshold": ("FLOAT", {"default": 0.30, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": "Confidence threshold for the aux_prompt grounding."}),
                "refine_iterations": ("INT", {"default": 2, "min": 0, "max": 5, "step": 1,
                    "tooltip": "SAM decoder passes that sharpen each detector mask.\n"
                               "0 = raw detector masks (faster, blockier edges)."}),
            },
            "optional": {
                **{f"reference_{i}": ("IMAGE", {"tooltip": f"Reference image(s) for person {i}."})
                   for i in range(1, cls.MAX_REFERENCES + 1)},
                "depth_map": ("IMAGE", {"tooltip": "Depth map for render order sorting."}),
                "depth_sort_order": (["front_last", "front_first", "off"], {"default": "front_last",
                    "tooltip": "Rendering order handed to the PersonDetailer."}),
            },
        }

    def execute(self, sam3_model, sam3_clip, current_image, threshold, det_size,
                match_weights="70/15/15", aux_preset="none", aux_custom_prompt="",
                aux_threshold=0.30, aux_prompt="", aux_prompt_threshold=0.30,
                refine_iterations=2,
                depth_map=None, depth_sort_order="front_last", **kwargs):
        """Bundle MODEL+CLIP into the native backend, then run the inherited pipeline.

        `threshold == 0` maps onto the parent's auto_threshold flag — one widget
        instead of the parent's auto_threshold/threshold pair.
        """
        backend = NativeSAM3(sam3_model, sam3_clip, refine_iterations=refine_iterations)

        # match_weights here is face/hair/head; the parent parses face/hair/head/outfit.
        weights = [p.strip() for p in str(match_weights).split("/") if p.strip()]
        if len(weights) == 3:
            match_weights = "/".join(weights + ["0"])

        result = super().execute(
            sam3_model=backend,
            current_image=current_image,
            auto_threshold=(threshold <= 0.0),
            threshold=threshold,
            guaranteed_refs=0,
            aggregation="max",
            det_size=det_size,
            aux_preset=aux_preset,
            aux_custom_prompt=aux_custom_prompt,
            aux_threshold=aux_threshold,
            match_weights=match_weights,
            outfit_palettes=None,
            depth_map=depth_map,
            depth_sort_order=depth_sort_order,
            aux_yolo_model="None",
            sam_model=None,
            **kwargs,
        )

        person_data, preview = result[0], result[5]
        aux_data, preview = self._build_aux(
            backend, current_image, person_data, preview,
            aux_prompt, aux_prompt_threshold)
        return result[:5] + (preview,) + result[6:] + (aux_data,)

    # ── Free-text aux channel ──

    def _build_aux(self, backend, images, person_data, preview, prompt, threshold):
        """Ground `prompt` and hand back a PERSON_DATA of the hits plus a marked preview.

        The aux region is written into EVERY mask type of the returned
        PERSON_DATA, so whatever `mask_type` a downstream PersonDetailer is set
        to, it inpaints the aux region. Assignment to a person is by largest
        overlap with that person's body mask; a hit that overlaps nobody is
        dropped from aux_data but still drawn in the preview.
        """
        # `or` short-circuits: don't touch images.shape when person_data already
        # carries the size (a dict.get default is evaluated eagerly).
        h = int(person_data.get("image_height") or images.shape[1])
        w = int(person_data.get("image_width") or images.shape[2])
        batch_size = int(person_data.get("batch_size", 1)) or 1
        num_refs = int(person_data.get("num_references", 0))
        mask_types = [k[:-len("_masks")] for k, v in person_data.items()
                      if k.endswith("_masks") and isinstance(v, list)
                      and k not in ("per_face_masks", "aux_unassigned_masks")]

        empty = [[empty_mask(h, w) for _ in range(max(num_refs, 0))]
                 for _ in range(batch_size)]
        prompt = (prompt or "").strip()
        if not prompt or num_refs == 0:
            return self._pack_aux(person_data, empty, mask_types, h, w, batch_size,
                                  num_refs), preview

        per_batch = []
        unassigned = []
        for b in range(batch_size):
            slots = [np.zeros((h, w), dtype=np.float32) for _ in range(num_refs)]
            loose = np.zeros((h, w), dtype=np.float32)
            if b < images.shape[0]:
                rgb = tensor2np(images[b:b + 1])
                state, base = sam3_prepare(backend, rgb)
                results = sam3_ground(state, base, rgb.shape, prompt, threshold) if state else []

                bodies = []
                for ri in range(num_refs):
                    lst = person_data.get("body_masks") or person_data.get("face_masks")
                    if lst and ri < len(lst) and b < lst[ri].shape[0]:
                        bodies.append(lst[ri][b].cpu().numpy() > 0.5)
                    else:
                        bodies.append(None)

                for mask_np, _score, _bbox in results:
                    mb = mask_np > 0.5
                    area = int(mb.sum())
                    if area == 0:
                        continue
                    best_ri, best = None, 0
                    for ri, body in enumerate(bodies):
                        if body is None:
                            continue
                        inter = int(np.logical_and(mb, body).sum())
                        if inter > best:
                            best_ri, best = ri, inter
                    if best_ri is None or best == 0:
                        loose = np.maximum(loose, mask_np)
                    else:
                        slots[best_ri] = np.maximum(slots[best_ri], mask_np)

            per_batch.append(slots)
            unassigned.append(loose)

        tensors = [[torch.from_numpy(m).unsqueeze(0) for m in slots] for slots in per_batch]
        aux_data = self._pack_aux(person_data, tensors, mask_types, h, w,
                                  batch_size, num_refs)
        aux_data["aux_unassigned_masks"] = torch.stack(
            [torch.from_numpy(u) for u in unassigned])
        preview = self._draw_aux(preview, per_batch[0], unassigned[0], prompt)
        return aux_data, preview

    @staticmethod
    def _pack_aux(person_data, per_batch_tensors, mask_types, h, w, batch_size, num_refs):
        """Wrap per-slot aux masks in a PERSON_DATA with the same layout as the input."""
        out = {
            "batch_size": batch_size,
            "num_references": num_refs,
            "image_height": h,
            "image_width": w,
            "depth_sort_order": person_data.get("depth_sort_order", "front_last"),
            "matches": [[True] * num_refs for _ in range(batch_size)],
            "face_to_ref": [list(range(num_refs)) for _ in range(batch_size)],
            "ref_depths": person_data.get("ref_depths",
                                          [{} for _ in range(batch_size)]),
        }
        per_ref = []
        for ri in range(num_refs):
            per_ref.append(torch.cat([per_batch_tensors[b][ri] for b in range(batch_size)], dim=0))
        for mt in (mask_types or ["face", "head", "body"]):
            out[f"{mt}_masks"] = list(per_ref)
        out["aux_masks"] = list(per_ref)
        out["per_face_masks"] = [
            [{mt: per_batch_tensors[b][ri] for mt in (mask_types or ["face", "head", "body"])}
             for ri in range(num_refs)]
            for b in range(batch_size)
        ]
        union = [torch.max(torch.cat(per_batch_tensors[b], dim=0), dim=0, keepdim=True)[0]
                 if num_refs else empty_mask(h, w) for b in range(batch_size)]
        out["all_faces_mask"] = torch.cat(union, dim=0)
        out["matched_faces_mask"] = out["all_faces_mask"].clone()
        out["aux_part_counts"] = [
            {ri: int(float(per_batch_tensors[b][ri].max()) > 0.5) for ri in range(num_refs)}
            for b in range(batch_size)
        ]
        return out

    @classmethod
    def _draw_aux(cls, preview, slots, unassigned, prompt):
        """Outline + tint the aux regions on the preview returned by the parent."""
        try:
            rgb = tensor2np(preview).copy()
            h, w = rgb.shape[:2]
            overlay = rgb.copy()
            drawn = 0
            for m in list(slots) + [unassigned]:
                binm = (m > 0.5).astype(np.uint8)
                if binm.sum() == 0:
                    continue
                drawn += 1
                overlay[binm > 0] = cls.AUX_COLOR
                contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, contours, -1, cls.AUX_COLOR, max(2, h // 400))
            if drawn == 0:
                return preview
            rgb = cv2.addWeighted(overlay, 0.30, rgb, 0.70, 0)
            scale = max(0.5, min(w, h) / 900.0)
            label = f"aux: {prompt}"
            cv2.putText(rgb, label, (8, int(28 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (0, 0, 0), max(3, int(4 * scale)), cv2.LINE_AA)
            cv2.putText(rgb, label, (8, int(28 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, cls.AUX_COLOR, max(1, int(2 * scale)), cv2.LINE_AA)
            return np2tensor(rgb)
        except Exception as e:
            print(f"[PersonSelectorSAM3Native] aux preview overlay failed: {e}")
            return preview
