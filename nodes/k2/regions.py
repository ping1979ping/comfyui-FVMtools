"""FVM K2 Lab — Regionsdefinition und -verkettung.

Jede Region ist eine eigene Node. Über den optionalen ``regions``-Eingang
lassen sie sich zu einer Kette hängen, ohne dass eine Sammel-Node nötig wäre.
"""

import json

import numpy as np
import torch

from ...core.k2.geometry import PixelBox
from ...core.k2.prompt import ROLES, EmphasisRequest, GLOBAL_SCOPE, RegionDefinition

CATEGORY = "FVM Tools/K2"


def _next_id(regions, prefix="region"):
    return f"{prefix}-{len(regions) + 1}"


class FVM_K2_Region:
    """Eine benannte Bildregion mit eigenem Prompt."""

    DESCRIPTION = (
        "Defines one named pixel-space region for Krea 2 regional prompting.\n\n"
        "The box is given in output pixels of the final image. Its prompt is compiled "
        "into the unified Krea prompt with an explicit location clause, and the K2 "
        "attention router binds exactly those text tokens to the image tokens inside "
        "this box.\n\n"
        "Chain several Region nodes through the 'regions' input to build a layout."
    )
    CATEGORY = CATEGORY
    FUNCTION = "build"
    RETURN_TYPES = ("K2_REGION",)
    RETURN_NAMES = ("regions",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": (
                    "STRING",
                    {
                        "default": "Subject",
                        "tooltip": "Human readable label. Used in the generated spatial "
                        "instructions ('Anna is to the left of Bea'), in face "
                        "assignment and in the report. Keep it short and distinct.",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "What should appear inside this box. An empty prompt "
                        "disables the region.",
                    },
                ),
                "x": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8,
                              "tooltip": "Left edge in output pixels."}),
                "y": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8,
                              "tooltip": "Top edge in output pixels."}),
                "width": ("INT", {"default": 512, "min": 16, "max": 16384, "step": 8,
                                  "tooltip": "Box width in output pixels."}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8,
                                   "tooltip": "Box height in output pixels."}),
                "role": (
                    list(ROLES),
                    {
                        "default": "auto",
                        "tooltip": "subject: full outside penalty, competes with other "
                        "subjects for overlapping tokens.\n"
                        "background: softer penalty, may feather beyond its box.\n"
                        "auto: boxes covering >=70% of canvas width become background.",
                    },
                ),
                "priority": (
                    "INT",
                    {
                        "default": 100, "min": -1000, "max": 1000,
                        "tooltip": "Higher compiles first and claims ambiguous detected "
                        "faces first. It is NOT a strength and NOT an image z-index.",
                    },
                ),
                "enabled": ("BOOLEAN", {"default": True,
                                        "tooltip": "Exclude the region without deleting it."}),
            },
            "optional": {
                "regions": ("K2_REGION", {"tooltip": "Chain input: previously defined regions."}),
                "identity_prompt": (
                    "STRING",
                    {
                        "multiline": True, "default": "",
                        "tooltip": "Face/identity description. It is placed first in the "
                        "clause, can be protected from the projector delta, and is "
                        "preferred by K2 Face Detail.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True, "default": "",
                        "tooltip": "Region-local negative text. Stored for tooling; Krea 2 "
                        "Turbo has no separate regional negative branch (CFG-free).",
                    },
                ),
                "region_id": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Stable internal id. Leave empty for an automatic one. "
                        "Regional LoRAs reference regions by name or id.",
                    },
                ),
            },
        }

    def build(self, name, prompt, x, y, width, height, role, priority, enabled,
              regions=None, identity_prompt="", negative_prompt="", region_id=""):
        collected = list(regions or [])
        definition = RegionDefinition(
            region_id=region_id.strip() or _next_id(collected),
            name=name.strip() or _next_id(collected),
            box=PixelBox.from_xywh(x, y, width, height),
            prompt=prompt,
            identity_prompt=identity_prompt,
            negative_prompt=negative_prompt,
            enabled=bool(enabled),
            priority=int(priority),
            role=role,
        )
        if any(r.name == definition.name for r in collected):
            raise ValueError(
                f"Region name {definition.name!r} is already used — names must be unique "
                "because they appear in the generated spatial instructions."
            )
        collected.append(definition)
        return (collected,)


class FVM_K2_RegionFromBBox:
    """Erzeugt eine Region aus einer Detektor-Bounding-Box."""

    DESCRIPTION = (
        "Turns a detector bounding box into a K2 region. Accepts BBOX/BOUNDING_BOX "
        "lists as produced by KJNodes' Ideogram prompt builder, YOLO detectors or "
        "FVM person selectors.\n\n"
        "Supports xywh and xyxy, absolute pixels or normalized 0..1 coordinates."
    )
    CATEGORY = CATEGORY
    FUNCTION = "build"
    RETURN_TYPES = ("K2_REGION",)
    RETURN_NAMES = ("regions",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "Subject",
                         "tooltip": "Label for this detected region. Must be unique."}),
                "prompt": ("STRING", {"multiline": True, "default": "",
                           "tooltip": "What should appear inside the detected box."}),
                "bbox_index": ("INT", {"default": 0, "min": 0, "max": 64,
                                       "tooltip": "Zero-based index into the bbox list. "
                                       "Out-of-range clamps to the last entry."}),
                "bbox_format": (["xyxy", "xywh", "auto"], {"default": "xyxy",
                                "tooltip": "xyxy = left/top/right/bottom (what YOLO, "
                                "BOUNDING_BOX and the KJ prompt builder emit).\n"
                                "xywh = left/top/width/height.\n"
                                "auto tries xyxy first and only falls back to xywh when "
                                "that yields an impossible box — it cannot always tell "
                                "them apart, so set the format explicitly when you know it."}),
                "reference_width": ("INT", {"default": 1024, "min": 16, "max": 16384,
                                            "tooltip": "Canvas width used to expand "
                                            "normalized coordinates."}),
                "reference_height": ("INT", {"default": 1024, "min": 16, "max": 16384,
                                             "tooltip": "Canvas height for normalized "
                                             "coordinates. Must match the render size."}),
                "grow_px": ("INT", {"default": 0, "min": -512, "max": 512,
                                    "tooltip": "Expand (or shrink) every side."}),
                "role": (list(ROLES), {"default": "auto",
                         "tooltip": "subject / background / auto — same meaning as on "
                         "K2 Region."}),
                "priority": ("INT", {"default": 100, "min": -1000, "max": 1000,
                             "tooltip": "Higher compiles first and claims ambiguous faces "
                             "first."}),
            },
            "optional": {
                "regions": ("K2_REGION", {"tooltip": "Chain input: previously defined "
                            "regions."}),
                "bboxes": ("BBOX", {"tooltip": "KJNodes-style BBOX list."}),
                "bounding_box": ("BOUNDING_BOX", {"tooltip": "Standard BOUNDING_BOX input. "
                                 "Takes precedence over bboxes."}),
                "identity_prompt": ("STRING", {"multiline": True, "default": "",
                                    "tooltip": "Face/identity description for this region."}),
            },
        }

    @staticmethod
    def _flatten(source):
        """Zieht eine flache Liste von 4er-Tupeln aus den gängigen BBOX-Formen."""
        if source is None:
            return []
        if isinstance(source, dict):
            for key in ("bbox", "bboxes", "boxes"):
                if key in source:
                    return FVM_K2_RegionFromBBox._flatten(source[key])
            return []
        if isinstance(source, (list, tuple)):
            if len(source) == 4 and all(
                isinstance(v, (int, float)) for v in source
            ):
                return [tuple(float(v) for v in source)]
            collected = []
            for item in source:
                collected.extend(FVM_K2_RegionFromBBox._flatten(item))
            return collected
        if isinstance(source, np.ndarray):
            array = source.reshape(-1, 4) if source.size % 4 == 0 else source
            return [tuple(float(v) for v in row) for row in array]
        if torch.is_tensor(source):
            return FVM_K2_RegionFromBBox._flatten(source.detach().cpu().numpy())
        return []

    def build(self, name, prompt, bbox_index, bbox_format, reference_width,
              reference_height, grow_px, role, priority, regions=None,
              bboxes=None, bounding_box=None, identity_prompt=""):
        candidates = self._flatten(bounding_box) or self._flatten(bboxes)
        if not candidates:
            raise ValueError(
                "No usable bounding box found. Connect a BBOX or BOUNDING_BOX output."
            )
        raw = candidates[min(int(bbox_index), len(candidates) - 1)]
        a, b, c, d = raw

        normalized = max(abs(v) for v in raw) <= 1.5
        if normalized:
            a *= reference_width
            b *= reference_height
            c *= reference_width
            d *= reference_height

        # 'auto' kann xywh und xyxy nicht sicher unterscheiden — (100,100,400,800)
        # ist in beiden Lesarten gültig. Deshalb: xyxy bevorzugen (das Format von
        # BOUNDING_BOX/YOLO/KJ) und nur bei unmöglichem Ergebnis auf xywh fallen.
        if bbox_format == "xywh":
            box = PixelBox.from_xywh(a, b, c, d)
        elif bbox_format == "xyxy":
            box = PixelBox(a, b, c, d)
        elif c > a and d > b:
            box = PixelBox(a, b, c, d)
        else:
            box = PixelBox.from_xywh(a, b, c, d)
        if grow_px:
            box = box.grown(float(grow_px))
        box = box.clipped(int(reference_width), int(reference_height))

        collected = list(regions or [])
        collected.append(
            RegionDefinition(
                region_id=_next_id(collected),
                name=name.strip() or _next_id(collected),
                box=box,
                prompt=prompt,
                identity_prompt=identity_prompt,
                priority=int(priority),
                role=role,
            )
        )
        return (collected,)


class FVM_K2_RegionCombine:
    """Führt zwei Regionsketten zusammen."""

    DESCRIPTION = (
        "Merges two region chains into one. Useful when regions are produced by "
        "separate branches (for example manual boxes plus detector boxes)."
    )
    CATEGORY = CATEGORY
    FUNCTION = "combine"
    RETURN_TYPES = ("K2_REGION",)
    RETURN_NAMES = ("regions",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"regions_a": ("K2_REGION",
                         {"tooltip": "First region chain."})},
            "optional": {
                "regions_b": ("K2_REGION", {"tooltip": "Second chain to merge."}),
                "regions_c": ("K2_REGION", {"tooltip": "Third chain to merge."}),
                "regions_d": ("K2_REGION", {"tooltip": "Fourth chain to merge."}),
            },
        }

    def combine(self, regions_a, regions_b=None, regions_c=None, regions_d=None):
        merged = []
        seen = set()
        for chain in (regions_a, regions_b, regions_c, regions_d):
            for region in chain or []:
                if region.region_id in seen:
                    continue
                seen.add(region.region_id)
                merged.append(region)
        names = [r.name for r in merged]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"Duplicate region names after merge: {sorted(duplicates)}")
        return (merged,)


class FVM_K2_Emphasis:
    """Verstärkt eine exakte Phrase im kompilierten Prompt."""

    DESCRIPTION = (
        "Boosts the spatial binding of one exact phrase.\n\n"
        "The phrase must occur literally in the selected scope (global prompt or a "
        "region prompt). This adds attention bias between those text tokens and the "
        "image tokens of the scope — it does not rewrite Qwen token weights and it is "
        "not the same as (phrase:1.2) weighting."
    )
    CATEGORY = CATEGORY
    FUNCTION = "build"
    RETURN_TYPES = ("K2_EMPHASIS",)
    RETURN_NAMES = ("emphasis",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "phrase": ("STRING", {"default": "",
                                      "tooltip": "Case-sensitive exact phrase."}),
                "scope": ("STRING", {"default": "__global__",
                                     "tooltip": "'__global__' for the global prompt, "
                                     "otherwise the region name or region id."}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                                       "tooltip": "Additional attention bias. 0 disables."}),
                "occurrence": ("INT", {"default": 0, "min": 0, "max": 32,
                                       "tooltip": "Which occurrence to target when the "
                                       "phrase appears multiple times (0 = first)."}),
            },
            "optional": {"emphasis": ("K2_EMPHASIS",
                         {"tooltip": "Chain input: previously defined emphases."})},
        }

    def build(self, phrase, scope, strength, occurrence, emphasis=None):
        collected = list(emphasis or [])
        if phrase.strip():
            collected.append(
                EmphasisRequest(
                    scope_id=scope.strip() or GLOBAL_SCOPE,
                    phrase=phrase,
                    strength=float(strength),
                    occurrence=int(occurrence),
                )
            )
        return (collected,)


class FVM_K2_RegionPreview:
    """Visualisiert Boxen und Attention-Felder als Bild."""

    DESCRIPTION = (
        "Renders the region layout for inspection: hard boxes, the soft attention "
        "field per region, or the token ownership map used by strict isolation.\n\n"
        "Run this before generating — a wrong layout is much cheaper to spot here."
    )
    CATEGORY = CATEGORY
    FUNCTION = "preview"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("preview", "union_mask", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "regions": ("K2_REGION", {"tooltip": "The region chain to visualise."}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 16,
                          "tooltip": "Preview width — use the render size so boxes match."}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 16,
                           "tooltip": "Preview height — use the render size."}),
                "mode": (["boxes", "field", "ownership"], {"default": "field",
                         "tooltip": "boxes: hard rasterized token masks.\n"
                         "field: the soft attention field incl. subject competition.\n"
                         "ownership: which subject owns each image token."}),
                "falloff_pixels": ("FLOAT", {"default": 128.0, "min": 0.0, "max": 2048.0,
                                             "step": 8.0,
                                             "tooltip": "Must match the tuning value to "
                                             "preview the real field."}),
                "subject_competition": ("BOOLEAN", {"default": True,
                                        "tooltip": "Preview overlapping-subject ownership "
                                        "sharing."}),
                "subject_fill": ("BOOLEAN", {"default": True,
                                 "tooltip": "Preview the stronger field towards box edges."}),
            },
        }

    _PALETTE = (
        (0.95, 0.35, 0.35), (0.35, 0.65, 0.95), (0.45, 0.90, 0.50),
        (0.95, 0.80, 0.35), (0.80, 0.45, 0.95), (0.40, 0.90, 0.90),
    )

    def preview(self, regions, width, height, mode, falloff_pixels,
                subject_competition, subject_fill):
        from ...core.k2.prompt import compile_plan
        from ...core.k2.geometry import CanvasGeometry

        active = [r for r in regions if r.enabled and r.description]
        if not active:
            geometry = CanvasGeometry.resolve(width, height)
            empty = torch.zeros(1, geometry.aligned_height, geometry.aligned_width, 3)
            return (empty, torch.zeros(1, height, width), "no active regions")

        plan = compile_plan(
            width, height, "", active,
            falloff_pixels=falloff_pixels,
            subject_competition=subject_competition,
            subject_fill=subject_fill,
            spatial_instructions=False,
        )
        geometry = plan.geometry
        canvas = np.zeros((geometry.token_height, geometry.token_width, 3), dtype=np.float32)

        if mode == "ownership":
            owner = np.zeros(geometry.token_count, dtype=np.int32)
            index = 0
            for region in plan.regions:
                if region.role != "subject":
                    continue
                index += 1
                claim = (region.mask > 0.0) & (owner == 0)
                owner[claim] = index
            for slot in range(1, index + 1):
                colour = self._PALETTE[(slot - 1) % len(self._PALETTE)]
                selection = (owner == slot).reshape(canvas.shape[:2])
                for channel in range(3):
                    canvas[..., channel][selection] = colour[channel]
        else:
            for slot, region in enumerate(plan.regions):
                colour = self._PALETTE[slot % len(self._PALETTE)]
                values = region.mask if mode == "boxes" else region.field
                layer = values.reshape(geometry.token_height, geometry.token_width)
                for channel in range(3):
                    canvas[..., channel] = np.maximum(
                        canvas[..., channel], layer * colour[channel]
                    )

        image = torch.from_numpy(canvas).unsqueeze(0)
        image = torch.nn.functional.interpolate(
            image.permute(0, 3, 1, 2), size=(int(height), int(width)), mode="nearest"
        ).permute(0, 2, 3, 1).contiguous()

        union = torch.from_numpy(
            plan.union_field().reshape(geometry.token_height, geometry.token_width)
        ).float().unsqueeze(0).unsqueeze(0)
        union = torch.nn.functional.interpolate(
            union, size=(int(height), int(width)), mode="bilinear", align_corners=False
        ).squeeze(1)

        info = json.dumps(
            {
                "token_grid": [geometry.token_width, geometry.token_height],
                "regions": [
                    {
                        "name": r.name,
                        "role": r.role,
                        "box": [round(v, 1) for v in r.box.as_tuple()],
                        "hard_tokens": int((r.mask > 0).sum()),
                        "field_energy": round(float(r.field.sum()), 1),
                    }
                    for r in plan.regions
                ],
            },
            indent=2,
        )
        return (image, union, info)


NODE_CLASS_MAPPINGS = {
    "FVM_K2_Region": FVM_K2_Region,
    "FVM_K2_RegionFromBBox": FVM_K2_RegionFromBBox,
    "FVM_K2_RegionCombine": FVM_K2_RegionCombine,
    "FVM_K2_Emphasis": FVM_K2_Emphasis,
    "FVM_K2_RegionPreview": FVM_K2_RegionPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FVM_K2_Region": "K2 Region",
    "FVM_K2_RegionFromBBox": "K2 Region from BBox",
    "FVM_K2_RegionCombine": "K2 Region Combine",
    "FVM_K2_Emphasis": "K2 Prompt Emphasis",
    "FVM_K2_RegionPreview": "K2 Region Preview",
}
