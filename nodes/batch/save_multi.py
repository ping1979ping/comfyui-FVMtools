"""FVM_BatchSaveMulti — send one source picture to several folders in one go.

Two jobs, one node:

* **Keep variants apart** — the original, the refined version, the inpainted
  one: each slot gets its own image input and its own folder.
* **Sort by features** — hang a detector on a slot's ``gate`` (a SAM3 mask, a
  face count, a boolean) and the picture only lands in that slot's folder when
  the gate fires. Pictures no gate wanted go to ``fallback_dir``.

A slot without an image input copies the original file byte for byte, so pure
sorting never re-encodes anything. The original itself is handled once, at the
end, and only when every slot was written — a failed write leaves it in place so
nothing is lost.
"""

import os
import shutil

try:
    from ...core.batch_state import IMAGE_EXTENSIONS
except ImportError:
    from core.batch_state import IMAGE_EXTENSIONS

try:
    from .saver import tensor_to_pil, unique_path
except ImportError:
    from nodes.batch.saver import tensor_to_pil, unique_path


MAX_SLOTS = 8


class _AnyType(str):
    """Matches every socket type, so a gate accepts masks, counts and booleans."""

    def __ne__(self, other):
        return False


ANY = _AnyType("*")


def gate_state(value, mask_min_area=0.001):
    """Whether a gate fires. ``None`` (unconnected) means "not asked".

    BOOLEAN as is, numbers when above zero, masks when at least
    ``mask_min_area`` of the pixels are set, strings unless empty or a spelled
    out "false". A PERSON_DATA fires when it carries any aux region — assigned
    to a person or unassigned — so a Person Selector's aux detection can gate
    directly. Anything else counts by Python truthiness.
    """
    if value is None:
        return None
    if isinstance(value, dict) and ("num_references" in value or "aux_masks" in value
                                    or "aux_unassigned_masks" in value):
        # PERSON_DATA. Only the aux channel counts; one without aux data (the
        # selector ran with aux off) has found nothing — it must not fall
        # through to "a non-empty dict is true" and pass every picture.
        regions = list(value.get("aux_masks") or [])
        if value.get("aux_unassigned_masks") is not None:
            regions.append(value["aux_unassigned_masks"])
        return any(gate_state(region, mask_min_area) for region in regions)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "none")
    if hasattr(value, "numel") and hasattr(value, "float"):   # torch tensor
        if value.numel() == 0:
            return False
        if value.ndim in (2, 3):                              # MASK [H,W] / [B,H,W]
            return float((value.float() > 0.5).float().mean()) >= float(mask_min_area)
        return bool((value != 0).any())
    if isinstance(value, (list, tuple)):
        return any(gate_state(item, mask_min_area) for item in value) if value else False
    return bool(value)


def resolve_dir(folder, base):
    """``folder`` as given when absolute, else below ``base``."""
    folder = (folder or "").strip().strip('"')
    if not folder:
        return ""
    if os.path.isabs(folder):
        return folder
    return os.path.join(base, folder) if base else ""


class FVM_BatchSaveMulti:
    """Write up to eight variants of one picture into their own folders."""

    CATEGORY = "FVM Tools/Batch"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "source_path": ("STRING", {
                "default": "", "forceInput": True,
                "tooltip": "The loader's source_path. Needed for slots without "
                           "an image, for the original handling and as the "
                           "base of relative folders.",
            }),
        }
        for i in range(1, MAX_SLOTS + 1):
            optional[f"image_{i}"] = ("IMAGE", {
                "tooltip": f"Slot {i}: picture to save. Unconnected = copy the "
                           "original file untouched.",
            })
            optional[f"gate_{i}"] = (ANY, {
                "tooltip": f"Slot {i}: condition. BOOLEAN, a count (> 0), or a "
                           "MASK (fires above mask_min_area). Unconnected = "
                           "always write.",
            })
            optional[f"subdir_{i}"] = ("STRING", {
                "default": "refined" if i == 1 else f"slot_{i}",
                "tooltip": f"Slot {i}: target folder. Relative to base_dir (or "
                           "the source folder), or an absolute path.",
            })

        return {
            "required": {
                "slots": ("INT", {
                    "default": 2, "min": 1, "max": MAX_SLOTS,
                    "tooltip": "How many slots are in use.",
                }),
                "route": (["all_matches", "first_match"], {
                    "default": "all_matches",
                    "tooltip": "all_matches: every slot whose gate fires gets the "
                               "picture. first_match: only the first gated slot "
                               "that fires. Slots without a gate always write.",
                }),
                "base_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Where relative folders live. Empty = the folder "
                               "the source picture came from.",
                }),
                "fallback_dir": ("STRING", {
                    "default": "unsorted",
                    "tooltip": "Gets a copy of the original when gates are wired "
                               "but none fired. Empty = do nothing.",
                }),
                "original": (["keep", "move", "copy", "delete"], {
                    "default": "keep",
                    "tooltip": "What happens to the source file once every slot "
                               "is written. keep: stays. move/copy: into "
                               "original_dir. delete: removed for good (no "
                               "recycle bin).",
                }),
                "original_when": (["always", "on_match"], {
                    "default": "always",
                    "tooltip": "always: handle the original on every run. "
                               "on_match: only when at least one gate fired — "
                               "pictures without a hit stay where they are.",
                }),
                "original_dir": ("STRING", {
                    "default": "done",
                    "tooltip": "Target for original = move/copy.",
                }),
                "format": (["keep", "png", "jpg", "webp"], {
                    "default": "keep",
                    "tooltip": "Encoding for slots with an image input. keep: the "
                               "source file's extension, PNG if unknown.",
                }),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100,
                                    "tooltip": "JPEG/WebP quality."}),
                "mask_min_area": ("FLOAT", {
                    "default": 0.001, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Share of set pixels a MASK gate needs to fire. "
                               "0.001 = 0.1 % of the picture.",
                }),
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Off: a colliding name gets _001, _002, …",
                }),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("saved_paths", "matched", "matched_count", "report")
    FUNCTION = "execute"
    OUTPUT_NODE = True
    DESCRIPTION = ("Saves up to 8 variants of one picture into their own folders, "
                   "optionally gated by detections, then keeps, moves or deletes "
                   "the original.")

    def execute(self, slots, route, base_dir, fallback_dir, original, original_dir,
                format, quality, mask_min_area, overwrite, source_path="",
                original_when="always", **kwargs):
        source_path = (source_path or "").strip().strip('"')
        has_source = bool(source_path) and os.path.isfile(source_path)
        base = (base_dir or "").strip().strip('"') or (
            os.path.dirname(source_path) if source_path else "")
        stem, source_ext = os.path.splitext(os.path.basename(source_path) or "image.png")

        lines, written, matched, errors = [], [], [], []
        any_gated, first_taken = False, False

        for i in range(1, int(slots) + 1):
            subdir = kwargs.get(f"subdir_{i}", "")
            image = kwargs.get(f"image_{i}")
            state = gate_state(kwargs.get(f"gate_{i}"), mask_min_area)

            if state is not None:
                any_gated = True
                if not state:
                    lines.append(f"slot {i}: gate off → skipped")
                    continue
                if route == "first_match" and first_taken:
                    lines.append(f"slot {i}: gate on, but first_match already taken")
                    continue
                first_taken = True
                matched.append((subdir or f"slot_{i}").strip())

            target_dir = resolve_dir(subdir, base)
            if not target_dir:
                errors.append(f"slot {i}: no folder (empty subdir or no base)")
                continue
            try:
                if image is not None:
                    target = self._encode(image, target_dir, stem, source_ext,
                                          format, quality, overwrite)
                elif has_source:
                    target = self._copy(source_path, target_dir, overwrite)
                else:
                    errors.append(f"slot {i}: neither image nor source file")
                    continue
            except (OSError, ValueError) as error:
                errors.append(f"slot {i}: {error}")
                continue
            written.append(target)
            lines.append(f"slot {i}: → {target}")

        if any_gated and not matched and (fallback_dir or "").strip():
            target_dir = resolve_dir(fallback_dir, base)
            if has_source and target_dir:
                try:
                    target = self._copy(source_path, target_dir, overwrite)
                    written.append(target)
                    lines.append(f"fallback: → {target}")
                except OSError as error:
                    errors.append(f"fallback: {error}")
            else:
                errors.append("fallback: needs source_path")

        if original_when == "on_match" and not matched:
            lines.append("original: no gate fired — left in place")
        else:
            lines.append(self._handle_original(original, original_dir, base, source_path,
                                               has_source, errors, overwrite))

        report = "\n".join(lines + [f"ERROR {e}" for e in errors])
        for line in report.splitlines():
            print(f"[FVM Save Multi] {line}")
        summary = f"{len(written)} written" + (f", {len(errors)} errors" if errors else "")
        if matched:
            summary += "  · matched: " + ", ".join(matched)
        return {"ui": {"text": [summary + "\n" + report]},
                "result": ("\n".join(written), ", ".join(matched), len(matched), report)}

    def _handle_original(self, original, original_dir, base, source_path, has_source,
                         errors, overwrite):
        if original == "keep":
            return "original: kept"
        if not has_source:
            return "original: no source file — nothing to do"
        if errors:
            # Something did not land; do not touch the only intact copy.
            return f"original: left in place because of {len(errors)} error(s)"
        try:
            if original == "delete":
                os.remove(source_path)
                return f"original: deleted {source_path}"
            target_dir = resolve_dir(original_dir, base)
            if not target_dir:
                errors.append("original: no original_dir")
                return "original: kept"
            if original == "move":
                os.makedirs(target_dir, exist_ok=True)
                target = unique_path(os.path.join(target_dir, os.path.basename(source_path)),
                                     overwrite)
                shutil.move(source_path, target)
                return f"original: moved → {target}"
            return f"original: copied → {self._copy(source_path, target_dir, overwrite)}"
        except OSError as error:
            errors.append(f"original: {error}")
            return "original: kept"

    @staticmethod
    def _copy(source_path, directory, overwrite):
        os.makedirs(directory, exist_ok=True)
        target = unique_path(os.path.join(directory, os.path.basename(source_path)),
                             overwrite)
        if os.path.abspath(target) != os.path.abspath(source_path):
            shutil.copy2(source_path, target)
        return target

    @staticmethod
    def _encode(image, directory, stem, source_ext, format, quality, overwrite):
        if format == "keep":
            extension = source_ext if source_ext.lower() in IMAGE_EXTENSIONS else ".png"
        else:
            extension = "." + format
        os.makedirs(directory, exist_ok=True)
        target = unique_path(os.path.join(directory, f"{stem}{extension}"), overwrite)
        pil = tensor_to_pil(image)
        lowered = extension.lower()
        if lowered in (".jpg", ".jpeg"):
            pil.save(target, quality=int(quality), subsampling=0)
        elif lowered == ".webp":
            pil.save(target, quality=int(quality))
        else:
            pil.save(target)
        if not os.path.getsize(target):
            raise OSError(f"empty file written: {target}")
        return target
