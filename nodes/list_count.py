"""List / Batch Count — how many items does whatever is plugged in carry.

ComfyUI has two different "many": a *list* runs the downstream node once per
element (``OUTPUT_IS_LIST``), a *batch* is one tensor with a leading batch
dimension. With ``INPUT_IS_LIST`` this node sees the list wrapper itself, so it
can tell the two apart: a real list arrives with ``len > 1``, a batch arrives as
one element whose tensor carries the count.

A CONDITIONING is itself a Python list of ``[tensor, dict]`` pairs, so a single
conditioning counts its entries (Combine / Concat results), while a list of
conditionings counts the list.
"""


class _AnyType(str):
    """Matches every socket type."""

    def __ne__(self, other):
        return False


ANY = _AnyType("*")


def _is_conditioning(item):
    return (
        isinstance(item, list)
        and len(item) > 0
        and all(
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and isinstance(entry[1], dict)
            for entry in item
        )
    )


def describe_item(item):
    """``(type_label, inner_count, detail)`` for one list element."""
    try:
        import torch
    except ImportError:  # pragma: no cover - ComfyUI always has torch
        torch = None

    if torch is not None and isinstance(item, torch.Tensor):
        shape = list(item.shape)
        if item.ndim == 4 and shape[-1] in (1, 3, 4):
            return "IMAGE", shape[0], f"{shape[0]} x {shape[1]}x{shape[2]}x{shape[3]}"
        if item.ndim == 3:
            return "MASK", shape[0], f"{shape[0]} x {shape[1]}x{shape[2]}"
        return "TENSOR", shape[0] if shape else 1, f"shape {shape}"

    if isinstance(item, dict):
        samples = item.get("samples")
        if samples is not None and hasattr(samples, "shape"):
            shape = list(samples.shape)
            return (
                "LATENT",
                shape[0],
                f"{shape[0]} x {'x'.join(str(d) for d in shape[1:])}",
            )
        waveform = item.get("waveform")
        if waveform is not None and hasattr(waveform, "shape"):
            shape = list(waveform.shape)
            return (
                "AUDIO",
                shape[0],
                f"{shape[0]} x {'x'.join(str(d) for d in shape[1:])}",
            )
        return "DICT", 1, f"{len(item)} keys"

    if _is_conditioning(item):
        return (
            "CONDITIONING",
            len(item),
            f"{len(item)} entr{'y' if len(item) == 1 else 'ies'}",
        )

    if isinstance(item, (list, tuple)):
        return type(item).__name__.upper(), len(item), f"{len(item)} elements"
    if isinstance(item, str):
        return "STRING", 1, f"{len(item)} chars"
    if isinstance(item, bool):
        return "BOOLEAN", 1, str(item)
    if isinstance(item, int):
        return "INT", 1, str(item)
    if isinstance(item, float):
        return "FLOAT", 1, str(item)
    if item is None:
        return "NONE", 0, "None"
    return type(item).__name__, 1, type(item).__name__


def analyse(items):
    """Return ``(count, kind, text)``.

    ``kind`` is LIST (several elements, one run per element), BATCH (one element
    carrying several items), SINGLE (exactly one item) or EMPTY.
    """
    if items is None:
        return 0, "EMPTY", "nothing connected"
    if not isinstance(items, list):
        items = [items]
    if len(items) == 0:
        return 0, "EMPTY", "EMPTY list"

    described = [describe_item(item) for item in items]
    types = sorted({label for label, _, _ in described})
    type_text = types[0] if len(types) == 1 else "mixed: " + ", ".join(types)

    if len(items) > 1:
        inner = {count for _, count, _ in described}
        detail = ""
        if inner != {1}:
            detail = (
                f", {described[0][2]} each"
                if len(inner) == 1
                else ", sizes " + "/".join(str(c) for _, c, _ in described[:6])
            )
        return len(items), "LIST", f"LIST of {type_text} ({len(items)} items{detail})"

    label, count, detail = described[0]
    if count > 1:
        return count, "BATCH", f"BATCH {label} ({detail})"
    return max(count, 0), "SINGLE", f"SINGLE {label} ({detail})"


class FVM_ListCount:
    """Outputs the list or batch count of any input and shows what it saw."""

    CATEGORY = "FVM Tools/Utils"
    FUNCTION = "doit"
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("count", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"value": (ANY, {"forceInput": True})}}

    @classmethod
    def VALIDATE_INPUTS(cls, input_types=None, **kwargs):
        return True

    def doit(self, value=None):
        count, kind, text = analyse(value)
        info = f"{text} -> count {count}"
        return {
            "ui": {"fvm_list_count": [{"kind": text, "count": str(count)}]},
            "result": (count, info),
        }


NODE_CLASS_MAPPINGS = {"FVM_ListCount": FVM_ListCount}
NODE_DISPLAY_NAME_MAPPINGS = {"FVM_ListCount": "List / Batch Count"}
