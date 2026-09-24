"""FVM_DatasetPromptList — prompt list for character-LoRA datasets.

Drop-in replacement for Comfyroll's ``CR Prompt List`` with wildcard
support: one prompt per line, ``[close]`` / ``[half]`` / ``[full]`` tags
for shot-size filtering, and ``variations`` to resolve each line several
times with fresh ``__wildcard__`` draws. The editor UI (presets, preview,
wildcard autocomplete) is ``web/js/fvm_dataset_prompts.js``.
"""

from __future__ import annotations

try:
    from ..core import dataset_prompts as dp
except ImportError:  # pragma: no cover
    from core import dataset_prompts as dp

DEFAULT_PRESET = "quick_20"


def _default_text() -> str:
    return dp.read_preset(DEFAULT_PRESET) or (
        "# One prompt per line. Optional tag: [close] / [half] / [full]\n"
        "[close] Straight-on frontal close portrait of the subject, "
        "wearing __dataset/upper__, __dataset/setting__\n"
    )


class FVM_DatasetPromptList:
    CATEGORY = "FVM Tools/Prompt"
    FUNCTION = "build"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("prompt", "caption", "shot", "index", "count", "listing")
    OUTPUT_IS_LIST = (True, True, True, True, False, False)
    OUTPUT_TOOLTIPS = (
        "LIST of finished prompts (prefix + line + suffix, wildcards resolved). "
        "Downstream nodes run once per entry — like CR Prompt List.",
        "LIST of the same prompts WITHOUT prefix/suffix — use as the training "
        "caption (.txt next to the image). Describing clothes/background in the "
        "caption keeps them from being baked into the character.",
        "LIST of shot sizes per prompt: close / half / full (empty if untagged). "
        "Handy for file names or subfolders.",
        "LIST of line numbers (position in the text, 0-based) per prompt.",
        "Total number of prompts produced.",
        "Readable overview of every resolved prompt with counts per shot size.",
    )
    DESCRIPTION = (
        "Dataset prompt list for character LoRA training.\n\n"
        "One prompt per line. Lines starting with # are comments. An optional "
        "leading tag [close], [half] or [full] marks the shot size.\n\n"
        "Wildcards work like in the JB nodes: __name__, __cat/*__, {a|b}, "
        "{2$$a|b|c}, __name^var__ / __^var__.\n\n"
        "variations = how often each line is used. Every copy draws new "
        "wildcards, so 20 lines × 10 variations = 200 different images.\n\n"
        "Toolbar: Presets (load/save lists), Preview (resolve without "
        "generating), Wildcards (edit files), Syntax (reference)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_text": (
                    "STRING",
                    {
                        "default": _default_text(),
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "One prompt per line. '#' lines are comments. "
                        "Optional tag at line start: [close], [half], [full]. "
                        "Type __ for wildcard autocomplete.",
                    },
                ),
                "shot_filter": (
                    list(dp.SHOT_FILTERS),
                    {
                        "default": "all",
                        "tooltip": "Only use lines with this shot tag.\n"
                        "all = every line (incl. untagged)\n"
                        "close = head & shoulders\n"
                        "half = waist-up / seated\n"
                        "full = whole body, head to feet",
                    },
                ),
                "start_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9999,
                        "tooltip": "First line to use (0 = first). Counted AFTER the "
                        "shot filter — the Preview shows the numbers.",
                    },
                ),
                "max_rows": (
                    "INT",
                    {
                        "default": 1000,
                        "min": 1,
                        "max": 9999,
                        "tooltip": "How many lines to take from start_index on.",
                    },
                ),
                "variations": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 200,
                        "tooltip": "How often each line is used. Every copy re-rolls "
                        "all wildcards (clothes, background, light …) — "
                        "the pose/angle of the line stays the same.",
                    },
                ),
                "order": (
                    list(dp.ORDERS),
                    {
                        "default": "rounds",
                        "tooltip": "rounds = all lines once, then all again (safe to "
                        "cancel early — you still have every angle)\n"
                        "per_prompt = all variations of line 1, then line 2 …\n"
                        "shuffle = random order (seeded)",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Seed for all wildcard draws. Same seed + same text "
                        "+ same wildcard files → identical prompts. Change "
                        "it for a completely new set.",
                    },
                ),
            },
            "optional": {
                "prefix": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "dynamicPrompts": False,
                        "tooltip": "Put in front of every prompt, e.g. trigger word or "
                        "'the woman from the reference image'. Wildcards "
                        "allowed; variables bound here (^var) can be "
                        "recalled in the lines with __^var__.",
                    },
                ),
                "suffix": (
                    "STRING",
                    {
                        "default": "__dataset/photo__",
                        "multiline": False,
                        "dynamicPrompts": False,
                        "tooltip": "Appended to every prompt, e.g. photo quality terms. "
                        "Default __dataset/photo__ picks a realistic camera "
                        "description. Not part of the caption output.",
                    },
                ),
                "context_from_prompt_generator": (
                    "DICT",
                    {
                        "tooltip": "Optional variable bag from a Prompt Generator "
                        "(adaptiveprompts `context` output), same as the JB "
                        "Builder. Values bound upstream with ^VAR are "
                        "recallable in every line with __^VAR__.",
                    },
                ),
            },
        }

    def build(
        self,
        prompt_text,
        shot_filter,
        start_index,
        max_rows,
        variations,
        order,
        seed,
        prefix="",
        suffix="",
        context_from_prompt_generator=None,
    ):
        items = dp.expand(
            prompt_text,
            shot_filter=shot_filter,
            start_index=start_index,
            max_rows=max_rows,
            variations=variations,
            order=order,
            seed=seed,
            prefix=prefix,
            suffix=suffix,
            context=context_from_prompt_generator,
        )
        if not items:
            raise ValueError(
                f"Dataset Prompt List: no lines selected (shot_filter={shot_filter}, "
                f"start_index={start_index}). Check the tags or lower start_index."
            )
        return (
            [it["prompt"] for it in items],
            [it["caption"] for it in items],
            [it["shot"] for it in items],
            [it["index"] for it in items],
            len(items),
            dp.listing(items),
        )


def register_routes(server) -> None:
    """Preset + preview endpoints for the editor JS."""
    from aiohttp import web

    routes = server.instance.routes

    @routes.get("/fvmtools/dataset-presets")
    async def _list_presets(request):
        return web.json_response({"presets": dp.list_presets()})

    @routes.get("/fvmtools/dataset-preset")
    async def _get_preset(request):
        text = dp.read_preset(request.rel_url.query.get("name", ""))
        if text is None:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({"text": text})

    @routes.post("/fvmtools/dataset-preset")
    async def _save_preset(request):
        body = await request.json()
        if not dp.write_preset(body.get("name", ""), body.get("text", "")):
            return web.json_response(
                {"error": "invalid name (letters, digits, _ and - only)"}, status=400
            )
        return web.json_response({"success": True})

    @routes.post("/fvmtools/dataset-preview")
    async def _preview(request):
        b = await request.json()
        try:
            items = dp.expand(
                b.get("text", ""),
                shot_filter=b.get("shot_filter", "all"),
                start_index=int(b.get("start_index", 0)),
                max_rows=int(b.get("max_rows", 1000)),
                variations=int(b.get("variations", 1)),
                order=b.get("order", "rounds"),
                seed=int(b.get("seed", 0)),
                prefix=b.get("prefix", ""),
                suffix=b.get("suffix", ""),
            )
        except (TypeError, ValueError) as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"listing": dp.listing(items), "count": len(items)})


NODE_CLASS_MAPPINGS = {"FVM_DatasetPromptList": FVM_DatasetPromptList}
NODE_DISPLAY_NAME_MAPPINGS = {"FVM_DatasetPromptList": "FVM · Dataset Prompt List"}
