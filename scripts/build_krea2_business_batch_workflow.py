"""Build examples/krea2_business_batch_reading.json — a Krea 2 batch workflow.

Every queued run draws a different location set, outfit set and outfit colour
mood, with FVMtools nodes only: the JB Location / Outfit Block multi-select
(hidden ``*_multi`` widgets, one choice per line) draws one choice per seed,
and the block seeds randomize. Run it with a batch count (Run ×N).

The older version chained an Impact ``ImpactWildcardProcessor`` and an rgthree
``Any Switch`` into the combo. That only worked because the links were written
into the JSON: the Any Switch takes on its input's STRING type, so it cannot be
wired to a COMBO by hand. A Primitive with ``control_filter_list`` does not
filter in frontend 1.53 either.

The cielbleu checkpoint already contains the turbo distillation; the turbo LoRA
node stays in the graph but bypassed (stacking it smeared bright floors).

    venv/Scripts/python.exe scripts/build_krea2_business_batch_workflow.py [--out path]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.location_engine import get_available_location_sets  # noqa: E402

LOCATION_CATEGORIES = (
    "reading_pennsylvania_corporate_office", "reading_pennsylvania_downtown_spot",
    "senior_living_community", "hospital_campus", "business_event_venue",
    "business_dining_spot", "suburban_office_park",
)
# awards_gala_evening_with_heels is left out on purpose: it only fits an
# evening event venue. Tick it in Multi-Select Sets when you want it.
OUTFIT_SETS = [f"female/business/{x}_with_heels" for x in (
    "board_meeting_suit", "office_day_business_casual", "senior_community_event",
    "hospital_campus_visit", "health_expo_booth_day", "client_lunch_dress",
    "winter_workday_layers")]
COLOR_MOODS = ["everyday_muted", "cool_muted", "warm_earth", "neutral_basics"]
# Edit List, Edit Overrides, Multi-Select Sets, Multi-Select Moods — buttons
# serialise as null after the Python widgets.
BUTTONS = [None, None, None, None]
BYPASS = 4


def location_sets():
    return [s for s in get_available_location_sets() if s.split("/")[1] in LOCATION_CATEGORIES]


FIRST_LOCATION = "indoor/reading_pennsylvania_corporate_office/open_plan_marketing_department"
FIRST_OUTFIT = OUTFIT_SETS[0]

PERSON = ("A 45-year-old woman, a marketing director for a senior living company, stands "
          "relaxed with a friendly, confident expression, seen full body from head to shoes.")

NOTE = """Krea 2 · Business-Batch (Reading, PA)

Run mit Batch-Anzahl > 1 (z.B. Run ×12): jeder Lauf zieht ein anderes Location-Set, Outfit-Set und eine andere Outfit-Farbstimmung. Der erste Lauf nutzt die gespeicherten Seeds, danach würfeln alle neu.

• Location Block → 'Multi-Select Sets': 42 Sets aus 7 Kategorien angehakt, pro Seed wird eins gezogen. Leuchtet das Dropdown, ist die Mehrfachauswahl aktiv und der Dropdown-Wert wird ignoriert. Sets dazu/weg = Haken setzen; ein Ordner-Haken nimmt alles darunter.
• Outfit Block → 'Multi-Select Sets': die 7 Business-Sets mit Heels. awards_gala_evening_with_heels ist absichtlich nicht drin (passt nur zum Abend-Event) — bei Bedarf anhaken.
• Outfit Block → 'Multi-Select Moods': everyday_muted / cool_muted / warm_earth / neutral_basics.
• Welches Set gezogen wurde, steht im JSON (set_name); die Stimmung am Ende von palette_summary.
• Seeds von Location Block, Outfit Block und Sampler stehen auf randomize. Gleiches Set + gleicher Seed = gleiches Ergebnis, auch einzeln gewählt.
• Turbo-LoRA ist auf Bypass: der cielbleu-Checkpoint enthält Turbo schon, doppelt gab es Flecken auf hellen Böden.
• Person: Text oben links ersetzen (oder LoRA-Trigger). Keine Personenwörter in Location/Outfit — die Sets sind darauf ausgelegt.
• style_preset business_executive: Headwear wird immer gezogen — Frisur ohne Farbe/Länge + 'face turned toward the camera' hält Kopf und Gesicht im Bild.
• Prompt-Länge bleibt < 2400 Zeichen (Krea 2 wird darüber schwarz/verrauscht).

Regeln & Test: .claude/commands/build-location-set.md (Step 2c), /krea2-set-test, scripts/krea2_set_check.py"""


class Graph:
    def __init__(self):
        self.nodes, self.links = [], []
        self._nid, self._lid = 0, 0

    def node(self, type_, pos, size, inputs=(), outputs=(), widgets=None, title=None, props=None):
        self._nid += 1
        n = {"id": self._nid, "type": type_, "pos": list(pos), "size": list(size), "flags": {},
             "order": self._nid - 1, "mode": 0,
             "inputs": [dict(i, link=None) for i in inputs],
             "outputs": [dict(o, links=[]) for o in outputs],
             "properties": dict({"Node name for S&R": type_}, **(props or {}))}
        if widgets is not None:
            n["widgets_values"] = widgets
        if title:
            n["title"] = title
        self.nodes.append(n)
        return n

    def link(self, src, src_slot, dst, dst_input_name):
        self._lid += 1
        slot = next(i for i, inp in enumerate(dst["inputs"]) if inp["name"] == dst_input_name)
        typ = src["outputs"][src_slot]["type"]
        self.links.append([self._lid, src["id"], src_slot, dst["id"], slot, typ])
        src["outputs"][src_slot]["links"].append(self._lid)
        dst["inputs"][slot]["link"] = self._lid

    def dump(self):
        return {"last_node_id": self._nid, "last_link_id": self._lid, "nodes": self.nodes,
                "links": self.links, "groups": [], "config": {},
                "extra": {"ds": {"scale": 0.75, "offset": [40, 40]}}, "version": 0.4}


def sock(name, type_):
    return {"name": name, "type": type_}


def wid(name, type_, optional=False):
    d = {"name": name, "type": type_, "widget": {"name": name}}
    if optional:
        d["shape"] = 7
    return d


def out(name, type_):
    return {"name": name, "type": type_}


def build():
    g = Graph()
    note = g.node("Note", (20, 20), (420, 420), widgets=[NOTE], title="Anleitung")
    note["color"], note["bgcolor"] = "#432", "#653"

    person = g.node("PrimitiveStringMultiline", (20, 470), (420, 160),
                    inputs=[wid("value", "STRING")], outputs=[out("STRING", "STRING")],
                    widgets=[PERSON], title="Person (ersetzen)")

    loc_inputs = [wid("location_set", "COMBO"), wid("seed", "INT")] + [
        wid(f"enable_{e}", "BOOLEAN") for e in ("background", "midground", "architecture_detail",
                                                "props", "foreground_element", "time_of_day", "weather")
    ] + [wid("color_mood", "COMBO"), wid("output_format", "COMBO"),
         wid("overrides", "STRING", True), wid("color_tone", "COMBO", True),
         wid("num_colors", "INT", True), wid("harmony_type", "COMBO", True),
         wid("palette_style", "COMBO", True), wid("vibrancy", "FLOAT", True),
         wid("contrast", "FLOAT", True), wid("warmth", "FLOAT", True),
         wid("location_set_multi", "STRING", True), wid("color_mood_multi", "STRING", True)]
    loc = g.node("FVM_JB_LocationBlock", (470, 20), (420, 660), inputs=loc_inputs,
                 outputs=[out("location_json", "STRING"), out("location_string", "STRING"),
                          out("palette_summary", "STRING")],
                 widgets=[FIRST_LOCATION, 0, "randomize", True, True, True, True, True, True, True,
                          "everyday_muted", "sentences", "", "", 5, "auto", "general",
                          0.5, 0.5, 0.5, "\n".join(location_sets()), ""] + BUTTONS)

    out_inputs = [wid("outfit_set", "COMBO"), wid("seed", "INT"), wid("style_preset", "COMBO"),
                  wid("formality", "FLOAT"), wid("coverage", "FLOAT")] + [
        wid(f"enable_{s}", "BOOLEAN") for s in ("headwear", "top", "bottom", "footwear",
                                                "outerwear", "accessories", "bag")
    ] + [wid("print_probability", "FLOAT"), wid("text_mode", "COMBO"), wid("color_mood", "COMBO"),
         wid("output_format", "COMBO"), wid("overrides", "STRING", True),
         wid("num_colors", "INT", True), wid("harmony_type", "COMBO", True),
         wid("palette_style", "COMBO", True), wid("vibrancy", "FLOAT", True),
         wid("contrast", "FLOAT", True), wid("warmth", "FLOAT", True),
         wid("text_probability", "FLOAT", True),
         wid("outfit_set_multi", "STRING", True), wid("color_mood_multi", "STRING", True)]
    outfit = g.node("FVM_JB_OutfitBlock", (920, 20), (420, 790), inputs=out_inputs,
                    outputs=[out("outfit_json", "STRING"), out("outfit_string", "STRING"),
                             out("palette_summary", "STRING")],
                    widgets=[FIRST_OUTFIT, 0, "randomize", "business_executive", 0.7, 0.5,
                             True, True, True, True, True, True, True,
                             0.15, "off", "everyday_muted", "sentences", "", 5, "auto", "general",
                             0.5, 0.5, 0.5, 0.0, "\n".join(OUTFIT_SETS), "\n".join(COLOR_MOODS)]
                    + BUTTONS)

    stitch = g.node("FVM_JB_Stitcher", (1370, 20), (300, 220),
                    inputs=[wid("title", "STRING"), wid("output_format", "COMBO"),
                            sock("input_1", "STRING"), sock("input_2", "STRING"),
                            sock("input_3", "STRING"), sock("input_4", "STRING")],
                    outputs=[out("raw_json", "STRING"), out("string", "STRING")],
                    widgets=["scene", "sentences"])
    preview = g.node("PreviewAny", (1370, 270), (300, 470),
                     inputs=[sock("source", "*")], outputs=[], title="Prompt (Kontrolle)")

    unet = g.node("UNETLoader", (1700, 20), (360, 82),
                  inputs=[wid("unet_name", "COMBO"), wid("weight_dtype", "COMBO")],
                  outputs=[out("MODEL", "MODEL")],
                  widgets=["krea2\\cielbleuKrea2_v1.safetensors", "default"])
    lora1 = g.node("LoraLoaderModelOnly", (1700, 130), (360, 82),
                   inputs=[sock("model", "MODEL"), wid("lora_name", "COMBO"), wid("strength_model", "FLOAT")],
                   outputs=[out("MODEL", "MODEL")],
                   widgets=["krea2\\helper\\krea2_turbo_lora_rank_64_bf16.safetensors", 0.95],
                   title="Turbo-LoRA (Bypass: im Checkpoint enthalten)")
    lora1["mode"] = BYPASS
    lora2 = g.node("LoraLoaderModelOnly", (1700, 240), (360, 82),
                   inputs=[sock("model", "MODEL"), wid("lora_name", "COMBO"), wid("strength_model", "FLOAT")],
                   outputs=[out("MODEL", "MODEL")],
                   widgets=["krea2\\helper\\inline-skin-lora-krea-2-raw.safetensors", 0.35])
    clip = g.node("CLIPLoader", (1700, 350), (360, 106),
                  inputs=[wid("clip_name", "COMBO"), wid("type", "COMBO"), wid("device", "COMBO", True)],
                  outputs=[out("CLIP", "CLIP")],
                  widgets=["qwen3vl_4b_int8_convrot.safetensors", "krea2", "default"])
    vae = g.node("VAELoader", (1700, 480), (360, 58), inputs=[wid("vae_name", "COMBO")],
                 outputs=[out("VAE", "VAE")], widgets=["Wan2_1_VAE_bf16.safetensors"])

    pos = g.node("CLIPTextEncode", (2090, 20), (300, 100),
                 inputs=[sock("clip", "CLIP"), wid("text", "STRING")],
                 outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[""], title="Positiv")
    neg = g.node("CLIPTextEncode", (2090, 150), (300, 100),
                 inputs=[sock("clip", "CLIP"), wid("text", "STRING")],
                 outputs=[out("CONDITIONING", "CONDITIONING")], widgets=[""], title="Negativ (leer)")
    latent = g.node("EmptyLatentImage", (2090, 280), (300, 106),
                    inputs=[wid("width", "INT"), wid("height", "INT"), wid("batch_size", "INT")],
                    outputs=[out("LATENT", "LATENT")], widgets=[720, 1280, 1])
    ks = g.node("KSampler", (2420, 20), (300, 262),
                inputs=[sock("model", "MODEL"), sock("positive", "CONDITIONING"),
                        sock("negative", "CONDITIONING"), sock("latent_image", "LATENT"),
                        wid("seed", "INT"), wid("steps", "INT"), wid("cfg", "FLOAT"),
                        wid("sampler_name", "COMBO"), wid("scheduler", "COMBO"), wid("denoise", "FLOAT")],
                outputs=[out("LATENT", "LATENT")],
                widgets=[1, "randomize", 10, 1.2, "euler", "beta", 1.0])
    dec = g.node("VAEDecode", (2420, 310), (300, 46),
                 inputs=[sock("samples", "LATENT"), sock("vae", "VAE")], outputs=[out("IMAGE", "IMAGE")])
    save = g.node("SaveImage", (2750, 20), (420, 760),
                  inputs=[sock("images", "IMAGE"), wid("filename_prefix", "STRING")],
                  outputs=[], widgets=["fvm_business_batch/reading"])

    g.link(person, 0, stitch, "input_1")
    g.link(outfit, 0, stitch, "input_2")
    g.link(loc, 0, stitch, "input_3")
    g.link(stitch, 1, preview, "source")
    g.link(stitch, 1, pos, "text")
    g.link(unet, 0, lora1, "model")
    g.link(lora1, 0, lora2, "model")
    g.link(lora2, 0, ks, "model")
    g.link(clip, 0, pos, "clip")
    g.link(clip, 0, neg, "clip")
    g.link(pos, 0, ks, "positive")
    g.link(neg, 0, ks, "negative")
    g.link(latent, 0, ks, "latent_image")
    g.link(ks, 0, dec, "samples")
    g.link(vae, 0, dec, "vae")
    g.link(dec, 0, save, "images")
    return g.dump()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "examples", "krea2_business_batch_reading.json"))
    args = ap.parse_args()
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(build(), fh, indent=1, ensure_ascii=False)
    print(args.out)


if __name__ == "__main__":
    main()
