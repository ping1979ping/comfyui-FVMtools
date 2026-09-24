"""Build the 'SAM3 sort + replace' batch workflow (UI format).

Walks a folder one picture per run. SAM3 native looks for a free-text term;
pictures with a hit get the term inpainted into something else by the Person
Detailer, the edited version is saved and the original moved into a target
folder. Pictures without a hit stay where they are.

    venv/Scripts/python.exe scripts/build_sam3_sort_workflow.py [out.json]

widgets_values follow the order the frontend builds, including the separator
and row widgets FVMtools' JS inserts (see web/js/inpaint_options.js,
person_detailer.js, fvm_batch_loader.js).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "examples", "batch_sam3_sort_replace.json")

nodes, links = [], []
_next_link = [1]


def node(nid, ntype, pos, size, widgets, title=None, inputs=(), outputs=(), color=None):
    entry = {
        "id": nid, "type": ntype, "pos": list(pos), "size": list(size), "flags": {},
        "order": nid, "mode": 0, "properties": {"Node name for S&R": ntype},
        "widgets_values": widgets,
        "inputs": [dict(i) for i in inputs],
        "outputs": [{"name": n, "type": t, "links": []} for n, t in outputs],
    }
    if title:
        entry["title"] = title
    if color:
        entry["color"], entry["bgcolor"] = color
    nodes.append(entry)
    return entry


def link(src, src_slot, dst, dst_input):
    """Connect src.outputs[src_slot] to the input named dst_input on dst."""
    lid = _next_link[0]
    _next_link[0] += 1
    out = src["outputs"][src_slot]
    out["links"].append(lid)
    for i, inp in enumerate(dst["inputs"]):
        if inp["name"] == dst_input:
            inp["link"] = lid
            links.append([lid, src["id"], src_slot, dst["id"], i, out["type"]])
            return
    raise KeyError(f"{dst['type']} has no input {dst_input}")


def sock(name, typ, widget=False):
    entry = {"name": name, "type": typ, "link": None}
    if widget:
        entry["widget"] = {"name": name}
    return entry


INPUT_COLOR = ("#232", "#353")
S = "STRING"

# ── Eingaben (die vier Textfelder) ──────────────────────────────────────────
src_dir = node(1, "PrimitiveString", (0, 0), (420, 58), ["D:/pfad/zum/ordner"],
               title="Quellordner", outputs=[("STRING", S)], color=INPUT_COLOR)
search = node(2, "PrimitiveString", (0, 110), (420, 58), ["tattoo"],
              title="Suchbegriff (SAM3)", outputs=[("STRING", S)], color=INPUT_COLOR)
replace = node(3, "PrimitiveString", (0, 220), (420, 58),
               ["natural skin"],
               title="Ersetzen durch (Prompt)", outputs=[("STRING", S)], color=INPUT_COLOR)
target = node(4, "PrimitiveString", (0, 330), (420, 58), ["treffer"],
              title="Zielordner (relativ zum Quellordner oder absolut)",
              outputs=[("STRING", S)], color=INPUT_COLOR)
edited_dir = node(5, "StringConcatenate", (0, 440), (420, 150), ["", "ersetzt", "/"],
                  title="Unterordner für die bearbeiteten Bilder",
                  inputs=[sock("string_a", S, widget=True)], outputs=[("STRING", S)])

# ── Laden ───────────────────────────────────────────────────────────────────
loader = node(10, "FVM_BatchLoadImage", (480, 0), (330, 420),
              ["", "keep", "reject", "stop", "name", False, False, "sam3_sort",
               "", "reset", "refresh"],
              inputs=[sock("directory", S, widget=True)],
              outputs=[("image", "IMAGE"), ("mask", "MASK"), ("source_path", S),
                       ("pass_dir", S), ("fail_dir", S), ("filename", S),
                       ("index", "INT"), ("total", "INT"), ("progress", S)])

sam3_ckpt = node(11, "CheckpointLoaderSimple", (480, 480), (330, 100),
                 ["sam3\\sam3.1_multiplex_fp16.safetensors"],
                 title="SAM3.1 (Modell + Text-Encoder)",
                 outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")])

# ── Erkennen ────────────────────────────────────────────────────────────────
selector = node(20, "PersonSelectorSAM3Native", (880, 0), (360, 420),
                [0.4, "640", "70/15/15", "none", "", 0.3, "", 0.5, 2, "front_last"],
                title="SAM3 Native — Suche (aux_prompt → aux_data)",
                inputs=[sock("sam3_model", "MODEL"), sock("sam3_clip", "CLIP"),
                        sock("current_image", "IMAGE"),
                        sock("aux_prompt", S, widget=True)],
                outputs=[("person_data", "PERSON_DATA"), ("face_masks", "MASK"),
                         ("head_masks", "MASK"), ("body_masks", "MASK"),
                         ("aux_masks", "MASK"), ("preview", "IMAGE"),
                         ("similarities", S), ("matches", S), ("matched_count", "INT"),
                         ("face_count", "INT"), ("report", S),
                         ("aux_data", "PERSON_DATA")])
preview_sam = node(21, "PreviewImage", (880, 480), (360, 300), [],
                   title="Vorschau: gefunden", inputs=[sock("images", "IMAGE")])

# ── Ersetzen (Krea2) ────────────────────────────────────────────────────────
unet = node(30, "UNETLoader", (1300, 520), (330, 82),
            ["krea2\\krea2_raw_int8_convrot.safetensors", "default"],
            outputs=[("MODEL", "MODEL")])
clip = node(31, "CLIPLoader", (1300, 640), (330, 106),
            ["qwen3vl_4b_int8_convrot.safetensors", "krea2", "default"],
            outputs=[("CLIP", "CLIP")])
vae = node(32, "VAELoader", (1300, 780), (330, 58), ["Wan2_1_VAE_bf16.safetensors"],
           outputs=[("VAE", "VAE")])
diffdiff = node(33, "DifferentialDiffusion", (1300, 880), (330, 58), [1],
                inputs=[sock("model", "MODEL")], outputs=[("MODEL", "MODEL")])

ref_rows = []
for i in range(1, 6):
    ref_rows += [i, "head", 1, False]
inpaint = node(34, "InpaintOptions", (1300, 0), (330, 480),
               [1.0, "blurry, artifacts", 0.0, "linear", "both", 0.35, True, 1.2, 32,
                ".7|.4", "5|3", "── References ──", "header", None, None, None, None, None]
               + ref_rows + [None, "aux", 2, False],     # 2 rounds -> progression applies
               title="Inpaint Options (Generic = aux)",
               outputs=[("inpaint_options", "INPAINT_OPTIONS")])

refs = []
for i in range(1, 6):
    refs += [False, "None", 1.0, ""]
detailer = node(40, "PersonDetailer", (1700, 0), (380, 900),
                ["── Sampler ──", 0, "fixed", 10, 0.75, "er_sde", "beta",
                 "── Detail Daemon ──", False, 0.2, True,
                 "── Inpaint ──", 32, 16, 1024, 1024,
                 "── References ──"] + refs + [True, True, "None", 1.0, ""],
                title="Person Detailer — ersetzt die Treffer",
                inputs=[sock("images", "IMAGE"), sock("person_data", "PERSON_DATA"),
                        sock("model", "MODEL"), sock("clip", "CLIP"), sock("vae", "VAE"),
                        sock("inpaint_options", "INPAINT_OPTIONS"),
                        sock("generic_prompt", S, widget=True)],
                outputs=[("images", "IMAGE"), ("refined", "IMAGE"),
                         ("refined_references", "IMAGE"), ("refined_generic", "IMAGE")])
preview_out = node(41, "PreviewImage", (1700, 960), (380, 300), [],
                   title="Vorschau: ersetzt", inputs=[sock("images", "IMAGE")])

# ── Einsortieren ────────────────────────────────────────────────────────────
subdirs = [""] + [f"slot_{i}" for i in range(2, 9)]
saver = node(50, "FVM_BatchSaveMulti", (2140, 0), (360, 480),
             [1, "all_matches", "", "", "move", "on_match", "", "keep", 95, 0.001, False]
             + subdirs,
             title="Batch Save Multi — nur Treffer verschieben",
             inputs=[sock("source_path", S), sock("image_1", "IMAGE"),
                     sock("gate_1", "*"), sock("subdir_1", S, widget=True),
                     sock("original_dir", S, widget=True)],
             outputs=[("saved_paths", S), ("matched", S), ("matched_count", "INT"),
                      ("report", S)])

# ── Verbindungen ────────────────────────────────────────────────────────────
link(src_dir, 0, loader, "directory")
link(target, 0, edited_dir, "string_a")
link(sam3_ckpt, 0, selector, "sam3_model")
link(sam3_ckpt, 1, selector, "sam3_clip")
link(loader, 0, selector, "current_image")
link(search, 0, selector, "aux_prompt")
link(selector, 5, preview_sam, "images")
link(unet, 0, diffdiff, "model")
link(loader, 0, detailer, "images")
link(selector, 11, detailer, "person_data")   # aux_data
link(diffdiff, 0, detailer, "model")
link(clip, 0, detailer, "clip")
link(vae, 0, detailer, "vae")
link(inpaint, 0, detailer, "inpaint_options")
link(replace, 0, detailer, "generic_prompt")
link(detailer, 0, preview_out, "images")
link(loader, 2, saver, "source_path")
link(detailer, 0, saver, "image_1")
link(selector, 11, saver, "gate_1")          # aux_data: fires only on hits
link(edited_dir, 0, saver, "subdir_1")
link(target, 0, saver, "original_dir")

workflow = {
    "id": "fvm-batch-sam3-sort-replace",
    "revision": 0,
    "last_node_id": max(n["id"] for n in nodes),
    "last_link_id": _next_link[0] - 1,
    "nodes": nodes,
    "links": links,
    "groups": [
        {"id": 1, "title": "Eingaben", "bounding": [-20, -60, 460, 680],
         "color": "#3f5f3f", "flags": {}},
    ],
    "config": {},
    "extra": {"ds": {"scale": 0.6, "offset": [80, 120]}},
    "version": 0.4,
}

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(workflow, handle, indent=1, ensure_ascii=False)
    print(f"{len(nodes)} nodes, {len(links)} links -> {os.path.abspath(out)}")
