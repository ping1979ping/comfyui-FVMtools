"""Generate the Batch Tools example workflow.

Hand-writing ComfyUI's UI format is error-prone — link ids have to line up with
slot indices in three places at once — so the example is generated from a small
graph description instead. Re-run after changing a node's inputs or outputs::

    python scripts/build_batch_workflow.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "examples", "batch_reality_sort.json")

SOURCE_DIR = r"D:\\AI\\ComfyUI\\ComfyUI\\output\\2025-12-30\\qe-lingerie-solo3\\test"
REFERENCE = "FACE1510060.jpg"

# id, type, title, pos, size, widget values
NODES = [
    (1, "FVM_BatchLoadImage", "1 · Walk the folder", [40, 120], [400, 300],
     [SOURCE_DIR, "keep", "reject", "stop", "name", False, True, "default"]),
    (2, "FVM_RealityCheck", "2 · Is it physically possible?", [500, 60], [420, 560],
     ["http://localhost:1234/v1", "qwen3-8b-vl-instruct-abliterated", 1, 0.5, 1,
      0.1, "median", True, True, True, False, False, False, 1024, 180, 0,
      "fixed", "pass"]),
    (3, "LoadImage", "Reference face (the intended person)", [40, 470], [320, 320],
     [REFERENCE, "image"]),
    # threshold 0.75: measured against the four real reference portraits, the
    # right person scored 0.787–0.855 and the worst-rendered faces 0.66–0.69, so
    # the cut sits in the gap. Widget order is
    # threshold, aggregation, mask_mode, mask_fill_holes, mask_blur, det_size.
    (4, "PersonSelector", "3 · Is it the right person?", [500, 660], [400, 260],
     [0.75, "max", "none", True, 0, "640"]),
    (5, "FVM_BatchRouter", "4 · Decide the folder", [980, 200], [380, 300],
     ["all", "reality", "identity", "extra", 1, 1, False]),
    (6, "FVM_BatchSaveImage", "5 · File it away", [1420, 200], [380, 260],
     ["move", "", "keep", 95, False, True]),
    (7, "PreviewImage", "Current picture", [980, 560], [380, 320], []),
    (8, "PreviewAny", "Verdict", [1420, 520], [380, 240], []),
]

# (from_node, from_slot, to_node, to_slot, type)
LINKS = [
    (1, 0, 2, 0, "IMAGE"),          # image      → reality check
    (1, 0, 4, 0, "IMAGE"),          # image      → person selector (current)
    (3, 0, 4, 1, "IMAGE"),          # reference  → person selector (references)
    (1, 3, 5, 0, "STRING"),         # pass_dir   → router
    (1, 4, 5, 1, "STRING"),         # fail_dir   → router
    (2, 0, 5, 2, "BOOLEAN"),        # passed     → router gate_a
    (2, 3, 5, 5, "STRING"),         # report     → router report_in
    (4, 1, 5, 3, "BOOLEAN"),        # match      → router gate_b
    (4, 4, 5, 4, "INT"),            # face_count → router count
    (5, 0, 6, 0, "STRING"),         # target_dir → save
    (1, 0, 6, 1, "IMAGE"),          # image      → save
    (1, 2, 6, 2, "STRING"),         # source_path→ save
    (5, 3, 6, 3, "STRING"),         # report     → save sidecar
    (1, 0, 7, 0, "IMAGE"),          # image      → preview
    (5, 3, 8, 0, "STRING"),         # report     → preview
]

# Which slots are inputs on the receiving node, in declaration order.
INPUT_SLOTS = {
    2: [("image", "IMAGE")],
    4: [("current_image", "IMAGE"), ("reference_images", "IMAGE")],
    5: [("pass_dir", "STRING"), ("fail_dir", "STRING"), ("gate_a", "BOOLEAN"),
        ("gate_b", "BOOLEAN"), ("count", "INT"), ("report_in", "STRING")],
    6: [("directory", "STRING"), ("image", "IMAGE"), ("source_path", "STRING"),
        ("report", "STRING")],
    7: [("images", "IMAGE")],
    8: [("source", "*")],
}

OUTPUT_SLOTS = {
    1: [("image", "IMAGE"), ("mask", "MASK"), ("source_path", "STRING"),
        ("pass_dir", "STRING"), ("fail_dir", "STRING"), ("filename", "STRING"),
        ("index", "INT"), ("total", "INT"), ("progress", "STRING")],
    2: [("passed", "BOOLEAN"), ("failed", "BOOLEAN"), ("score", "FLOAT"),
        ("report", "STRING"), ("json", "STRING"), ("image", "IMAGE")],
    3: [("IMAGE", "IMAGE"), ("MASK", "MASK")],
    4: [("similarity", "FLOAT"), ("match", "BOOLEAN"), ("mask", "MASK"),
        ("best_reference", "IMAGE"), ("face_count", "INT"),
        ("matched_face_index", "INT"), ("report", "STRING")],
    5: [("target_dir", "STRING"), ("passed", "BOOLEAN"), ("failed", "BOOLEAN"),
        ("report", "STRING")],
    6: [("image", "IMAGE"), ("saved_path", "STRING"), ("written", "BOOLEAN")],
    7: [],
    8: [],
}

NOTE = (
    "Batch Tools — sort a folder of renders\n"
    "\n"
    "Queue this with a run count equal to the number of pictures (or just a big\n"
    "number — node 1 stops the run when the folder is done).\n"
    "\n"
    "Each run: node 1 hands out the next picture, node 2 asks the vision model\n"
    "whether it is physically possible, node 4 checks it is the right person and\n"
    "counts the faces, node 5 picks the folder, node 6 moves the file there.\n"
    "\n"
    "Set-up\n"
    "  • node 1 · directory: the folder to work through\n"
    "  • node 2 · needs LM Studio running with a VISION model loaded\n"
    "  • node 3 · a reference portrait of the person the batch is about\n"
    "  • node 6 · mode 'move' tidies the source folder; 'copy' leaves it intact\n"
    "\n"
    "Gates in node 5 (all must pass)\n"
    "  gate_a  reality check\n"
    "  gate_b  face matches the reference (PersonSelector threshold)\n"
    "  count   exactly one face (count_min 1, count_max 1) — this alone catches\n"
    "          the duplicated-body renders\n"
    "\n"
    "Measured on the 17-image acceptance set: 8 broken pictures rejected, 8 good\n"
    "ones kept, no false alarms, twice with different seeds.\n"
    "Re-run it with tests/live/reality/pipeline_run.py --acceptance"
)


def build():
    nodes, links = [], []
    link_id = 1
    inputs_by_node = {nid: [] for nid, *_ in NODES}
    outputs_by_node = {nid: {} for nid, *_ in NODES}

    for from_node, from_slot, to_node, to_slot, kind in LINKS:
        links.append([link_id, from_node, from_slot, to_node, to_slot, kind])
        name, type_name = INPUT_SLOTS[to_node][to_slot]
        inputs_by_node[to_node].append(
            {"name": name, "type": type_name, "link": link_id})
        outputs_by_node[from_node].setdefault(from_slot, []).append(link_id)
        link_id += 1

    for node_id, node_type, title, pos, size, widgets in NODES:
        outputs = []
        for index, (name, type_name) in enumerate(OUTPUT_SLOTS[node_id]):
            outputs.append({
                "name": name, "type": type_name,
                "links": outputs_by_node[node_id].get(index, []) or None,
                "slot_index": index,
            })
        node = {
            "id": node_id, "type": node_type, "title": title, "pos": pos,
            "size": size, "flags": {}, "order": node_id - 1, "mode": 0,
            "inputs": inputs_by_node[node_id], "outputs": outputs,
            "properties": {"Node name for S&R": node_type},
            "widgets_values": widgets,
        }
        nodes.append(node)

    nodes.append({
        "id": 99, "type": "Note", "title": "Read me first",
        "pos": [40, 830], "size": [520, 420], "flags": {}, "order": 99, "mode": 0,
        "inputs": [], "outputs": [], "properties": {"text": ""},
        "widgets_values": [NOTE], "color": "#432", "bgcolor": "#653",
    })

    return {
        "id": "fvm-batch-reality-sort", "revision": 0, "last_node_id": 99,
        "last_link_id": link_id - 1, "nodes": nodes, "links": links,
        "groups": [
            {"id": 1, "title": "Walk the folder", "bounding": [20, 40, 450, 800],
             "color": "#3f789e", "font_size": 24, "flags": {}},
            {"id": 2, "title": "Judge", "bounding": [485, 0, 450, 950],
             "color": "#8A8", "font_size": 24, "flags": {}},
            {"id": 3, "title": "Decide and file", "bounding": [960, 140, 860, 760],
             "color": "#b58b2a", "font_size": 24, "flags": {}},
        ],
        "config": {}, "extra": {"ds": {"scale": 0.7, "offset": [80, 40]}},
        "version": 0.4,
    }


#: Types that arrive over a wire rather than as a widget. ``*`` is the wildcard
#: any-type socket, which is always a wire.
WIRE_TYPES = {"IMAGE", "MASK", "LATENT", "MODEL", "CLIP", "VAE", "CONDITIONING",
              "SEGS", "PERSON_DATA", "SAM_MODEL", "*"}


def widget_names(spec):
    """Widget slots of a node, in the order ComfyUI serialises them.

    A required input is a widget unless it arrives over a wire — either because
    its type is a tensor type, or because it declares ``forceInput``. A widget
    named ``seed`` additionally carries a control_after_generate slot.
    """
    names = []
    for name, definition in spec.get("required", {}).items():
        type_name, options = definition[0], (definition[1] if len(definition) > 1 else {})
        if isinstance(type_name, str) and type_name in WIRE_TYPES:
            continue
        if isinstance(options, dict) and options.get("forceInput"):
            continue
        names.append(name)
        if name in ("seed", "noise_seed"):
            names.append(f"{name}_control")
        if isinstance(options, dict) and options.get("image_upload"):
            names.append(f"{name}_upload")
    return names


def validate(workflow, base_url="http://127.0.0.1:8189"):
    """Check the generated graph against a live server's node definitions."""
    import urllib.request

    with urllib.request.urlopen(f"{base_url}/object_info", timeout=60) as response:
        info = json.load(response)

    problems = []
    for node in workflow["nodes"]:
        node_type = node["type"]
        if node_type == "Note":
            continue
        if node_type not in info:
            problems.append(f"{node_type}: not installed on the server")
            continue
        spec = info[node_type]["input"]
        # A widget that has been converted to an input no longer carries a value.
        wired = {connected["name"] for connected in node.get("inputs", [])}
        expected = [name for name in widget_names(spec) if name not in wired]
        got = node.get("widgets_values") or []
        if len(expected) != len(got):
            problems.append(f"{node_type}: {len(expected)} widgets "
                            f"({', '.join(expected)}) but {len(got)} values")
        known = set(spec.get("required", {})) | set(spec.get("optional", {}))
        for connected in node.get("inputs", []):
            if connected["name"] not in known:
                problems.append(f"{node_type}: no input named {connected['name']!r}")
        outputs = info[node_type].get("output_name", [])
        for index, slot in enumerate(node.get("outputs", [])):
            if index < len(outputs) and slot["name"] != outputs[index]:
                problems.append(f"{node_type}: output {index} is "
                                f"{outputs[index]!r}, graph says {slot['name']!r}")
    return problems


def main():
    import sys

    workflow = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(workflow, handle, indent=1, ensure_ascii=False)
    print(f"wrote {os.path.abspath(OUT)} — {len(workflow['nodes'])} nodes, "
          f"{len(workflow['links'])} links")

    if "--validate" in sys.argv:
        try:
            problems = validate(workflow)
        except OSError as error:
            print(f"could not reach ComfyUI to validate: {error}")
            return 0
        for problem in problems:
            print(f"  ! {problem}")
        print("VALID — every node, widget and slot matches the server"
              if not problems else f"{len(problems)} problem(s)")
        return 0 if not problems else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
