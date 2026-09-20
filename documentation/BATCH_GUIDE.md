# Batch Tools — work through a folder and sort the renders

Five nodes that walk a folder one picture per queue run, judge each one, and file
it into a keep or reject folder. Built for the job of tidying up a night's worth
of generated images without opening every one of them.

```
Batch Load ──image──▶ Reality Check ──passed──┐
    │                                          │
    │        ──image──▶ Person Selector ───────┤
    │                    (match, face_count)   ├──▶ Batch Router ──▶ Batch Save
    ├── pass_dir ─────────────────────────────┘      target_dir  ▲
    ├── fail_dir ────────────────────────────────────────────────┤
    └── source_path ─────────────────────────────────────────────┘
```

Ready-made graph: `examples/batch_reality_sort.json`.
Regenerate and check it against a running server with
`python scripts/build_batch_workflow.py --validate`.

---

## The nodes

### Batch Load Image

Hands out the next image each time the graph runs, and remembers how far it got.

| Setting | What it does |
|---|---|
| `directory` | The folder to work through. Non-image files are ignored. |
| `pass_subdir` / `fail_subdir` | Subfolders of the source directory, created on demand and passed downstream as absolute paths. |
| `on_finish` | `stop` ends the run when the folder is done; `loop` forgets the progress and starts over. |
| `sort_by` | `name` or `modified`. |
| `include_subdirs` | Walk subfolders too. Dot-folders are always skipped. |
| `tracker` | Name of the progress marker. Different names walk the same folder independently. |

Outputs: `image`, `mask`, `source_path`, `pass_dir`, `fail_dir`, `filename`,
`index`, `total`, `progress`.

The node shows a progress bar with the absolute position, the total, and the
percentage. The counts are read straight from disk, so they are there before the
first run. **Reset progress** forgets the marker and starts the folder over.

**Progress is stored as a list of finished filenames**, in
`.fvm_batch_state.json` inside the source folder — not as a counter. That matters
as soon as Batch Save is set to `move`: the listing shrinks under the loader's
feet, and a counter would skip every other picture. Delete that file to reset by
hand; the batch is resumable across ComfyUI restarts.

Queue the graph with a high run count — the loader ends the run itself when the
folder is done, the same way Cancel does.

### Reality Check (LM Studio)

Asks a local vision model whether the picture is physically possible, and answers
`passed` / `failed` / `score` / `report` / `json`.

Needs LM Studio running with a **vision** model. Measured on
`qwen3-8b-vl-instruct-abliterated`; an uncensored model is required for lingerie
or nude material, or the model refuses and every picture reads as unavailable.

The probes, and why they are the ones that are on:

| Probe | Question | Default |
|---|---|---|
| `parts` | How many heads, arms, hands, legs, feet are visible? | on |
| `people` | How many people, and are any two bodies fused? | on |
| `landmarks` | Do you see breasts or shoulder blades? the groin or the buttock cleft? | on |
| `hands` | Finger counts | **off** |
| `physics` | Floating bodies, limbs through furniture | **off** |
| `text` | Garbled lettering | off — turn on for scenes with signage |

`hands` and `physics` are off because they *measured worse than useless* on the
acceptance set: they never fired on a broken picture and occasionally fired on a
clean one. An 8B model cannot resolve fingers at any sane resolution, and it
reads a compressed mattress as a body sinking through it.

If the model cannot be reached, `on_error` decides the answer. The default is
`pass`: an inspector that cannot see should not silently reject a whole batch.

### Batch Router

ANDs the checks together and answers with the folder the picture belongs in.
Every gate is optional — **an unconnected gate is "not asked", not "false"**, so
adding a check you have not wired up yet cannot reject the batch.

- `gate_a` / `gate_b` / `gate_c` — booleans, with editable labels for the report
- `count` + `count_min` / `count_max` — bound a number, e.g. PersonSelector's
  `face_count` at 1..1 for "exactly one person"
- `invert` — swap the two folders

### Batch Save Image

- `move` — move the original file. The source folder empties as the batch runs;
  what is left is what nothing was decided about.
- `copy` — copy the original, source stays intact.
- `save` — encode the image tensor as a new file (use when the graph changed the
  picture).

`move` and `copy` hand the original bytes over untouched, so the generator's own
JPEG is preserved rather than re-encoded. A `report` input is written as a `.txt`
sidecar next to the picture, so the folder records why each file landed there.

### Reality Check Probe

Reachability check for LM Studio — returns whether the endpoint answers and which
models it offers.

---

## How the reality check works, and why it is built this way

**A small vision model cannot judge, but it can count and name what it sees.**

Asked "is this pose anatomically impossible?" an 8B model answered *no* for every
picture in the acceptance set, including a torso rotated through 180 degrees. It
does not judge; it reassures. Asked "do you see breasts or shoulder blades?" and
"do you see the groin or the buttock cleft?" it answered both correctly — and the
contradiction between those two answers **is** the impossible pose.

So every probe asks a perception question with a closed set of answers, and the
inference happens in Python, where it can be read, tested and argued with:

- more heads than one body can have → a second person
- front-of-body and back-of-body landmarks in one frame → a spine rotated about
  180°, which no back does

**Two stages, because one wording was not enough.** The landmark question alone
produced one false alarm: a woman sitting with open legs, whose visible crotch
the model called a "buttock cleft". A differently worded confirmation — a forced
choice between two concrete descriptions — got that picture right. So a twist
suspicion is only upheld when the second wording agrees. The confirmation can
only ever *clear* a suspicion, never raise one, and it is skipped entirely when
nothing was flagged.

Repeating the *same* question does not help: at temperature 0.1 all three passes
returned byte-identical answers on all 17 images. `passes` is 1 by default for
that reason — robustness comes from asking differently, not from asking again.

### Measured result

17 images, 8 known broken. Reproduced with two different seeds, and once more
end-to-end through the actual nodes:

```
caught 8/8 broken   kept 8/8 good   misses 0   false alarms 0
```

| Caught | Why |
|---|---|
| 0030, 0034, 0046, 0078 | two heads / merged bodies |
| 0042, 0082, 0110, 0150 | front chest above a rear pelvis — impossible torsion |

Reproduce:

```bash
# the model prompts, scored against ground truth
python tests/live/reality/calibrate.py live

# the whole chain — load, judge, route, move the files
python tests/live/reality/pipeline_run.py --acceptance
```

`calibrate.py run --tag x` caches every raw model answer, and
`calibrate.py score --tag x` replays it through the scorers — so rules and
thresholds can be re-tuned in a second instead of an hour of inference.

---

## The identity gate, and its measured limit

PersonSelector answers `match` (is this the right face) and `face_count` (how
many faces). Wire both into the router: `face_count` bounded at 1..1 catches the
duplicated-body renders on its own, independently of the vision model.

Threshold: **0.75**. Measured against four real reference portraits, the right
person scored 0.787–0.855 and the worst-rendered faces 0.66–0.69, so the cut
sits in the gap.

**What this gate cannot do.** One acceptance image (0136) is a clean, plausible
photograph whose face reads as the wrong person to a human eye. Four independent
methods all failed to separate it:

| Method | Result for 0136 |
|---|---|
| InsightFace vs. images from the same batch | 0.924 — the **highest** of the set |
| InsightFace vs. four real reference portraits | 0.841 — inside the good range (0.787–0.855) |
| Facial proportions from landmarks (z-distance) | 0.28 — the **most typical** face of the set |
| VLM asked to compare faces | 0.98 for every image; it separates nothing |

This is not a threshold that needs adjusting. The recognition embedding is
trained to be invariant to exactly the drift that changed here, and the picture
is frontal and well lit — the ideal case for the detector, which puts it closer
to the prototype than the obliquely lit good ones. A face gate will not catch
this class of reject; it needs a human eye, or a model trained on the specific
identity.
