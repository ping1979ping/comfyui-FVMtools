# Wildcards, variables & Dataset Prompt List — quick start

One engine for everything: `core/jb/wildcards.py`. The **JB Builder**, the
**Ideogram nodes** and the **Dataset Prompt List** all resolve through it, so
everything below applies to all of them.

---

## 1. Where things live

| What | Path | Git |
|---|---|---|
| Wildcard root | `wildcards/` (set in `wildcards.ini`, personal override `wildcards.ini.local`) | **ignored** — your files survive every update |
| Shipped dataset wildcards | `dataset_presets/wildcards/dataset/*.txt` | tracked (template) |
| Dataset presets (prompt lists) | `dataset_presets/presets/*.txt` | tracked |
| Engine | `core/jb/wildcards.py` | |
| Dataset logic | `core/dataset_prompts.py`, node `nodes/dataset_prompts.py`, UI `web/js/fvm_dataset_prompts.js` | |

On ComfyUI start FVMtools copies the shipped `dataset/*.txt` into the wildcard
root **only if the file is missing there**. Edits made in the wildcard editor
are never overwritten. To get the factory version back, delete the file in the
root and restart.

Editing needs no restart: the **Wildcards** button (JB Builder or Dataset
Prompt List). One file = one list, one option per line.

---

## 2. Syntax cheat sheet

Also available in every node via the **Syntax** / **AdvPmptInfo** button.

### Wildcards
| Token | Meaning |
|---|---|
| `__name__` | random line from `name.txt` |
| `__dataset/outfit__` | subfolder |
| `__dataset/setting_*__` | random file whose name starts with `setting_` |
| `__name^var__` | draw a line **and** bind it to `var` |
| `__^var__` | recall the bound value (same value!) |

### Brackets
| Token | Meaning |
|---|---|
| `{a\|b\|c}` | pick one |
| `{%3%a\|b}` | weighted (a three times as likely) |
| `{2$$a\|b\|c}` | 2 distinct, joined with `, ` |
| `{1-3$$a\|b\|c}` | 1 to 3 distinct |
| `{2$$ and $$a\|b\|c}` | custom separator |
| `{a\|b}^var` | bind the result to `var` |

### Inside `.txt` files
- `# comment` (also at line end). A literal `#` is written `\#`.
- `%2.5%silk` — per-line weight.
- Lines may contain wildcards/brackets themselves (nested, max. 16 levels).

### Misc
- `##…##` — resolved (variables get bound), then removed from the output.
  Example: `##__dataset/color^c__## a __^c__ shirt with __^c__ socks`
- `\__name__` / `\{` — literal, never resolved.

### Context (variables from outside)
The optional `context_from_prompt_generator` input (DICT) takes the `context`
output of an adaptiveprompts **PromptGenerator**. Values bound there with `^VAR`
can be recalled in every line with `__^VAR__`. A plain text node does **not** work.

### Seeds
Same seed + same text + same wildcard files → identical result. Every call site
(JB: JSON path, Dataset: line + variation) gets its own salt and rolls
independently.

---

## 3. FVM · Dataset Prompt List

Replacement for Comfyroll **CR Prompt List**, built for character-LoRA
datasets. Category: `FVM Tools/Prompt`.

### Line format
```
# comment
[close] Left three-quarter portrait …, wearing __dataset/upper__, __dataset/setting__
[half, back] Waist-up view from behind …
[full] Full-body side view walking …, wearing __dataset/outfit__, __dataset/setting_outdoor__
```
`[close]` head & shoulders · `[half]` waist-up / seated · `[full]` head to feet.
Extra tags are allowed and only serve as bookkeeping.

### Widgets
| Widget | Effect |
|---|---|
| `shot_filter` | only close / half / full lines (all = everything incl. untagged) |
| `start_index`, `max_rows` | range, counted **after** the filter (0-based, see Preview) |
| `variations` | every line N times, each with fresh wildcards → 20 lines × 10 = 200 images |
| `order` | `rounds` (every angle first, then again — safe to stop early) · `per_prompt` · `shuffle` |
| `seed` | new seed = completely new set |
| `prefix` | e.g. trigger word; wildcards and `^var` allowed |
| `suffix` | default `__dataset/photo__` (photo realism) |

### Outputs
| Output | Type | Use |
|---|---|---|
| `prompt` | list | to the text encoder |
| `caption` | list | training caption **without** prefix/suffix — clothes/background are named, so they don't get baked into the character |
| `shot` | list | file name / subfolder |
| `index` | list | line number |
| `count` | INT | number of prompts |
| `listing` | STRING | overview for Show Text |

Toolbar: **Presets ▾** (Load / + Append) · **Save Preset** · **Preview**
(resolves everything without generating; has a "New Seed" button) ·
**Wildcards** · **Syntax** · **?**. Typing `__` in the text box opens the
wildcard autocomplete.

### Shipped presets
| Preset | Content |
|---|---|
| `quick_20` | 20 lines (8 close / 6 half / 6 full), the default |
| `character_complete` | 72 lines (26 / 24 / 22): all angles, profiles, back views, high/low angle, seated, walking |
| `original_25_wildcards` | the original 25 prompts with clothes/background replaced by slots |

### Dataset wildcard slots (`__dataset/…__`)
| Slot | Content | Used in |
|---|---|---|
| `upper` | `top`, or an open `outer` over a `top` | close, half |
| `outfit` | top + bottom + shoes, sometimes with a jacket | full |
| `top`, `outer`, `bottom`, `shoes` | single garments, each pulls a `color` | |
| `color` | 30 muted colours (no neon — strong colours cast onto skin) | |
| `setting` | mix studio 2 : indoor 3 : outdoor 4 | |
| `setting_studio` / `_indoor` / `_outdoor` | background + **matching** light | |
| `bg_*`, `light_*` | the parts of the above | |
| `expression` | 18 expressions, described by the face | lines with a free expression |
| `arms`, `stance` | arm / body posture | half, full |
| `photo` | realism suffix | suffix |

**Clothes are deliberately unisex, without logos, lettering or headwear.**
Lettering gets rendered as (broken) text; hats and glasses hide identity
features. For female characters, add dresses/skirts to `outfit.txt` /
`bottom.txt`.

### Building your own set
1. Load a preset, adjust lines, **Save Preset** under your own name.
2. Own slots: new file in the wildcard editor, e.g. `mychar/hair_accessory`,
   then use `__mychar/hair_accessory__` in the lines.
3. Do **not** randomise character traits (hair, eye colour, tattoos) — they
   must stay constant and belong in the `prefix` or the reference image.

### Rules of thumb for a LoRA dataset
- Roughly 40 % close, 30 % half, 30 % full. Close-ups only → a LoRA that can't
  do bodies.
- Every angle several times, each with different clothes and surroundings —
  exactly what `variations` does.
- No "different outfit": each prompt is generated in isolation, "different"
  has nothing to compare against.
- Left/right always from the subject's point of view ("toward the subject's left").
