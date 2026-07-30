# ComfyUI PromptForge — Technical Specification

> **Working title:** PromptForge
> **Repository name (Vorschlag):** `ComfyUI-PromptForge`
> **Schema version:** 1.0
> **Status:** Draft 1 (Konzept-Briefing für Claude Code)

---

## 1. Vision & Ziele

PromptForge ist eine modulare Custom-Node-Suite für ComfyUI, die strukturierte JSON-Prompts erzeugt. Die Suite richtet sich an Power-User, die mit modernen Diffusionsmodellen (Z-Image-Turbo, Nano Banana, Flux 2, SDXL) reproduzierbar und in Batches arbeiten wollen, und denen klassische "Prompt-als-Textwurst" zu unflexibel ist.

**Designprinzipien:**

1. **Eine Datenstruktur, mehrere Output-Formate.** Intern fließt immer ein `PROMPT_DICT` (JSON) durch die Pipeline. Am Ende wird modellabhängig serialisiert (raw JSON, Natural Language, Qwen-ChatML, Comma-Tags).
2. **Ein Konzept pro Node.** Jede Sektion (Subject, Pose, Clothing, Environment, Camera, Lighting, ...) ist eine eigene Node. Keine Mega-Node mit 80 Widgets.
3. **Alles randomisierbar.** Auf Wert-Ebene (Inline-Wildcards), auf Datei-Ebene (externe Wildcard-Files), auf Feld-Ebene (BatchVariator), auf Preset-Ebene (Preset-Bibliothek mit Vererbung).
4. **Voll kompatibel.** Alle Strings sind ComfyUI-Standard-Strings; jeder externe Encoder, Wildcard-Pack oder Sampler funktioniert weiter. PromptForge ersetzt nichts, sondern ergänzt.
5. **Sidecar-Saving.** Jeder gerenderte Prompt wird als `.json` neben dem Output-Bild gespeichert — bei Wildcards und Variation absolut unverzichtbar, weil ComfyUI sonst nur den Workflow speichert, nicht den aufgelösten Prompt.
6. **Live-Preview als Top-Priorität.** Der User sieht *bevor* der KSampler läuft, was tatsächlich an den Encoder geht.

---

## 2. Architektur: Dictionary-Pipeline-Pattern

```mermaid
flowchart LR
    A[Subject Builder] --> AGG
    B[Pose Builder] --> AGG
    C[Clothing Builder] --> AGG
    D[Environment Builder] --> AGG
    E[Camera Builder] --> AGG
    F[Lighting Builder] --> AGG
    G[Composition Builder] --> AGG
    H[Scene/Style Builder] --> AGG

    PRESET[Preset Library<br/>Loader] -.optional Override.-> AGG
    WILDCARDS[Wildcard Resolver<br/>__file__ + a|b|c] -.transparent.-> AGG

    AGG[Aggregator<br/>Validator] --> VAR
    VAR[BatchVariator<br/>Lock/Vary] --> SER
    SER[Serialize<br/>raw_json / NL / chatml] --> PRE

    PRE[Live Preview<br/>+ Sidecar Save] --> ENC
    ENC[CLIP Text Encode<br/>or ZImageTextEncoder]

    style AGG fill:#4a5568,stroke:#2d3748,color:#fff
    style VAR fill:#4a5568,stroke:#2d3748,color:#fff
    style SER fill:#4a5568,stroke:#2d3748,color:#fff
    style PRE fill:#d69e2e,stroke:#975a16,color:#fff
```

**Kernidee:** Jeder Builder produziert einen partiellen `PROMPT_DICT`. Der Aggregator macht einen Deep-Merge mit Konfliktauflösung (siehe §6.2). Der BatchVariator interveniert pro Batch-Iteration und zerwürfelt definierte Felder. Die Serialize-Node ist der einzige Ort, an dem JSON in einen finalen String umgewandelt wird.

**Datentypen zwischen Nodes:**
- Eigener Type `PROMPT_DICT` für strenge Typprüfung zwischen PromptForge-Nodes
- Type `STRING` als finaler Output, kompatibel mit allen ComfyUI-Encodern
- Type `INT` für Seed-Propagation

---

## 3. Repository-Struktur

```
ComfyUI-PromptForge/
├── __init__.py                 # NODE_CLASS_MAPPINGS Registrierung
├── pyproject.toml              # Dependencies, ComfyUI Manager Registry-Eintrag
├── README.md
├── LICENSE
├── requirements.txt
│
├── promptforge/                # Python-Package
│   ├── __init__.py
│   ├── types.py                # PROMPT_DICT-Definition, Validierung
│   ├── schema.py               # Pydantic-Schema v1.0
│   ├── merge.py                # Deep-Merge-Logik
│   ├── wildcards.py            # Wrapper um dynamicprompts-Library
│   ├── serialize/
│   │   ├── __init__.py
│   │   ├── raw_json.py
│   │   ├── natural_language.py # JSON → NL Flattening
│   │   ├── qwen_chatml.py      # Z-Image Chat-Template
│   │   └── comma_tags.py       # SDXL-style
│   │
│   └── nodes/
│       ├── __init__.py
│       ├── builders/           # Tier 1
│       │   ├── subject.py
│       │   ├── pose.py
│       │   ├── clothing.py
│       │   ├── environment.py
│       │   ├── camera.py
│       │   ├── lighting.py
│       │   ├── composition.py
│       │   └── scene_style.py
│       ├── library/            # Tier 2
│       │   ├── preset_loader.py
│       │   └── wildcard_injector.py
│       ├── composition/        # Tier 3
│       │   ├── aggregator.py
│       │   ├── batch_variator.py
│       │   └── randomizer.py
│       └── output/             # Tier 4
│           ├── serialize.py
│           ├── preview.py
│           ├── sidecar_saver.py
│           └── llm_enhance.py  # optional, später
│
├── presets/                    # YAML-Preset-Bibliothek mit Vererbung
│   ├── _base/
│   │   ├── photoreal.yaml
│   │   └── editorial.yaml
│   ├── styles/
│   │   ├── fashion_editorial.yaml
│   │   ├── street_photography.yaml
│   │   └── studio_portrait.yaml
│   ├── locations/
│   │   ├── brutalist_concrete.yaml
│   │   ├── rooftop_sunset.yaml
│   │   └── industrial_loft.yaml
│   └── poses/
│       ├── seated_dynamic.yaml
│       └── standing_confident.yaml
│
├── wildcards/                  # Klassische Wildcard-Files (kompatibel mit dynamicprompts)
│   ├── colors.txt
│   ├── hair_styles.txt
│   ├── camera_lenses.txt
│   └── moods.txt
│
├── web/                        # JS-Frontend (UX-Priorität!)
│   ├── promptforge.js          # Extension-Entrypoint
│   ├── widgets/
│   │   ├── json_editor.js      # Monaco/CodeMirror-Wrapper
│   │   ├── live_preview.js     # WebSocket-basiert
│   │   ├── lock_vary_toggle.js
│   │   └── preset_browser.js
│   └── styles.css
│
├── examples/
│   ├── workflows/
│   │   ├── 01_basic_zimage.json
│   │   ├── 02_batch_variations.json
│   │   ├── 03_multi_person.json
│   │   └── 04_preset_inheritance.json
│   └── presets/
│
└── tests/
    ├── test_schema.py
    ├── test_merge.py
    ├── test_wildcards.py
    └── test_serialize.py
```

---

## 4. Datenschema (PROMPT_DICT v1.0)

Implementiert via **Pydantic v2** für automatische Validierung, JSON-Schema-Generierung und IDE-Support.

```python
# promptforge/schema.py
from pydantic import BaseModel, Field
from typing import Literal, Optional, Any

class Meta(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    target_model: Literal["z-image-turbo", "nano-banana", "flux", "sdxl", "generic"] = "generic"
    seed: int = 0
    label: Optional[str] = None  # Frei wählbarer Identifier für die Generation

class Scene(BaseModel):
    overall_style: Optional[str] = None
    mood: Optional[str] = None
    color_palette: Optional[str] = None
    photography_technique: Optional[str] = None

class Environment(BaseModel):
    location: Optional[str] = None
    background: Optional[str] = None
    foreground: Optional[str] = None
    time_of_day: Optional[str] = None
    weather: Optional[str] = None
    props: Optional[str] = None

class Camera(BaseModel):
    angle: Optional[str] = None
    lens: Optional[str] = None
    perspective: Optional[str] = None
    aspect_ratio: Optional[str] = None

class Lighting(BaseModel):
    type: Optional[str] = None
    direction: Optional[str] = None
    quality: Optional[str] = None
    key_fill_rim: Optional[dict[str, str]] = None  # für Studio-Setups

class Composition(BaseModel):
    framing: Optional[str] = None
    rule: Optional[str] = None  # rule_of_thirds, golden_ratio, etc.
    focus: Optional[str] = None
    depth: Optional[str] = None

class Subject(BaseModel):
    id: str = "subject_1"
    identity: Optional[dict[str, Any]] = None  # description, age_range, ethnicity, ...
    hair: Optional[dict[str, Any]] = None
    face: Optional[dict[str, Any]] = None
    body: Optional[dict[str, Any]] = None
    pose: Optional[dict[str, Any]] = None
    clothing: Optional[dict[str, Any]] = None
    accessories: Optional[list[dict[str, Any]]] = None
    interaction_with: Optional[str] = None  # ID einer anderen Subject
    visibility: Literal["full", "partial", "background"] = "full"

class PostProcessing(BaseModel):
    negative_prompt: Optional[str] = None
    loras: Optional[list[dict[str, Any]]] = None
    quality_tags: Optional[list[str]] = None

class PromptDict(BaseModel):
    """Master container — das Schema, das durch die Pipeline fließt."""
    meta: Meta = Field(default_factory=Meta)
    scene: Scene = Field(default_factory=Scene)
    environment: Environment = Field(default_factory=Environment)
    camera: Camera = Field(default_factory=Camera)
    lighting: Lighting = Field(default_factory=Lighting)
    composition: Composition = Field(default_factory=Composition)
    subjects: list[Subject] = Field(default_factory=list)
    post_processing: PostProcessing = Field(default_factory=PostProcessing)
    extras: dict[str, Any] = Field(default_factory=dict)  # Escape-Hatch für Custom-Felder
```

**Wichtige Designentscheidungen:**

- Alle Felder optional → ein einzelner Builder kann teilweise gefüllt sein, der Rest wird default-leer befüllt.
- `subjects` ist ein Array → Multi-Person-Szenen funktionieren ohne Schema-Änderung.
- `extras` als Escape-Hatch → User kann eigene Felder hinzufügen, die später Custom-Serializer verarbeiten.
- `id` pro Subject → ermöglicht `interaction_with: "subject_2"` für relationale Posing-Beschreibungen.

---

## 5. Node-Inventar

### 5.1 Tier 1 — Builder-Nodes

Jeder Builder hat dieses Grundgerüst:

```python
class PromptForge_SubjectBuilder:
    CATEGORY = "PromptForge/Builders"
    RETURN_TYPES = ("PROMPT_DICT",)
    RETURN_NAMES = ("prompt_dict",)
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_dict_in": ("PROMPT_DICT", {"forceInput": False}),  # optional Eingang
            },
            "optional": {
                "subject_id": ("STRING", {"default": "subject_1"}),
                "identity_json": ("STRING", {"multiline": True, "default": "{}"}),
                "hair_json": ("STRING", {"multiline": True, "default": "{}"}),
                "face_json": ("STRING", {"multiline": True, "default": "{}"}),
                "pose_json": ("STRING", {"multiline": True, "default": "{}"}),
                "clothing_json": ("STRING", {"multiline": True, "default": "{}"}),
                "lock_field_paths": ("STRING", {"multiline": False, "default": ""}),
                "vary_field_paths": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    def build(self, prompt_dict_in=None, **kwargs):
        # 1. Bestehenden PROMPT_DICT laden oder neu erstellen
        # 2. Subject-Sub-Schema befüllen
        # 3. Wildcards in den Strings auflösen (delegiert an wildcards.py)
        # 4. Lock/Vary-Markierungen setzen
        # 5. Mergen und zurückgeben
        ...
```

| Node | Zweck | Wichtigste Inputs |
|---|---|---|
| `PromptForge_SubjectBuilder` | Eine Person (mehrfach instanziierbar) | `subject_id`, `identity_json`, `hair_json`, `face_json` |
| `PromptForge_PoseBuilder` | Pose getrennt, weil oft randomisiert | `pose_json` oder Preset-Picker `pose_preset` |
| `PromptForge_ClothingBuilder` | Outer-Layer, Hosiery, Footwear, Accessories | strukturierte Sub-Felder |
| `PromptForge_EnvironmentBuilder` | Location, Background, Time, Weather | `location`, `background`, `time_of_day` |
| `PromptForge_CameraBuilder` | Angle, Lens, Perspective | meist statisch pro Shoot |
| `PromptForge_LightingBuilder` | Lichtsetup | `type`, `direction`, `quality` |
| `PromptForge_CompositionBuilder` | Framing & Tiefe | `framing`, `rule`, `focus` |
| `PromptForge_SceneStyleBuilder` | Style, Mood, Color-Palette | übergeordneter Stil |
| `PromptForge_NegativePromptBuilder` | Quality-Tags & Anti-Artefakte | wichtig: separater Zweig |

**Multi-Person-Workflow:** Zwei `SubjectBuilder` werden hintereinander geschaltet, jeweils mit eigener `subject_id`. Der zweite Builder mergt zur subjects-Liste, statt zu überschreiben (siehe §6.2).

### 5.2 Tier 2 — Library- & Preset-Nodes

| Node | Zweck |
|---|---|
| `PromptForge_PresetLoader` | Lädt YAML aus `presets/`, mergt in den PROMPT_DICT-Stream. Mit Inheritance-Support (`extends:` Feld in YAML). |
| `PromptForge_WildcardInjector` | Explizite Wildcard-Auflösung an einem Pipeline-Punkt (normalerweise auto im Aggregator). |

**Preset-Vererbung — Beispiel:**

```yaml
# presets/styles/fashion_editorial_brutalist.yaml
extends: _base/editorial.yaml
scene:
  overall_style: "photorealistic fashion editorial"
  mood: "elegant, modern minimalist, confident"
  color_palette: "neutral concrete grays with bold accent"
environment:
  location: "modern brutalist concrete architecture exterior"
  background: "large plain light-gray concrete wall"
camera:
  angle: "low-angle shot from slightly below"
  lens: "35mm prime"
lighting:
  type: "bright natural daylight"
  quality: "soft diffused, high key"
```

```yaml
# presets/_base/editorial.yaml
scene:
  photography_technique: "fashion shoot, sharp focus on subject"
composition:
  rule: "rule_of_thirds"
  focus: "sharp on eyes and face"
post_processing:
  quality_tags: ["8k", "highly detailed", "professional photography"]
  negative_prompt: "blurry, low quality, distorted anatomy, extra limbs"
```

### 5.3 Tier 3 — Composition-Nodes

| Node | Zweck |
|---|---|
| `PromptForge_Aggregator` | Sammelt alle PROMPT_DICTs vom Stream, validiert, löst Wildcards auf, gibt finalen merged DICT zurück. |
| `PromptForge_BatchVariator` | Erzeugt N Varianten. Nimmt Lock-Liste (was konstant bleibt) und Vary-Liste (was randomisiert) entgegen. |
| `PromptForge_Randomizer` | Pickt aus mehreren Inputs einen aus (Multi-Input → Single-Output). Nützlich für "5 Outfits, eines wird gewählt". |

**BatchVariator — wichtigste Node für deinen Use Case:**

```python
class PromptForge_BatchVariator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_dict": ("PROMPT_DICT",),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "vary_paths": ("STRING", {
                    "multiline": True,
                    "default": "subjects[0].pose\nsubjects[0].clothing.outer_layer\nenvironment.location",
                }),
                "lock_paths": ("STRING", {
                    "multiline": True,
                    "default": "subjects[0].identity\nscene.overall_style\ncamera",
                }),
                "seed_strategy": (["increment", "random", "fixed"], {"default": "increment"}),
            },
        }

    RETURN_TYPES = ("PROMPT_DICT", "INT")
    RETURN_NAMES = ("prompt_dict", "current_seed")
```

Pro Batch-Run wird der DICT geklont, die `vary_paths` neu aus ihrem Wildcard-Pool gezogen (Seed wird gemäß `seed_strategy` propagiert), die `lock_paths` bleiben unverändert.

### 5.4 Tier 4 — Output-Nodes

| Node | Output | Zweck |
|---|---|---|
| `PromptForge_Serialize` | `STRING` (positive), `STRING` (negative), `STRING` (raw_json) | **Drei Output-Pins gleichzeitig**, User entscheidet welcher in den Encoder geht |
| `PromptForge_LivePreview` | (kein Output, nur UI) | Zeigt aufgelösten Prompt in Echtzeit im Node-Body |
| `PromptForge_SidecarSaver` | `IMAGE` (durchgereicht) | Sitzt zwischen VAEDecode und SaveImage, schreibt `.json` parallel zum Bild |
| `PromptForge_LLMEnhance` | `PROMPT_DICT` | Optional: schickt DICT durch lokales LLM (Ollama) zur Anreicherung |

**Serialize-Node — Detail:**

```python
class PromptForge_Serialize:
    CATEGORY = "PromptForge/Output"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "raw_json")
    FUNCTION = "serialize"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_dict": ("PROMPT_DICT",),
                "format": (
                    ["natural_language", "raw_json", "qwen_chatml", "comma_tags", "auto"],
                    {"default": "auto"},
                ),
                "include_negative": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }
```

**`auto`-Modus:** wertet `meta.target_model` aus → wählt automatisch das passende Format (`z-image-turbo` → `qwen_chatml`, `flux`/`nano-banana` → `natural_language`, `sdxl` → `comma_tags`).

**Drei Outputs gleichzeitig** ist wichtig: User kann `positive_prompt` direkt in CLIPTextEncode stecken UND `raw_json` für Debugging in eine PreviewText-Node hängen, ohne erneut serialisieren zu müssen.

---

## 6. Variations-Mechanismen (alle vier)

### 6.1 Inline-Wildcards `{a|b|c}`

Funktioniert auf jeder String-Ebene. Der Aggregator löst auf:

```json
{
  "subjects": [{
    "clothing": {
      "outer_layer": {
        "color": "{red|black|emerald|burgundy}",
        "type": "{mini dress|midi dress|jumpsuit}"
      }
    }
  }]
}
```

Optional gewichtet: `{2::red|1::black|1::emerald}` — `red` wird doppelt so wahrscheinlich gewählt. Implementierung: **direkter Wrapper um `dynamicprompts.RandomPromptGenerator`**, kein Re-Implement.

### 6.2 Externe Wildcard-Files

Klassische `__path__`-Syntax, kompatibel mit comfyui-dynamicprompts:

```json
{
  "environment": {
    "location": "__locations/brutalist__",
    "time_of_day": "__time_of_day__"
  }
}
```

Sucht in `wildcards/locations/brutalist.txt` (eine Zeile = ein Eintrag) bzw. `wildcards/time_of_day.txt`. Auch YAML mit strukturierten Blöcken wird unterstützt — dann wird der ganze Block in den DICT gemergt.

### 6.3 Field-Level BatchVariator (siehe §5.3)

Der präziseste Mechanismus: User definiert genau, welche Felder über den Batch variieren und welche eingefroren werden. Beispiel-Workflow:

> "Ich will 50 Bilder. Modell, Stil und Kamera gleich. Pose, Outfit-Farbe und Location randomisieren."

```
lock_paths:
  subjects[0].identity
  subjects[0].hair
  subjects[0].face
  scene
  camera

vary_paths:
  subjects[0].pose
  subjects[0].clothing.outer_layer.color
  environment.location
```

### 6.4 Preset-Bibliothek mit Vererbung

Siehe §5.2. Der `PresetLoader` unterstützt:
- **`extends:`** — String oder Liste von Pfaden, die in Reihenfolge gemergt werden
- **`overrides:`** — Felder, die selbst beim erbenden Preset Vorrang haben
- **`tags:`** — Klassifikation für den Preset-Browser im Frontend

**Konflikt-Auflösung im Aggregator:**
1. Spätere Builder im Stream überschreiben frühere (Standard-Pipeline-Reihenfolge).
2. Felder mit Wert `null` löschen explizit das Feld (statt es auf null zu setzen).
3. Arrays werden **konkateniert**, nicht überschrieben — wichtig für `subjects`.
4. Felder mit `lock`-Marker werden vom BatchVariator ignoriert, aber von späteren Buildern noch überschrieben.

---

## 7. Output-Serialisierung (beide Modi prominent)

### 7.1 Natural Language (Standard für Nano Banana, Flux, Generic)

Der NL-Serializer wandelt JSON in einen kohärenten Prompt um. Strategie: **Hierarchischer Walk** mit Templates pro Sektion.

```
A {scene.overall_style} photograph. {scene.mood}.
The image shows {subjects[0].identity.description}, with {subjects[0].hair.color} {subjects[0].hair.style} hair.
She is {subjects[0].pose.body_position}, wearing {subjects[0].clothing.outer_layer.color} {subjects[0].clothing.outer_layer.type}.
Setting: {environment.location}, {environment.time_of_day}.
{lighting.quality} {lighting.type}, {lighting.direction}.
Shot with {camera.lens}, {camera.angle}.
{composition.framing}, focus on {composition.focus}.
Quality: {post_processing.quality_tags joined}.
```

Templates sind separat editierbar in `promptforge/serialize/templates/`. Per Modell ein Template-Set, weil z.B. Nano Banana eher konversationell auf "Show me X wearing Y in Z" reagiert.

### 7.2 Raw JSON (Standard für Z-Image, Flux 2, Power-User)

Serialisiert den DICT direkt als JSON-String mit aufgelösten Wildcards. Forschung (Republic Labs, Diffusion Doodles, Imagine Art) zeigt: moderne Modelle mit LLM-basierten Encodern performen mit raw JSON oft **besser** als mit NL bei komplexen Szenen, weil die hierarchische Struktur als Gewichtungs-Hinweis interpretiert wird.

### 7.3 Qwen-ChatML (Z-Image-spezifisch)

Z-Image's Qwen3-4B-Encoder erwartet:

```
<|im_start|>system
You are a photorealistic image generator. Render with high detail.
<|im_end|>
<|im_start|>user
{NL-flattened prompt or raw JSON}
<|im_end|>
<|im_start|>assistant
<think>
{optional reasoning block — wirkt als implizite Bildbeschreibungs-Verlängerung}
</think>
```

Der Serializer baut diese Struktur auf, optional mit `<think>`-Block. **Wichtig:** Wenn der User `martin-rizzo/ComfyUI-ZImagePowerNodes` (`ZImageTextEncoder`) bereits installiert hat, kann unser Serializer im Modus `qwen_chatml_raw` einfach den NL-Prompt zurückgeben und die ZImagePowerNode kümmert sich um das Template-Wrapping. Empfohlene Default-Einstellung.

### 7.4 Comma-Tags (SDXL/CLIP-Modelle)

Klassisches `tag1, tag2, tag3, ...`-Format. Serializer extrahiert die wichtigsten Werte aus dem DICT und konkateniert. Weniger expressiv, aber für SDXL-LoRAs notwendig.

---

## 8. UI/UX-Spezifikation (priorisiert nach User-Vorgabe)

### 8.1 [PRIO 1] Live-Preview des aufgelösten Prompts

**Implementierung:** `PromptForge_LivePreview`-Node mit JS-Frontend-Widget, das per WebSocket vom Backend bei Änderungen aktualisiert wird.

- **Lage:** typischerweise zwischen `Serialize` und `CLIPTextEncode`. Pure Pass-Through-Node mit großem Text-Display.
- **Anzeige:** Tabs für `positive` / `negative` / `raw_json` / `wildcards_resolved_log`.
- **Trigger-Update:** beim Workflow-Queue (oder Live bei "Auto-refresh on change"-Toggle).
- **Diff-Modus:** zeigt zwischen zwei Batch-Iterationen nur die geänderten Werte hervorgehoben.

```javascript
// web/widgets/live_preview.js (Skizze)
app.registerExtension({
  name: "PromptForge.LivePreview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "PromptForge_LivePreview") return;

    nodeType.prototype.onNodeCreated = function() {
      const widget = this.addCustomWidget({
        type: "preview",
        name: "preview_display",
        // ... Tabs, Syntax-Highlighting via Prism.js, Diff via diff2html
      });
    };

    api.addEventListener("promptforge.preview", ({detail}) => {
      // detail = { node_id, positive, negative, raw_json, wildcards_log }
      widget.update(detail);
    });
  },
});
```

### 8.2 [PRIO 2] JSON-Editor mit Syntax-Highlight

In jeder Builder-Node ersetzt ein **CodeMirror-basierter Editor** das Standard-Multiline-Textfeld für `*_json`-Inputs.

- Syntax-Highlighting (CodeMirror 6, JSON-Mode)
- Live-Validation gegen Pydantic-Schema (Server-Roundtrip oder Client-Side via JSON-Schema)
- Auto-Vervollständigung der Schema-Keys
- Format-Button (Pretty-Print)
- Fehleranzeige bei invalidem JSON, ohne die Workflow-Ausführung zu blockieren (Builder ignoriert dann das Feld und loggt Warning)

### 8.3 [PRIO 3] Lock/Vary-Toggle pro Sektion

Zwei kleine Buttons (🔒 Lock / 🔀 Vary) am Header jeder Builder-Node. Klick → setzt entsprechenden Marker im DICT. Der BatchVariator liest diese Marker auf einer Pipeline-Ebene und überschreibt seine eigenen `lock_paths` / `vary_paths` damit. Default: kein Marker (= Variator-Konfig gilt).

Visuell: kleine Badges am Node-Rand, farblich (🔒 grau, 🔀 blau) markiert.

### 8.4 [PRIO 4] Preset-Browser mit Thumbnails

Sidebar-Tab in der ComfyUI-UI. Listet alle Presets aus `presets/`, gruppiert nach Verzeichnis. Pro Preset:

- Thumbnail (`presets/thumbnails/<preset-name>.webp`, optional)
- Name, Beschreibung, Tags
- "Apply"-Button → erzeugt `PresetLoader`-Node mit dem Preset auf dem Canvas

**Phase-2-Feature** (nicht Phase 1) — kann einfach mit statischen Markdown-Listen und Bildern starten und später ein vollwertiger Browser werden.

---

## 9. Wildcards & Preset-Konventionen

### 9.1 Wildcard-Files

- **`.txt`** — eine Zeile = ein Eintrag, Standard.
- **`.yaml`** — strukturierte Blöcke. Jeder Top-Level-Key ist ein wählbarer Wert; der Block dahinter wird in den DICT gemergt.
- **`.json`** — wie YAML, aber JSON.

Beispiel `wildcards/poses/seated.yaml`:

```yaml
casual_lean:
  body_position: "seated on a wide step"
  torso: "leaning slightly back"
  legs: "extended forward, casually crossed"
  arms: "one resting on knee, other supporting on step behind"

dynamic_curl:
  body_position: "seated on edge of step"
  torso: "elegant S-curve, leaning to one side"
  legs: "one bent up, one extended down"
  arms: "asymmetric, one supporting, one draping"
```

Im PoseBuilder wählbar via `__poses/seated__` oder `pose_preset: "casual_lean"`.

### 9.2 Preset-Vererbungs-Regeln

1. `extends:` wird vor allem anderen verarbeitet (Basis aufbauen).
2. Eigene Felder überschreiben.
3. `overrides:` (Top-Level) hat höchste Priorität, gewinnt auch gegen spätere Builder im Stream — Escape-Hatch für "dieses Preset zwingt die Kamera".
4. Diamond-Inheritance erlaubt; Reihenfolge der `extends:`-Liste bestimmt Auflösung (links = niedrigste Prio).

---

## 10. Modell-spezifische Adapter

| Modell | Empfohlener Format | Encoder-Node | Anmerkung |
|---|---|---|---|
| **Z-Image-Turbo** | `qwen_chatml` oder `raw_json` | `ZImageTextEncoder` (martin-rizzo) oder Standard `CLIPTextEncode` mit Lumina-2-Modus | System-Prompt im ChatML-Block kann Stilrichtung verstärken |
| **Nano Banana** (über API-Bridge) | `natural_language` | externer API-Call, nicht klassischer ComfyUI-Encoder | konversationell, max. 14 Referenzbilder |
| **Flux 2** | `raw_json` (offiziell unterstützt) | `CLIPTextEncode` mit T5/CLIP | unterstützt `reference_images`, `regions` natively |
| **Flux 1** / **SDXL** | `natural_language` oder `comma_tags` | `CLIPTextEncode` | klassisches Verhalten |

Der **Auto-Modus** in `Serialize` liest `meta.target_model` und wählt entsprechend.

---

## 11. Beispiel-Workflows (im Repo unter `examples/workflows/`)

### 11.1 `01_basic_zimage.json`
Einfachster Fall: SubjectBuilder → EnvironmentBuilder → CameraBuilder → Aggregator → Serialize (auto, Target=z-image-turbo) → ZImageTextEncoder → KSampler.

### 11.2 `02_batch_variations.json`
Wie oben + BatchVariator mit `batch_size=20`, `vary_paths` für Pose+Outfit+Location, `lock_paths` für Identity+Style+Camera. Plus SidecarSaver für JSON-Logs.

### 11.3 `03_multi_person.json`
Zwei SubjectBuilder mit unterschiedlichen IDs, einer mit `interaction_with: "subject_1"`. Aggregator merged zur Subjects-Liste. NL-Serializer fügt im Output beide Personen mit Bezug auf einander ein.

### 11.4 `04_preset_inheritance.json`
PresetLoader lädt `fashion_editorial_brutalist.yaml` (das selbst von `_base/editorial.yaml` erbt), dann ein zusätzlicher CameraBuilder, der ein Detail überschreibt.

---

## 12. Implementierungs-Roadmap für Claude Code

### Phase 1 — Skelett & Core (1-2 Tage)
- Repo-Struktur, `pyproject.toml`, `__init__.py`
- `PROMPT_DICT`-Type-Registrierung in ComfyUI
- Pydantic-Schema (`schema.py`)
- `merge.py` mit Deep-Merge + Konfliktauflösung
- Genau **eine** Builder-Node (`SubjectBuilder`) als Referenz
- `Aggregator`-Node (minimal)
- `Serialize`-Node mit `raw_json` und `natural_language`
- Test-Workflow für Z-Image-Turbo
- Unit-Tests für Schema und Merge

**Akzeptanzkriterium:** Ein `SubjectBuilder` → `Aggregator` → `Serialize` → `ZImageTextEncoder` → `KSampler` produziert ein sinnvolles Bild mit dem Schema aus dem User-Beispiel.

### Phase 2 — Sektion-Builder & Presets (2-3 Tage)
- Alle anderen Builder-Nodes (Pose, Clothing, Environment, Camera, Lighting, Composition, SceneStyle, NegativePrompt)
- `PresetLoader` mit YAML-Parsing und `extends:`-Vererbung
- Erste Beispiel-Presets in `presets/`
- `qwen_chatml`- und `comma_tags`-Serializer

### Phase 3 — Wildcards & Variation (2 Tage)
- Integration mit `dynamicprompts`-Library (als Dependency)
- `WildcardInjector`-Node und transparente Auflösung im Aggregator
- `BatchVariator` mit Lock/Vary-Logik und Seed-Strategien
- `SidecarSaver` für JSON-Logs neben Output-Bildern

**Akzeptanzkriterium:** Ein Batch-Run mit 20 Variationen, bei dem Identity/Style/Camera gelocked sind und Pose/Outfit/Location variieren, produziert 20 unterschiedliche Bilder mit konsistentem Subjekt — alle mit Sidecar-JSON.

### Phase 4 — UX-Frontend (3-4 Tage)
- `LivePreview`-Node mit WebSocket-Bridge
- CodeMirror-basierter JSON-Editor in Builder-Nodes
- Lock/Vary-Toggle-Buttons im Frontend
- Statische Preset-Liste als erste Browser-Iteration

### Phase 5 — Polish & Modell-Adapter (2-3 Tage)
- `auto`-Modus im Serializer (Modell-Detection)
- Alle vier Beispiel-Workflows lauffähig dokumentiert
- README mit Screenshots
- Optionale `LLMEnhance`-Node mit Ollama-Anbindung
- Publishing-Vorbereitung: ComfyUI-Manager-Eintrag, GitHub-Actions für Tests

**Gesamt-Aufwand:** ~10–14 Tage Entwicklung mit Claude Code.

---

## 13. Dependencies

**Python:**
```toml
[project]
dependencies = [
    "pydantic>=2.5",
    "pyyaml>=6.0",
    "dynamicprompts>=0.31",  # für Wildcards, statt Eigen-Implementation
]
```

**JavaScript (Frontend):**
- CodeMirror 6 (oder Monaco) — JSON-Editor
- Prism.js — Syntax-Highlighting in Preview
- diff2html — Diff-Anzeige im BatchVariator

**Optional / Empfohlen** (für Endnutzer):
- `comfyui-dynamicprompts` (adieyal) — für Standalone-Wildcard-Features
- `ComfyUI-ZImagePowerNodes` (martin-rizzo) — für nahtlose Z-Image-Integration
- `comfyui-ollama` oder `comfyui-LLM-party` — für `LLMEnhance`

---

## 14. Test-Strategie

- **Unit-Tests** (`tests/`): Schema-Validation, Merge-Konflikte, Wildcard-Auflösung, alle vier Serializer
- **Workflow-Snapshot-Tests**: vier Beispiel-Workflows werden headless gerendert, Output-JSON wird gegen erwarteten Snapshot verglichen
- **Manual-Test-Checkliste** im README für UX-Features (LivePreview, Editor, Lock/Vary)

---

## 15. Future Extensions (nicht in v1.0)

- **Negative-Prompt-Builder mit Auto-Tags** basierend auf Modell (SDXL braucht andere Negativs als Flux)
- **Image-zu-Schema-Reverse**: ein Bild rein, ein gefüllter PROMPT_DICT raus (über VLM wie LLaVA oder Florence-2)
- **Style-Mixing**: zwei Presets mit Gewichtung mergen
- **A/B-Testing-Node**: zwei DICT-Varianten parallel rendern, im Frontend nebeneinander zeigen
- **Workflow-Templates**: PromptForge wird im ComfyUI-Manager als "Recipe" registriert, das ganze Workflows installiert

---

## 16. Lizenzempfehlung

**MIT** oder **Apache 2.0** — kompatibel mit ComfyUI-Ökosystem und den genutzten Dependencies.

---

## Anhang A: Beispiel — vollständiger PROMPT_DICT (basiert auf User-Schema)

```json
{
  "meta": {
    "schema_version": "1.0",
    "target_model": "z-image-turbo",
    "seed": 42,
    "label": "fashion_editorial_brutalist_v1"
  },
  "scene": {
    "overall_style": "photorealistic fashion editorial photography",
    "mood": "elegant, confident, modern minimalist",
    "color_palette": "monochrome accents on neutral concrete grays",
    "photography_technique": "fashion shoot, sharp focus on subject, shallow depth of field"
  },
  "environment": {
    "location": "modern brutalist concrete architecture exterior",
    "background": "large plain light-gray concrete wall with subtle geometric panel lines",
    "foreground": "wide concrete steps with visible texture",
    "time_of_day": "bright daytime",
    "weather": "clear, soft natural light"
  },
  "camera": {
    "angle": "low-angle shot from slightly below",
    "lens": "35mm prime",
    "perspective": "dynamic upward perspective",
    "aspect_ratio": "2:3"
  },
  "lighting": {
    "type": "bright natural daylight",
    "direction": "soft diffused overhead with subtle side highlights",
    "quality": "minimal shadows, high key"
  },
  "composition": {
    "framing": "full-body with emphasis on legs and upper body",
    "rule": "rule_of_thirds",
    "focus": "sharp on eyes, face, and key garment details",
    "depth": "three-layer: foreground steps, subject, background wall"
  },
  "subjects": [
    {
      "id": "subject_1",
      "identity": {
        "description": "{model_a|model_b|model_c}",
        "age_range": "20s",
        "build": "slender"
      },
      "hair": {
        "style": "long straight, center-parted",
        "color": "{blonde|brunette|black}",
        "length": "waist-length",
        "details": "healthy shine, slight movement"
      },
      "face": {
        "expression": "subtle confident gaze",
        "gaze_direction": "{direct at camera|slightly off-camera}",
        "makeup": "natural glamorous"
      },
      "pose": "__poses/seated_dynamic__",
      "clothing": {
        "outer_layer": {
          "type": "{form-fitting mini dress|tailored jumpsuit|midi dress}",
          "color": "{red|black|emerald|burgundy}",
          "material": "matte stretch fabric"
        },
        "footwear": {
          "type": "high stiletto",
          "color": "glossy patent leather"
        }
      },
      "visibility": "full"
    }
  ],
  "post_processing": {
    "negative_prompt": "blurry, low quality, distorted anatomy, extra limbs, watermark",
    "quality_tags": ["8k", "highly detailed", "professional photography", "sharp focus"]
  },
  "extras": {}
}
```

Mit aktiviertem BatchVariator (`vary_paths`: `subjects[0].clothing.outer_layer.color`, `subjects[0].pose`, `subjects[0].hair.color`) erzeugt das z.B. 12 verschiedene Bilder derselben Person in derselben Szene mit Pose/Haar/Outfit-Variation.

---

**Ende des Specs.** Fragen, Änderungswünsche oder gleich an Claude Code übergeben?
