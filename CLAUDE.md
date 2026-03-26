# CLAUDE.md — ComfyUI Custom Node Entwicklung

> Projektanleitung für Claude Code. Dieses Dokument definiert den Entwicklungs- und Test-Workflow
> für ComfyUI Custom Nodes — mit dem Ziel, Neustarts zu minimieren und schnelle Iterationszyklen zu ermöglichen.

---

## Projektstruktur

```
my-comfyui-nodes/
├── CLAUDE.md                    # ← Diese Datei
├── __init__.py                  # Node-Registrierung (NODE_CLASS_MAPPINGS)
├── nodes/
│   ├── __init__.py
│   ├── my_node.py               # Node-Implementierung
│   └── utils.py                 # Shared Utilities
├── web/                         # Frontend JS (optional)
│   └── js/
│       └── my_widget.js
├── tests/
│   ├── conftest.py              # pytest Fixtures + ComfyUI-Mocks
│   ├── mocks/
│   │   └── comfy_mocks.py       # Mock-Objekte für Tensor, IMAGE, MODEL etc.
│   ├── unit/
│   │   └── test_my_node.py      # Standalone Unit-Tests (kein ComfyUI nötig)
│   └── integration/
│       └── test_workflow.py      # Tests gegen laufende ComfyUI-Instanz
├── test_workflows/
│   └── test_my_node.json        # ComfyUI-Workflow JSON für manuelle/automatische Tests
├── requirements.txt
└── pyproject.toml
```

### Wichtige Konventionen

- **Node-Dateien** gehören nach `nodes/` — eine Datei pro Node oder thematischer Gruppe.
- **`__init__.py` im Root** enthält nur `NODE_CLASS_MAPPINGS` und `NODE_DISPLAY_NAME_MAPPINGS`.
- **Tests** sind strikt getrennt in `unit/` (ohne ComfyUI) und `integration/` (mit ComfyUI).
- **Test-Workflows** als `.json` exportieren und in `test_workflows/` ablegen.

---

## Node-Anatomie (V1 Schema — aktueller Standard)

```python
class MeinNode:
    """Jede Node braucht: CATEGORY, INPUT_TYPES, RETURN_TYPES, FUNCTION."""

    CATEGORY = "mein-paket/kategorie"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "stärke": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "maske": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("bild",)
    FUNCTION = "execute"

    def execute(self, image, stärke, maske=None):
        # Geschäftslogik hier — möglichst in eigene Funktionen auslagern
        result = self._verarbeite(image, stärke, maske)
        return (result,)  # Tuple!

    def _verarbeite(self, image, stärke, maske):
        """Kernlogik ausgelagert → separat testbar."""
        # ... Verarbeitung ...
        return image * stärke
```

### Registrierung in `__init__.py`

```python
from .nodes.my_node import MeinNode

NODE_CLASS_MAPPINGS = {
    "MeinNode": MeinNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeinNode": "Mein Node (Anzeigename)",
}
```

---

## V3 Schema (Zukunft — ab Mitte 2026 empfohlen)

V3 wird das neue Standard-Schema. Wesentliche Änderungen:
- `execute` wird `@classmethod` (stateless, kein `self`)
- Schema-Definition über `DEFINE_SCHEMA()` statt Dict-Konventionen
- Versionierte, stabile Public API (`from comfy_api.v0_0_3 import ComfyAPI`)
- Caching über `cls`-Parameter statt Instanz-Variablen

```python
# V3 Beispiel (Vorschau — kann sich noch ändern)
from comfy_api.v3 import Schema, ImageInput, ImageOutput

class MeinNodeV3:
    @classmethod
    def DEFINE_SCHEMA(cls):
        return Schema(
            node_id="MeinNode_v1",
            inputs=[ImageInput("image")],
            outputs=[ImageOutput("processed_image")]
        )

    @classmethod
    def execute(cls, image):
        return image * 1.5
```

**Empfehlung:** Neue Nodes jetzt V1-kompatibel schreiben, aber Logik in statische/classmethod-Funktionen auslagern, um die V3-Migration zu vereinfachen. Die `execute`-Methode sollte kein `self`-State nutzen.

---

## Entwicklungs-Workflow

### Schritt 1: Hot Reload installieren (einmalig)

```bash
cd /pfad/zu/ComfyUI/custom_nodes
git clone https://github.com/LAOGOU-666/ComfyUI-LG_HotReload.git
```

**Was es tut:** Überwacht `custom_nodes/` auf Dateiänderungen und lädt geänderte Module automatisch nach. Nach dem Speichern reicht ein Node-Reset oder Browser-Refresh — kein ComfyUI-Neustart nötig.

**Einschränkungen:**
- Neue Inputs/Outputs erfordern manchmal einen vollen Neustart.
- Syntaxfehler im Code unterbrechen den Reload-Prozess.
- Für komplexe strukturelle Änderungen (neue Nodes registrieren) → Neustart nötig.

**Alternative:** [ComfyUI-HotReloadHack](https://github.com/logtd/ComfyUI-HotReloadHack) (braucht nur `pip install watchdog`).

### Schritt 2: Entwicklungszyklus

```
┌─────────────────────────────────────────────────────────┐
│  Schneller Zyklus (90% der Arbeit):                     │
│                                                         │
│  Code ändern → Speichern → pytest → Browser refreshen   │
│  (< 5 Sekunden)                                         │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Voller Zyklus (nur bei strukturellen Änderungen):      │
│                                                         │
│  Code ändern → ComfyUI neustarten → Browser refreshen   │
│  (30-60 Sekunden)                                       │
└─────────────────────────────────────────────────────────┘
```

---

## Test-Strategie

### Ebene 1: Unit-Tests (KEIN ComfyUI nötig)

Die schnellste Feedback-Schleife. Testet die reine Logik einer Node isoliert.

**`tests/conftest.py`** — Zentrale Fixtures und Mocks:

```python
import pytest
import torch
import sys
from unittest.mock import MagicMock

# ComfyUI-Module mocken, damit Imports nicht fehlschlagen
def mock_comfy_modules():
    """Mockt ComfyUI-Importe, sodass Nodes ohne laufendes ComfyUI importiert werden können."""
    modules = [
        "comfy", "comfy.model_management", "comfy.utils",
        "comfy.sd", "comfy.samplers", "comfy.sample",
        "folder_paths", "server", "execution",
    ]
    for mod in modules:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

mock_comfy_modules()


# ──── Tensor-Fixtures ────

@pytest.fixture
def dummy_image():
    """Standard ComfyUI IMAGE-Tensor: [B, H, W, C] float32 0-1."""
    return torch.rand(1, 512, 512, 3, dtype=torch.float32)

@pytest.fixture
def dummy_image_batch():
    """Batch mit 4 Bildern."""
    return torch.rand(4, 512, 512, 3, dtype=torch.float32)

@pytest.fixture
def dummy_mask():
    """Standard ComfyUI MASK-Tensor: [B, H, W] float32 0-1."""
    return torch.rand(1, 512, 512, dtype=torch.float32)

@pytest.fixture
def dummy_latent():
    """Standard ComfyUI LATENT-Dict."""
    return {"samples": torch.rand(1, 4, 64, 64, dtype=torch.float32)}

@pytest.fixture
def small_image():
    """Kleines Bild für schnelle Tests."""
    return torch.rand(1, 64, 64, 3, dtype=torch.float32)
```

**`tests/unit/test_my_node.py`** — Beispiel Unit-Tests:

```python
import pytest
import torch

# conftest.py mockt ComfyUI automatisch → Import funktioniert
from nodes.my_node import MeinNode


class TestMeinNode:
    """Unit-Tests für MeinNode — laufen ohne ComfyUI."""

    def test_input_types_vollstaendig(self):
        """Prüft, dass INPUT_TYPES korrekt definiert sind."""
        inputs = MeinNode.INPUT_TYPES()
        assert "required" in inputs
        assert "image" in inputs["required"]
        assert inputs["required"]["image"][0] == "IMAGE"

    def test_return_types(self):
        """Prüft RETURN_TYPES-Definition."""
        assert MeinNode.RETURN_TYPES == ("IMAGE",)

    def test_execute_basic(self, dummy_image):
        """Grundlegender Funktionstest."""
        node = MeinNode()
        result = node.execute(image=dummy_image, stärke=1.0)

        assert isinstance(result, tuple), "execute() muss Tuple zurückgeben"
        assert len(result) == len(MeinNode.RETURN_TYPES)
        assert result[0].shape == dummy_image.shape

    def test_execute_staerke_null(self, dummy_image):
        """Stärke 0 → Ergebnis sollte Null-Tensor sein."""
        node = MeinNode()
        result = node.execute(image=dummy_image, stärke=0.0)
        assert torch.allclose(result[0], torch.zeros_like(dummy_image))

    def test_execute_mit_maske(self, dummy_image, dummy_mask):
        """Test mit optionaler Maske."""
        node = MeinNode()
        result = node.execute(image=dummy_image, stärke=1.0, maske=dummy_mask)
        assert result[0].shape == dummy_image.shape

    def test_batch_verarbeitung(self, dummy_image_batch):
        """Batch-Bilder müssen korrekt verarbeitet werden."""
        node = MeinNode()
        result = node.execute(image=dummy_image_batch, stärke=1.0)
        assert result[0].shape[0] == 4, "Batch-Größe muss erhalten bleiben"

    def test_output_wertebereich(self, dummy_image):
        """Ausgabewerte müssen im gültigen Bereich liegen."""
        node = MeinNode()
        result = node.execute(image=dummy_image, stärke=1.0)
        assert result[0].min() >= 0.0, "Werte dürfen nicht negativ sein"
        assert result[0].max() <= 1.0, "Werte dürfen 1.0 nicht überschreiten"

    def test_dtype_erhalten(self, dummy_image):
        """Ausgabe-dtype muss float32 bleiben."""
        node = MeinNode()
        result = node.execute(image=dummy_image, stärke=0.5)
        assert result[0].dtype == torch.float32


class TestMeinNodeEdgeCases:
    """Randfälle und Fehlerbehandlung."""

    def test_einzelnes_pixel(self):
        """1x1 Bild darf keinen Fehler werfen."""
        node = MeinNode()
        tiny = torch.rand(1, 1, 1, 3)
        result = node.execute(image=tiny, stärke=1.0)
        assert result[0].shape == (1, 1, 1, 3)

    def test_grosse_staerke(self, dummy_image):
        """Hohe Stärke → Werte werden geclampt."""
        node = MeinNode()
        result = node.execute(image=dummy_image, stärke=10.0)
        assert result[0].max() <= 1.0, "Clamp auf [0,1] erwartet"
```

**Tests ausführen:**

```bash
# Alle Unit-Tests (schnell, < 2 Sekunden)
cd /pfad/zu/my-comfyui-nodes
pytest tests/unit/ -v

# Einzelne Datei
pytest tests/unit/test_my_node.py -v

# Nur fehlgeschlagene Tests wiederholen
pytest tests/unit/ --lf

# Mit Coverage
pytest tests/unit/ --cov=nodes --cov-report=term-missing
```

### Ebene 2: Integration-Tests (gegen laufendes ComfyUI)

Testet die Node im echten ComfyUI-Kontext über die API.

**`tests/integration/test_workflow.py`:**

```python
import pytest
import json
import requests
import time

COMFY_URL = "http://127.0.0.1:8188"


def comfy_erreichbar():
    try:
        r = requests.get(f"{COMFY_URL}/system_stats", timeout=2)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


@pytest.fixture
def comfy_api():
    """Prüft ob ComfyUI läuft, sonst Test überspringen."""
    if not comfy_erreichbar():
        pytest.skip("ComfyUI nicht erreichbar auf :8188")
    return COMFY_URL


def lade_workflow(pfad):
    """Lädt einen exportierten Workflow."""
    with open(pfad) as f:
        return json.load(f)


def sende_prompt(api_url, workflow):
    """Sendet Workflow an ComfyUI und wartet auf Ergebnis."""
    resp = requests.post(f"{api_url}/prompt", json={"prompt": workflow})
    assert resp.status_code == 200, f"Prompt fehlgeschlagen: {resp.text}"
    return resp.json()


class TestMeinNodeIntegration:

    def test_workflow_laeuft_durch(self, comfy_api):
        """Prüft, dass der Test-Workflow ohne Fehler ausgeführt wird."""
        workflow = lade_workflow("test_workflows/test_my_node.json")
        result = sende_prompt(comfy_api, workflow)
        assert "error" not in result, f"Workflow-Fehler: {result}"
        assert "prompt_id" in result

    def test_node_ist_registriert(self, comfy_api):
        """Prüft, dass die Node im System bekannt ist."""
        resp = requests.get(f"{comfy_api}/object_info/MeinNode")
        assert resp.status_code == 200
        info = resp.json()
        assert "MeinNode" in info

    def test_node_inputs_korrekt(self, comfy_api):
        """Validiert Input-Definition über die API."""
        resp = requests.get(f"{comfy_api}/object_info/MeinNode")
        info = resp.json()["MeinNode"]
        assert "image" in info["input"]["required"]
```

**Integration-Tests ausführen:**

```bash
# Nur Integration-Tests (ComfyUI muss laufen!)
pytest tests/integration/ -v

# Alle Tests zusammen
pytest tests/ -v

# Timeout für langsame Workflows
pytest tests/integration/ -v --timeout=120
```

### Ebene 3: Manueller Schnelltest (im Browser)

Für visuellen Feedback und UI-Debugging:

1. ComfyUI mit Hot-Reload starten
2. Test-Workflow laden: `test_workflows/test_my_node.json`
3. Code ändern → Speichern → Node resetten → Queue Prompt
4. Ergebnis prüfen

---

## Nützliche pytest-Patterns für Nodes

### Parametrisierte Tests

```python
@pytest.mark.parametrize("staerke,erwartet_min,erwartet_max", [
    (0.0, 0.0, 0.0),
    (0.5, 0.0, 0.5),
    (1.0, 0.0, 1.0),
])
def test_staerke_skalierung(dummy_image, staerke, erwartet_min, erwartet_max):
    node = MeinNode()
    result = node.execute(image=dummy_image, stärke=staerke)
    assert result[0].min() >= erwartet_min
    assert result[0].max() <= erwartet_max
```

### Performance-Tests

```python
import time

def test_performance_akzeptabel(dummy_image):
    """Node muss unter 100ms für ein 512x512 Bild laufen."""
    node = MeinNode()
    start = time.perf_counter()
    for _ in range(10):
        node.execute(image=dummy_image, stärke=1.0)
    elapsed = (time.perf_counter() - start) / 10
    assert elapsed < 0.1, f"Zu langsam: {elapsed:.3f}s pro Durchlauf"
```

### GPU-Tests (optional)

```python
@pytest.fixture
def gpu_image():
    if not torch.cuda.is_available():
        pytest.skip("Kein CUDA verfügbar")
    return torch.rand(1, 512, 512, 3, dtype=torch.float32).cuda()

def test_gpu_kompatibel(gpu_image):
    node = MeinNode()
    result = node.execute(image=gpu_image, stärke=1.0)
    assert result[0].is_cuda, "Ergebnis muss auf GPU bleiben"
```

---

## Debugging-Tipps

### Console-Logging in Nodes

```python
import logging

logger = logging.getLogger("MeinNode")

class MeinNode:
    def execute(self, image, stärke, maske=None):
        logger.info(f"Input Shape: {image.shape}, Stärke: {stärke}")
        logger.info(f"Maske: {'ja' if maske is not None else 'nein'}")
        # ... Verarbeitung ...
        logger.info(f"Output Shape: {result.shape}, Range: [{result.min():.3f}, {result.max():.3f}]")
        return (result,)
```

### Tensor-Inspektion (Hilfsfunktion)

```python
def tensor_info(name, t):
    """Gibt Tensor-Infos für Debugging aus."""
    if t is None:
        print(f"  {name}: None")
        return
    print(f"  {name}: shape={t.shape}, dtype={t.dtype}, "
          f"range=[{t.min().item():.4f}, {t.max().item():.4f}], "
          f"mean={t.mean().item():.4f}, device={t.device}")
```

### Häufige Fehlerquellen

| Problem | Ursache | Lösung |
|---------|---------|--------|
| Node erscheint nicht | `NODE_CLASS_MAPPINGS` fehlt/falsch | `__init__.py` prüfen, Neustart |
| `execute() got unexpected keyword argument` | Falscher `FUNCTION`-Name | `FUNCTION` muss exakt der Methodenname sein |
| Rote Node nach Änderung | Hot Reload hat Syntaxfehler | Terminal/Console auf Fehler prüfen |
| Leeres Ergebnis | Kein Tuple returned | `return (result,)` nicht `return result` |
| Shape Mismatch | Falsches Tensor-Format | ComfyUI: IMAGE=[B,H,W,C], MASK=[B,H,W], LATENT=[B,C,H,W] |
| Node-Inputs verschwinden | Input-Typ nicht erkannt | Standardtypen: IMAGE, MASK, LATENT, MODEL, CLIP, VAE, CONDITIONING, INT, FLOAT, STRING |

---

## Befehle für Claude Code

### Neue Node erstellen

```
Erstelle eine neue Node "BildVerdunkler" in nodes/bild_verdunkler.py mit:
- Input: IMAGE (required), FLOAT "intensität" (0.0-1.0, default 0.5)
- Output: IMAGE
- Logik: Multipliziert Bildwerte mit (1 - intensität)
- Registriere in __init__.py
- Schreibe Unit-Tests in tests/unit/test_bild_verdunkler.py
```

### Tests schreiben

```
Schreibe Unit-Tests für die Node in nodes/mein_filter.py.
Nutze die Fixtures aus conftest.py.
Teste: Basis-Funktionalität, Edge Cases, Batch-Verarbeitung, Wertebereich.
```

### Bestehende Node debuggen

```
Die Node "MeinNode" gibt falsche Werte aus.
Füge tensor_info() Debugging hinzu und schreibe einen gezielten Test
der das erwartete Ergebnis für einen bekannten Input prüft.
```

### Test-Workflow generieren

```
Generiere einen minimalen ComfyUI-Workflow als JSON in test_workflows/
der MeinNode mit einem LoadImage-Node verbindet und das Ergebnis an PreviewImage ausgibt.
```

---

## Projekt-Setup (einmalig)

```bash
# 1. Projekt-Verzeichnis in custom_nodes erstellen
cd /pfad/zu/ComfyUI/custom_nodes
mkdir my-comfyui-nodes && cd my-comfyui-nodes

# 2. Struktur anlegen
mkdir -p nodes tests/{unit,integration,mocks} test_workflows web/js

# 3. Dev-Dependencies
pip install pytest pytest-cov pytest-timeout torch --break-system-packages

# 4. Hot Reload (falls noch nicht installiert)
cd /pfad/zu/ComfyUI/custom_nodes
git clone https://github.com/LAOGOU-666/ComfyUI-LG_HotReload.git

# 5. pyproject.toml für pytest-Config
cat > pyproject.toml << 'EOF'
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
markers = [
    "slow: Langsame Tests (> 5s)",
    "gpu: Tests die CUDA brauchen",
    "integration: Braucht laufendes ComfyUI",
]
EOF
```

---

## Schnellreferenz: ComfyUI Tensor-Formate

| Typ | Shape | dtype | Wertebereich | Bemerkung |
|-----|-------|-------|-------------|-----------|
| IMAGE | `[B, H, W, 3]` | float32 | 0.0 – 1.0 | RGB, Batch-first |
| MASK | `[B, H, W]` | float32 | 0.0 – 1.0 | Kein Channel-Dim |
| LATENT | `{"samples": [B, 4, H/8, W/8]}` | float32 | beliebig | Dict mit "samples" Key |
| CONDITIONING | `[(tensor, dict)]` | — | — | Liste von Tupeln |

---

## Checkliste vor dem Release

- [ ] Alle Unit-Tests grün: `pytest tests/unit/ -v`
- [ ] Integration-Test mit echtem ComfyUI durchgeführt
- [ ] `NODE_CLASS_MAPPINGS` und `NODE_DISPLAY_NAME_MAPPINGS` korrekt
- [ ] `INPUT_TYPES` haben sinnvolle Defaults und Min/Max
- [ ] `RETURN_TYPES` und `RETURN_NAMES` stimmen überein
- [ ] Edge Cases getestet (leere Inputs, extreme Werte, verschiedene Batch-Größen)
- [ ] Tensor-Formate korrekt (IMAGE=[B,H,W,C], nicht [B,C,H,W])
- [ ] `execute()` gibt immer ein Tuple zurück
- [ ] Logging/Debug-Prints entfernt oder auf logger umgestellt
- [ ] requirements.txt aktuell
