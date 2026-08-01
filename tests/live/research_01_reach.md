# Recherche 01 — Reichweite: alle textführenden Flächen erfassen

Externe Recherche (Web, Stand August 2026) zum offenen Hauptproblem: Flächen, für die der
Selector keine Region ausgibt, behalten ihre Fantasieschrift. Kein Code geändert, keine
ComfyUI-Läufe, keine GPU belegt.

---

## 0 · Was vor der Recherche im Repo nachgemessen wurde

Drei Befunde, die die Bewertung der Ansätze verschieben. Alle direkt aus dem Repo bzw.
dem venv, nicht aus dem Web:

**(a) Das OCR-Backend ist auf dieser Maschine gar nicht installiert.**

```
backend_status()["onnx"]["reason"]
  "no OCR model directory found - searched '<root>/ocr' for every registered 'onnx'
   model root, '<ComfyUI>/models/onnx/ocr', and outfit_config.ini [models] ocr_path"
backend_status()["easyocr"]["reason"]
  "the optional 'easyocr' package is not installed"
```

Passend dazu steht in jedem Selector-Log der Diagnoseläufe `OCR backends available: none`
und in jeder Regionzeile `slop=0.00 (unknown) text=''`. Die komplette DBNet-Maschinerie in
`nodes/utils/ocr_backend.py` (Detektion, Unclip, Quad-Warp, CTC-Recognizer) ist derzeit
toter Code. `scripts/fetch_ocr_models.py` existiert und holt ~16 MB — das ist der billigste
Hebel im ganzen Projekt und er ist nicht gezogen.

**(b) Die Detektion läuft auf dieser Maschine garantiert nicht auf der GPU — und das ist gut.**

```
onnxruntime 1.24.1   providers: ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
torch 2.8.0+cu128    NVIDIA GeForce RTX 5090
```

Kein `CUDAExecutionProvider`. Das deckt sich mit dem allgemeinen Stand: die offiziellen
`onnxruntime-gpu`-Wheels liefern (Stand Anfang 2026) keine sm_120-Kernel für Blackwell, es
gibt nur Community-Builds ([Issue #26177](https://github.com/microsoft/onnxruntime/issues/26177),
[Natfii/onnxruntime-gpu-blackwell](https://github.com/Natfii/onnxruntime-gpu-blackwell)).
Für uns ist das ein **Vorteil**: jede ONNX-Detektion läuft auf CPU/iGPU und kann per
Konstruktion nicht mit dem geladenen Krea 2 um VRAM streiten.

**(c) SAM3-Grounding sättigt, und zwar deutlich unterhalb dessen, was der Zensus zählt.**

Aus `diag_street_run.txt` (real_street.png, 1280×768):

| Einstellung | Regionen | Bildfläche abgedeckt |
|---|---|---|
| `threshold_scale 0.7, min_height 18, max_regions 4` (Suite-Default) | 4 | **8,5 %** |
| `threshold_scale 0.7, min_height 18, max_regions 100` | 23 | — |
| `threshold_scale 0.35, min_height 8, max_regions 100` | 27 | **23,1 %** |
| dito, `merge_iou 1.0, min_area_ratio 0.0` | 80 | — |

Alle 23 bzw. 27 Treffer tragen die Klasse `sign`. SAM3 groundet **Objekte**, keine
Textzeilen. Weiter aufdrehen bringt ab `0.35 / 8 px` nichts mehr an Fläche, nur noch
Duplikate (80 Regionen bei gleicher Abdeckung). Das ist die eigentliche Diagnose: die
Erfassungslücke lässt sich mit den vorhandenen Reglern **nicht** schließen.

Ergänzend, weil es die Kostenrechnung aller Ansätze betrifft — zwei Stellen im vorhandenen
Detektionspfad arbeiten aktuell gegen genau diesen Anwendungsfall:

- `_det_preprocess(..., limit_side_len=960)` skaliert 1280×768 auf 960 herunter (0,75×).
  Eine 24-px-Zeile wird zu 18 px, eine 20-px-Zeile zu 15 px — unterhalb der Größe, bei der
  DBNet zuverlässig anspricht.
- `_run_onnx` verwirft **jede Box, deren Recognizer nichts liest** (`if not text: continue`).
  Das ist exakt der Pseudo-Glyphen-Fall, um den es hier geht: DBNet sagt „hier ist Text",
  der CTC-Decoder liefert leer — und die Box fällt raus, statt gemeldet zu werden.

Beides sind Einzeiler, aber ohne sie ist ein Sweep wertlos.

---

## Ansatz A — Textzeilen-Sweep (DBNet, gekachelt) + optische Auflösungsgrenze

### Name + Quelle

- Detektion: **DBNet / DB++** — [Liao et al., *Real-time Scene Text Detection with
  Differentiable Binarization*](https://github.com/MhLiao/DB); im Repo bereits als
  PP-OCRv4-ONNX-Pfad implementiert. Aktueller: **PP-OCRv5**
  ([PaddlePaddle/PP-OCRv5_mobile_det](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det),
  [ONNX-Export](https://huggingface.co/monkt/paddleocr-onnx), Apache-2.0), seit Juni 2026
  **PP-OCRv6** ([arXiv 2606.13108](https://arxiv.org/abs/2606.13108), 1,5–34,5 M Parameter,
  +4,6 Punkte Detection-Hmean gegenüber v5 — Gewichte-Verfügbarkeit im Paper nicht genannt,
  *unbestätigt*).
- Kachelung: **SAHI** ([obss/sahi](https://github.com/obss/SAHI)) als Muster, nicht als
  Abhängigkeit — überlappende Kacheln, Merge über NMS. Für unsere Bildgrößen reichen 30
  Zeilen eigener Code.
- Behandlung: keine Fremdquelle. Physikalisches Argument, gestützt auf die
  OCR-Auflösungsliteratur: Erkennungsgenauigkeit bricht unterhalb ~7 px Zeichenhöhe
  scharf ein, empfohlen werden 20–30 px x-Höhe
  ([Cognex](https://support.cognex.com/en/help-articles/in-sight-what-is-the-minimum-pixel-resolution-recommended-to-ocr-human-readable-text),
  [Nuance](https://nuance.custhelp.com/app/answers/detail/a_id/6346/)).

### Was es hier genau tun würde

**Im Selector**, als zweite Erfassungsquelle neben SAM3:

1. Bild in überlappende Kacheln (z. B. 3×2 bei 1280×768, 20 % Überlappung), jede Kachel bei
   ihrer **nativen** Auflösung durch das DBNet — nicht das ganze Bild auf 960 px gestaucht.
   Damit sieht der Detektor eine 20-px-Zeile als 20-px-Zeile.
2. `_run_onnx` so ändern, dass Boxen **mit leerem Recognizer-Ergebnis erhalten bleiben** und
   als `text=""` gemeldet werden. Genau die sind der Slop.
3. Quads über Kachelgrenzen zusammenführen, Quads verwerfen, die zu ≥ 50 % in einer bereits
   vorhandenen SAM3-Region liegen. Rest = **Waisentext**.
4. Jeder Waisenquad bekommt eine Route:
   - Kurzseite ≥ ~24 px **und** lokale Schärfe hoch → normale Kette (Proposer/Detailer),
     also ein echtes Wort.
   - alles andere → **optische Behandlung**, kein Sampler.

**Die optische Behandlung** (Ansatz-Kern, komplett ohne Diffusion): Patch ausschneiden,
gemessene Strichhöhe bestimmen (`measure_ink_height` existiert bereits in `utils/glyph.py`),
so weit herunterskalieren, dass die Strichhöhe unter ~5 px fällt, mit passendem Tiefpass
(Defokus-Scheibe, nicht Gauß — Gauß sieht nach Weichzeichner aus, eine Kreisscheibe nach
Objektiv) wieder hochskalieren, das Korn der Umgebung wieder aufmodulieren, mit weicher,
der Flächengeometrie folgender Kante einkomponieren.

Das Ergebnis fällt in die Zensus-Kategorie **`unreadable`** — „man sieht dass da Text ist,
kann aber keine Buchstaben ausmachen" — und die zählt laut Abnahmekriterium **ausdrücklich
nicht** als Fehler. Das ist der springende Punkt: der billigste Handler ist gleichzeitig ein
legitimer.

### Kosten

| Posten | Wert |
|---|---|
| Gewichte | PP-OCRv4-Det 4,7 MB + Rec 10,9 MB + Keys (bereits von `scripts/fetch_ocr_models.py` abgedeckt), optional PP-OCRv5-Mobile-Det-ONNX ~84 MB |
| Pakete | `onnxruntime` ✓, `pyclipper` ✓, `shapely` ✓, `cv2` ✓ — **alle schon im venv** |
| Laufzeit | Detektion CPU: pro Kachel grob 0,1–0,3 s, 6 Kacheln ≈ **1–2 s**; optische Behandlung < 50 ms pro Quad. **Nicht gemessen** — muss beim Bau nachgewiesen werden |
| VRAM | **0 MB.** Läuft nachweislich auf CPU/OpenVINO (siehe 0b). Kein Konflikt mit Krea 2 |

### Erwarteter Effekt auf die Messgröße

- Beseitigt aller Voraussicht nach: `PAXTRES`, `SAMYERK`, `POJUITRDOT`, `FUAVNLE`,
  `EHRITVITYDUC`, `PAIXTROE`, `SANE SEKRI`, `LHNC` — die Hintergrund-Schaufenster der
  Straßenszene. Die liegen ohnehin hinter der Schärfeebene; Defokus ist dort nicht nur
  billig, sondern **physikalisch richtig**.
- Beseitigt **nicht** zuverlässig: die Pinnwand-Zettel (`H Ohiec Sabl`, `Piatda Zook`,
  `Bod indiuals`). Die stehen in derselben Schärfeebene wie der Zieltext. Weichzeichnen wäre
  dort physikalisch falsch und würde als Grafikartefakt gelesen. Diese Fälle brauchen Route
  „echtes Wort" (≥ 24 px) oder Ansatz B.
- Erzeugt **keinen** neuen Slop, weil kein Sampler beteiligt ist.

Grobe Erwartung an der letzten Messung: street A von 11 → ~3, street B 9 → ~3, street C 1 → 0,
board unverändert. Das reicht rechnerisch für street C und schiebt A/B in Reichweite, aber
board bleibt offen. **Das ist eine Schätzung, keine Messung.**

### Aufwand + was schiefgehen kann

**1,5 Halbtage.** 0,5 für Modelle holen + die zwei Einzeiler + Kachelung, 1,0 für die
optische Behandlung und den Router.

Risiken:
- **Der Sweep findet die Slop-Strings nicht.** DBNet ist auf echter Schrift trainiert; ob es
  auf Krea-Fantasieglyphen genauso anspricht, ist plausibel (DBNet lernt „Textartigkeit" aus
  Strichbild, nicht Wörter), aber ich habe **keine Publikation gefunden, die das misst** —
  explizit *unbestätigt*. Das ist der Grund, warum dieser Ansatz zuerst gebaut werden muss:
  er beantwortet die Frage in einem einzigen billigen Lauf über `selector_probe.py`.
- **Ein weichgezeichneter Fleck im scharfen Vordergrund** liest sich als Fehler, nicht als
  Ferne. Der Schärfe-Gate im Router ist nicht optional.
- **Das VLM nennt einen halbherzigen Schmier trotzdem `gibberish`.** Wenn Buchstabenumrisse
  auch nur ansatzweise überleben, ist nichts gewonnen. Lieber deutlich zu weit
  herunterskalieren als knapp.
- Der Nutzer will vorhandene Glyphen **nicht** als maßgeblichen Input. Hier ist die
  Abgrenzung wichtig: DBNet nutzt Buchstabenformen zum **Finden**, aber an keiner Stelle als
  generativen Input — die Formen werden zerstört, nicht nachgezeichnet. Wer auch das
  vermeiden will, ersetzt die Detektionsquelle durch VLM-Grounding (siehe „Variante" unten),
  bei schlechterer Lokalisierung.

**Variante ohne jeden Glyphenbezug:** Qwen3-VL kann 2D-Grounding
([Qwen3-VL Technical Report, arXiv 2511.21631](https://arxiv.org/pdf/2511.21631)), das LM
Studio hängt ohnehin schon in der Kette. Eine Anfrage „liste jede textführende Fläche mit
Bounding Box" liefert semantisch begründete Regionen statt strichbasierter. Kosten: eine
zusätzliche VLM-Anfrage pro Bild (2–5 s), dafür VRAM auf der LM-Studio-Seite. Die
Lokalisierungsgenauigkeit von VLM-Boxen liegt deutlich unter der eines Detektors — für
20-px-Zeilen im Hintergrund vermutlich zu grob. *Unbestätigt für Kleinschrift.*

---

## Ansatz B — Textzeilen-Sweep + LaMa-Löschung (Schrift verschwindet ganz)

### Name + Quelle

- **LaMa** — [Suvorov et al., *Resolution-robust Large Mask Inpainting with Fourier
  Convolutions*, WACV 2022](https://arxiv.org/abs/2109.07161),
  [advimman/lama](https://github.com/advimman/lama).
  Gewichte: `big-lama.pt`, TorchScript, **~206 MB**
  ([JosephCatrambone/big-lama-torchscript](https://huggingface.co/JosephCatrambone/big-lama-torchscript),
  [Sanster/models Release](https://github.com/Sanster/models/releases/tag/add_big_lama)).
  Python-Paket: [`simple-lama-inpainting`](https://pypi.org/project/simple-lama-inpainting)
  (nicht im venv), oder direkt über
  [Acly/comfyui-inpaint-nodes](https://github.com/Acly/comfyui-inpaint-nodes) (`LoadInpaintModel`
  + `InpaintWithModel`, Modell nach `ComfyUI/models/inpaint`).
- Billigste Stufe: **`cv2.inpaint`** (Telea/Navier-Stokes). Genau das macht der einzige
  existierende ComfyUI-Textentferner,
  [huwenkai26/comfyui-remove-text](https://github.com/huwenkai26/comfyui-remove-text): eigenes
  4-MB-DBNet-ONNX + OpenCV-Repair, Parameter `short_size` (Default 960) und `inpaint_radius`.
  **0 Stars, 1 Fork, keine Lizenz** — als Referenzimplementierung lesenswert, nicht als
  Abhängigkeit.
- Spezialisierte Scene-Text-Removal-Netze, falls LaMa nicht reicht: **ViTEraser**
  (AAAI 2024, [arXiv 2306.12106](https://arxiv.org/abs/2306.12106),
  [Repo](https://github.com/shannanyinxiang/ViTEraser)), **CTRNet**
  ([arXiv 2207.10273](https://arxiv.org/pdf/2207.10273)), **DeepEraser** (TMM 2024,
  nur 1,4 M Parameter, [Repo](https://github.com/fh2019ustc/DeepEraser)).
- Neuestes und qualitativ bestes: **OSOR — One-Step Diffusion Inpainting for Effect-Aware
  Object Removal** ([arXiv 2606.28094](https://arxiv.org/abs/2606.28094), Juni 2026,
  [Repo](https://github.com/Zhouqm-Git/osor),
  [Gewichte](https://huggingface.co/QinmingZhou/OSOR)). Ein Schritt, 1024² in unter 1 s auf
  A100, eigener **TextEraseBench** (185 Bilder, Zieltext-Regionen). Zwei Familien:
  OSOR-FLUX-Fill (**nicht-kommerzielle Lizenz**) und OSOR-SDXL-Inpainting (CreativeML
  OpenRAIL++-M). **Keine ComfyUI-Node.**

### Was es hier genau tun würde

Erfassung identisch zu Ansatz A. Unterschied ist nur die Behandlung des Waisentexts: statt
unlesbar zu machen wird die Schrift **entfernt** und die Fläche rekonstruiert.

- **Im Selector oder als eigene Node** zwischen Selector und Detailer: Waisenquads leicht
  dilatieren → binäre Maske → LaMa auf dem Bildausschnitt (Crop mit Kontext, damit LaMa
  Material zum Fortsetzen hat) → zurückkomponieren.
- Für sehr kleine Quads (< 12 px Kurzseite) genügt `cv2.inpaint` mit Radius 3 — bei der
  Größe ist der Unterschied zu LaMa nicht sichtbar und es kostet Millisekunden.

Der entscheidende strukturelle Vorteil: **LaMa kann per Konstruktion keine Buchstaben
malen.** Es ist ein FFC-GAN ohne Textprior und ohne Prompt. Es gibt keinen Mechanismus, über
den neue Pseudoschrift entstehen könnte — anders als bei jedem Diffusions-Inpainting.

### Kosten

| Posten | Wert |
|---|---|
| Gewichte | `big-lama.pt` 206 MB (oder 0 MB für den reinen `cv2.inpaint`-Pfad) |
| Pakete | `simple-lama-inpainting` **oder** nur `torch` + eigenes TorchScript-Laden; `cv2` ✓ |
| Laufzeit | LaMa 512²: ~0,1 s GPU / ~1–2 s CPU. Bei 5–15 Waisenquads pro Bild und Crop-Verarbeitung: **1–4 s**. `cv2.inpaint`-Pfad: < 100 ms gesamt |
| VRAM | LaMa auf GPU ~1–2 GB — **Konfliktrisiko mit Krea 2**, aber vermeidbar: LaMa kann auf CPU laufen (langsamer, aber unkritisch), oder wird nach Gebrauch entladen. `cv2.inpaint`: 0 MB |

OSOR wäre qualitativ besser, kostet aber ein zweites mehrere-GB-Diffusionsmodell neben
Krea 2 auf derselben Karte, hat keine ComfyUI-Anbindung und die bessere Variante ist
nicht-kommerziell lizenziert. Für dieses Projekt: **notieren, nicht bauen.**

### Erwarteter Effekt auf die Messgröße

- Beseitigt **alle** genannten Slop-Strings, auch die Pinnwand-Zettel — das ist der
  Unterschied zu Ansatz A. Es gibt keine Schärfeebenen-Ausrede, die Schrift ist einfach weg.
- Preis: eine Pinnwand ohne Zettelbeschriftung und ein Schaufenster ohne Beschriftung. Ob
  das als „Foto" durchgeht, hängt vom Motiv ab. Bei einer Pinnwand, deren Zweck Zettel sind,
  ist das inhaltlich fragwürdig.
- **Bekanntes Fehlerbild:** LaMa erzeugt auf großen Masken wiederkehrende Textur statt
  Struktur; Big-LaMa mildert das mit 18 Residual-Blöcken, behebt es nicht
  ([arXiv 2206.13644](https://arxiv.org/pdf/2206.13644) — Refinement-Verfahren mit 15
  Iterationen pro Skala, das brauchen wir für Textzeilen nicht). In der Literatur zu
  Objektentfernung wird LaMa außerdem „ghosty artifacts" durch fehlenden generativen Prior
  attestiert ([RePainter, arXiv 2510.07721](https://arxiv.org/pdf/2510.07721)). Auf
  Textzeilen — dünne, längliche Masken — ist LaMa dagegen im günstigsten Teil seines
  Arbeitsbereichs.

### Aufwand + was schiefgehen kann

**2 Halbtage** (1,5 wenn nur der `cv2.inpaint`-Pfad, weil dann Modellverwaltung und
Entladelogik entfallen).

Risiken:
- **Skepsis gegenüber den STR-Netzen ist angebracht.** ViTEraser/CTRNet/DeepEraser sind
  praktisch ausschließlich auf **SCUT-EnsText** und **SCUT-Syn** evaluiert. Die OTR-Arbeit
  ([arXiv 2510.02787](https://arxiv.org/html/2510.02787)) hält SCUT-EnsText drei konkrete
  Mängel vor: rund 8 % der Nicht-Text-Pixel weichen durch die manuelle Photoshop-Bearbeitung
  vom Original ab (PSNR 42,7 dB statt ~148 dB bei identischen Bildern); der Text liegt „auf
  relativ gleichförmigem Hintergrund", also nur leicht zu inpaintende Fälle; und PSNR/SSIM
  bestrafen überzeugende Rekonstruktionen, die vom Ground Truth abweichen. Übersetzt: die
  Bestenlisten sagen wenig über Fotos mit Schaufensterspiegelungen und Pinnwandtextur.
  **LaMa ist die konservativere Wahl** — kein Textprior, keine Domänenanpassung, kein
  Overfitting auf saubere Schrift.
- **Bildinhalt geht verloren.** Beim Schaufenster egal, bei der Pinnwand nicht.
- **VRAM-Spitze**, falls LaMa auf der GPU läuft, während Krea 2 geladen ist. Vermeidbar,
  aber muss bewusst gelöst werden.

---

## Ansatz C — Sweep + die vorhandene Kette für alles, mit Budget

### Name + Quelle

Keine neue Fremdkomponente. Die vorhandene Kette (Proposer → Detailer, Glyph Guidance),
angewendet auf jeden Waisenquad statt nur auf SAM3-Regionen.

### Was es hier genau tun würde

Sweep wie in A. Jeder Waisenquad wird geclustert (`core/signs/cluster.py` existiert), pro
Cluster **eine** VLM-Anfrage (`one_call_per_cluster` existiert), dann normal gerendert. Am
Ende steht überall ein echtes Wort — genau das, was das Abnahmekriterium wörtlich verlangt.

### Kosten

| Posten | Wert |
|---|---|
| Gewichte | keine neuen |
| Pakete | keine neuen |
| Laufzeit | **Der Killer.** Aktuell 25–45 s pro Pass bei 4–12 Regionen. Ein Sweep liefert realistisch 15–40 Textzeilen. Selbst mit Clustering und `one_call_per_cluster` landet man bei 2–5 Minuten pro Pass |
| VRAM | unverändert |

### Erwarteter Effekt auf die Messgröße

- **Höchste Decke, schlechtestes Risikoprofil.** Jede zusätzlich gerenderte Region ist eine
  neue Chance, Slop zu erzeugen — genau der Effekt, der im README dokumentiert ist („eine
  leergefegte Fläche bei vollem Denoise bekommt die eigenen Notizen des Modells").
- Bei 17–35 px Kurzseite (der Größenbereich der Straßenszenen-Treffer, siehe Tabelle in 0c)
  **kann** kein Modell lesbar rendern: die OCR-Literatur setzt die Untergrenze bei ~7 px für
  Erkennbarkeit überhaupt und 20–30 px x-Höhe für Zuverlässigkeit. Eine 20-px-Zeile mit drei
  Wörtern hat 5 px pro Glyphe. Der Detailer weiß das (`too_small_policy`), aber der Sweep
  würde ihm hunderte solcher Fälle vorlegen.
- Netto erwarte ich hier eine **Verschlechterung** gegenüber A, nicht eine Verbesserung.

### Aufwand + was schiefgehen kann

**1 Halbtag** (die Teile existieren alle), aber die Laufzeit macht die Testschleife
unbrauchbar und die Fehlerquelle wächst mit jeder Region. Nur sinnvoll als Route **innerhalb**
von A, für die wenigen Quads oberhalb der Lesbarkeitsgrenze.

---

## Achse 3 — Diffusion daran hindern, neue Nebentexte zu erfinden

Kurz, weil es nach der Zensus-Auswertung **die kleinere Hälfte des Problems** ist: der
Zieltext steht in allen 12 Durchgängen korrekt, und `glyph_surface_restyle=0.35` hat den
selbst erzeugten Slop bereits von 7/12 auf 11/12 gehoben. Der Rest kommt aus nicht
adressierten Flächen. Trotzdem, für die Vollständigkeit:

**Negative Prompts wirken bei cfg 1 nicht — das ist bestätigt.** Bei CFG 1,0 wird der
negative Zweig schlicht ignoriert; distillierte Turbo-Modelle sind auf `guidance_scale`
0,0–1,0 trainiert und verlassen sich zur Inferenz nicht auf Classifier-Free Guidance
([Z-Image-Turbo Prompting Guide](https://gist.github.com/illuminatianon/c42f8e57f1e3ebf037dd58043da9de32)).
Ein `negative_prompt: "text, letters, writing"` in den Sign Options ist bei der aktuellen
Konfiguration **wirkungslos**. Das gehört gesagt, weil es leicht als vermeintliche Lösung
eingebaut wird.

Es gibt zwei Verfahren, die genau diese Lücke schließen:

| | NAG | VSF |
|---|---|---|
| Quelle | [arXiv 2505.21179](https://arxiv.org/abs/2505.21179v2), [ComfyUI-NAG](https://github.com/ChenDarYen/ComfyUI-NAG) | [arXiv 2508.10931](https://arxiv.org/html/2508.10931v6) |
| Prinzip | Extrapolation im Attention-Raum, L1-Normalisierung, α-Blending | Vorzeichen der Attention-Values des Negativprompts flippen |
| Kosten | **verdoppelt die Inferenz** (zweiter Forward-Pass) | Einzelpass, „kleiner Overhead" |
| Getestete Architekturen | Flux, Flux Kontext, Wan, Vace Wan, Hunyuan Video, Chroma, SD3.5, SDXL, SD | SD3.5-Turbo, Flux Schnell, Wan |
| Negativ-Score (Paper VSF) | 0,320 | **0,545** (CFG: 0,300) |

Für uns relevant, und beides negativ:

- **Krea 2 steht auf keiner der beiden Listen.** Krea 2 Turbo läuft mit
  `qwen3vl_4b_fp8_scaled` + `qwen_image_vae`, also einer Qwen-Image-nahen DiT — ob NAG/VSF
  dort greifen, ist **unbestätigt**.
- ComfyUI hat seit 0.15.1 eine native NAG-Node, und es gibt einen Bugreport, dass sie auf
  Flux.2 Klein 4B und Flux.2 9B **nichts bewirkt, obwohl sie die Inferenzzeit verdoppelt**
  ([Issue #12707](https://github.com/Comfy-Org/ComfyUI/issues/12707), geschlossen mit Label
  „Potential Bug"). Das ist das exakte Fehlerbild, das man hier ohne A/B-Vergleich nicht
  bemerken würde.

**Empfehlung: nicht jetzt.** Ein Verfahren, das die Laufzeit verdoppelt, die Zielarchitektur
nicht abdeckt und dessen ComfyUI-Implementierung nachweislich stumm ausfallen kann, ist kein
erster Schritt. Wenn später doch, dann VSF vor NAG (einzelpass, besserer gemessener
Negativ-Score) und nur mit gepaartem Seed-für-Seed-Vergleich als Wirksamkeitsnachweis.

Zu den Glyph-Renderern der Vollständigkeit halber: **AnyText2**
([arXiv 2411.15245](https://arxiv.org/abs/2411.15245), Code seit März 2025),
**Glyph-ByT5/Glyph-SDXL-v2** ([glyph-byt5.github.io](https://glyph-byt5.github.io/)),
TextDiffuser-2 — alle hängen an SDXL bzw. eigenen ControlNet-Zweigen und lösen das
**Rendern** von Zieltext, nicht das **Unterdrücken** von Nebentext. Die vorhandene
Glyph-Guidance des Detailers macht dasselbe ohne Zusatzmodell. Die ComfyUI-Anbindung
([zmwv823/ComfyUI_Anytext](https://github.com/zmwv823/ComfyUI_Anytext)) trägt selbst den
Hinweis „Test failed on latest version transformers". Kein Handlungsbedarf.

---

## Rangfolge

### 1. Ansatz A — Textzeilen-Sweep + optische Auflösungsgrenze  ← **das wird gebaut**

Vier Gründe, in dieser Reihenfolge:

1. **Es ist der einzige Ansatz, der die offene Frage beantwortet, statt sie
   vorauszusetzen.** Niemand weiß derzeit, ob ein Textdetektor die Strings `PAXTRES`,
   `FUAVNLE`, `POJUITRDOT` überhaupt findet — DBNet ist auf echter Schrift trainiert, und
   ich habe keine Publikation gefunden, die die Detektion von generierten Pseudoglyphen
   misst. Ansatz B und C setzen dieselbe Erfassung voraus. Wenn der Sweep nicht trägt, sind
   B und C ebenfalls tot, nur teurer. Ein `selector_probe.py`-Lauf mit eingeschaltetem Sweep
   liefert diese Antwort in Minuten, ohne einen einzigen Sampler-Durchgang.
2. **Nachweislich null VRAM-Risiko.** Auf dieser Maschine hat onnxruntime keinen
   CUDA-Provider (verifiziert), also läuft die Detektion auf CPU/OpenVINO und kann Krea 2
   nicht stören. Die Behandlung ist reines OpenCV.
3. **Die benötigten Modelle sind 16 MB und das Abholskript existiert bereits.** Dass das
   OCR-Backend gar nicht installiert ist, ist der peinlichste und billigste Befund dieser
   Recherche — die halbe Infrastruktur liegt ungenutzt im Repo.
4. **Die Zensus-Regeln arbeiten für uns.** `unreadable` zählt ausdrücklich nicht als Fehler.
   Für Hintergrundflächen ist Defokus zudem physikalisch korrekt und nicht nur zulässig —
   das Ergebnis sieht mehr nach Foto aus als jede Neurenderung, nicht weniger.

Konkret zuerst: `fetch_ocr_models.py` laufen lassen → `_run_onnx` behält leere Boxen →
Kachelung bei nativer Auflösung → `selector_probe.py` über `real_street.png` und
`real_noticeboard.png` → **die gefundenen Quads gegen die bekannten Slop-Strings halten**.
Erst wenn das trägt, die optische Behandlung bauen. Ein Artefakt pro Schritt, wie im
Live-Test-README gefordert.

### 2. Ansatz B — Sweep + LaMa-Löschung

Die richtige Ergänzung, sobald A gemessen ist, und die einzige Antwort auf die
Pinnwand-Zettel. Nicht zuerst, weil er dieselbe unbewiesene Erfassung voraussetzt, 206 MB
Gewichte und eine Entladelogik mitbringt und weil das Löschen von Bildinhalt eine
inhaltliche Entscheidung ist, die der Nutzer treffen sollte, nachdem er das Ergebnis von A
gesehen hat. Der `cv2.inpaint`-Pfad ist der billige Vorgeschmack und kostet fast nichts —
wenn A gebaut wird, kann er als zweite Route für scharfe Kleinflächen gleich mitlaufen.

### 3. Ansatz C — alles durch die vorhandene Kette

Erfüllt das Abnahmekriterium am wörtlichsten und wird es in der Praxis am ehesten
verschlechtern: 2–5 Minuten pro Pass, und jede neu gerenderte Kleinstfläche ist eine neue
Slop-Quelle, bei Größen, in denen kein Modell lesbar rendern kann. Sinnvoll nur als Route
innerhalb von A, für die wenigen Quads oberhalb ~24 px.

**Nicht empfohlen:** NAG/VSF (Zielarchitektur nicht abgedeckt, verdoppelte Laufzeit bzw.
unbestätigt, ComfyUI-Node nachweislich stumm ausfallend), OSOR (zweites Diffusionsmodell auf
derselben Karte, keine ComfyUI-Node, bessere Variante nicht-kommerziell), AnyText2/Glyph-ByT5
(lösen das Rendern, nicht das Unterdrücken; SDXL-gebunden).

---

## Quellen

- [MhLiao/DB — Real-time Scene Text Detection with Differentiable Binarization](https://github.com/MhLiao/DB)
- [PP-OCRv5 Einführung, PaddleOCR-Doku](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html)
- [PaddlePaddle/PP-OCRv5_mobile_det (HF)](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det) · [PP-OCRv5_server_det](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det) · [ONNX-Export monkt/paddleocr-onnx](https://huggingface.co/monkt/paddleocr-onnx)
- [PP-OCRv6, arXiv 2606.13108](https://arxiv.org/abs/2606.13108)
- [PaddleOCR 3.0 Technical Report, arXiv 2507.05595](https://arxiv.org/pdf/2507.05595)
- [obss/SAHI — sliced/tiled inference](https://github.com/obss/SAHI)
- [EasyOCR API-Doku (detect-only, text_threshold/low_text)](https://www.jaided.ai/easyocr/documentation/)
- [Cognex — minimale Pixelauflösung für OCR](https://support.cognex.com/en/help-articles/in-sight-what-is-the-minimum-pixel-resolution-recommended-to-ocr-human-readable-text) · [Nuance — empfohlene Zeichenhöhe](https://nuance.custhelp.com/app/answers/detail/a_id/6346/)
- [LaMa, arXiv 2109.07161](https://arxiv.org/abs/2109.07161) · [advimman/lama](https://github.com/advimman/lama) · [big-lama TorchScript, 206 MB](https://huggingface.co/JosephCatrambone/big-lama-torchscript) · [simple-lama-inpainting](https://pypi.org/project/simple-lama-inpainting)
- [Acly/comfyui-inpaint-nodes](https://github.com/Acly/comfyui-inpaint-nodes)
- [huwenkai26/comfyui-remove-text](https://github.com/huwenkai26/comfyui-remove-text)
- [ViTEraser, arXiv 2306.12106](https://arxiv.org/abs/2306.12106) · [CTRNet, arXiv 2207.10273](https://arxiv.org/pdf/2207.10273) · [DeepEraser](https://github.com/fh2019ustc/DeepEraser)
- [OTR: Synthesizing Overlay Text Dataset for Text Removal, arXiv 2510.02787](https://arxiv.org/html/2510.02787)
- [OSOR, arXiv 2606.28094](https://arxiv.org/abs/2606.28094) · [Repo](https://github.com/Zhouqm-Git/osor) · [Gewichte](https://huggingface.co/QinmingZhou/OSOR)
- [RePainter, arXiv 2510.07721](https://arxiv.org/pdf/2510.07721)
- [Feature Refinement for High Resolution Inpainting, arXiv 2206.13644](https://arxiv.org/pdf/2206.13644)
- [NAG, arXiv 2505.21179](https://arxiv.org/abs/2505.21179v2) · [ComfyUI-NAG](https://github.com/ChenDarYen/ComfyUI-NAG) · [ComfyUI Issue #12707](https://github.com/Comfy-Org/ComfyUI/issues/12707)
- [VSF, arXiv 2508.10931](https://arxiv.org/html/2508.10931v6)
- [Z-Image-Turbo Prompting Guide (cfg 1 ⇒ Negativprompt wirkungslos)](https://gist.github.com/illuminatianon/c42f8e57f1e3ebf037dd58043da9de32)
- [AnyText2, arXiv 2411.15245](https://arxiv.org/abs/2411.15245) · [Glyph-ByT5](https://glyph-byt5.github.io/) · [zmwv823/ComfyUI_Anytext](https://github.com/zmwv823/ComfyUI_Anytext)
- [Qwen3-VL Technical Report, arXiv 2511.21631](https://arxiv.org/pdf/2511.21631)
- [SAM 3: Segment Anything with Concepts, arXiv 2511.16719](https://arxiv.org/abs/2511.16719)
- [onnxruntime sm_120/Blackwell Issue #26177](https://github.com/microsoft/onnxruntime/issues/26177) · [Community-Build](https://github.com/Natfii/onnxruntime-gpu-blackwell)
