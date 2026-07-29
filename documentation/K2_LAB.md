# K2 Lab — Krea 2 Regionalsteuerung in ComfyUI

Graph-native Nachbildung der K2Lab-Desktopanwendung als Teil von FVMtools:
regionales Prompting, strikte regionale LoRA-Führung, räumliche Attention,
Token-Emphase, Projector-Kontrolle, Gesichtsverfeinerung und regionales
Editieren — ohne zusätzliche Python-Abhängigkeiten.

Alle Knoten liegen unter **FVM Tools/K2**.

---

## 1. Warum das nicht einfach „zwei Prompts" sind

Krea 2 ist ein Single-Stream-MMDiT: Text- und Bildtoken laufen in **einer**
Sequenz `[text | image]` durch dieselben 28 Blöcke. Es gibt keinen zweiten
Konditionierungszweig, an den man eine zweite Person hängen könnte.

Genau das macht regionale Kontrolle aber möglich — man kann direkt in die
Attention-Logits eingreifen:

| Ebene | Mechanismus |
| --- | --- |
| Weicher Bias | Tokenpaare (Regionsklausel ↔ Bildtoken der Box) werden angehoben, Paare außerhalb abgesenkt. |
| Harte Partition | Subjekteigene Texttoken sind für fremde Subjekte und für Bildtoken außerhalb ihrer Box gesperrt. |
| LoRA-Delta-Gate | Der LoRA wird ungefusioniert als Forward-Adapter installiert; sein Delta wird pro Token maskiert. |

**Bild-zu-Bild-Attention bleibt unangetastet** — sonst entstünden Kachelkanten
an den Boxgrenzen. Alles läuft in *einem* Denoising-Pass, ohne Crop-Compositing
und ohne zweiten Sampler.

Ein Bildtoken entspricht bei Krea 2 genau **16 × 16 Ausgabepixeln**
(VAE-Faktor 8 × Patchgröße 2). Boxen werden immer in Ausgabepixeln angegeben.

---

## 2. Minimaler Aufbau

```
UNETLoader (krea2)  ─┐
CLIPLoader (krea2)  ─┼→ K2 Compose ─→ K2 Regional Sampler ─→ VAE Decode ─→ Save
VAELoader           ─┘      ↑
K2 Region ── K2 Region ─────┘
```

1. `K2 Region` je Person/Objekt, über den `regions`-Eingang aneinanderhängen.
2. `K2 Compose` mit MODEL und CLIP verbinden, Größe setzen.
3. `K2 Regional Sampler` an dessen `model`/`positive`/`negative`/`latent`/`plan`.
4. Normal decodieren und speichern.

Krea 2 Turbo: **8 Steps, CFG 1.0, euler / simple**.

`K2 Load Krea 2` fasst die drei Loader zusammen — die nativen Loader tun es
genauso, dann lassen sich Quantisierungs- oder Cache-Knoten dazwischenhängen.

---

## 3. Knotenübersicht

### Aufbau
| Knoten | Zweck |
| --- | --- |
| `K2 Load Krea 2` | Transformer + Qwen-Encoder + VAE in einem Knoten. |
| **`K2 Region Builder`** | **Visueller Editor: alle Boxen, ihre Prompts und ihre LoRAs in einer Node.** |
| `K2 Region` | Eine benannte Box mit Prompt, Identität, Rolle, Priorität. |
| `K2 Region from BBox` | Region aus Detektor-/KJ-Bounding-Box. |
| `K2 Region Combine` | Mehrere Regionsketten zusammenführen. |
| `K2 Region Preview` | Boxen, weiches Feld oder Token-Ownership als Bild. |
| `K2 Prompt Emphasis` | Exakte Phrase räumlich verstärken. |
| `K2 Regional LoRA` | LoRA global oder an Regionen binden. |
| `K2 LoRA Info` | Header-Diagnose: passt die LoRA auf Krea 2? |
| `K2 Projector` / `K2 Projector from LoRA` | Delta auf die 12er-Layer-Mischung. |
| `K2 Spatial Tuning` | Feineinstellung des Routers. |
| `K2 Compose` | Kompiliert alles zu Standard-ComfyUI-Objekten. |

### Ausführung
| Knoten | Zweck |
| --- | --- |
| `K2 Regional Sampler` | KSampler + Fortschrittsmeldung an den Plan. |
| `K2 Plan Report` | Plan prüfen, ohne zu rendern. |
| `K2 Regional Face Detail` | Gesichter je Region nachschärfen. |
| `K2 Edit Latent` / `K2 Edit Composite` | Regionales Editieren. |
| `K2 Latent Pin` | Außenbereich an eine Referenztrajektorie binden. |
| `K2 Post Upscale` | Lanczos oder neuronaler Upscaler auf exakte Größe. |
| `K2 Project Import` / `K2 Project Export` | K2Lab-Projekt-JSON lesen/schreiben. |

---

## 3a. Der Region Builder

`K2 Region Builder` ersetzt die Kette aus vielen Einzelknoten: eine Node hält
das komplette Layout und speist `K2 Compose` direkt.

```
K2 Region Builder ──regions──┐
       │           ──loras────┤
       │       ──global_prompt┼──► K2 Compose ──► K2 Regional Sampler
       │       ──width/height─┘
```

**Editorfenster.** `Edit layout ⤢` (oder Klick auf die Vorschau) öffnet ein
losgelöstes Fenster — verschiebbar am Titel, größenverstellbar an der Ecke
unten rechts, `Esc` schließt. Links die Zeichenfläche, rechts Regionsliste und
Details der ausgewählten Region.

| Aktion | Bedienung |
| --- | --- |
| Neue Region | Auf freier Fläche aufziehen, oder `+ Region` |
| Verschieben / Größe ändern | Box ziehen bzw. an Kante/Ecke ziehen (8 Griffe) |
| Auswählen | Klick auf Box oder Listeneintrag |
| Löschen | `✕` in der Liste oder `Entf` bei ausgewählter Region |
| Gleichmäßig verteilen | `Fit` |
| Reihenfolge | `↑` in der Liste (bestimmt die Kompilierreihenfolge) |

**LoRAs pro Region.** Im Detailbereich beliebig viele — je mit eigener Stärke,
Routing-Modus und Trigger-Phrase. Sie landen als regionale Zuweisungen am
`loras`-Ausgang und brauchen keinen einzigen zusätzlichen Knoten.

**Hintergrundbild.** `Pull latest` lädt den neuesten Render aus dem
Ausgabeordner hinter die Zeichenfläche, das Auswahlfeld daneben die letzten 24.
`Dim` regelt die Helligkeit, damit die Boxkanten lesbar bleiben.

**Seitenverhältnis.** Boxen werden normalisiert gespeichert *und* beim Ändern
von `width`/`height` formerhaltend umgerechnet: eine quadratische Box bleibt
quadratisch, ihr relativer Mittelpunkt bleibt erhalten, und passt sie nicht mehr
in die neue Leinwand, wird sie gleichmäßig verkleinert statt einseitig gestaucht.
Die Umrechnung ist umkehrbar — 1:1 → 16:9 → 1:1 landet exakt wieder beim
Ausgangsrechteck.

> Zum Vergleich: der Ideogram-4-Prompt-Builder von KJNodes speichert reine
> Canvas-Anteile ohne Kompensation. Dort wird aus einer 0.3×0.3-Box beim Wechsel
> von 1:1 auf 16:9 still ein breites Rechteck, und ein geladenes Hintergrundbild
> überschreibt zusätzlich `width`/`height` — beides verzieht ein fertiges Layout.

## 4. Regionen

**Rolle** steuert, wie hart die Region gebunden wird:

- `subject` — voller Außen-Penalty, konkurriert mit anderen Subjekten um
  überlappende Token, Text ist hart auf die Box begrenzt.
- `background` — nur ein Viertel des Penalties, darf über die Box hinaus
  ausfransen (`falloff_pixels`).
- `auto` — Boxen ab 70 % Canvasbreite werden Hintergrund, schmalere Subjekt.

**Priorität** entscheidet die Kompilierreihenfolge und wer ein mehrdeutig
erkanntes Gesicht zuerst beansprucht. Sie ist *keine* Stärke und *kein*
Bild-Z-Index.

**Identity prompt** wird als Attribut angehängt („… , with a freckled face"),
nicht als eigener Satz vorangestellt — sonst malt das Modell buchstäblich ein
zweites Gesicht in die Region.

### Prompt-Disziplin

Der globale Prompt beschreibt **die Szene**, nicht die Personen — und zwar
strenger, als man zunächst denkt. Schon ein Gattungsbegriff wie „full body
photo" genügt, damit das Modell außerhalb der Boxen eine weitere Person malt:
Bildtoken außerhalb aller Regionen sehen nur den globalen Prompt, und der
verlangt dann eben eine Person.

Nachgemessen mit drei Regionen, gleicher Seed:

| Globaler Prompt | Ergebnis |
| --- | --- |
| `candid full body photo in a sunlit park, 35mm` | 3 Regionen **+ eine zusätzliche Person** im Hintergrund |
| `a sunlit park lawn with tall trees, 35mm photo, natural light` | genau 3 Personen |

Also:

```
global:  a sunlit park lawn with tall trees, 35mm photo, natural light
Anna:    a woman in a bright red dress, shown full length
Bea:     a woman in a bright blue coat, shown full length
```

Bildausschnitt und Kameradistanz gehören in die **Regionsprompts**
(„shown full length"), nicht in den globalen Prompt.

---

## 5. Regionale LoRAs

```
K2 Regional LoRA (Dirndl, global_scope=off, regions="Anna")
   └→ K2 Regional LoRA (Leggings, global_scope=off, regions="Bea")
        └→ K2 Compose.loras
```

`regions` nimmt Regionsnamen (oder -IDs), kommagetrennt. Ein Tippfehler wird
mit einer Liste der vorhandenen Namen abgewiesen, statt still nichts zu tun.

**Global** (`global_scope=on`) wird als normaler, gefusionierter Patch
angewandt — schnell, wirkt auf alles.
**Regional** läuft ungefusioniert über einen Forward-Adapter; das funktioniert
dadurch auch auf FP8-/INT8-Checkpoints, deren Gewichte man nicht überschreiben
kann.

### Was strikte Isolation auslässt

Im strikten Modus lässt eine regionale Route die Ziele `wk`/`wv` des
Hauptstroms **aus**. Grund: diese Key/Value-Projektionen werden von *jeder*
Bild-Query gelesen — ein regionaler Delta darauf wäre nicht mehr ortsgebunden.
Der Report zählt das unter `locality_skipped_targets` (typisch 56 von 256).
Text-Fusions-Ziele bleiben erlaubt, weil sie die Textpartition schützt.

### Routing-Modus

- `standard` — Delta auf Klausel-Token und Box-Bildtoken begrenzt.
- `character_identity` — zusätzlich wird ein Identitätsanker in den Prompt
  eingefügt („<trigger> identifies the person in this region …"). Erfordert
  regionalen Scope und eine Trigger-Phrase.

### Grenzen ehrlich benannt

Der Delta endet an der Boxkante, die Trajektorie nicht: Bild-zu-Bild-Attention
bleibt durchlässig (mit Absicht), deshalb verschiebt ein regionaler LoRA das
Bild außerhalb seiner Box noch leicht. Wer harte Lokalität braucht, sampelt
denselben Seed zweimal — mit und ohne regionale LoRAs — und bindet den
Außenbereich mit `K2 Latent Pin` an den sauberen Lauf.

---

## 6. Projector

`txtfusion.projector` ist ein `Linear(12 → 1)`: es mischt die zwölf
Qwen3-VL-Ebenen zum Textvektor. Ein Delta darauf verschiebt, welche
Sprachebene das Bild dominiert — der stärkste Einzelhebel in Krea 2 und die
Grundlage der bekannten Projector-LoRAs.

`K2 Projector` bietet die veröffentlichten Presets. `K2 Projector from LoRA`
liest die exakten zwölf Werte direkt aus einer Projector-LoRA und ist
vorzuziehen: die Presettabelle ist eine Näherung, bei `skc3vo` und `z0jglf`
weichen mehrere Vorzeichen von den tatsächlichen LoRA-Werten ab. Unterstützt
werden beide Formate (`…projector.diff` und `…projector.lora_A/lora_B`).

`identity_protection = 1.0` nimmt Identitäts-Token vom Delta aus, damit ein
starker Stil-Shift kein Gesicht mitverbiegt.

---

## 7. Gesichtsverfeinerung

In einer Ganzkörperkomposition deckt ein Gesicht nur wenige Latent-Token ab —
Identität geht auch bei perfektem Routing verloren. `K2 Regional Face Detail`
erkennt Gesichter, ordnet jedes der Region zu, in deren Box es liegt, und
sampelt einen gepolsterten Ausschnitt mit **nur** deren LoRAs nach.

Detektor: jedes Ultralytics-Gesichtsmodell (empfohlen) oder das NanoDet
`face_det.onnx` aus FantasyPortrait. Beides ist in üblichen Installationen
vorhanden.

Wichtig: an `model` das **Basismodell** hängen, nicht die Compose-Ausgabe —
auf einem isolierten Crop ergibt regionales Gating keinen Sinn.

---

## 8. Regionales Editieren

```
IMAGE ─→ K2 Edit Latent ─→ K2 Regional Sampler ─→ VAE Decode ─→ K2 Edit Composite ─→ Save
                │                                                      ↑
                └──────────────── mask ────────────────────────────────┘
```

- **latent_feather** ist der Übergangskragen *während* des Denoisings; er lässt
  das Modell Struktur um die Box herum verblenden, statt auf eine harte
  Latentkante zu treffen.
- **composite_feather** ist die schmalere Pixelüberblendung *danach*.
- Pixel außerhalb der Maske werden exakt aus der Quelle kopiert.

Startpunkt: niedriges `denoise`, 64 px Latent-Feather, 48 px Composite-Feather.
Bei sichtbarer Kante eher die Box vergrößern als `denoise` erhöhen.

> **Hinweis zum VAE:** Krea 2 teilt sich den Autoencoder mit Qwen-Image, und
> der ist ein 3D-VAE — `encode()` liefert `[B, C, 1, H, W]`. Ohne Entfernen
> dieser Zeitachse kommt aus jedem img2img-Pfad Farbrauschen heraus (auch aus
> ComfyUIs eigenem `VAEEncode` → `KSampler`). `K2 Edit Latent` normalisiert das
> auf 4D.

---

## 9. Räumliche Feineinstellung

| Regler | Wirkung |
| --- | --- |
| `inside_strength` | Positiver Bias innerhalb der Region. Höher = härtere Bindung, sehr hoch = flaches Bild. |
| `outside_penalty` | Zentrum-zu-Rand-Kontrast; Hintergrundregionen nutzen ein Viertel davon. |
| `falloff_pixels` | Weiche Randzone jenseits einer Hintergrundbox. Subjekttext bleibt hart begrenzt. |
| `late_step_scale` | Anteil der räumlichen Stärke im letzten Schritt. Lockerung beginnt bei 55 % Fortschritt. Braucht `K2 Regional Sampler`. |
| `subject_competition` | Überlappende Subjekte teilen Token nach quadratischer Feldstärke — verhindert verschmolzene Personen. |
| `subject_fill` | Hält das Feld bis zum Boxrand stark. |
| `spatial_instructions` | Generierte Ortsangaben im Prompt. Aus = reines Attention-Routing. |
| `strict_isolation` | Harte Textpartition. Aus = nur weicher Bias (Attribute können überlaufen). |
| `lora_delta_adaptation` | Skaliert Regionen anhand der gemessenen LoRA-Delta-Energie nach. |

`spatial_enabled=off` ist verboten, solange ein regionaler LoRA aktiv ist —
ohne Router würde dessen Textdelta zu geteilter Szenenkonditionierung und liefe
in jede Region.

---

## 10. Diagnose

`K2 Compose` liefert einen JSON-Report, `K2 Regional Sampler` denselben plus
Laufzeitdaten. Wichtige Felder:

- `spatial_attention.main_stream_attention_calls` — muss > 0 sein
  (28 Blöcke × Steps). Ist es 0, passt die Sequenzlänge nicht.
- `spatial_attention.text_refiner_attention_calls` — 2 × Steps bei strikter
  Isolation.
- `lora_reports[].applied_model_targets` / `locality_skipped_targets`
- `lora_delta_statistics[].delta_rms` — 0 bedeutet, der LoRA wirkt nicht.
- `warnings` — leere Liste ist gut.

`K2 LoRA Info` prüft eine Datei ohne Laden: ein anderer Namensraum als
`diffusion_model` / `blocks` / `txtfusion` heißt „für eine andere Architektur
trainiert".

---

## 11. Kompatibilität

- Der Router belegt ComfyUIs `optimized_attention_override`. Kein zweiter
  Knoten mit demselben Hook auf demselben MODEL-Zweig.
- Ausgänge sind gewöhnliche Typen — ControlNet, Guider, FreeU, beliebige
  Sampler und Save-Knoten funktionieren normal.
- Krea 2 Turbo läuft CFG-frei; regionale Negativprompts werden gespeichert,
  haben aber keinen eigenen Konditionierungszweig.
- Geprüft mit `krea2_turbo_int8_convrot` + `qwen3vl_4b_int8_convrot` +
  `qwen_image_vae` auf ComfyUI 0.28.3.

---

## 12. Verifikation

Nachvollziehbar geprüft (Messwerte, nicht Augenschein):

- **Router trägt die Platzierung**: ohne Ortsangaben im Prompt landeten rote
  und blaue Kleider seitenrichtig (rot links 13,9 % / rechts 0,4 %); nach
  Vertauschen der Boxen wanderten sie mit (rot rechts 15,7 % / links 0,1 %).
- **Regionales LoRA-Routing**: derselbe Alters-Slider auf die rechte Region
  gelegt veränderte die rechte Bildhälfte doppelt so stark wie die linke; auf
  die linke Region gelegt alterte er die linke Person. Zwei verschiedene LoRAs
  (Alter links, Haarvolumen rechts) wirkten in einem Pass getrennt.
- **Delta-Statistik** skaliert linear mit der Stärke (rms 0,095 → 0,197 bei 1,0 → 2,0).
- **Edit-Lokalität**: 234-fach stärkere Änderung innerhalb der Editbox als außerhalb.
- **Region Builder mit 2, 3 und 4 Regionen** gerendert, jeweils mit korrekten
  Pixelboxen und regionaler LoRA-Zuordnung; das 4er-Layout zusätzlich auf
  1536×640 mit formerhaltender Umrechnung.
- 129 Unit-Tests unter `tests/unit/test_k2_*.py`.

### Grenzen bei vielen Regionen

Zwei bis drei Subjekte sitzen zuverlässig. Ab vier schmalen Ganzkörper-Boxen
wird es eng: Farben und LoRAs bleiben korrekt getrennt, aber das Modell staffelt
die Personen gern in die Tiefe, statt sie auf gleiche Größe zu bringen. Was hilft:

- Querformat statt 1:1 (mehr Breite pro Box),
- `inside_strength` ≈ 2.6 und `outside_penalty` ≈ 2.0 (gemessen: bester
  In-Box-Anteil im Vergleich zu 1.0/1.0 und 1.8/1.5),
- „shown full length" in jedem Regionsprompt.

Der mitgelieferte `k2_builder_4regions_wide_api.json` nutzt genau diese Werte.
