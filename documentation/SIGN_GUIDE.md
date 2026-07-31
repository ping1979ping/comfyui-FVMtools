# Sign Tools — glaubwürdige Schrift in generierten Bildern

Vier Nodes, die Schilder, Etiketten, Aufdrucke und andere Textflächen finden,
beurteilen und mit lesbarem Text neu rendern.

```
LoadSAM3Model ─┐
       image ──┼→ Sign Selector SAM3 → SIGN_DATA →  Sign Text Proposer  → SIGN_DATA ─┐
restrict_mask ─┘        ↓ masks, crops, preview       (LM Studio)  ↓ proposed_texts   │
                                                                                      │
                              model / clip / vae ───────→  Sign Detailer  ←───────────┘
                                                             ↑ sign_options
                                                             ↓ images, glyph_preview
```

Die Trennung ist Absicht: zwischen Erkennung und Rendering kannst du sehen, was das
Sprachmodell vorschlägt, und einzelne Texte überschreiben, bevor gesampelt wird.

---

## Warum drei statt einer Node

Der Selector ist teuer (SAM3-Encode), der Proposer ist langsam (Sprachmodell), der
Detailer ist der einzige Teil, den man beim Feintuning oft wiederholt. Getrennt
kannst du am Detailer schrauben, ohne jedes Mal neu zu erkennen und neu zu fragen.

---

## 1 · Sign Selector SAM3

Findet Textflächen über SAM3-Textgrounding. Alle aktivierten Klassen teilen sich
**einen** Vision-Encode pro Bild — neun Klassen kosten also kaum mehr als eine.

### Die neun Klassen

| Klasse | SAM3-Prompts | Schwelle | Mindesthöhe |
|---|---|---|---|
| `sign` | sign, street sign, shop sign | 0.30 | 32 px |
| `label` | bottle label, product label, packaging label | 0.28 | 24 px |
| `garment_print` | printed text on clothing, t-shirt print, logo on shirt | 0.30 | 40 px |
| `poster` | poster, banner, billboard | 0.28 | 40 px |
| `screen` | phone screen, computer monitor, display screen | 0.30 | 32 px |
| `book` | book cover, magazine cover | 0.30 | 32 px |
| `plate` | license plate | 0.35 | 20 px |
| `paper` | document, menu, price tag, receipt | 0.28 | 24 px |
| `graffiti` | graffiti, handwritten text | 0.30 | 40 px |

Jede Klasse bringt eigene Schwelle, Mindestgröße, VLM-Anweisung, Prompt-Vorlage und
einen Denoise-Versatz mit. Ein Kennzeichen wird härter neu gerendert als ein
T-Shirt-Aufdruck, der sich in Stofffalten verzieht.

### Wichtige Regler

- **`threshold_scale`** skaliert alle Klassenschwellen auf einmal. Unter 1.0 findet
  mehr (und mehr Fehltreffer), über 1.0 ist strenger. Der bequemste erste Regler.
- **`min_height_px`** ist die Untergrenze für die Texthöhe im **Originalbild**,
  gemessen als kurze Seite des `minAreaRect`. Darunter kann kein Modell lesbar rendern.
  Regionen darunter werden **markiert, nicht verworfen** — der Detailer entscheidet.
- **`merge_iou`** fasst Treffer zusammen, wenn mehrere Klassen dasselbe Objekt finden
  (ein Plakat wird gern gleichzeitig `poster` und `paper`). Der höher bewertete Treffer
  behält seine Klasse.
- **`cluster_similar`** gruppiert nahezu identische Regionen. Zwölf gleiche Flaschen im
  Regal werden zu einer Entscheidung statt zu zwölf.
- **`only_slop`** wirft Regionen weg, deren Schrift bereits glaubwürdig ist.

### Slop-Erkennung

Das stärkste Signal ist ein Widerspruch, kein Einzelwert: **SAM3 sagt „hier ist Text"
und OCR liest nichts** — das ist die Signatur von Pseudo-Buchstaben. Dazu kommen
OCR-Konfidenz, Wörterbuch-Trefferquote und Bigramm-Plausibilität, die `SHOPPINQ` und
`RESTAURENT` von echten Wörtern trennen.

`slop_detection` wählt die Methode. Ohne installiertes OCR-Backend fällt alles sauber
auf das Urteil des Sprachmodells zurück — nie ein Fehler, nur ein Hinweis im `report`.

---

## 2 · Sign Text Proposer (LM Studio)

Fragt ein Vision-Modell, was auf dem Schild stehen sollte. Spricht direkt HTTP mit dem
OpenAI-kompatiblen Endpunkt (Standard `http://localhost:1234/v1`) — keine andere
Extension nötig.

**`context_mode` ist der wichtigste Regler.** `crop_only` ist am billigsten und
erfindet Text, der nicht zur Szene passt. `crop+scene` schickt zusätzlich das ganze
Bild — erst dadurch passt der Vorschlag zum Ort. `crop+scene+neighbors` hält eine
Ladenzeile in sich stimmig.

**Rangfolge:** `manual_override` > Modellvorschlag > `fallback_texts` > vorhandener OCR-Text.

Die Texte reisen im `SIGN_DATA` weiter: der Proposer legt in jede Region ein
`proposal`-Dict (`text`, `style`, `font_hint`, `legible_original`, `confidence`,
`source`), aus dem der Detailer den Zieltext, den Stil für den Prompt und den
Schriftschnitt zieht. Der Proposer **kopiert** die Regionen dabei, statt das
eingehende `SIGN_DATA` zu verändern — sonst würden zwei Proposer am selben Selector
sich gegenseitig überschreiben, weil ComfyUI allen Abnehmern dasselbe Objekt reicht.
Wer eigene Nodes anschließt, muss also die **zurückgegebenen** Regionen lesen.

`manual_override` nutzt die Nummern aus dem Preview-Bild:

```
3: ACHTUNG
7: Café Mozart
```

`one_call_per_cluster` schickt nur den Repräsentanten eines Clusters; die Geschwister
erben den Text. Zwölf Flaschen = eine Anfrage.

Ist LM Studio nicht erreichbar, läuft der Graph weiter und nutzt die Fallback-Liste.
Der `report` sagt deutlich, was passiert ist.

### `temperature` gehört auf 0.2 — und zwar genau

Das ist keine Geschmacksfrage, sondern eine gemessene Kante:

| Temperatur | 0.0 | 0.1 | 0.2 | **0.25** | 0.3 | 0.4 | 0.6 |
|---|---|---|---|---|---|---|---|
| Abgeschriebener Kauderwelsch | 0/4 | 0/6 | 0/6 | **3/6** | 3/6 | 3/6 | 2/4 |

Kein Verlauf, sondern ein Sprung zwischen 0.2 und 0.25. Oberhalb landet das Sampling in
der Nachbarschaft der Beinahe-Treffer und liefert `CAFFEE` statt `CAFE` — weil eine
Wiener oder italienische Szene diese Schreibweise authentisch wirken lässt. Den richtigen
Text für ein Schild zu wählen ist eine Aufgabe mit wenig Entropie; sie profitiert nicht
von Sampling-Vielfalt.

Der Node warnt im `report`, wenn du darüber gehst. Bei 0.2 waren es in einem
Durchlauf über 5 Szenen × 4 Seeds **0 von 20** Abschriften.

---

## 3 · Sign Detailer

Rendert die Regionen neu. Nutzt dieselbe Inpaint-Pipeline wie der Person Detailer.

### Glyph Guidance — der entscheidende Hebel

Der Zieltext wird mit PIL gesetzt, per `minAreaRect` auf den tatsächlichen Winkel des
Schilds gedreht, perspektivisch auf sein Viereck gewarpt und **vor dem Encoden** in den
Crop komponiert. Das Modell muss dann nur noch Material, Perspektive und Beleuchtung
anpassen, statt Buchstabenformen zu erfinden. Das ist die Idee hinter AnyText und
GlyphControl, ohne zusätzliches Modell — und es funktioniert deshalb auch mit
Checkpoints, die von sich aus keinen Text können.

| Modus | Wirkung |
|---|---|
| `init` | Text eingebettet, gesampelt mit `glyph_denoise` |
| `init_strong` | Dasselbe mit 0,15 niedrigerem Denoise — maximale Texttreue, weniger Anpassung ans Material |
| `off` | Nur Prompt. Braucht ein wirklich textfähiges Modell |

**`glyph_strength` gehört auf 1.0.** Gemessen: schon bei 0,95 bleibt die alte,
verkorkste Schrift klar lesbar unter der neuen stehen — und landet damit im
Init-Latent, also genau der Slop, den wir loswerden wollen. Nur senken, wenn du die
alte Oberflächenschattierung bewusst mitnehmen willst.

**Nur die Schrift wird ersetzt, nicht die Fläche.** `glyph_preserve_surface`
(Standard an) tauscht die Buchstaben aus und lässt alles andere stehen. Ohne das
wurde die gesamte Maskenfläche mit der Plattenfarbe gefüllt, und der Sampler
musste Rahmen, Textur und alles Durchscheinende aus einer einfarbigen Fläche neu
erfinden. An echten Fotos gemessen: aus einem Emailleschild mit blauem
Doppelrand wurde eine schlichte weiße Scheibe, aus einer geschwungenen
Schaufensterbeschriftung ein grünes Banner, das Glas und Regale verdeckte.

### Das Schriftband — warum Striche abdecken nicht reicht

Die erste Fassung übermalte gezielt die alten Buchstaben. Genau daran scheiterte
der **zweite** Durchgang über dieselbe Fläche: was die Tintenerkennung nicht
findet — ein blasser Strichrand, ein weggerundeter Buchstabe — bleibt stehen und
addiert sich über die Durchgänge auf.

Ersetzt wird deshalb nicht Strich für Strich, sondern die **Fläche, auf der
Schrift steht**. Alte und neue Buchstaben werden zu Blöcken verschmolzen, vom
Rand der Region zurückgesetzt (dort sitzen Rahmen und Einfassung) und komplett
geleert. Ein Rest kann nicht überleben, weil nichts einzeln gefunden werden muss.

Geleert heißt nicht „einfarbig zugekleistert": die Fläche wird von ihrem eigenen
Rand nach innen rekonstruiert, damit Beleuchtung und Verlauf erhalten bleiben.

Ins Band kommt **alles**, was als Tinte erkannt wird, unabhängig von der Höhe.
Der Versuch, hohe Formen zu schonen (Wasserzeichen, Falten), schonte auch eine
Überschrift, die dreimal so groß war wie das Kleingedruckte darunter — und die
überlebte den Durchgang als Geisterschrift. Gemessen über vier Szenen: schonen
10/12, alles abdecken 12/12. Der Preis ist das Papiermuster unter der Schrift.

### `glyph_surface_restyle` — der Regler für Mehrfach-Bearbeitung

Bisher bekam die **ganze** Region vollen Denoise. Dreimal hintereinander heißt
drei Generationen Drift: am Emailleschild wanderte das Blau nach Dunkelgrau, der
Rahmen brach auf. Die Pipeline trennt jetzt zwei Dinge, die vorher dieselbe
Maske waren: **wo** das Ergebnis eingeblendet wird, und **wie stark** neu
gerechnet wird (`noise_mask_2d` in `inpaint_slot`).

Vollen Denoise bekommt nur, **wo die neue Schrift steht** — nicht das ganze
geleerte Band. Das ist der Unterschied zwischen 7 und 11 von 12: gibt man dem
Modell ein leergefegtes Blatt bei vollem Denoise, schreibt es seine eigenen
Notizen darauf, und die liest sich wie durchscheinende Altschrift.

| Wert | Wirkung |
|---|---|
| 0,0 | Fläche unberührt — treueste Wiederholung |
| **0,35** | **Standard** — genug, damit eine flache Fläche Material bekommt |
| 1,0 | altes Verhalten, ganze Region wird neu gerechnet |

### Ein Klumpen darf die Schrift nicht überstimmen

`glyph_match_source_size` misst die Zeilenhöhe flächengewichtet. Ein
Wasserzeichen oder Aufdruck kommt als **eine** zusammenhängende Form und hat
mehr Fläche als alle Buchstaben zusammen: an einem gemusterten Pinnwand-Zettel
gemessen 15 597 von 24 684 Tintenpixeln, Schätzung 171 px für eine Schrift, die
50 px hoch steht — und der nächste Durchgang setzte sie dreimal zu groß.

Zwei Schranken, beide nötig: keine Form darf mehr wiegen als alle anderen
zusammen, und keine mehr als ein Viertel der Tintenfläche. Gemessen ergibt das
54 / 37 / 30 / 30 px gegen Sollwerte von rund 50 / 40 / 30 / 30 — und ändert
nichts, wo kein Klumpen dominiert. Eine eng gezogene Maske um ein einzelnes Wort
hat nichts zu überstimmen und behält ihr volles Gewicht.

Gegenläufig zu `glyph_plate_color`: eine Fläche neu einfärben und die alte Fläche
erhalten sind entgegengesetzte Anweisungen. Setzt du eine Plattenfarbe, schaltet
der Detailer den Oberflächenerhalt für diesen Lauf ab und schreibt es in den `report`.

**Gewölbte und ausgefranste Flächen** kann ein Vier-Ecken-Warp grundsätzlich nicht
abbilden — eine Homographie beschreibt immer eine Ebene. Ein Etikett um eine Flasche
oder ein zerrissenes Plakat bekommt deshalb einen **spaltenweisen Warp**: für jede
Bildspalte werden Ober- und Unterkante der Maske abgetastet, und der Textblock wird
dazwischen eingepasst. Die Grundlinie wölbt sich dann mit dem Etikett und wandert mit
der Papierkante.

`glyph_fit` steuert das:

| Wert | Wann |
|---|---|
| `auto` (Standard) | misst, wie stark ein Vier-Ecken-Fit die Kontur verfehlt, und schaltet ab 12 % um |
| `perspective` | ebene Schilder — gibt schrägen Flächen ihre Fluchtlinie |
| `contour` | gewölbte Etiketten, Stofffalten, gerissenes Papier |

Gemessener Fehler des Vier-Ecken-Fits: **34 %** bei einem gewölbten Flaschenetikett,
**23 %** bei einem gerissenen Plakat, **~2 %** bei einem ebenen Schild — daher die
Schwelle bei 12 %.

`glyph_cylinder` (0–1) staucht zusätzlich zu den Seiten hin, wie Schrift, die sich um
eine Flasche wickelt und zum Rand hin perspektivisch verkürzt. Für Flaschenetiketten
0,4–0,6, für alles Ebene 0.

Die Kantenprofile werden bewusst **geglättet**: ohne das würde der Text jeder Zacke
einer Risskante folgen und sich zerreißen. Gesucht ist die Drift der Form, nicht ihr
Rauschen.

**Der Text bekommt eine Fluchtlinie.** Die vier Ecken werden aus der Maskenkontur
selbst gewonnen, nicht aus `minAreaRect` — das liefert immer ein Rechteck, und
damit stünde der Text auf einem schräg wegstehenden Schild waagerecht mit
gleichbleibender Buchstabenhöhe. Lesbar, aber wie ein aufgeklebter Sticker. An
einem um 33 % verjüngten Schild gemessen: das gefittete Rechteck behauptete 0 %
Verjüngung und 20 % zu viel Fläche. Ist die Maske erkennbar rechteckig (unter 6 %
Kantenunterschied), bleibt alles gerade — es wird keine Verzerrung erfunden, wo
keine ist. Abschaltbar über `perspective=False` in `mask_quad`.

Die Glyph-Ebene folgt der **SAM3-Silhouette**, nicht ihrem Begrenzungsrechteck. Bei
runden Schildern, gewölbten Flaschenetiketten oder eingerissenen Plakaten würde ein
Rechteck 20–27 % über die Objektfläche hinausragen — und der Sampler bekäme ein
rechteckiges Schild ins Init-Latent, wo ein rundes steht. Ist die Maske annähernd
quadratisch, wird der Winkel aus `minAreaRect` verworfen: ein Kreis hat keine
Vorzugsrichtung, OpenCV liefert dann willkürliche 45° und der Text stünde diagonal.

`glyph_autocolor` liest Schrift- und Grundfarbe aus dem Originalschild, damit das
Ergebnis sein Farbschema behält. Die Trennung nutzt keinen einfachen Median — auf
einem echten Schild decken die Buchstaben nur 10–15 % der Fläche, da läge der Median
mitten in der Plattenfarbe und beide Werte kämen fast identisch heraus.

### Verifizierte Krea-2-Einstellungen

Live gerendert auf Krea 2 Turbo (`krea2_turbo_fp8` + `qwen3vl_4b_fp8_scaled` +
`qwen_image_vae`), 8 Steps, cfg 1, `er_sde` / `simple` — die Krea-Standardwerte
gelten unverändert auch hier:

| `glyph_denoise` | Ergebnis |
|---|---|
| 0,35 | Text sauber, aber die Fläche bleibt flach und charakterlos |
| **0,55** | **bestes Ergebnis** — scharfer Text *und* echtes Material (Emaille, Schrauben, Abnutzung) |
| 0,65 | eine zweite, schwache Kopie des Wortes erscheint hinter der ersten |
| 0,70+ | das Schild wird neu erfunden: verzogen, erfundene Beschläge, Schrift verblasst |

Der Detailer warnt oberhalb von 0,60. Der Grund für die Kante: darüber behandelt
der Sampler die Glyph-Ebene nicht mehr als Vorlage, sondern als groben Anhalt.

Krea 2 macht aus einer flachen Fläche ein glaubwürdiges Objekt — im Testlauf wurde
aus einem einfarbigen Rechteck ein Emailleschild mit Schrauben in den Ecken und
verwitterten Kanten. Genau dafür ist der Spielraum zwischen 0,45 und 0,60 da.

### Modellwahl — nicht optional

**Z-Image Turbo, SDXL und andere textschwache Checkpoints rendern keinen lesbaren
Text.** Sie würden Slop durch anderen Slop ersetzen. Nimm **Qwen-Image, Ideogram 4
oder Krea 2**. Der Detailer prüft die Modellkonfiguration und schreibt eine Warnung in
den `report`, wenn er ein bekannt textschwaches Modell erkennt.

### Denoise

Der Standard ist **0.85** und damit viel höher als beim Person Detailer. Unter etwa 0.7
scheinen die alten, verkorksten Buchstabenstriche durch. Ist Glyph Guidance an,
übernimmt `glyph_denoise` (Standard 0.55) — dort ist niedrig richtig, weil die
Buchstabenform ja schon im Init-Bild steht.

### Zu kleine Regionen

`too_small_policy` entscheidet, was mit Regionen unter der Lesbarkeitsgrenze passiert:

- **`soften`** (Standard) rendert sie als glaubwürdig unscharfen Text. Echte Fotos haben
  in der Ferne auch unlesbare Schrift — aber mit korrekter Textur statt Fantasiebuchstaben.
- **`skip`** lässt sie unberührt.
- **`render`** versucht es trotzdem.

`max_upscale` deckelt, wie weit eine kleine Region hochskaliert wird. Darüber
halluziniert der Sampler Details, die beim Zurücksetzen ins Bild brechen.

### Verifikation

`verify_after = ocr` liest das Ergebnis zurück und vergleicht es unscharf mit dem
Zieltext. Bleibt die Ähnlichkeit unter `verify_similarity`, wird mit neuem Seed erneut
versucht, bis `max_attempts` erreicht ist. Ohne OCR-Backend bleibt das stillschweigend aus.

---

### Der Untergrund kommt automatisch mit

Das Sprachmodell liefert neben `text` auch ein **`style`**-Feld, und das beschreibt
Schrift **und** Fläche. Bei einem gelben Post-it mit verkorkster Handschrift kommt
zurück:

```json
{"text": "TELEFONNUMMER 123456789", "style": "black ink on yellow sticky note", ...}
```

`build_prompt` setzt das in die Klassenvorlage ein, das Ergebnis geht so an den Sampler:

```
a piece of paper showing the clear text "TELEFONNUMMER 123456789",
black ink on yellow sticky note, legible writing, sharp focus
```

Du musst dafür **nichts** einstellen — `style` fließt immer in den Prompt. Kommt der
Untergrund einmal nicht deutlich genug durch, schärf ihn über `class_instructions`
im Proposer nach:

```
paper: Sag im style-Feld immer, was für ein Papier das ist - Haftnotiz, Kassenbon,
       Speisekarte - samt Farbe und Oberfläche.
```

Die `paper`-Vorlage ist bewusst neutral gehalten (`a piece of paper showing…`), weil
die Klasse auch Haftnotizen, Kassenbons und Handschrift abdeckt. Ein festes
„printed typography" würde jedem gemeldeten Untergrund widersprechen.

### Einen anderen Untergrund erzwingen

Das ist der andere Fall: du willst **nicht** das übernehmen, was im Bild ist,
sondern eine Fläche vorgeben, die es dort noch gar nicht gibt — aus einem grauen
Schmierzettel ein gelbes Post-it machen. Dann brauchst du beide Hebel gleichzeitig:

| Wo | Was |
|---|---|
| `prompt_suffix` in den **Sign Options** | beschreibt die Fläche für das Diffusionsmodell |
| `glyph_plate_color` im **Detailer** | malt sie schon in der Glyph-Ebene |
| `glyph_ink_color` im **Detailer** | passende Schriftfarbe dazu |

```
prompt_suffix:     on a bright yellow post-it note stuck to the wall,
                   slight paper curl at one corner, soft drop shadow,
                   matte paper texture
glyph_plate_color: #ffe680
glyph_ink_color:   #3a3630
```

Nur den Prompt zu ändern reicht nicht: `glyph_autocolor` leitet die Farben aus dem
**Original** ab, die Glyph-Ebene wäre also weiter grau und würde das Modell in
Richtung des alten Untergrunds ziehen. Umgekehrt reicht auch die Farbe allein nicht
— dann steht ein gelbes Rechteck im Init-Latent, während der Prompt von Papier
redet. Der Detailer weist im `report` darauf hin, wenn du eine Plattenfarbe setzt,
aber den `prompt_suffix` leer lässt.

Die Farbfelder verstehen `#ffe680`, `ffe680`, `#fe8` und `255,230,128`. Leer heißt
„aus dem Bild ableiten".

Damit SAM3 die **ganze** Fläche maskiert und nicht nur die Schriftzeile, hilft ein
passender Prompt in `custom_prompts`, etwa `post-it note:0.3` oder `sticky note:0.3`.

---

## 4 · Sign Options

Hält die Widget-Liste des Detailers beherrschbar. Per-Klasse-Denoise
(`plate: 0.95`), `skip_classes`, Negativ-Prompt, Kontext-Faktor, Rundenprogression.

---

## OCR-Backend installieren

Optional. Ohne OCR funktioniert alles, nur die Slop-Erkennung stützt sich dann allein
auf das Sprachmodell.

```bash
"D:/AI/ComfyUI/ComfyUI/venv/Scripts/python.exe" scripts/fetch_ocr_models.py --dry-run
"D:/AI/ComfyUI/ComfyUI/venv/Scripts/python.exe" scripts/fetch_ocr_models.py
```

Die Modelle landen in einem `ocr`-Unterordner des `onnx`-Modellpfads. Weil
`extra_model_paths.yaml` `onnx:` in beiden Roots abbildet, funktioniert das ohne
YAML-Änderung sowohl im Haupt- als auch im Archiv-Root:

```
<onnx-root>/ocr/
├── ch_PP-OCRv4_det_infer.onnx    ~4,7 MB   Detektor
├── ch_PP-OCRv4_rec_infer.onnx   ~10,9 MB   Recognizer
├── ch_ppocr_mobile_v2.0_cls_infer.onnx     Rotation
└── ppocr_keys_v1.txt                       Zeichensatz
```

Gesucht wird in dieser Reihenfolge: `onnx`-Modellpfade aus `folder_paths` →
`ComfyUI/models/onnx/ocr` → `ocr_path` in `outfit_config.ini`. Dasselbe Muster wie
beim BiSeNet-Modell.

**Es wird nie zur Laufzeit heruntergeladen.** Fehlt ein Modell, sagt der `report`,
welche Datei wohin gehört.

Der ONNX-Pfad braucht zusätzlich `pyclipper` und `shapely` für den DBNet-Unclip-Schritt.
Beide liegen im ComfyUI-venv bereits vor, stehen aber bewusst nicht in
`requirements.txt` — OCR ist optional, und ein fehlendes Paket wird als Grund im
`backend_status()` gemeldet statt still zu einem leeren Ergebnis zu führen.

---

## Fonts

Glyph Guidance braucht Schriftdateien. Gesucht wird in `fonts/` im Paket und in den
Windows-Schriftverzeichnissen — auf einem normalen Windows also sofort hunderte
Schnitte. Der `font_hint` aus der Modellantwort wird unscharf auf einen vorhandenen
Schnitt abgebildet.

---

## Typische Fehlerbilder

| Symptom | Ursache | Lösung |
|---|---|---|
| Neuer Text ist wieder Kauderwelsch | textschwacher Checkpoint | Qwen-Image/Ideogram/Krea 2, oder `init_strong` mit niedrigem `glyph_denoise` |
| Alte Buchstaben scheinen durch | Denoise zu niedrig | `denoise` ≥ 0.8, oder Glyph Guidance an |
| Ab dem zweiten Durchgang Reste der Vorschrift | `glyph_preserve_surface` aus | anlassen — sonst wird Strich für Strich gearbeitet |
| Fläche driftet über mehrere Durchgänge | `glyph_surface_restyle` zu hoch | auf 0,35 oder 0,0 |
| Modell erfindet zusätzliche Kritzeleien | leergefegte Fläche bei vollem Denoise | `glyph_surface_restyle` senken |
| Schrift wächst von Durchgang zu Durchgang | Muster/Wasserzeichen als Tinte gezählt | im `report` die gemessene Zeilenhöhe prüfen |
| Schrift steht waagerecht auf schrägem Schild | Maske zu zerfranst für `minAreaRect` | `mask_expand_pixels` erhöhen, `mask_fill_holes` an |
| Keine Regionen gefunden | Schwelle zu hoch | `threshold_scale` auf 0.7, `min_height_px` senken |
| Zu viele Fehltreffer | Schwelle zu niedrig | `threshold_scale` erhöhen, Klassen abschalten, `max_regions` senken |
| Schild wirkt aufgeklebt | Kontext zu knapp | `context_expand_factor` in den Sign Options erhöhen |
| Zwölf Flaschen, zwölf verschiedene Etiketten | Clustering aus oder zu streng | `cluster_similar` an, `cluster_distance` erhöhen |
