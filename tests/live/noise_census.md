# Rauschmessung des Slop-Zensus

Wie viel einer Slop-Zahl ist das Bild, und wie viel ist das VLM, das würfelt?
Jedes Bild wurde **unverändert** N-mal hintereinander durch
`slop_census.census(path, rows=2, cols=2, targets=...)` geschickt — exakt die
Parameter der Suite. `slop_census.py` wurde nicht angefasst; gemessen wurde mit
einer Kopie, `noise_census_variants.py`.

VLM `qwen3-8b-vl-instruct-abliterated` auf LM Studio `:1234`.
Rohdaten je Wiederholung: `noise_census.json`.

## Kurzfassung

In **50 Wiederholungen** auf unveränderten Bildern ist das
Urteil PASS/FAIL (`slop == 0`) **0-mal** gekippt. Die *Zahl*
schwankt (σ bis 1,2 auf textreichen Bildern), das *Urteil* nicht.

Beide Zeilen der beanstandeten Tabelle haben eine andere Ursache als Rauschen:

- **`board_A` 4 → 13** ist ein **verschluckter Kachelfehler**. Scheitert einer
  der vier HTTP-Aufrufe, gibt `slop_census.ask()` stillschweigend `[]` zurück und
  ein Viertel des Bildes fehlt in der Zählung. Reproduziert: bei temperature 0.0
  lieferte `board_A` in **9 von 9 fehlerfreien Läufen exakt 12** und im einen Lauf
  mit Kachelfehler **4**. Siehe Abschnitt 5.
- **`board_B` 0 → 4** ist ein **echter Bildunterschied**. Die beiden Läufe
  unterscheiden sich in **22,65 %** aller Pixel, nicht in 0,64 %. Der Zensus auf
  dem anderen Lauf ergibt reproduzierbar 1–3 Slop-Strings, auf dem aktuellen
  10×0. Siehe Abschnitt 6.

## 1. Streuung je Bild (temperature 0.1, wie in der Suite)

| Bild | Charakter | N | min | Median | max | σ | Läufe mit 0 | verschiedene Strings | echte Blobs | stabiler Kern | Einmal-Treffer | s/Lauf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `board_A` | viel Kleinschrift (Zettel-Pinnwand) | 10 | 11 | 13.0 | 14 | 0.78 | 0/10 | 32 | 16 | 6 | 12 | 13 |
| `board_B` | viel Kleinschrift (Zettel-Pinnwand) | 10 | 0 | 0.0 | 0 | 0.00 | 10/10 | 0 | 0 | 0 | 0 | 12 |
| `street_A` | mittel (Ladenschilder) | 10 | 7 | 8.0 | 11 | 1.22 | 0/10 | 21 | 12 | 4 | 10 | 12 |
| `shelf_A` | mittel (viele kleine Etiketten) | 10 | 1 | 2.0 | 2 | 0.30 | 0/10 | 2 | 1 | 1 | 0 | 12 |
| `synth_A` | wenig (ein synthetisches Schild) | 10 | 0 | 0.0 | 0 | 0.00 | 10/10 | 0 | 0 | 0 | 0 | 9 |

Rohe Zählfolgen in Reihenfolge der Wiederholungen:

- `board_A`: `[13, 13, 12, 14, 12, 13, 13, 11, 13, 13]`
- `board_B`: `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
- `street_A`: `[8, 7, 7, 8, 9, 11, 7, 8, 7, 9]`
- `shelf_A`: `[1, 2, 2, 2, 2, 2, 2, 2, 2, 2]`
- `synth_A`: `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`

„verschiedene Strings" zählt jede Schreibweise einzeln; „echte Blobs" fasst
Schreibvarianten desselben Gekritzels zusammen (Ähnlichkeit ≥ 0,72). Der
Unterschied ist der Kern des Rauschens — siehe Abschnitt 1.2.

### 1.1 Stabiler Kern gegen Einmal-Treffer

**`board_A`** — 32 verschiedene Strings über 10 Läufe, davon **6 in allen 10 Läufen** und **12 in genau einem**.

- Kern: `ATRAFY.`, `HOHIECSABL`, `ITZPLT`, `KUHETAG`, `PIATDAZOOK`, `RUHETAG`

- nur einmal: `20°F.`, `ADINDIUALP`, `ADINIDUOLN`, `DADINIDUOLN`, `EAIL.EONITILK`, `EAILBAINTILL`, `EAILEOTITIK`, `ETAILEOTITIK`, `ETALLEONITILK`, `HECAMOUNED`, `NAMPIN`, `SUCCE`

**`board_B`** — 0 verschiedene Strings über 10 Läufe, davon **0 in allen 10 Läufen** und **0 in genau einem**.

**`street_A`** — 21 verschiedene Strings über 10 Läufe, davon **4 in allen 10 Läufen** und **10 in genau einem**.

- Kern: `ANNAHAN`, `LHNC`, `PAKTRDE`, `PERITYTOMO`

- nur einmal: `BITTE`, `DAYILE`, `DAYILESANEBITTE`, `FUAIVNLE`, `FUVNILE`, `FUVNILEPOJUITRDOT`, `PAXTRDE`, `PAXTRON`, `PAXTROS`, `WHNC`

**`shelf_A`** — 2 verschiedene Strings über 10 Läufe, davon **1 in allen 10 Läufen** und **0 in genau einem**.

- Kern: `SLIN`

**`synth_A`** — 0 verschiedene Strings über 10 Läufe, davon **0 in allen 10 Läufen** und **0 in genau einem**.

### 1.2 Woher die Streuung kommt

Der Zensus dedupliziert auf den *exakten* String. Zwei Läufe, die dasselbe
Gekritzel minimal anders abtippen, erzeugen deshalb zwei Einträge. Die
Einmal-Treffer sind fast alle Schreibvarianten eines Blobs, den auch der Kern
schon enthält:

**`board_A`**

- `DADINUOLP`×3, `ADINDIUALS`×2, `ADINIDUOLP`×2, `ADINDIUALP`×1, `ADINIDUOLN`×1, `DADINIDUOLN`×1
- `ETALLEONITIK`×3, `ETAILEONITIK`×2, `EAIL.EONITILK`×1, `EAILEOTITIK`×1, `ETAILEOTITIK`×1, `ETALLEONITILK`×1
- `HECONONISED`×5, `HECOUNED`×4, `HECAMOUNED`×1
- `KUHETAG`×10, `RUHETAG`×10
- `NANPIA`×7, `NAMPIA`×2, `NAMPIN`×1
- `PITUNITE.HE`×5, `PITUNITE.IT`×5

**`street_A`**

- `EHRITVITYDUC`×5, `EHRITIVYDOLIC`×3
- `FUAVNLE`×7, `FUAIVNLE`×1, `FUVNILE`×1
- `POJUITRDOT`×9, `FUVNILEPOJUITRDOT`×1
- `LHNC`×10, `WHNC`×1
- `PAITROS`×2, `PAXTRON`×1, `PAXTROS`×1
- `PAKTRDE`×10, `PAXTRDE`×1
- `SANESEKRI`×3, `SANIESEKRI`×2

**`shelf_A`**

- `SLIN`×10, `IESLIN`×9

## 2. Wie oft kippt ein Bild fälschlich?

Ein Durchgang besteht nur bei `slop == 0` (`--max-slop 0`).

| Bild | Mehrheitsurteil | Läufe mit 0 | Läufe mit >0 | Fehlurteile |
|---|---|---|---|---|
| `board_A` | FAIL | 0/10 | 10/10 | **0/10** |
| `board_B` | PASS | 10/10 | 0/10 | **0/10** |
| `street_A` | FAIL | 0/10 | 10/10 | **0/10** |
| `shelf_A` | FAIL | 0/10 | 10/10 | **0/10** |
| `synth_A` | PASS | 10/10 | 0/10 | **0/10** |

**0 Fehlurteile in 50 Wiederholungen.** Kein Bild,
das in einem Lauf 0 ergab, ergab in einem anderen Lauf > 0 — und umgekehrt.

### 2.1 Wie groß muss ein Unterschied in „X/12" sein?

Ein Nulltreffer erlaubt keine Punktschätzung, aber eine obere Schranke
(Dreierregel: 0 Ereignisse in n Versuchen -> p <= 3/n bei 95 %).

| Größe | nur temperature 0.1 | beide Einstellungen zusammen |
|---|---|---|
| Wiederholungen | 50 | 100 |
| beobachtete Fehlurteile | **0** | **0** |
| obere 95-%-Schranke p(Kipp) je Durchgang | 0.060 | 0.030 |
| σ der Zahl bestandener Durchgänge bei 12 Läufen | ≤ 0.82 | ≤ 0.59 |
| σ der *Differenz* zweier Suite-Läufe | ≤ 1.16 | ≤ 0.84 |
| Mindestunterschied über 2σ | ≥ 2.3 | **≥ 1.7 Durchgänge** |

**Antwort: ab 2/12 liegt ein Unterschied über dem Rauschen, 1/12 nicht.**
Das ist die vorsichtige Lesart aus der oberen Schranke. Der Messwert selbst ist
strenger: **0 Fehlurteile in 100 Wiederholungen** — „4/12 gegen 2/12" ist in
diesen Daten kein Rauschen.

Diese Schranke gilt jedoch **nur für fehlerfreie Läufe**. Ein durchgerutschter
Kachelfehler (Abschnitt 5) kippt ein Urteil auf einen Schlag und trat in
**4 von 100** Läufen auf. Er ist die einzige beobachtete Quelle von Fehlurteilen.

## 3. Die zwei billigen Verbesserungen

### a) temperature 0.0 statt 0.1

| Bild | σ 0.1 (alle) | σ 0.0 (alle) | σ 0.1 (nur fehlerfreie) | σ 0.0 (nur fehlerfreie) | Einmal-Treffer 0.1 | Einmal-Treffer 0.0 | s/Lauf 0.1 | s/Lauf 0.0 |
|---|---|---|---|---|---|---|---|---|
| `board_A` | 0.78 | 2.40 | 0.78 | 0.00 | 12 | 0 | 13 | 14 |
| `board_B` | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 12 | 11 |
| `street_A` | 1.22 | 1.80 | 1.22 | 1.66 | 10 | 0 | 12 | 12 |
| `shelf_A` | 0.30 | 0.00 | 0.00 | 0.00 | 0 | 0 | 12 | 11 |
| `synth_A` | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 9 | 10 |

Rohe Zählfolgen bei 0.0:

- `board_A`: `[12, 4, 12, 12, 12, 12, 12, 12, 12, 12]`  (der Ausreißer stammt aus einem Lauf mit verschlucktem Kachelfehler: [4])
- `board_B`: `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
- `street_A`: `[11, 7, 5, 11, 7, 7, 7, 7, 7, 7]`  (der Ausreißer stammt aus einem Lauf mit verschlucktem Kachelfehler: [5])
- `shelf_A`: `[2, 2, 2, 2, 2, 2, 2, 2, 2, 2]`
- `synth_A`: `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`

**Der klare Gewinn liegt nicht im σ, sondern im Wegfall der Zufallsstrings.**
Über alle fünf Bilder tauchten bei 0.1 **22 Strings genau einmal** auf; bei
0.0 sind es **0**. `board_A` liefert in *jedem* fehlerfreien Lauf exakt
dieselben 12 Strings, `shelf_A` exakt dieselben 2. Nur `street_A` bleibt
bimodal (7 oder 11) — dort schwankt eine einzelne Kachel zwischen zwei
stabilen Lesarten, was σ dort sogar leicht anhebt (1,22 → 1,66).

**Laufzeitkosten: keine.** Gleiche Anzahl Aufrufe, gleiche Dauer
(583s gegen 584s für je 50 Zensus-Läufe).

### b) Mehrheitsentscheid: 3 Läufe, String zählt ab 2 Nennungen

Aus den 10 Wiederholungen je Bild wurden alle 120 Dreier-Kombinationen
ausgewertet — eine echte Bootstrap-Verteilung des Mehrheitsentscheids, ohne
zusätzliche VLM-Aufrufe.

| Bild | σ einzeln | σ Mehrheit-3 | Median einzeln | Median Mehrheit-3 | Spanne einzeln | Spanne Mehrheit-3 | Urteil kippt? | Laufzeit |
|---|---|---|---|---|---|---|---|---|
| `board_A` | 0.78 | 1.23 | 13.0 | 10.0 | 3 | 5 | nein | 13s → 39s |
| `board_B` | 0.00 | 0.00 | 0.0 | 0.0 | 0 | 0 | nein | 12s → 36s |
| `street_A` | 1.22 | 0.85 | 8.0 | 7.0 | 4 | 4 | nein | 12s → 35s |
| `shelf_A` | 0.30 | 0.00 | 2.0 | 2.0 | 1 | 0 | nein | 12s → 37s |
| `synth_A` | 0.00 | 0.00 | 0.0 | 0.0 | 0 | 0 | nein | 9s → 28s |

Der Mehrheitsentscheid senkt den **Median** (er wirft die Einmal-Treffer weg),
aber nicht die **Streuung** — σ bleibt gleich oder steigt leicht, weil die
Schwelle „2 von 3" selbst eine Zufallsgröße ist. Für den Preis von **3× Laufzeit**
ist das kein guter Tausch; die Zahl wird nur kleiner, nicht ruhiger.

## 4. Zieltext als Kauderwelsch gewertet

| Bild | Läufe mit Ziel-als-Slop | Treffer gesamt | betroffene Strings |
|---|---|---|---|
| `board_A` | **10/10** | 10 | `RUHETAG`×10 |
| `board_B` | **0/10** | 0 | — |
| `street_A` | **0/10** | 0 | — |
| `shelf_A` | **0/10** | 0 | — |
| `synth_A` | **0/10** | 0 | — |

Das ist **kein** Zufall, sondern systematisch: `RUHETAG` wird in **10 von 10**
Läufen als Kauderwelsch eingestuft, obwohl es der eingesetzte Zieltext ist und
der Zensus es korrekt abliest. Dazu kommt in allen 10 Läufen `KUHETAG` — dasselbe
Wort ein zweites Mal, einmal verlesen. Ein Zieltext kostet damit zwei Punkte.

Daneben gibt es Fragmente eines Zielworts, die als eigener Slop-Eintrag zählen:

| Bild | Fragment | in Läufen | gehört zu |
|---|---|---|---|
| `shelf_A` | `SLIN` | 10/10 | `RIESLING` |
| `shelf_A` | `IESLIN` | 9/10 | `RIESLING` |

`shelf_A` hat damit gar keinen echten Slop: seine Zahl 1–2 besteht ausschließlich
aus `SLIN` und `IESLIN` — abgeschnittene Lesungen des Zielworts `RIESLING`.
Ein Bild, das die Suite als FAIL führt, ist in Wahrheit sauber.

Bei temperature 0.0:

| Bild | Läufe mit Ziel-als-Slop | betroffene Strings |
|---|---|---|
| `board_A` | 10/10 | `RUHETAG`×10 |
| `board_B` | 0/10 | — |
| `street_A` | 0/10 | — |
| `shelf_A` | 0/10 | — |
| `synth_A` | 0/10 | — |

## 5. Der eigentliche Fehler: verschluckte Kachelfehler

```python
# slop_census.ask()
if not res.get("ok"):
    return []          # <- eine gescheiterte Kachel sieht aus wie eine leere Kachel
```

Scheitert ein Kachel-Aufruf (beobachtet: `HTTP 400 {"error":"terminated"}`),
verschwindet ein Viertel des Bildes spurlos aus der Zählung. Der Lauf meldet
keine Störung — er meldet **weniger Slop**.

| Bild | Läufe mit Kachelfehler | Kachelfehler / Aufrufe | fehlerfreie Läufe | Läufe mit Fehler |
|---|---|---|---|---|
| `board_B` (temp_0_1) | 1/10 | 1/40 | `[0, 0, 0, 0, 0, 0, 0, 0, 0]` | **`[0]`** |
| `shelf_A` (temp_0_1) | 1/10 | 1/40 | `[2, 2, 2, 2, 2, 2, 2, 2, 2]` | **`[1]`** |
| `board_A` (temp_0_0) | 1/10 | 1/40 | `[12, 12, 12, 12, 12, 12, 12, 12, 12]` | **`[4]`** |
| `street_A` (temp_0_0) | 1/10 | 1/40 | `[11, 7, 11, 7, 7, 7, 7, 7, 7]` | **`[5]`** |

Das ist der berichtete Sprung. Bei temperature 0.0 lieferte `board_A` in **jedem
fehlerfreien Lauf exakt 12** — und im einen Lauf mit verschlucktem Kachelfehler
**4**. Genau die Zahl aus der beanstandeten Bewertung.

Was eine einzelne verlorene Kachel kostet (Probeläufe, Slop nach Kachel):

| Bild | Lauf | gesamt | je Kachel 0/1/2/3 | Zahl bei Verlust von Kachel 0/1/2/3 |
|---|---|---|---|---|
| `board_A` | 0 | 14 | 9/2/1/2 | 5/12/13/12 |
| `board_A` | 1 | 14 | 9/2/1/2 | 5/12/13/12 |
| `board_A` | 2 | 13 | 9/2/1/1 | 4/11/12/12 |
| `board_B` | 0 | 0 | 0/0/0/0 | 0/0/0/0 |
| `board_B` | 1 | 0 | 0/0/0/0 | 0/0/0/0 |
| `board_B` | 2 | 0 | 0/0/0/0 | 0/0/0/0 |
| `street_A` | 0 | 7 | 2/1/2/2 | 5/6/5/5 |
| `street_A` | 1 | 9 | 6/1/1/1 | 3/8/8/8 |
| `street_A` | 2 | 5 | 0/1/2/2 | 5/4/3/3 |
| `shelf_A` | 0 | 2 | 0/0/1/1 | 2/2/1/1 |
| `shelf_A` | 1 | 1 | 0/0/1/0 | 1/1/0/1 |
| `shelf_A` | 2 | 2 | 0/0/1/1 | 2/2/1/1 |
| `synth_A` | 0 | 0 | 0/0/0/0 | 0/0/0/0 |
| `synth_A` | 1 | 0 | 0/0/0/0 | 0/0/0/0 |
| `synth_A` | 2 | 0 | 0/0/0/0 | 0/0/0/0 |

## 6. Waren die Bilder des 4→13-Vergleichs wirklich gleich?

`tests/live/` ist byteidentisch mit der Sicherung `off/`. Der zweite Lauf
liegt unter `floor/`. Verglichen wurde jedes Bildpaar pixelweise.

| Bild | Vergleich | Pixel mit Unterschied | davon > 8/255 | max Δ | mittleres \|Δ\| |
|---|---|---|---|---|---|
| `suite_board_A.png` | live vs floor | 1.266 % | 0.305 % | 122 | 0.048 |
| `suite_board_B.png` | live vs floor | 22.650 % | 2.112 % | 136 | 0.719 |
| `suite_board_C.png` | live vs floor | 33.587 % | 11.455 % | 172 | 3.086 |
| `suite_street_A.png` | live vs floor | 1.043 % | 0.176 % | 134 | 0.077 |
| `suite_street_B.png` | live vs floor | 6.066 % | 0.748 % | 150 | 0.318 |
| `suite_street_C.png` | live vs floor | 12.664 % | 2.860 % | 196 | 1.055 |
| `suite_shelf_A.png` | live vs floor | 3.674 % | 0.683 % | 180 | 0.338 |
| `suite_shelf_B.png` | live vs floor | 7.300 % | 1.423 % | 199 | 0.548 |
| `suite_shelf_C.png` | live vs floor | 8.921 % | 2.503 % | 200 | 0.847 |
| `suite_synth_A.png` | live vs floor | 0.000 % | 0.000 % | 0 | 0.000 |
| `suite_synth_B.png` | live vs floor | 0.000 % | 0.000 % | 0 | 0.000 |
| `suite_synth_C.png` | live vs floor | 0.000 % | 0.000 % | 0 | 0.000 |

Die beiden Läufe unterscheiden sich deutlich stärker als „0,09 %" bzw.
„0,64 %" — `board_A` um 1,27 % aller Pixel (0,31 % davon sichtbar > 8/255),
`board_B` um **22,65 %** (2,11 % > 8/255), `board_C` um **33,59 %**. Nur die
`synth_*`-Bilder sind byteidentisch. Die Bilder waren also nicht „praktisch
dasselbe Bild".

### 6.1 Zensus auf dem *anderen* Lauf desselben Motivs

| Bild | Zählfolge auf `floor/` | min | max | zum Vergleich `tests/live/` |
|---|---|---|---|---|
| `board_A@floor` | `[13, 13, 13, 13, 12, 13]` | 12 | 13 | `[13, 13, 12, 14, 12, 13, 13, 11, 13, 13]` |
| `board_B@floor` | `[3, 3, 2, 1, 2, 2]` | 1 | 3 | `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` |

Damit ist beides sauber getrennt:

- `board_A` liefert auf **beiden** Läufen 12–14. Die berichtete `4` ist in
  keinem der beiden Bilder eine reproduzierbare Messung — sie ist der
  Kachelfehler.
- `board_B` liefert auf `floor/` reproduzierbar 1–3 und auf `tests/live/`
  reproduzierbar 0. Das ist ein **echter Unterschied zwischen den Bildern**,
  kein Messrauschen — passend zu den 22,65 % abweichenden Pixeln.

## 7. Empfehlung

1. **Kachelfehler zum harten Fehler machen.** `ask()` darf bei `not res['ok']`
   nicht `[]` zurückgeben. Entweder Retry (2–3 Versuche), oder der Durchgang
   wird als `ERROR` markiert und zählt weder als PASS noch als FAIL. Das ist
   die einzige Änderung, die den beobachteten 4→13-Sprung wirklich beseitigt.
2. **temperature 0.0.** Kostet nichts und macht fehlerfreie Läufe auf dem
   getesteten Bild exakt reproduzierbar.
3. **Zielwörter vom Slop-Zähler ausnehmen** — sowohl exakte Treffer als auch
   Fragmente/Ein-Buchstaben-Verleser eines Zielworts (Ähnlichkeit ≥ 0,72 zu
   einem Ziel). `RUHETAG`, `KUHETAG`, `SLIN`, `IESLIN` sind keine Slop-Funde.
4. **Schreibvarianten zusammenfassen** vor dem Zählen (Blob-Clustering statt
   exakter String-Deduplikation). Das ändert die Streuung kaum, macht die Zahl
   aber interpretierbar: 16 Blobs statt 32 Strings auf `board_A`.
5. **Kein Mehrheitsentscheid.** 3× Laufzeit für keine Streuungsreduktion.
6. **Für ein Urteil „besser/schlechter" nicht die Zahl bestandener Durchgänge
   vergleichen, sondern die Slop-Zahl je Bild** — die ist die eigentliche
   Messgröße und hat σ ≤ 1,2 statt eines Ja/Nein. Ein Unterschied von
   **1 Durchgang in „X/12" ist nicht belastbar, ab 2/12 ist er es.** Wer
   feiner auflösen will, vergleicht die Slop-Summe über alle 12 Durchgänge;
   die löst Verbesserungen auf, die die 0/≠0-Schwelle gar nicht sieht.
7. **Jeden Suite-Lauf mit der Zahl der Kachelfehler protokollieren.** Solange
   die nicht 0 ist, ist die Gesamtzahl nicht vergleichbar.

