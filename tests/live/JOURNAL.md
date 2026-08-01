# Journal — Sign Tools, Abnahme nach dem strengen Maßstab

Ein Eintrag je Abschnitt. Format: **wirkt** (Zahl vorher→nachher, committet) /
**gleich** (nur committet wenn es Code vereinfacht) / **verworfen** (Grund +
Messwert, damit es niemand ein zweites Mal versucht).

Messgröße ist ausschließlich `suite.py` ohne `--fast`: vier Szenen, je A→B→A,
zwölf Durchgänge. Ein Durchgang besteht nur, wenn alle drei Bedingungen gelten —
Zieltext da, Geisterschrift ≤ 0,2, **und kein einziges erfundenes Glyphencluster
irgendwo im Bild**. Die dritte Bedingung ist die, die zählt; die ersten beiden
waren sechzehn Iterationen lang der ganze Test und haben deshalb fast nichts
gemessen.

---

## Ausgangsstand (2026-08-01, `16ba2c6`)

**4 / 12**, roh in `final_census_run.txt`.

| Szene | A | B | C | Slop A/B/C |
|---|---|---|---|---|
| street | FAIL | FAIL | FAIL | 11 / 9 / 1 |
| shelf  | FAIL | PASS | FAIL | 2 / 0 / 1 |
| board  | FAIL | PASS | FAIL | 12 / 0 / 5 |
| synth  | PASS | FAIL | PASS | 0 / 0 / 0 |

Was schon steht:
- Zieltext in **allen zwölf** Durchgängen (`contains=True` durchgehend)
- Geisterschrift ≤ 0,2 in **elf von zwölf** — einziger Ausreißer `synth B` mit 0,8
- Regal und synth weitgehend sauber

Was fehlt — die Zahl wird nicht mehr vom Rendern bestimmt:
1. **Reichweite (dominant, verifiziert).** Flächen, für die keine Region
   ausgegeben wurde, behalten ihre Fantasieschrift. Hintergrund-Schaufenster
   (`PAXTRES`, `SAMYERK`, `EHRITIVYDOLIC`, `POJUITRDOT`, `FUAVNLE`), kleine
   Zettel auf der Pinnwand (`H Ohiec Sabl`, `Piatda Zook`, `Bod indiuals`).
   Gehört in den Selector, nicht in den Detailer.
2. **`synth B` ghosting 0,8** — einzelner offener Punkt, eigene Ursache.

---

## Gepinnt — nicht ohne neuen Sweep ändern (alles live gemessen)

| Wert | Stand | Warum |
|---|---|---|
| Krea 2 Turbo, 8 Steps, cfg 1, er_sde/simple | — | Basis aller Messungen |
| `glyph_denoise` | 0,55 | ab 0,65 Geisterkopie, ab 0,70 Schild neu erfunden |
| LM-Studio `temperature` | 0,2 | ab 0,25 wird Kauderwelsch abgeschrieben |
| `glyph_surface_restyle` | 0,35 | 1,0 lässt die Fläche über Durchgänge driften |
| `HOT_ZONE_MARGIN` | 1,2 | bei 0,5 fehlen erster und letzter Buchstabe |

**cfg 1 heißt: der Negativ-Prompt wirkt nicht** — aber cfg 1 ist eine Wahl, keine
Eigenschaft des Modells. Korrektur vom 2026-08-01 (Nutzerangabe): Krea 2 verträgt
cfg > 1 bei mehr Steps, und **selbst die Turbo-Variante läuft bei cfg 1.5
problemlos**. Damit ist die Negativ-Prompt-Achse offen, nicht tot. Diese Zeile
stand vorher als harte Schranke hier und hat eine Recherche in die Irre geführt —
sie hat NAG/VSF als einzige Auswege behandelt, obwohl schlicht cfg anheben
genügt.

Noch nicht gemessen: ob cfg 1.5 die gepinnten Werte darüber (`glyph_denoise`,
`glyph_surface_restyle`) verschiebt. Vermutlich ja — sie wurden alle bei cfg 1
bestimmt. Ein Wechsel auf cfg > 1 macht einen neuen Sweep über diese beiden
nötig, sonst misst man zwei Änderungen auf einmal (siehe „Band + Maske", 9→7).

## Verworfen — nicht wiederholen

| Versuch | Messwert | Grund |
|---|---|---|
| Schärferer Negativ-Prompt (`additional text, extra words`) | — | Modell lässt ersten und letzten Buchstaben weg |
| Rauschmaske nicht weichzeichnen | — | harte Kante in Latent-Auflösung, schlechter als der Weichzeichner, den sie ersetzen sollte |
| Band + volle Bandmaske im Sampler zusammen | 9/12 → 7/12 | getrennt gemessen: Band +1, Maske −3 |
| Feste Größengrenze gegen Klumpen | — | zerlegt ein Kennzeichen, dessen Ziffern zwei Drittel des Schilds füllen |
| Hohe Formen aus dem Band heraushalten | 10 → 12 (alter Maßstab) | schonte auch Überschriften, die als Geisterschrift überlebten |

---

## Fallen, die je eine Fehldiagnose gekostet haben

1. `$pid` ist in PowerShell die PID der Shell selbst — nie so nennen.
2. `GET /object_info/<Node>` liefert **HTTP 200 mit leerem Body**, wenn die Node
   fehlt. Auf den Node-Key prüfen, nicht auf den Statuscode.
3. Eine Ground Truth für Tintenerkennung, die Rahmen und Textur mitzählt, meldet
   „60 % der Schrift fehlen" — das ist der korrekt geschonte Rahmen.
4. numpy-dtype: eine **uint8**-Maske statt **bool** an eine Funktion geben, die
   damit indiziert, ergibt Zeilen-Indizierung statt Maskierung — still falsch.
   Der Debug-Harnisch übergab bool und zeigte deshalb das richtige Ergebnis,
   der echte Aufrufpfad nicht.
5. Erkennung muss auf dem Begrenzungsrechteck der Region rechnen, nicht auf dem
   ganzen Bild: sonst 460 s statt 39 s pro Durchgang.
6. `suite.py` mitten im Lauf editieren ist gefahrlos (Python hat die Datei schon
   geladen). **Node-Code editieren wirkt erst nach ComfyUI-Neustart** — sonst
   misst man den alten Stand.

## Neustart ComfyUI :8189

```powershell
$conn   = Get-NetTCPConnection -LocalPort 8189 -State Listen
$owner  = ($conn.OwningProcess | Select-Object -First 1)
$parent = (Get-CimInstance Win32_Process -Filter "ProcessId=$owner").ParentProcessId
Stop-Process -Id $parent -Force; Stop-Process -Id $owner -Force

# Verwaiste Hüllen früherer Starts mit aufräumen. Ohne das bleibt je Neustart
# ein cmd + ein conhost liegen (~20 MB) — nach einem Arbeitstag mit knapp
# zwanzig Neustarts waren es 23 tote Konsolen. Nur die ohne Python-Kind.
$py = (Get-CimInstance Win32_Process | Where-Object Name -like 'python*').ParentProcessId
Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" |
  Where-Object { $_.CommandLine -like '*GPU1_P8189_cu130.bat*' -and $py -notcontains $_.ProcessId } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

Start-Process cmd.exe -ArgumentList "/c","D:\AI\ComfyUI\LaunchScripts\GPU1_P8189_cu130.bat" -WindowStyle Minimized
```

Auf :8188 läuft eine **zweite, fremde** ComfyUI-Instanz (Start 30.07.). Nicht
anfassen — die Aufräumzeile filtert deshalb auf `GPU1_P8189`, nicht auf alle
Launcher.

Danach auf `/system_stats` pollen. Python immer absolut:
`D:/AI/ComfyUI/ComfyUI/venv/Scripts/python.exe`.

Unit-Tests: 6067 grün, **48 Failures sind vorbestehend** (`location_lists`, `jb`,
`smp`) und haben nichts mit den Sign Tools zu tun — immer diffen, nie zählen.

## Bilderstrecke

<https://claude.ai/code/artifact/2c478a03-3d22-434a-86d7-b022f0e73b67> —
Quelle liegt als `documentation/sign_strecke.html` im Repo.

---

## Abschnitt 1 — Reichweite

**Diagnose: Kappung, nicht Detektion.** SAM3 findet die fehlenden Flächen bei den
unveränderten Suite-Schwellen bereits alle. Einzige zählende Drossel ist
`selector.py:431` (`[:max_regions]`); `min_height_px` filtert nichts, es setzt nur
ein Flag (`selector.py:295`).

| Szene | detektiert | ausgegeben (Suite) | Kauderwelsch außerhalb der Regionen |
|---|---|---|---|
| street | 23 | 4 | 6 → **0** ohne Kappung |
| board  | 11 | 7 | 7 → **0** ohne Kappung |
| shelf  | 38 | 8 | — |
| synth  | 1  | 1 | — (wird nicht gekappt) |

Methode, die es entscheidet: nicht-selektierte Pixel schwärzen und den Zensus
darauf laufen lassen. Das macht die Abdeckung zur Zahl statt zum Augenmaß.
Rohdaten `diag_*_run.txt`, `diag_outside_census.txt`, Overlays `dst_*`/`dnb_*`.

**`threshold_scale` NICHT senken.** Bei 0.35 kommen +4/+3 Regionen dazu, aber
ausschließlich niedrig bewertete Mega-Blobs (score 0.12–0.18) über ganzen
Schaufenstern, die unter `sort_order=area_desc` ganz nach oben sortieren. Beleg
`dst_wide_uncapped_overlay.png`.

**Notwendig, aber nicht hinreichend:** auch *innerhalb* belegter Regionen
entsteht Slop — `PAIXTROE`/`PAKTRDE` saßen auf dem Sturzband, das bei mr=4 schon
selektiert war. Nach der Kappungsbehebung ist der Rest ein Render-Problem.

**Gate DBNet — bestanden, aber jetzt zweitrangig.** Die Frage war, ob ein auf
echter Schrift trainierter Detektor auf erfundene Glyphen anspringt. Antwort: ja,
deutlich — Straße 22 rohe Boxen (21 auf Fantasieschrift), Pinnwand 96 von 96.
Modelle liegen jetzt unter `ComfyUI/models/onnx/ocr` (16,2 MB, CPU-Provider, kein
VRAM). Kein Widerspruch zur Diagnose: SAM3 liefert Objektregionen (ein ganzer
Zettel), DBNet Zeilenboxen (die ~9 Zeilen darauf) — 11 SAM3-Regionen können 96
DBNet-Zeilen abdecken.

Zwei Befunde daraus, die unabhängig gelten:
- `limit_side_len=960` (`ocr_backend.py:257`) staucht 1280 px auf 0,75× und
  kostet Boxen: nativ 22/96 gegen 20/92. Straße gewinnt zusätzlich bei 1,5×,
  Pinnwand verliert dort (Zeilen verschmelzen) — also szenenabhängig, nicht global.
- `if not text: continue` (`ocr_backend.py:417`, 469, 493) verwirft 1/22 bzw.
  17/96 Boxen — genau die unlesbarsten. Für einen Selektorpfad muss der raus.
- **Ein echter Detektor-Ausfall:** defokussierte gemalte Bogenschrift
  (`RAHELE`/`EHRITIVYDOLIC`, Bildmitte) hat in der Prob-Map null Heat. DBNet lebt
  von Kantengradienten, die Scheibe liegt außerhalb der Schärfeebene. Absenken
  von `box_thresh` holt sie nicht zurück. Also SAM3 ergänzen, nicht ersetzen.

**Größenverteilung, relevant für alles Weitere:** Pinnwand-Zeilen Median 12,3 px,
**75 von 96 unter 20 px**. Straße Median 25,1 px, 9 von 22 unter 20 px. Unter
~20 px rendert kein Diffusionsmodell ein lesbares Wort — dort ist Wegretuschieren
oder die Auflösungsgrenze die ehrliche Antwort, nicht Neubeschriften.

### VERWORFEN: Kappung aufheben — 4/12 → **2/12**

Zurückgenommen mit `git checkout -- tests/live/suite.py`. Roh in
`tests/live/regions_run.txt`. Lauf deterministisch (synth-Bilder byte-identisch
zur Baseline), also Messung und nicht Rauschen.

| Szene/Pass | Slop vorher → nachher | |
|---|---|---|
| street A/B/C | 11→4, 9→4, 1→5 | Summe 21→13, numerisch **besser** |
| shelf A/B/C | 2→8, **0→5**, 1→9 | Summe 3→22, **PASS verloren** |
| board A/B/C | 12→7, **0→1**, 5→4 | Summe 17→12, **PASS verloren** |
| synth | 0/0/0 | unverändert |

**Die Ursache ist nicht die Regionenzahl, sondern was in zu kleine Regionen
geschrieben wird.** 18 der 46 Slop-Einträge sind abgeschnittene Bruchstücke der
Wörter, die das Werkzeug **selbst** gesetzt hat: `CHOENBURGLE`, `PINZIPA`,
`UXELRE`, `NSBURC`, `RAUBURGUNDEI`, `FRAENK`, `EIGELT`, `ESPRECHUN`, `RUHETA`,
`SPUeldIENS`, `CHULUN T`, `KONDITORE`. Die beiden verlorenen PASSes fielen an je
einem einzigen solchen Bruchstück — board B an `CHULUN T` (SCHULUNG), shelf B an
`FRAENK` und `EIGELT`.

Damit hat sich die Fehlerquelle verschoben: **der dominierende Slop stammt jetzt
vom Werkzeug selbst, nicht mehr aus dem Original.**

Zwei Nebenbefunde, die bleiben:
- *Fragmentrisiko war es nicht.* Nur 7 von 47 Einträgen sind ≤ 2 Zeichen. Würde
  man jeden Streubuchstaben verzeihen, bestünde **kein einziger** Durchgang
  zusätzlich. Die Hypothese ist gemessen und erledigt.
- *Bildurteil weicht von der Zahl ab.* street ist numerisch besser (21→13), aber
  visuell schlechter: riesige durchscheinende Buchstaben über den Schaufenstern,
  ein leergeräumtes Hängeschild. Auf dem Weinregal sitzt gestochen scharfe
  Schrift auf Bokeh-Flaschen — aufgeklebt, nicht fotografiert. Der Zensus
  bestraft das kaum. **Die Zahl allein hätte hier in die Irre geführt.**
- *board B war als Bild besser als die Baseline* — alle Kritzelzettel durch echte
  Wörter ersetzt, während die Baseline drei Kritzelnotizen trug, die das VLM als
  `unreadable` durchwinkte. Es verlor den PASS an einem selbst erzeugten Fehler,
  nicht an einer Härtung des Maßstabs.

**Folgerung für den nächsten Bauschritt:** eine Region, die kein lesbares Wort
fassen kann, darf keins bekommen. Nicht die Regionenzahl begrenzen — die
Zuweisung. Das deckt sich mit der DBNet-Messung (Pinnwand-Median 12,3 px, 75 von
96 Zeilen unter 20 px) und mit der Nutzeridee „erst Untergrund, dann Text":
für solche Flächen ist Stufe A die ganze Lösung, nicht die halbe.

Behalten wurde aus dem Versuch nur `tests/unit/test_live_suite_scenes.py`,
gekürzt auf 22 Tests — die Invarianten, die auch auf der Baseline gelten
(A/B disjunkt, keine Dubletten, Liste ≥ Regionen, jede Region ein eigenes Wort).
Die vier Tests, die die verworfene Hypothese kodierten, sind raus.

**Bau (verworfen):** `suite.py` Regionen 4→23, 8→38, 7→11;
Wortlisten auf ≥ Regionenzahl verschiedene Wörter verlängert, weil
`suite.py:86` mit `texts[i % len(texts)]` zyklisch verteilt und der Zensus
Wortwiederholung als Füllsel-Kauderwelsch wertet. 39 neue Unit-Tests
(`test_live_suite_scenes.py`) sichern genau diese Falle. pytest-Diff gegen
`baseline_failures.txt` leer. Gemessen: **33 s Fixkosten + 1,6 s je Region** —
die Regionenzahl ist viel billiger als befürchtet, das Modellladen dominiert.
Voller Zensus-Lauf ≈ 17–18 min statt 13.

---

## Abschnitt 2 — Lesbarkeitsboden

### VERWORFEN: `min_legible_px=12` — 4/12 → **2/12**

Zurückgenommen (`git checkout` auf `detailer.py`, `glyph.py`; Testdatei gelöscht).
Roh in `tests/live/legible_run.txt`.

Gebaut waren zwei Teile: **Reparatur** (der Deckel `target_line_height * 1.35` in
`_fit_text` drückte den Satz auf die Größe der ALTEN Schrift — auf defokussierten
Etiketten 4–6 px, obwohl die Fläche 13–23 px trüge) und **Gate** (trägt auch die
volle Breite das Wort nicht, dann `too_small_policy`).

| Szene | Slop vorher → nachher |
|---|---|
| street | 11→11, 9→8, 1→2 |
| shelf | 2→**5**, **0→8**, 1→**4** |
| board | 12→**15**, **0→6**, 5→**10** |
| synth | 0/0/0 unverändert |

### Der Befund, der bleibt: `soften` ist grundsätzlich unvereinbar mit dem Kriterium

Der neue Slop besteht **nicht** aus Bruchstücken der Zielwörter, sondern aus
frisch erfundenem Text:

- shelf A: `TOKL CUBA ANY RODA`, `TEGLING AVYAT`, `AL RVASI`, `BUTORT DOATRKR`
- shelf B: `Tho pog Neinnt LLRLING Yoo tie Culti Ratee Piebua` — eine ganze Zeile
- shelf C: `& Tag..!`, `taoewids`, `NER..`

Das ist der `soften`-Pfad (`detailer.py:578-581`: `SOFTEN_PROMPT`, denoise 0.35,
kein Glyph-Layer). Sein Zweck laut Tooltip: „render as believable out-of-focus
text (recommended)".

**„Glaubwürdige unscharfe Schrift" IST Pseudoschrift.** Der Zensus stuft sie als
`gibberish` ein, nicht als `unreadable` — und das ist richtig, nicht ein Fehler
des Maßstabs: was lesbar genug ist, um transkribiert zu werden, ist lesbare
Fantasieschrift. Die Kategorie `unreadable` meint Flächen, auf denen man *keine
Buchstaben ausmachen kann*; ein Modell, das absichtlich schriftähnliche Formen
malt, produziert das Gegenteil.

Damit ist nicht der Schwellwert falsch, sondern die Politik. **Jede Behandlung,
die textähnliche Struktur rendert, wird gezählt** — egal wie unscharf.

Für eine Fläche, die kein lesbares Wort tragen kann, bleiben genau zwei ehrliche
Ausgänge:
1. **löschen** — Band leeren, Fläche aus ihrem eigenen Rand rekonstruieren, nichts
   daraufsetzen. Die Mechanik existiert bereits (`text_band` +
   `reconstruct_surface`, Bilderstrecke Schritt 12).
2. das Original stehen lassen — behält aber dessen Fantasieschrift, fällt also
   ebenso durch.

Also: löschen. Das ist zugleich Stufe A der Nutzer-Taktik „erst Untergrund, dann
Text" — zwei unabhängige Wege führen auf dieselbe Maßnahme.

**Die Reparatur ist davon unberührt und weiterhin plausibel richtig**
(`BUCHLADEN` auf 4 px zu setzen, wo 23 px Platz sind, ist ein Bug). Beim
Wiederaufbau beide Teile über Widgets schaltbar machen, damit `--set` sie ohne
neuen Bau trennen kann.

**Korrektur (nachgetragen):** der Satz „nie allein gemessen" stand hier zu
Unrecht. Sie **wurde** isoliert gemessen, in derselben Sitzung:

| Lauf | Einstellung | bestanden | Slop |
|---|---|---|---|
| `floor_off_run.txt` (17:00) | `min_legible_px=0` | **4/12** | 37 |
| `floor_run.txt` (16:45) | `min_legible_px=12` | **2/12** | 52 |

Der Boden allein, bei Standardregionen, fiel also durch. Beide Läufe fanden
**vor** der Zensus-Reparatur (18:00) statt und sind mit allem danach nicht
vergleichbar — untereinander aber schon, und dort ist das Urteil eindeutig.

### Widerlegt: „zu klein zum Rendern" — das Gate ist in JEDER Höhe falsch

Der Offline-Boden-Sweep hatte ausgerechnet, das Weinregal könne physisch nicht
mehr (~100 px Etikettenbreite, 8–15 Zeichen, also 5–9 px Versalhöhe). **Das
Baseline-BILD widerlegt die Rechnung.** `suite_shelf_B.png` der Baseline trägt
`SCHWARZRIESLING`, `LEMBERGER`, `SPAETBURGUNDER`, `DORNFELDER`, `TROLLINGER`,
`PORTUGIESER` — alle sieben gestochen scharf und lesbar, **slop 0**. Genau diese
Regionen hat das Gate als unrenderbar verworfen. `ZWEIGELT` fiel bei 11 px raus,
1 px unter der Schwelle.

Der Boden 12 trennt also nicht zwei Populationen, er schneidet mitten durch die
funktionierende. Eine Fläche, die das Wort nur klein trägt, muss es trotzdem
bekommen. **Kein breitenbasiertes Gate — in keiner Höhe.**

Lehre über den Einzelfall hinaus: eine Geometrierechnung sagt, was *hineinpasst*,
nicht was *lesbar herauskommt*. Die zweite Frage beantwortet nur ein Bild. Beide
verworfenen Abschnitte gehen auf dieselbe Verwechslung zurück.

### Falle: `contains=True` beweist nicht, dass der Zieltext da ist

`judge()` (`suite.py:147`) übergibt dem Sichtmodell die GANZE Wortliste und
fragt „eines davon". Bei shelf B war `SPAETBURGUNDER` nachweislich nicht im Bild
— `REGENT` stand drin, und das genügte für `contains=True`. Je länger die
Wortliste, desto schwächer die Bedingung. Bei Bewertungen nie auf `contains`
allein stützen; das Bild ansehen oder gezielt auf das erste Wort prüfen.

### `soften` erzeugt keine Unschärfe, sondern scharfe Erfindung

Nachgesehen im Bild: denoise 0.35 reicht nicht, um die vorhandenen Lettern zu
löschen — das Modell malt sie als neue Fantasieschrift nach, scharf und
kontrastreich. Direkte Zuordnung Bild→Zensus belegt: alle fünf Slop-Strings von
shelf A stammen von *einer* weichgezeichneten Flasche.

### Was messbar funktioniert hat

Die **Reparatur** des Größendeckels, isoliert belegt: shelf A, Flasche 4 zeigt
danach scharf `RIESLING`, wo die Baseline `IESLIN`/`SLIN` trug. Das ist der
einzige Teil beider Abschnitte mit positivem Beleg — und er wird als Nächstes
allein gebaut und allein gemessen.

### Nebenbefund: `OVERLAP_SKIP_SHARE` reagiert auf Größenänderungen

street B meldete „1 skipped — 100% of it was already rewritten by an earlier
region". Nicht das Gate: größer gesetzte Schrift in Durchgang A ändert das
Regionen-Layout, das SAM3 in B findet, und eine Region fällt vollständig in eine
schon bemalte. Jede Änderung an der Schriftgröße wirkt über den A→B→C-Pfad auf
die Regionenaufteilung zurück.

---

## Der erhöhte Regionenzustand — reproduzierbar, aber NICHT der Default

`SCENES` in `suite.py` steht committet auf den **Referenzwerten** 4/8/7/1, weil
alle historischen Zahlen (Baseline 4/12) daran gemessen sind. Der experimentelle
Zustand mit voller Reichweite lieferte die besten Slop-Werte, aber nie mehr
bestandene Durchgänge — er gehört deshalb nicht in die Abnahme, sondern hierher.

Zum Wiederherstellen: `regions` auf **street 23, shelf 38, board 11, synth 1**
und die Listen auf mindestens so viele VERSCHIEDENE Wörter (sonst zyklisch
wiederholte Füllsel, die der Zensus als Kauderwelsch wertet):

**street A** WEINHANDEL, GOLDSCHMIED, SCHREIBWAREN, BUCHLADEN, SPIELWAREN,
METZGEREI, KAFFEEHAUS, APOTHEKE, UHRMACHER, SCHUHHAUS, WEINSTUBE, TEESTUBE,
HUTMACHER, GALERIE, OPTIKER, IMBISS, BLUMEN, GASTHOF, KAESE, BROT, HONIG,
TABAK, WOLLE
**street B** KONDITOREI, EISENWAREN, FAHRRADLADEN, BAECKEREI, LEDERWAREN,
DROGERIE, BUCHBINDER, FRISEUR, GLASEREI, SCHNEIDER, KRAEUTER, TISCHLER,
GEWUERZE, MALEREI, KELLEREI, KAKAO, RAHMEN, GARTEN, WEBEREI, BIER, SEIFE,
KERZEN, MUEHLE

**shelf A** (weiße Rebsorten) WEISSBURGUNDER, GRAUBURGUNDER, SCHOENBURGER,
WUERZGARTEN, EHRENFELSER, JOHANNITER, SIEGERREBE, ARNSBURGER, HUXELREBE,
FABERREBE, SCHEUREBE, AUXERROIS, FREISAMER, PRINZIPAL, ALBALONGA, RIESLANER,
SILVANER, RIESLING, TRAMINER, MERZLING, HIBERNAL, OSTEINER, GUTEDEL, BACCHUS,
ELBLING, NOBLING, RIVANER, SOLARIS, PHOENIX, SAPHIRA, STAUFER, KANZLER, KERNER,
ORTEGA, OPTIMA, HELIOS, JUWEL, PERLE
**shelf B** (rote) SPAETBURGUNDER, FRUEHBURGUNDER, BLAUFRAENKISCH,
TAUBERSCHWARZ, HELFENSTEINER, FRANKENTHALER, DUNKELFELDER, PORTUGIESER,
AFFENTHALER, DORNFELDER, TROLLINGER, HEROLDREBE, BLAUBURGER, LEMBERGER,
ROTBERGER, VELTLINER, ZWEIGELT, CABERNET, CABERTIN, REBERGER, SCHIELER,
MONARCH, ROESLER, LAURENT, ALLEGRO, LAGREIN, ROTLING, DOMINA, ACOLON, REGENT,
MERLOT, DAKAPO, ACCENT, BOLERO, PRIOR, BARON, PALAS, KOLOR

**board A** KAFFEEKASSE, SPUELDIENST, MITTAGESSEN, BESPRECHUNG, KUEHLSCHRANK,
GEBURTSTAG, ANMELDUNG, PUTZPLAN, RUHETAG, EINKAUF, URLAUB
**board B** MUELLDIENST, MILCHKASSE, ABENDESSEN, WARTUNGSPLAN, DIENSTPLAN,
TERMINLISTE, FIRMENFEIER, TEEKUECHE, LIEFERUNG, SCHULUNG, FEIERTAG

Gemessene Läufe in diesem Zustand: `regions_run` (soften) 2/12 Slop 47,
`erase_run` 2/12 Slop 41, `combo_run` (+Boden 12) 2/12 **Slop 27**,
`ratchet_run` 3/12 Slop 36, `noclip_run` 3/12 Slop 36.

---

## Rückschau nach Abschnitt 3 — das Messgerät selbst

Anlass: eine Bewertung meldete 0,09 % Pixelunterschied bei Slop 4 → 13, und ich
hielt den Zensus für unbrauchbar. **Der Verdacht ist widerlegt.** 100
Wiederholungen auf unveränderten Bildern, roh in `noise_census.md` /
`noise_census.json`.

| Bild | N | min | Median | max | σ | Läufe mit 0 |
|---|---|---|---|---|---|---|
| board_A | 10 | 11 | 13 | 14 | 0,78 | 0/10 |
| board_B | 10 | 0 | 0 | 0 | 0,00 | 10/10 |
| street_A | 10 | 7 | 8 | 11 | 1,22 | 0/10 |
| shelf_A | 10 | 1 | 2 | 2 | 0,30 | 0/10 |
| synth_A | 10 | 0 | 0 | 0 | 0,00 | 10/10 |

**Das PASS/FAIL-Urteil kippte in 100 Wiederholungen kein einziges Mal.** Die Zahl
schwankt, das Urteil nicht. Mindest-Differenz, die über dem Rauschen liegt:
**2/12**. 1/12 ist nicht belastbar. Die Urteile der Abschnitte 1–3 (je 4/12 → 2/12)
stehen damit.

Auch meine Pixelzahl war falsch: `board_B` unterschied sich um **22,65 %** der
Pixel, nicht 0,64 %. Der Sprung war ein echter Bildunterschied.

### Zwei echte Fehler im Messgerät, beide gefunden

**1. Verschluckte Kachelfehler.** `slop_census.ask()` gibt bei
`not res.get("ok")` stillschweigend `[]` zurück. Eine per HTTP gescheiterte
Kachel verschwindet spurlos — der Lauf meldet keine Störung, sondern **weniger
Slop**. In 4 von 100 Läufen aufgetreten. Erklärt den 13→4-Sprung auf `board_A`
vollständig (9 von 14 Slop-Strings sitzen allein auf Kachel 0). Das ist die
gefährliche Richtung: der Fehler lässt Ergebnisse *besser* aussehen.

**2. Das Zielwort zählt als Kauderwelsch.** `RUHETAG` in **10 von 10** Läufen als
`gibberish`, dazu `KUHETAG` (dasselbe Wort verlesen) — ein korrekt gerendertes
Zielwort kostet zwei Punkte. Die Pinnwand kann unter dieser Regel nicht bestehen.

Vorsicht bei der Behebung, sonst versteckt man echte Fehler: `IESLIN`/`SLIN` zu
`RIESLING` sind **kein** Messfehler, sondern vermatschte eigene Schrift. Deshalb
drei Eimer statt zwei — exakter Zieltreffer zählt nicht, `target_fragment`
(Ähnlichkeit ≥ 0,72, nicht exakt) zählt **weiter als Fehler**, aber getrennt
ausgewiesen.

### Weiteres

- **`temperature=0.0`** kostet nichts (583 s gegen 584 s auf 50 Läufe) und senkt
  die Einmal-Treffer von 22 auf 0.
- **Kein Mehrheitsentscheid:** senkt den Median, aber nicht die Streuung
  (board_A σ 0,78 → 1,23), und kostet dreifache Laufzeit. Gemessen, verworfen.
- **String-Deduplikation ist zu streng:** die 32 „verschiedenen Strings" auf
  `board_A` sind 16 Blobs — `PAKTRDE`/`PAXTRDE`/`PAXTRON`/`PAXTROS` zählen
  viermal. Clustern statt exakt vergleichen.
- Für „besser/schlechter" ist die **Slop-Summe über alle 12 Durchgänge** das
  feinere Maß als die Zahl bestandener Durchgänge.

### Messgerät repariert, Baseline neu vermessen — **4/12 bestätigt**

`rebaseline_run.txt`, 0 Kachelfehler, Exit 1. Alle zwölf Einzelurteile
**positionsgleich** mit dem alten Lauf. Produkt unverändert (`min_legible_px=0`).
Das reparierte Gerät bestätigt die Baseline, statt sie zu verschieben.

Gebaut wurde: Kachelfehler → Wiederholung, sonst ERROR und Exit 3;
`temperature=0.0`; drei Eimer (`gibberish` fremd / `target_fragment` eigene
verstümmelte Schrift, zählt weiter / exakter Zieltreffer, zählt nicht);
Clusterung von Schreibvarianten. 31 Unit-Tests.

**Die entscheidende neue Zahl — Aufschlüsselung des Restslops:**

| Szene | gesamt | fremd | eigenes Fragment |
|---|---|---|---|
| street | 24 | **23** | 1 |
| board | 15 | **15** | 0 |
| shelf | 2 | 1 | 1 |
| synth | 0 | 0 | 0 |

**95 % des Restslops ist unberührtes Original** (39 von 41). Die Kette misst
überwiegend Flächen, die sie nie angefasst hat. Die Hauptursache ist damit
wieder die Reichweite — aber Abschnitt 1 hat gezeigt, dass bloßes Aufdrehen
scheitert, weil mehr Regionen mehr kleine Flächen bedeuten und dort das eigene
Wort verstümmelt.

**Daraus folgt der Weg:** Reichweite erhöhen UND für Flächen, die kein lesbares
Wort tragen können, löschen statt schreiben. Erst beides zusammen ergibt Sinn —
einzeln ist jedes gemessen gescheitert.

### Reste im Messgerät, bewusst offen gelassen

- **`LUNHANDE` auf street C.** Das Schild trägt im Bild fehlerfrei `WEINHANDEL`;
  die Kachelüberlappung liest es ein zweites Mal falsch. Der Freispruch greift
  nicht: die korrekte Lesung landete im `word`-Eimer, der Blob enthält nur
  `LUNHANDE`, und die Ähnlichkeit ist 0,667 — knapp unter 0,72. Kostet 1 Punkt.
  Die Längenregel hilft hier nicht, weil die Fehllesung *kürzer* ist und damit
  wie ein echtes Fragment aussieht. Nicht weiter verfolgt: 1 Punkt.
- **Zuordnung zu pessimistisch.** `ITZPLT` (0,571), `FUTZEPLO` (0,625), `SBURGU`
  (0,600) sind nachweislich eigene verstümmelte Schrift, fallen aber unter 0,72
  und werden als „fremd" gebucht. Die 2 Zielfragmente sind eine **Untergrenze**,
  real eher 5. Die Gesamtzahl stimmt, nur die Aufteilung ist verschoben.
- **street bleibt bimodal** (8 gegen 11) auch bei temperature 0,0. Das
  Reststreuen sitzt konzentriert auf dieser einen Szene.

### Falle: Vergleiche über die Regionenzahl hinweg sind NICHT wortgleich

Die Wortlisten wurden für den Regionen-Bau verlängert, und `judge()` bekommt das
**erste** Wort der jeweiligen Liste als `target`. Dadurch heißt derselbe
Durchgang in beiden Zuständen anders: `street B` einmal `BAECKEREI`, einmal
`KONDITOREI`; `shelf A` einmal `RIESLING`, einmal `WEISSBURGUNDER`.

Die Gegenüberstellung „Baseline 41 gegen erhöhte Regionen 41" ist deshalb
**nicht wortgleich** — sie vergleicht dieselben Szenen mit anderer Beschriftung.
Die Aussage über die Aufteilung (fremd gegen eigenes Fragment) bleibt gültig,
die Gleichheit der Summe ist teilweise Zufall.

### Falle bestätigt: `target='X' contains=True` heißt NICHT, dass X im Bild steht

Im Bild nachgeprüft: **street A** meldet `target='WEINHANDEL' contains=True` —
im Bild steht **APOTHEKE**, WEINHANDEL kommt nirgends vor. **street B** meldet
`BAECKEREI`, im Bild steht **DROGERIE**. **shelf A** meldet `RIESLING`, und
genau dieses Etikett ist das defekte. `judge()` fragt nach der ganzen Liste, die
Zusammenfassung druckt aber das erste Wort. `contains` ist nie falsch, die Zeile
liest sich nur so. **Bei jeder Bewertung im Bild gegenprüfen.**

### WIRKT: `too_small_policy=erase` statt `soften` — Slop 47 → 41

Erster positiver Messwert. Gleicher Szenenzustand (erhöhte Regionen 23/38/11/1),
**eine Variable**: die Politik. Roh in `erase_run.txt`, Server-Log mit den
`erased`-Zählungen in `erase_serverlog.txt`. 0 Kachelfehler.

Bestandene Durchgänge 2/12 → 2/12 (unter der Rauschgrenze), aber die feinere
Slop-Summe fällt von 47 auf 41. Davon ist **−1 nachweislich Rauschen** (board war
zwischen beiden Läufen pixelidentisch, `meanabs=0.0`, und schwankte trotzdem um 1
durch einen zusammengeführten Doppeleintrag). **−5 sind der Politik zuzurechnen**,
verteilt auf street (−2) und shelf (−3).

**Der Bildbeleg ist stärker als die Zahl.** An jeder Stelle, an der sich die
Läufe unterscheiden, ist der Abstand zum **Originalfoto**:

| Ort | \|erase − orig\| | \|soften − orig\| |
|---|---|---|
| street A 691,436 | 0,53 | 16,61 |
| street A 682,663 | 0,00 | 10,85 |
| shelf A 175,445 | 3,20 | 13,34 |
| shelf A 696,31 | 1,08 | 13,95 |

Was `soften` dort tut: zwei gestochen scharfe schwarze Glyphen über ein blaues
Emailschild, ein grell orangeroter Block auf eine dunkelgrüne Tür (halluziniertes
Objekt), erfundene Schreibschrift quer über ein Weinetikett. `erase` lässt an
denselben Stellen Emailschild, Messingschild mit Metallglanz und Etikett mit
Papierkorn stehen.

**Ehrliche Einschränkung:** der Gewinn kommt überwiegend daraus, dass gar nichts
gemalt wird — an den meisten Löschstellen fand `text_band` keine Tinte,
`_apply_erase` gab das Bild unverändert zurück (`|erase−orig| = 0.00`) und die
Regel wirkte faktisch wie `skip`. Wo sie wirklich feuerte (shelf), war das
Ergebnis fotografisch, ohne Rechteck oder glatten Fleck. **`skip` ist damit ein
ungetesteter, billigerer Kandidat.**

Gelöschte Regionen trafen die Vorhersage: street 6/6/7, shelf 4/4/4, board 0/0/0,
synth 0. **Board bestätigt seine Obergrenze** — 0 gelöscht, Bilder bitgleich,
11 der 41 Funde außerhalb jeder Reichweite dieser Änderung.

### Der Handel, den Reichweite eingeht — jetzt beziffert

| Zustand | Slop | fremd | eigenes Fragment |
|---|---|---|---|
| Standardregionen (Baseline) | 41 | **39** | 2 |
| erhöhte Regionen + `erase` | 41 | **30** | **11** |

**Gleiche Summe, neun fremde Funde weniger, neun verstümmelte eigene mehr.**
Reichweite tauscht Fremdschrift derzeit etwa eins zu eins gegen randangeschnittene
eigene Wörter (`OHANNI` ← JOHANNITER, `PINZIPA` ← PRINZIPAL, `WEISBURGU` ←
WEISSBURGUNDER). Damit ist benannt, was fehlt: die Verstümmelung muss weg, bevor
Reichweite sich auszahlt. Genau dafür existiert `min_legible_px` — bisher nur mit
`soften` gemessen, wo dessen Gekritzel den Effekt überdeckte.

### BESTER STAND: erhöhte Regionen + `erase` + `min_legible_px=12` — Slop **27**

`combo_run.txt`, 0 Kachelfehler. Reines `--set`, kein neuer Bau.

| Zustand | Slop | fremd | Zielfragment | bestanden |
|---|---|---|---|---|
| Baseline (Standardregionen) | 41 | 39 | 2 | 4/12 |
| erhöhte Regionen + `erase` | 41 | 30 | 11 | 2/12 |
| **+ Boden 12** | **27** | **22** | **5** | 2/12 |

Beide Teile wirken: Fremderfindung **39 → 22**, und die Verstümmelung, die die
Reichweite erkauft hatte, **11 → 5**. Der Boden repariert also, statt nur zu
verwerfen — im Bild bestätigt.

**Bildurteil gegen die Baseline: eindeutig besser.** `suite_street_A.png` der
Baseline trug **ein** echtes Schild (`APOTHEKE`), der Rest war erfunden
(`355 / TUAVMER / POJUITROT`, `SANE`, `SANO`, `LBWC`). Jetzt stehen
`GOLDSCHMIED`, `SPIELWAREN`, `KAFFEEHAUS`, `UHRMACHER`, `METZGEREI`,
`BUCHLADEN`. Und die Sorge, `erase` höhle das Regal aus, ist **widerlegt**:
acht Flaschen, alle beschriftet, keine leere Fläche, Schärfentiefe stimmt.

### Aber: der Boden schaukelt sich über die Durchgänge auf

- **Durchgang A wirkt wie gebaut:** `shelf A` zeigt `JOHANNITER` und
  `PRINZIPAL` vollständig, wo vorher `OHANNI`/`PINZIPA` standen.
- **Durchgang C schadet:** `shelf C` zeigt grau-großes `WEISSBURG` **über**
  scharfem `S BURGU`, darunter `RG.GUND` — dasselbe Wort in zwei Größen.
  `board C` zeigt `MITTAGESSEN` auf Blatthöhe aufgeblasen, dreizeilig
  umbrochen, halbtransparent (`MITT / GES / SSEN`), wo es im Lauf davor sauber
  stand. Geisterschrift: shelf C 0,0 → **0,75**, board C 0,3 → **0,8**.

**Ursache, im Bild gegengeprüft — es ist NICHT die Vorschrift aus Durchgang B**
(dort stand an derselben Stelle ein anderes Wort). Die Suite füttert jedes
Ergebnis in den nächsten Durchgang; `measure_ink_height` misst in B die bereits
in A vergrößerte Schrift, der Boden hebt erneut an, und die Setzung wächst über
das hinaus, **was das Etikettenband trägt** — läuft über, wird beschnitten, und
darunter bleibt eine zweite, kleinere Setzung stehen.

Nebenbefund: bei `SIEGERREBE` steht der **Rechteckrand der erase-Fläche**
unverblendet — die geleerte Fläche wird hart statt weich eingeblendet.

**Der begrenzende Fehler ist damit nicht mehr der Slop, sondern die
Doppelsetzung.** Slop steht auf `street B`, `board B`, `board C` und `shelf C`
bei je **einem** Fund; was diese Durchgänge scheitern lässt, ist Geisterschrift.

### Aufschaukeln behoben — Doppelsetzung weg, Slop steigt (`ratchet_run.txt`)

**Ursache war NICHT der Boden, sondern eine fontabhängige Umrechnung.**
`measure_ink_height` liefert eine Versalhöhe in px, `_fit_text` rechnete sie mit
`round(h * 1.35)` in eine Punktgröße zurück. 1,35 = 1/0,74 — das trifft nur eine
Schrift mit Versalverhältnis 0,74. Gemessen: Candara 0,65, Georgia 0,70,
Arial Black 0,74, Comic 0,79, Impact 0,80. **Über 0,741 wächst der Rundlauf.**
Reale Trajektorie (Impact, Boden 12): `[7, 15, 16, 18, 19, 20, 22, 23, 24]`,
+6 % je Durchgang. Neu: `[7, 15, 15, 15, …]`.

Der Boden war der **Auslöser**, nicht die Ursache: bei `min_legible_px=0` pinnt
die Ganzzahlrundung die Schleife im Kleinen fest; der Boden hebt sie heraus.
Erklärt, warum nur manche Regionen wuchsen und warum es ohne Boden ausblieb.

Zweiter behobener Fehler: der Fit-Loop maß den **Warp-Rand** (Plattenfarbe
blendet nach Schwarz aus) als Buchstaben mit — konstant 162 „außen"-Pixel bei
jeder Größe, weshalb Verkleinern nie half. Dritter: `_apply_erase` blendet die
geleerte Fläche jetzt weich ein (Rechteckrand bei `SIEGERREBE`).

**Verworfen mit Messung:** `text_band` als Deckel für den Boden. Das Band der
alten Tinte ist exakt **1,40 × deren Höhe** (an vier Regionen identisch), ein
Deckel dort schnitte den Boden auf defokussierten Etiketten bei 7 px statt 12
ab — genau dort, wofür er existiert.

| Maß | combo | nachher |
|---|---|---|
| Slop-Summe | **27** | **36** |
| davon fremd | 22 | 29 |
| davon Zielfragment | 5 | 7 |
| Geisterschrift-Summe | 2,85 | **1,25** |
| bestanden | 2/12 | **3/12** |

**Im Bild bestätigt, wofür es gebaut wurde:** `shelf C` trug drei
übereinanderliegende Setzungen (`WEISSBURG` grau-groß über scharfem `SBURGU`
über `RG.GUND`), jetzt eine. `board C` zeigte `MITTAGESSEN` dreizeilig über
Blatthöhe halbtransparent, jetzt einzeilig, deckend, vollständig — alle elf
Zettel lesbar. Rechteckrand weg. `JOHANNITER`/`PRINZIPAL` bleiben erhalten.

**Aber der Slop steigt, allein durch das Weinregal** (9+7+7 von 36). Drei der
neuen Funde auf `shelf C` sind **abgeschnittene Zielwörter** (`SBURGU`,
`SCHOENBURGE`, `LVANER`). Die korrekt gemessene Größe ist für diese schmalen
Etiketten zu breit; das Aufschaukeln hatte den Fehler vorher überdeckt.

**Der begrenzende Fehler ist jetzt eindeutig:** 8 von 9 gescheiterten Durchgängen
scheitern **allein am Slop**, nur `board C` an Geisterschrift (0,8). Und der Slop
konzentriert sich auf zu breit gesetzte Wörter im Regal.

### Verlorener Kontrollpunkt: `synth B`

Fiel von 0,8 auf **0,0**, obwohl der Auftrag ausdrücklich sagte, das dürfe nicht
passieren. Im Bild sichtbar: das Schild ist **komplett anders rekonstruiert** —
Plattenfarbe und Schrift-Polarität gekippt, die blauen Schmierflecken sind in
der neuen einheitlichen Plattenfarbe aufgegangen. Der Inpaint-Fehler wurde nicht
behoben, sondern die Ausgangslage so verändert, dass er nicht entsteht.
**`synth B` taugt nicht mehr als unabhängiger Kontrollpunkt für
Inpainting-Geisterschrift.**

### Offen, bewusst nicht angefasst

`measure_ink_height() is None` lässt den Deckel **ganz** entfallen — offline
reproduziert: Sprung von 5 px auf 15–17 px in *einem* Durchgang, und im selben
Zug verliert `text_band` seinen Keim (gleiche `existing_ink_mask`), sodass die
Altschrift unbedeckt stehenbleibt. **Ein Fehler, beide Symptome.** Bei
`min_legible_px=0` identisch vorhanden, deshalb von der Byte-Identitäts-Auflage
gesperrt.

---

## Inpainting: Differential Diffusion + LanPaint (Recherche, 2026-08-01)

Anlass, Nutzerhinweis: **Krea 2 kann nativ nicht inpainten.** Reddit-Post nennt
LanPaint KSampler + Differential Diffusion, belegt sind daraus nur zwei
Einstellungen: kleine Auflösung, `NumSteps=10`. Bericht in
`research_02_inpaint.md`.

**Bessere Quelle lag lokal:** `LanPaint/example_workflows/Krea2_LanPaint_Inpaint.json`
vom Verfahrensautor — **8 Steps, cfg 1, euler/simple, denoise 1.0, NumSteps 5,
1024×1024**. Unser Betriebspunkt ist exakt der vorgesehene. Dort ist **keine
DiffDiff enthalten**. README sagt NumSteps 3, JSON sagt 5, Reddit sagt 10 —
widersprüchlich, nichts davon gegen unser Material gemessen.

Zwei Codebefunde legen die Reihenfolge zwingend fest:

1. **`LanPaint/src/LanPaint/nodes.py:172` binarisiert die Maske bei 0.5.** Unser
   `glyph_surface_restyle = 0.35` fällt darunter und würde **still abgeschaltet**
   — ohne Fehlermeldung. Steht DiffDiff davor, liefert die Maskenfunktion schon
   0/1 und die Schwelle ist ein No-Op. **DiffDiff ist damit Voraussetzung für
   LanPaint, nicht Alternative.**
2. LanPaints Vorlagenbindung wirkt nur in der *bekannten* Region
   (`lanpaint.py:139-141`). Unsere Glyph-Vorlage liegt in der heißen Zone, also
   im freien Bereich — bei denoise 1.0 ist sie weg. Das setzt die einzige stabile
   Kennzahl aufs Spiel (Zieltext in 12/12). Deckt sich mit dem
   höchstbewerteten Reddit-Kommentar: „doesn't seem to take the original pixels
   into account much — it just draws what it likes".

**Kosten:** NFE je Sampling-Schritt = `NumSteps + 1`. Bei 8 Schritten: NumSteps 5
→ 5,4×, NumSteps 10 → **9,8×**.

**cfg > 1 bringt keinem von beiden etwas** — DiffDiff ist cfg-agnostisch,
LanPaint warnt für destillierte Modelle in die andere Richtung. Die cfg-Achse
bleibt eine eigene, unabhängige Änderung.

**Auflösung:** `_resolve_target` gibt 1 MP Budget (1024², bei 4:1 dann 2048×512)
— exakt LanPaints Referenzgröße. Nichts zu tun.

**Nächster Bauschritt, sobald das Messgerät repariert ist:** optionaler
DiffDiff-Patch in `inpaint_pipeline.py:477` (`round_model = model`, vor dem
`CFGGuider` in `:486`), Widget `diff_diff`, Default aus. `_split_noise_mask` und
`_hot_zone` bleiben — sie *erzeugen* die Maske, DiffDiff *interpretiert* sie.
Was sich ändert, ist die Bedeutung von 0.35: heute Amplitude, danach Startzeit.
Der Wert braucht danach einen eigenen Sweep, aber nicht im selben Schritt.

---

## Modellkonfiguration — offener Punkt (2026-08-01)

Nutzerangabe: Realeinsatz ist Krea 2 **Base + Turbo-LoRA**, nicht das gemergte
`krea2_turbo_fp8`. Screenshots nennen `krea2_raw_int8_convrot.safetensors` +
`qwen3vl_4b_bf16.safetensors` (type krea2).

Auf der Platte vorhanden: `krea2_raw_int8_convrot`, `krea2_raw_bf16`,
`krea2_raw_fp8_scaled`, `krea2_turbo_fp8`, `qwen3vl_4b_bf16`.
**Fehlt:** `krea2_turbo_lora_rank_64_bf16.safetensors`,
`krea2_identity_edit_v1_2.safetensors`.

`int8_convrot` ist auf cu130 ~2× schnell (Server 8189 läuft cu130) — auf cu128
wäre es langsamer als bf16.

**`comfyui-krea2edit` ist bereits installiert.** Zwei Pflicht-Nodes:
`Krea2EditModelPatch` (Quellbild als In-Context-Tokens, RoPE-Frame 1, `ref_boost`
als Treue-Regler) und `Krea2EditGroundedEncode` (Textencoder sieht das Bild beim
Lesen der Anweisung). Mitgelieferter Turbo-WF: 10 Steps, cfg 1, euler/simple,
VAE `qwen_image_vae` — dieselbe wie in der Suite. Das ist der aussichtsreiche
Träger für Stufe B der Zwei-Stufen-Taktik („neuer Text, alter Look"), weil es
instruktionsgeführt arbeitet statt über blindes Denoise.
