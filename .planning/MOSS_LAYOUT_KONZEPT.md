# MOSS-Layout · Konzept (Stand 2026-10-23)

Ausgangspunkt: `JB · MOSS Layout from SAM3` (Commit df98dbb) baut aus Referenzbild + JB-Prompt ein
MOSS-Box-JSON. Der User hat den Beispiel-Workflow um eine zweite Person mit eigener Personen-LoRA
erweitert. Problem: die LoRA wirkt global und macht beide Personen gleich.

## Gemeinsames Datenformat `MOSS_BOXES`

Eine Python-Dict (Socket-Typ `MOSS_BOXES`), erzeugt von `core.jb.moss_layout.make_boxes_doc`,
gerendert von `render_moss(doc, w, h)`:

```
{"version": 1, "src_w", "src_h", "width", "height", "up_width", "up_height",
 "summary": {"caption", "meta": [...], "canvas"},
 "boxes": [{"slot", "label": character|object|text, "caption", "text", "prompt",
            "rect": [x1, y1, x2, y2] (0..1), "score", "source": sam3|user|ideogram}]}
```

`rect` ist relativ → dieselben Boxen gelten für Pass 1, Refine und Masken jeder Auflösung.

## Häppchen

| # | Inhalt | Wer | Dateien |
|---|---|---|---|
| H0 | Commit + Push des Ist-Stands | erledigt (df98dbb) | — |
| H1 | Test: LoRA nur in einer Box (Hook-LoRA + Masken-Conditioning) | erledigt, funktioniert | — |
| H2 | `MOSS_BOXES`-Format + Ausgang am SAM3-Layout-Node | erledigt | core/jb/moss_layout.py |
| H3 | **Regeln selbst vorschlagen**: Vokabular-Scan mit SAM3, `rules_mode` manual/auto/both, Ausgang `suggested_rules`, Button „Vorschlag übernehmen“; Regel-Spalte 4 = fester Box-Text; Ausgang `ideogram_json` zum Füttern des KJ-Editors | Agent A | nodes/jb/moss_layout.py, core/jb/moss_layout.py, web/js/fvm_moss_layout.js |
| H4 | **Editieren im Ideogram-Editor**: neuer Node `JB · MOSS Layout from Ideogram` — liest das Caption-JSON des KJ „Ideogram 4 Prompt Builder“ (verschieben, Texte überschreiben, neue Boxen zeichnen) bzw. von `FVM_Ideogram_BoxJitter` (zufällig verschieben); `desc` = Slot-Key (wird aus JB gefüllt), `character: …` oder freier Text | Agent B | nodes/jb/moss_from_ideogram.py, core/jb/moss_ideogram.py |
| H5 | **Boxen als Personen-Finder**: `JB · MOSS Boxes → Person Data` — SAM3 im gerenderten Bild, Treffer per Box-Überlappung den Slots zuordnen → PERSON_DATA für den PersonDetailer (reference_i = character_i) + Slot-Prompts | Agent C | nodes/jb/moss_box_person.py |
| H6 | **Regional-LoRA**: `JB · MOSS Regional LoRA` — Conditioning + MOSS_BOXES + bis zu 4 (Slot, LoRA, Stärke) → Hook-LoRA je Box, Rest ohne; Feather | Hauptlauf | nodes/jb/moss_regional_lora.py |
| H7 | Workflow aufräumen: mehr Abstand, Gruppen, Regional-LoRA in Pass 1 + Refine, PersonDetailer-Gruppe für die Personen, Editor-Strecke als Alternative, Anleitung | Hauptlauf | examples/, user/default/workflows/ |
| H8 | Live-Test 8188 (Bilder ansehen), UI-Ladetest, Commit + Push | Hauptlauf | — |

## Stand nach Umsetzung (2026-10-23)

Alle Häppchen H2–H7 umgesetzt, Unit-Suite 14636 grün, Kette live auf 8188 aus dem vom Frontend
serialisierten Workflow gelaufen (Bilder geprüft). Nachgezogen nach dem ersten Live-Lauf:
Vokabular-Scan hatte 21/35 Treffer mit Doppelboxen (stool/chair/bench/lamp = eine Box) →
`auto_threshold` 0,6, `pick_auto_hits` (eine Box pro Ding über alle Wörter, `max_auto_boxes` 8) → 6/35.

Offen: Button „use suggested rules“ nicht im Browser geklickt; `character: …`-Boxen aus dem Editor
bekommen keinen Slot (PersonDetailer sieht sie nicht); Trigger-Wörter in Captions landen gern als
Schrift im Bild (z. B. auf Tassen).

## H1 · Testergebnis (8188, gleicher Seed, 1 MP, 4 Steps)

- A global (jg 0,75 + dlbody 0,3): beide Frauen gleichen sich an (rechte wird älter, gleiche Gesichtszüge).
- B Hook-LoRA nur in der Box von character_1, Rest-Conditioning mit inverser Maske, Feather 48:
  links = LoRA-Person wie A, rechts = identisch mit C (ohne LoRA). Keine Naht sichtbar. 30 s statt 6 s.
- C ohne Personen-LoRA.

→ Regional-LoRA ist der Hauptweg für die Generierung; der PersonDetailer (H5) ist der Nachgang
für Gesicht/Details mit derselben LoRA je Person.

## Schnittstellen-Regeln

- Keine Imports aus anderen Extensions (KJ, moss-image-tools): nur über Workflow-Verbindungen.
- Ideogram-Caption-Format: `compositional_deconstruction.elements[]` mit `type` obj/text, `desc`,
  `bbox` [ymin, xmin, ymax, xmax] auf 0–1000.
- Slot-Erkennung im `desc`: exakter JB-Key → Text aus JB; `character: …` → Label character;
  Key mit Präfix `character`/`person` → Label character; sonst object; `type: text` → text.
