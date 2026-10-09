# Prompt: autonome Bereinigung der Unit-Tests (FVMtools)

> Zum Einfügen in eine neue Claude-Code-Sitzung im Repo `D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools`.

---

Bereinige die Unit-Tests von FVMtools autonom, bis `tests/unit` vollständig grün ist. Mach das ohne Rückfragen, solange du die Regeln unten einhältst. Melde dich nur, wenn eine Entscheidung wirklich bei mir liegt; die Fälle stehen unter „Stopp-Regeln“.

**Ausgangslage (Stand 2026-10-09, HEAD nach cbe1a35):** `tests/unit` hat 13289 grüne und 11 rote Tests:

1. `tests/unit/test_jb_builder.py` (8 Tests): `test_input_types_only_rows_and_format`, `test_rows_with_user_hosiery_example`, `test_rows_strip_quote_chars_from_value_in_loose_keys`, `test_loose_keys_output_has_no_extraneous_quotes`, `test_empty_rows_emits_empty_object`, `test_invalid_rows_falls_back_to_empty`, `test_compact_format`, `test_pretty_json_round_trips`.
   Vermutete Ursache: Seit Commit d4b32ed (2026-09-22) hat `FVM_JB_Builder` einen `seed`-Eingang, den die Tests weder erwarten noch übergeben.
2. `tests/unit/test_outfit_generator.py::TestSchema::test_outfit_set_contains_general_female` und `tests/unit/test_smp_outfit_v2.py::test_records_set_dropdown_includes_known_sets`. Beide erwarten den flachen Legacy-Slug `general_female`. Seit der Hierarchie-Migration heißen Sets `female/<category>/<leaf>`, flache Slugs liegen unter `outfit_lists/_archive/`.
3. `tests/unit/test_jb_sentences.py::TestMetadata::test_block_output_has_no_structure_characters`. Das Set `indoor/family_event/graduation_party_at_home` enthält seit Commit 8e668bb absichtlich `"CONGRATS"`, also wörtlichen Text für Krea 2. Der Test verbietet `"` aber pauschal in der Block-Ausgabe.

## Vorgehen

1. **Baseline:** `D:/AI/ComfyUI/ComfyUI/venv/Scripts/python.exe -m pytest tests/unit -q -p no:cacheprovider`. Speichere die Liste der roten Tests als Ausgangspunkt. Sind es nicht genau die 11 oben, arbeite mit der echten Liste und vermerke die Abweichung.
2. **Pro rotem Test zuerst entscheiden: Ist der Test veraltet oder der Code kaputt?**
   - Lies `git log -p` der beteiligten Dateien und die Commit-Message, die die Änderung eingeführt hat. War die Änderung beabsichtigt (neuer `seed`-Eingang, Hierarchie-Migration, absichtlich zitierter Text), dann passt du den Test an das gewollte Verhalten an.
   - Ist es eine Regression, reparierst du den Code.
   - Schreib die Entscheidung je Testgruppe in einem Satz in die Commit-Message.
3. **Tests anpassen, nicht aufweichen:**
   - Kein `skip`, `xfail` oder auskommentierter Assert, und keine Assertion so weit lockern, dass sie nichts mehr prüft.
   - Erwartet ein Test einen alten Slug, prüft er künftig das aktuelle Äquivalent, z. B. `female/casual/general`. Schau vorher in `scripts/rename_sets_map.txt` und `core/outfit_lists.py` (`_resolve_legacy_outfit_slug`), und behalte den Legacy-Fallback als eigenen Testfall, falls er noch existiert.
   - Für `"` in der Sentences-Ausgabe: Wörtlich zitierter Text (`reading "CONGRATS"`) ist erlaubt. Der Test soll weiterhin echte JSON-Strukturzeichen und Schlüssel-Leaks fangen. Die bessere Lösung sind gezielte Regeln wie keine `{`, `}`, `[` oder `":`, kein unpaariges `"`. Die Set-Daten änderst du nicht.
   - Für den Builder: Der Test übergibt `seed` und prüft dazu, dass derselbe Seed dieselbe Ausgabe liefert.
4. **Nach jeder Testgruppe** läuft die ganze Suite (`tests/unit -q -p no:cacheprovider`). Eine neue rote Stelle reparierst du, bevor du weitermachst.
5. **Pro Gruppe ein Commit** im Conventional-Stil (`test(jb): …`, `fix(outfit): …`) mit der Zeile `Co-Authored-By: …` laut Attribution-Vorgabe der Sitzung. **Nicht pushen.** Das Remote `origin` ist absichtlich kaputt, das echte heißt `manual`, und Pushen ist nicht Teil dieser Aufgabe.
6. **Abschluss:** Zeig die rohe letzte Zeile des vollen Laufs, keine Zusammenfassung. Dazu `git log --oneline` deiner Commits und je Gruppe einen Satz zu Ursache und Entscheidung.

## Stopp-Regeln (dann fragen statt handeln)

- Ein Fix würde das Verhalten eines Nodes für bestehende Workflows ändern, also Widget-Reihenfolge, `INPUT_TYPES`, Ausgabe-Text oder Seeds.
- Du findest keine Commit-Absicht und kannst nicht entscheiden, ob Test oder Code recht hat.
- Ein Test verlangt Daten-Änderungen an `location_lists/` oder `outfit_lists/`. Die passt du nicht an, nur um einen Test grün zu bekommen.

## Nicht anfassen

- `LaunchScripts/`, die laufenden ComfyUI-Instanzen und `git stash`. Prüf `git stash list` vorher und nachher und melde Änderungen.
- `tests/unit/test_location_krea2_rules.py` und `tests/unit/test_outfit_hairstyle_headwear.py` sind neu und grün. Schlagen sie fehl, liegt das an deiner Änderung.
