# Prompt: übrige Location-Sets auf die Krea-2-Regeln bringen (nach /clear)

> In eine neue Claude-Code-Sitzung im Repo `D:/AI/ComfyUI/ComfyUI/custom_nodes/comfyui-FVMtools` einfügen.

---

Bring die unten gelisteten 56 Location-Sets von FVMtools auf die render-geprüften Krea-2-Regeln. Arbeite autonom und agentisch, mit Sonnet-Agents. Teste so lange, bis jedes Set über mehrere Seeds reproduzierbar realistisch rendert.

## Grundlagen, zuerst vollständig lesen

- `.claude/commands/build-location-set.md`: vor allem **Step 2c „Krea 2 composition rules“**, dazu Step 3 bis 5b. Ergebnis der Messungen vom 2026-10-09:
  - Krea malt jedes Nomen groß und neben die Person. Hintergrund und Mittelgrund tragen deshalb die Szene, mit relativer Ortsangabe.
  - Props sind klein und fern verankert, im Vordergrund steht nur der Boden.
  - Sitzmöbel bekommen immer `empty`.
  - Kein frei stehendes `tiny in the distance`, das malt eine kleine Figur.
  - Keine Personenwörter, kein impliziter Text, keine Klammern, keine Teppichfliesen.
- `.claude/commands/krea2-set-test.md` zum Rendern und Beurteilen, dazu `scripts/krea2_set_check.py`.
- `tests/unit/test_location_krea2_rules.py`: Lint. Jedes fertige Set kommt in `KREA2_STRICT_EXTRA_SETS`. Kategorien kommen nicht in `KREA2_STRICT_CATEGORIES`, weil sie auch ausgeschlossene Sets enthalten.
- Vorbilder im Zielstil: `location_lists/indoor/luxury_hotel/hotel_bar/`, `location_lists/outdoor/american_everyday_scene/sidewalk_cafe_patio/`, `location_lists/indoor/reading_pennsylvania_corporate_office/open_plan_marketing_department/`.

## Scope: genau diese 56 Sets, Slugs und Dateinamen bleiben

- `indoor/american_everyday_scene`: classic_american_diner, dentist_waiting_room, gas_station_convenience_store, mcdonalds_dining_booth, neighborhood_sports_bar, old_fashioned_ice_cream_parlor, target_clothing_section, walmart_aisle
- `indoor/american_fitness_spot`: community_yoga_studio, strip_mall_gym
- `indoor/american_vacation_stay`: poconos_cabin_living_room, shore_rental_kitchen
- `indoor/german_everyday_scene`: german_bakery_cafe, german_supermarket_aisle, kitchen_while_cooking, living_room_with_sofa, traditional_german_pub
- `indoor/household_scene`: attic_storage, basement_utility_room, breakfast_nook, country_farmhouse_kitchen, garage_workshop, laundry_room, modern_kitchen, mudroom_entry, sun_room, walk_in_pantry
- `outdoor/american_everyday_scene`: apartment_complex_dumpster_area, chain_restaurant_drive_thru, cvs_parking_lot, dollar_general_storefront, gas_station_pump_island, subdivision_sidewalk, walmart_garden_center_exit
- `outdoor/american_fitness_spot`: township_track
- `outdoor/american_winter_vacation`: empty_ski_slope_edge, frozen_lake_shore, snowed_in_cottage_yard
- `outdoor/german_everyday_scene`: apartment_balcony, german_beer_garden, pedestrian_zone
- `outdoor/household_scene`: backyard_clothesline, classic_front_porch, compost_bins_corner, garden_tool_shed, vegetable_garden_bed
- `outdoor/luxury_hotel`: hotel_terrace_cafe, rooftop_hotel_bar
- `outdoor/pennsylvania_nature_spot`: fall_woods_walk, state_park_trailhead, summer_creek_bank
- `outdoor/pennsylvania_small_town`: diner_parking_lot, small_town_main_street, strip_mall_sidewalk
- `outdoor/pennsylvania_suburb`: backyard_deck_with_grill, vinyl_sided_front_porch

**Alle anderen Location-Sets fasst du nicht an.** Das gilt auch für Kategorien und Sets mit Schule, Teens, Kindern oder Familie, für `private_space`, `private_home_room`, Schlaf- und Badezimmer, Motel- und Hotelzimmer, Pool- und Strand-Settings, `college_scene` und `prom_night`. Outfit-Sets bleiben ebenfalls unverändert.

## Ablauf

1. **Baseline:** `git status` (muss sauber sein), `git stash list`, und `pytest tests/unit -q -p no:cacheprovider`. Bekannt sind 11 Fehler, siehe `.planning/PROMPT_unit_test_cleanup.md`. Speichere die Liste.
2. **Wellen:** höchstens 6 Sonnet-Agents parallel, je eine Gruppe mit 2 bis 6 Sets. Jeder Agent bekommt:
   - die Pfade seiner Sets und den Hinweis „in place umschreiben, Thema behalten, realistisch heute“;
   - die Pflichtlektüre oben;
   - die Fehlerbilder der alten Version: Objektlisten, Objekte im Vordergrund, Personen, Witz-„weather“, impliziter Text;
   - die Regel „nur in den eigenen Set-Ordnern schreiben, nicht committen“;
   - den Validierungsbefehl:
     `pytest tests/unit/test_location_krea2_rules.py tests/unit/test_location_lists_extended.py tests/unit/test_set_naming.py tests/unit/test_smp_location.py -q -p no:cacheprovider -k "<set1> or <set2> …"`
   - Vorher trägst du die Sets der Welle selbst in `KREA2_STRICT_EXTRA_SETS` ein, damit der Lint greift.
   - Sets einer Gruppe ohne natürliche Sitzmöbel brauchen kein `empty`. Der Lint verlangt es nur, wenn Sitzmöbel genannt werden.
3. **Generator-Fallen:** Prüfe `scripts/gen_*.py` (`gen_everyday_locations.py`, `gen_everyday_sets.py`, `gen_us_scenario_locations.py`, `gen_vacation_water_locations.py`, …), ob sie die alten Daten eines umgeschriebenen Sets enthalten. Wenn ja, entfernst du den Block. Statt dessen kommt ein Kommentar wie in `gen_us_scenario_locations.py` Zeile ~163 hinein, damit `--force` die Arbeit nicht zurückdreht.
4. **Render-Check pro Welle, das ist dein Job und nicht der der Agents:**
   `scripts/krea2_set_check.py --location <sets> --outfit female/business/office_day_business_casual_with_heels --seeds 11 22 --accessories --bag --port <port> --out "<scratchpad>/<welle>.jpg"`
   - Nutze **8189** (Testinstanz), wenn die Queue leer ist. Prüf das vorher mit `curl -s http://127.0.0.1:8189/queue`. 8188 nur, wenn ich es in der Sitzung freigebe.
   - **Sieh dir den Bogen selbst an** (Read auf das .jpg). Pro Bild prüfen:
     - genau eine Person, Kopf und Schuhe im Bild;
     - Props klein und fern, nichts riesig neben der Person;
     - der Hintergrund liest sich als der genannte Ort;
     - keine Kauderwelsch-Schrift, kein Schmierfleck, der vom Text kommt.
   - Ein vereinzelter dunkler Bodenfleck ist ein bekannter cielbleu-Effekt (ca. 1 von 30). Kommt er wiederholt im selben Set vor, prüf die Bodenformulierung.
   - Fehler: Eintrag fixen (die Prompts stehen im JSONL neben dem Bogen), mit denselben Seeds neu rendern, wiederholen.
5. **Commit pro Welle,** nachdem Lint und Render bestanden sind:
   `refactor(lists): rewrite <gruppe> sets to the Krea 2 rules`, mit Liste der Sets und Render-Befund. Nur explizite Pfade adden, kein `git add -A`. Hänge die Attribution-Zeile der Sitzung an.
6. **Abschluss:**
   - Volle Suite `pytest tests/unit -q -p no:cacheprovider`. Rot dürfen nur die 11 bekannten Fehler sein, gib die rohe letzte Zeile aus.
   - Dann `git push manual main`. `origin` ist absichtlich gesperrt.
   - Melde mir pro Gruppe einen Satz zum Befund und die Commit-Hashes.

## Regeln

- **Beweis ist der Bogen, nicht die Agent-Meldung.** Kein Set gilt als fertig, solange du seine Bilder nicht selbst angesehen hast.
- Nur Daten-Dateien in `location_lists/`, dazu der Lint-Eintrag und die Generator-Blöcke. Am Engine-Code und an Nodes änderst du nichts.
- Keine Personen- oder Rollenwörter (`busy`, `crowd`, `cashier`, `bartender`, `kids` …). Jahreszeiten-Details stehen nur in `weather.txt`, Hintergrund und Mittelgrund bleiben jahreszeitenneutral.
- Wenn ein Regex-Backslash wie Wortgrenze-b in eine Datei soll: über `chr(92)` erzeugen und mit `grep -c $'\x08'` prüfen. Über Tool-Calls wird er sonst zum Backspace-Zeichen.
