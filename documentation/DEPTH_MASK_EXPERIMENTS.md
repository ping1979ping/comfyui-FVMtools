# Depth & Mask Experiments — Was ausprobiert wurde

Chronologische Dokumentation der Ansätze für Depth-guided Masking und Body-Mask-Verbesserung.

---

## 1. Depth-Band Overlap Removal (v1) — VERWORFEN

**Ansatz:** Pro Person ein Depth-Band (Percentil 15-85) schätzen. Pixels außerhalb des Bands aus der Maske entfernen.

**Problem:** Beine hintereinander (gleiche Person, unterschiedliche Tiefe) wurden fälschlich entfernt. Die Depth-Band-Schätzung war zu rigide — eine Person mit ausgestrecktem Arm hat ein breites Depth-Band, was die Filterung nutzlos machte.

**Parameter:** `depth_tolerance`, `depth_remove_overlap` — hatten wenig Einfluss.

**Warum verworfen:** Fundamentaler Denkfehler — eine Person hat kein einheitliches Depth-Band. Der Ansatz funktioniert nur bei Personen die flach zur Kamera stehen.

---

## 2. Head-basierte SAM Mask Scoring — VERWORFEN

**Ansatz:** SAM gibt mehrere Mask-Kandidaten zurück. Statt die größte zu wählen, wurde nach Head-Overlap (60%) und Size-Penalty (40%) scored.

**Problem:** Bevorzugte zu kleine Masken die nur den Kopf/Oberkörper abdeckten. Unterkörper wurde abgeschnitten.

**Warum verworfen:** Head-Overlap als Scoring-Kriterium penalisiert genau die Masken die wir wollen (ganzer Körper weit vom Kopf entfernt).

---

## 3. SAM mit negativen Prompts — BEIBEHALTEN, aber nicht ausreichend

**Ansatz:** Andere erkannte Face-Zentren als negative SAM-Prompts übergeben, damit SAM weiß "das ist nicht meine Person".

**Status:** Implementiert und aktiv. Hilft bei der Mask-Generierung, aber SAM produziert trotzdem oft zu große Masken bei eng stehenden Personen (Restaurant-Szenen).

**Warum nicht ausreichend:** SAM's Prompt-basierte Segmentierung ist grundsätzlich nicht für Multi-Person-Trennung designed. Die negativen Prompts helfen, lösen aber nicht das Grundproblem.

---

## 4. Depth Edge Carving + Region Grouping — BEIBEHALTEN

**Ansatz:** Sobel auf Depth Map → Edge Map. Masken an Depth-Kanten aufschneiden. Danach Connected Components analysieren und Teile mit passender Person-Tiefe wieder zusammenführen (Pfahl-Problem).

**Status:** Implementiert, funktioniert gut für Head-Bereich (klare Depth-Kanten). Im Body-Bereich weniger effektiv weil Depth-Kanten zwischen Personen auf gleicher Tiefe schwach sind.

**Beobachtung:** `depth_edge_threshold=0.03-0.10` und `depth_carve_strength=0.80-0.95` sind gute Bereiche. Hoher `depth_grow_pixels` verbessert Head-Bereich, erweitert aber Body unkontrolliert.

---

## 5. Cross-Reference Deconfliction — BEIBEHALTEN

**Ansatz:** Nach Mask-Generierung: wo sich Masken verschiedener Referenzen überlappen, gewinnt die Person mit der näheren Depth (Core-Depth aus nicht-überlappenden Pixeln).

**Status:** Implementiert und funktioniert! Bug war: Preview zeigte die alten Masken VOR Deconfliction. Nach Fix (Preview zeigt deconflicted Masken) war das Ergebnis deutlich besser.

**Wichtig:** Core-Depth wird nur aus NICHT-überlappenden Pixeln berechnet, damit die Overlap-Pixels den Median nicht verfälschen.

---

## 6. Seed-Grow Body Masks (BiSeNet) — AKTUELL IN TEST

**Ansatz:** Statt SAM für Body-Masken → BiSeNet Labels als Seed (alle nicht-Background Labels 1-18: Kopf + Kleidung + Hals + Schmuck). Dann iterativ wachsen, gestoppt von Image-Edges (Canny) und Depth-Edges (Sobel).

**Vorteile:**
- Seeds sind per-Person exklusiv (kein Overlap möglich)
- Image-Edges sind das stärkste Signal für Körperkonturen
- Kein SAM nötig (schneller)
- BiSeNet Label 16 (cloth) war bisher ungenutzt

**Offene Fragen:**
- Wie gut funktioniert es bei Personen ohne sichtbare Kleidung?
- Reicht der Grow-Radius für ganzkörper?
- Canny-Parameter (50/150) — müssen evtl. angepasst werden

**Parameter:** `body_mask_mode` = "auto" | "seed_grow" | "sam"

---

## Erkenntnisse

1. **BiSeNet Head-Masken sind zuverlässig** — können als Vertrauensanker dienen
2. **SAM Body-Masken sind unzuverlässig** bei Multi-Person — zu groß, inkl. Möbel/andere Personen
3. **Depth-Edges sind nützlich** aber nur im Head-Bereich wirklich stark
4. **Image-Edges (Canny) sind unterschätzt** — Körperkonturen sind im Bild klar sichtbar
5. **Preview muss deconflicted Masken zeigen** — sonst sieht man den Effekt nicht
6. **Inside-Out (Seed→Grow) ist besser als Outside-In (SAM→Trim)**
