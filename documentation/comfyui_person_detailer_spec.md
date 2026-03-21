# ComfyUI PersonDetailer & Person Selector Multi Batch-Update — Spec

## Übersicht

Zwei Custom Nodes für einen automatisierten, LoRA-basierten Face-Detailing-Workflow in ComfyUI.
Ziel: **Keine For-Each-Schleife** mehr im Workflow. Alles batch-fähig, alles intern.

```
image batch [B] → Person Selector Multi (batch) → PersonDetailer → image batch [B]
```

**Keine externen Dependencies** außer ComfyUI-Core. Detail Daemon wird intern implementiert.

---

## Teil 1: Person Selector Multi — Batch-Update

### Aktueller Zustand
- Verarbeitet ein Einzelbild
- Gibt Masken pro Referenz aus (face_masks, head_masks als Batch indiziert nach Referenz)
- Outputs: face_masks, head_masks, combined_face, matched, faces, preview, similarities, matches, matched_count, face_count, report

### Änderungen für Batch-Verarbeitung

**Input `current_image`** akzeptiert jetzt auch Batches `[B, H, W, C]`.

**Interne Verarbeitung:**
- Loop über jedes Bild im Batch
- Pro Bild: bestehende Erkennungslogik ausführen (Face Detection, Embedding-Vergleich, SAM-Segmentierung)
- Ergebnisse in einem Custom-Datentyp `PERSON_DATA` bündeln

### Custom Type: PERSON_DATA

Wird als einzelne Verbindung zwischen Person Selector Multi und PersonDetailer übergeben.

```python
PERSON_DATA = {
    "batch_size": int,              # z.B. 4
    "num_references": int,          # z.B. 5 (Anzahl angeschlossener Referenzen)
    "image_height": int,
    "image_width": int,
    
    # Masken pro Referenz, pro Bild im Batch
    # Shape jeweils: [B, H, W]
    "face_masks": [
        tensor[B, H, W],   # Referenz 1
        tensor[B, H, W],   # Referenz 2
        tensor[B, H, W],   # Referenz 3
        tensor[B, H, W],   # Referenz 4
        tensor[B, H, W],   # Referenz 5
    ],
    "head_masks": [         # gleiche Struktur
        tensor[B, H, W],
        ...
    ],
    "body_masks": [         # gleiche Struktur
        tensor[B, H, W],
        ...
    ],
    
    # Matches pro Batch-Item, pro Referenz
    # matches[batch_idx][ref_idx] = True/False
    "matches": [
        [True, True, True, False, False],   # Bild 0
        [True, False, True, False, False],  # Bild 1
        [True, True, True, True, False],    # Bild 2
        [True, True, False, False, False],  # Bild 3
    ],
    
    # Kombinierte Maske aller erkannten Gesichter (matched + unmatched)
    "all_faces_mask": tensor[B, H, W],
    
    # Kombinierte Maske nur der gematchten Gesichter
    "matched_faces_mask": tensor[B, H, W],
    
    # Daraus ableitbar: unmatched = all_faces - matched_faces
}
```

### Outputs der Person Selector Multi (aktualisiert)

| Output | Typ | Beschreibung |
|--------|-----|-------------|
| person_data | PERSON_DATA | Custom Type für PersonDetailer |
| face_masks | MASK | Wie bisher, für Einzelbild-Kompatibilität (erstes Bild im Batch oder Gesamtbatch) |
| head_masks | MASK | Wie bisher |
| combined_face | MASK | Wie bisher |
| matched | INT | Anzahl Matches (Summe über Batch) |
| faces | INT | Anzahl Gesichter (Summe über Batch) |
| preview | IMAGE | Preview-Komposit |
| similarities | STRING | Similarity-Werte |
| matches | STRING | Match-Info als Text |
| matched_count | INT | Matched count |
| face_count | INT | Face count |
| report | STRING | Report als Text |

**Wichtig:** Die bisherigen Outputs bleiben für Abwärtskompatibilität erhalten. `person_data` ist der neue Haupt-Output für den PersonDetailer.

---

## Teil 2: PersonDetailer Node — Neubau

### Konzept

Eine All-in-One Node die pro Bild im Batch sequentiell über alle aktiven Referenz-Slots iteriert, jeweils mit eigenem LoRA und Prompt inpaintet, und optional unerkannte Gesichter generisch behandelt.

### Inputs (Sockets)

| Input | Typ | Beschreibung |
|-------|-----|-------------|
| images | IMAGE | Bild-Batch [B, H, W, C] |
| person_data | PERSON_DATA | Custom Type aus Person Selector Multi |
| model | MODEL | Basis-Modell (ohne LoRAs) |
| clip | CLIP | CLIP Model |
| vae | VAE | VAE |
| positive_base | CONDITIONING | Optionaler Basis-Positive-Prompt (wird mit Slot-Prompt kombiniert) |
| negative | CONDITIONING | Globaler Negative-Prompt |

### Widgets — Globale Sampler-Settings

| Widget | Typ | Default | Beschreibung |
|--------|-----|---------|-------------|
| seed | INT | 0 | Globaler Seed, fix für alle Durchläufe |
| steps | INT | 4 | Sampling Steps |
| denoise | FLOAT | 0.52 | Denoise Strength |
| sampler_name | COMBO | "euler" | Sampler-Auswahl |
| scheduler | COMBO | "simple" | Scheduler-Auswahl |

### Widgets — Detail Daemon (Global)

| Widget | Typ | Default | Beschreibung |
|--------|-----|---------|-------------|
| detail_daemon_enabled | BOOLEAN | True | Global an/aus |
| detail_amount | FLOAT | 0.20 | |
| dd_start | FLOAT | 0.20 | |
| dd_end | FLOAT | 0.80 | |
| dd_bias | FLOAT | 0.50 | |
| dd_exponent | FLOAT | 0.99 | |
| dd_start_offset | FLOAT | 0.00 | |
| dd_end_offset | FLOAT | 0.00 | |
| dd_fade | FLOAT | 0.00 | |
| dd_smooth | BOOLEAN | True | |

### Widgets — Inpaint Settings (Global)

| Widget | Typ | Default | Beschreibung |
|--------|-----|---------|-------------|
| mask_blend_pixels | INT | 32 | |
| mask_expand_pixels | INT | 0 | |
| mask_fill_holes | BOOLEAN | True | |
| mask_invert | BOOLEAN | False | |
| context_from_mask_extend_factor | FLOAT | 1.20 | |
| output_resize_to_target | BOOLEAN | True | |
| output_target_width | INT | 800 | |
| output_target_height | INT | 1200 | |
| output_padding | INT | 32 | |
| mask_hipass_filter | FLOAT | 0.10 | |
| device_mode | COMBO | "gpu (much faster)" | gpu/cpu |

### Widgets — Referenz-Slots (1–5 + Generisch)

Pro Slot identische Struktur. Slot 6 = Generisch (für unmatched Faces).

| Widget | Typ | Default | Beschreibung |
|--------|-----|---------|-------------|
| ref_N_enabled | BOOLEAN | False | Slot an/aus — wenn aus, wird komplett übersprungen |
| ref_N_detail_daemon | BOOLEAN | True | Detail Daemon für diesen Slot nutzen |
| ref_N_lora | COMBO | "None" | LoRA-Dateiauswahl (folder_paths.get_filename_list("loras") + ["None"]) |
| ref_N_lora_strength | FLOAT | 1.0 | LoRA Strength (0.0 – 2.0) |
| ref_N_prompt | STRING | "" | Positiver Prompt für diesen Charakter (multiline) |
| ref_N_mask_type | COMBO | "face" | Welche Maske verwenden: "face", "head", "body" |

Wobei N = 1, 2, 3, 4, 5, generic

**Hinweis Slot "generic":** Verwendet die Maske `all_faces - matched_faces` (unmatched Gesichter). Wird nur ausgeführt wenn unmatched Faces vorhanden UND Slot enabled.

### Widgets — Zusammenfassung

```
Globale Sampler:       5 Widgets
Detail Daemon:        10 Widgets  
Inpaint Settings:     11 Widgets
6 Slots × 6 Widgets:  36 Widgets
─────────────────────────────────
Gesamt:               ~62 Widgets
```

### Interne Verarbeitung (Pseudocode)

```python
def process(images, person_data, model, clip, vae, negative, **kwargs):
    batch_size = images.shape[0]
    results = []
    refined_parts = []
    
    total_steps = batch_size * count_active_slots()
    current_step = 0
    
    for batch_idx in range(batch_size):
        current_image = images[batch_idx]  # [H, W, C]
        
        # === REFERENZ-SLOTS 1-5 (sequentiell) ===
        for slot_idx in range(5):
            if not slot_enabled(slot_idx):
                continue
            
            # Maske für diese Referenz, dieses Bild
            mask_type = get_mask_type(slot_idx)  # "face", "head", "body"
            mask = person_data[f"{mask_type}_masks"][slot_idx][batch_idx]  # [H, W]
            
            # Prüfe ob Maske leer (kein Match)
            if mask.sum() < 1.0:
                print(f"  [Batch {batch_idx+1}/{batch_size}] Reference {slot_idx+1} — no match, skipping")
                continue
            
            current_step += 1
            print(f"  [{current_step}/{total_steps}] [Batch {batch_idx+1}/{batch_size}] Reference {slot_idx+1} — detailing...")
            
            # LoRA laden (temporär)
            lora_name = get_lora(slot_idx)
            lora_strength = get_lora_strength(slot_idx)
            if lora_name != "None":
                patched_model, patched_clip = load_lora(model, clip, lora_name, lora_strength)
            else:
                patched_model, patched_clip = model, clip
            
            # Prompt encodieren
            slot_prompt = get_prompt(slot_idx)
            if slot_prompt:
                positive_cond = clip_encode(patched_clip, slot_prompt)
            else:
                positive_cond = positive_base  # Fallback auf Basis-Prompt
            
            # Inpaint-Pipeline
            # 1. Crop (Inpaint-CropAndStitch vorbereiten)
            cropped_image, cropped_mask, stitch_data = inpaint_crop(
                current_image, mask, **inpaint_settings
            )
            
            # 2. VAE Encode
            latent = vae_encode(vae, cropped_image)
            
            # 3. Set Latent Noise Mask
            latent = set_noise_mask(latent, cropped_mask)
            
            # 4. Sigmas berechnen (mit oder ohne Detail Daemon)
            sigmas = compute_sigmas(patched_model, steps, scheduler, denoise)
            if detail_daemon_enabled and slot_detail_daemon(slot_idx):
                sigmas = apply_detail_daemon(sigmas, detail_amount, dd_start, dd_end, 
                                             dd_bias, dd_exponent, dd_fade, dd_smooth)
            
            # 5. Sample
            noise = generate_noise(seed)  # Fixer globaler Seed
            guider = BasicGuider(patched_model, positive_cond)
            sampled = ksampler(noise, guider, sampler, sigmas, latent)
            
            # 6. VAE Decode
            decoded = vae_decode(vae, sampled)
            
            # 7. Refined part speichern (vor dem Stitch)
            refined_parts.append(decoded)
            
            # 8. Stitch zurück
            current_image = inpaint_stitch(stitch_data, decoded)
        
        # === GENERISCHER SLOT (unmatched Faces) ===
        if generic_enabled:
            # Unmatched-Maske berechnen
            unmatched_mask = person_data["all_faces_mask"][batch_idx] - person_data["matched_faces_mask"][batch_idx]
            unmatched_mask = unmatched_mask.clamp(0, 1)
            
            if unmatched_mask.sum() >= 1.0:
                print(f"  [Batch {batch_idx+1}/{batch_size}] Generic (unmatched faces) — detailing...")
                
                # Gleiche Pipeline wie oben, mit Generic-LoRA/Prompt
                # ABER: unmatched_mask kann mehrere Gesichter enthalten
                # Option A: Alle auf einmal inpainten
                # Option B: Connected Components finden, einzeln inpainten
                # → Option A als Default, ist simpler und schneller
                
                current_image = run_inpaint_pipeline(
                    current_image, unmatched_mask, 
                    generic_lora, generic_prompt, ...
                )
        
        results.append(current_image)
    
    # Ergebnis als Batch stapeln
    output_images = torch.stack(results)       # [B, H, W, C]
    output_refined = torch.stack(refined_parts) # [N, H, W, C] (N = Anzahl tatsächlicher Detailings)
    
    return (output_images, output_refined)
```

### Outputs

| Output | Typ | Beschreibung |
|--------|-----|-------------|
| images | IMAGE | Fertige Bilder als Batch [B, H, W, C] |
| refined | IMAGE | Alle refined/cropped Parts als Batch (vor dem Stitch-Zurück) |
| preview | IMAGE | Zeigt immer den aktuell gedetailten Bereich (letzter Refined Part) |

### Terminal-Output

```
╔══════════════════════════════════════════╗
║         PersonDetailer v1.0              ║
╠══════════════════════════════════════════╣
║ Batch: 4 images | Active slots: 3 + gen ║
╠══════════════════════════════════════════╣
║ [Batch 1/4]                              ║
║   Reference 1 — detailing...     ✓       ║
║   Reference 2 — detailing...     ✓       ║
║   Reference 3 — no match, skip   ─       ║
║   Generic     — 1 unmatched face ✓       ║
║ [Batch 2/4]                              ║
║   Reference 1 — detailing...     ✓       ║
║   Reference 2 — no match, skip   ─       ║
║   Reference 3 — detailing...     ✓       ║
║   Generic     — no unmatched     ─       ║
║ ████████████████░░░░░░░░ 50%             ║
╚══════════════════════════════════════════╝
```

### Detail Daemon — Interne Implementierung

Sigma-Manipulation basierend auf den Detail-Daemon-Parametern. Keine externe Dependency.

```python
def apply_detail_daemon(sigmas, detail_amount, start, end, bias, exponent, 
                         start_offset, end_offset, fade, smooth):
    """
    Moduliert die Sigma-Kurve um Detail-Erhaltung zu steuern.
    
    - detail_amount: Stärke der Modulation (0 = keine Änderung)
    - start/end: Bereich der Sigma-Kurve der moduliert wird (0.0-1.0)
    - bias: Verschiebung der Modulations-Kurve
    - exponent: Form der Kurve
    - fade: Ein/Ausblendung an den Rändern
    - smooth: Glättung der Kurve
    
    Kern-Logik: Sigmas werden im definierten Bereich skaliert,
    so dass der Sampler in diesem Bereich mehr/weniger Detail erzeugt.
    """
    # Implementierung analog zum Detail Daemon Custom Node:
    # 1. Normalisiere Sigma-Range auf [0, 1]
    # 2. Erstelle Modulations-Maske basierend auf start/end/bias/exponent
    # 3. Wende Fade an den Rändern an
    # 4. Optional Smoothing
    # 5. Skaliere Sigmas: sigma_new = sigma * (1 + detail_amount * modulation)
    
    n = len(sigmas) - 1  # letztes sigma ist 0
    if n <= 0 or detail_amount == 0:
        return sigmas
    
    modified = sigmas.clone()
    
    for i in range(n):
        progress = i / n  # 0.0 bis 1.0
        
        # Innerhalb des aktiven Bereichs?
        if progress < start or progress > end:
            continue
        
        # Normalisierte Position innerhalb des Bereichs
        local_progress = (progress - start) / (end - start)
        
        # Bias anwenden
        local_progress = local_progress ** (1.0 / max(bias, 0.001))
        
        # Exponent für Kurvenform
        modulation = local_progress ** exponent
        
        # Fade an den Rändern
        if fade > 0:
            fade_in = min(local_progress / fade, 1.0) if fade > 0 else 1.0
            fade_out = min((1.0 - local_progress) / fade, 1.0) if fade > 0 else 1.0
            modulation *= fade_in * fade_out
        
        # Sigma skalieren
        modified[i] = sigmas[i] * (1.0 + detail_amount * modulation)
    
    if smooth:
        # Einfache Glättung über benachbarte Werte
        smoothed = modified.clone()
        for i in range(1, n - 1):
            smoothed[i] = (modified[i-1] + modified[i] * 2 + modified[i+1]) / 4
        modified = smoothed
    
    return modified
```

---

## Teil 3: Dateistruktur

```
ComfyUI/custom_nodes/comfyui-person-tools/
├── __init__.py                  # Node-Registrierung
├── nodes/
│   ├── __init__.py
│   ├── person_selector_multi.py # Bestehende Node (Batch-Update)
│   ├── person_detailer.py       # Neue Node
│   └── utils/
│       ├── __init__.py
│       ├── detail_daemon.py     # Detail Daemon Sigma-Manipulation
│       ├── inpaint_pipeline.py  # Inpaint Crop/Stitch/VAE/Sample Pipeline
│       └── mask_utils.py        # Masken-Operationen (leer prüfen, kombinieren, subtrahieren)
```

---

## Teil 4: Offene Punkte / Entscheidungen

1. **Mask-Type pro Slot:** Aktuell vorgesehen als Dropdown (face/head/body). Ist das nötig, oder reicht face für alle Referenz-Slots und nur der generische Slot braucht vielleicht head?

2. **positive_base Input:** Soll der Basis-Prompt mit dem Slot-Prompt kombiniert werden (concatenate) oder nur als Fallback dienen wenn Slot-Prompt leer? Vorschlag: Fallback.

3. **Refined Output:** Die refined Parts haben unterschiedliche Größen (je nach Crop). Für einen Batch müssen alle gleich groß sein. Optionen:
   - Auf output_target_width × output_target_height resizen
   - Nur den letzten refined Part als Preview ausgeben
   - Alle als Liste (kein Batch) — geht nicht als Standard-IMAGE-Output
   
4. **Generischer Slot — mehrere unmatched Faces:** Alle auf einmal inpainten (ein Durchlauf, combined Maske) oder einzeln per Connected Component? Combined ist schneller, einzeln potentiell besser.

5. **Person Selector Multi — Abwärtskompatibilität:** Soll die Node bei Einzelbild-Input (kein Batch) exakt wie bisher funktionieren? Vorschlag: Ja, automatische Erkennung ob Batch oder Einzelbild.

6. **Widget-Anzahl reduzieren:** 62 Widgets sind viel. Option: Seltene Inpaint-Settings (hipass_filter, mask_invert, extend_factors) in eine separate "Advanced"-Sektion oder als Default-Only ohne Widget?

---

## Teil 5: Implementierungsreihenfolge

1. **mask_utils.py** — Hilfsfunktionen (is_mask_empty, subtract_masks, combine_masks)
2. **detail_daemon.py** — Sigma-Manipulation (unabhängig testbar)
3. **inpaint_pipeline.py** — Crop, VAE, Sample, Stitch als wiederverwendbare Pipeline
4. **Person Selector Multi Batch-Update** — Bestehende Node erweitern
5. **PersonDetailer Node** — Hauptnode zusammenbauen
6. **__init__.py** — Registrierung, NODE_CLASS_MAPPINGS
