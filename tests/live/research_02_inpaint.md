# Recherche 02 — Differential Diffusion + LanPaint für den Sign Detailer

Stand 2026-08-01. Reine Recherche, kein Code geändert, kein Lauf gestartet, keine GPU belegt.
Belegte Aussagen tragen die Quelle (Datei:Zeile oder URL). Alles ohne Beleg ist als
**unbestätigt** markiert.

---

## 1 — Der Reddit-Post

**Beschafft.** `https://www.reddit.com/r/comfyui/comments/1v93i6n/krea2_inpainting_workflow/`
WebFetch und die `.json`-API liefern beide HTTP 403; über einen echten Browser
(Chrome-DevTools-MCP) geht die JS-Challenge durch und der Post ist lesbar.

Autor `Altruistic_Tax1317`, 2026-07-28T16:30Z, 64 Upvotes, 13 Kommentare.

Der Postkörper enthält **keine Parametertabelle**. Wörtlich:

> Krea2 doesn't support inpainting natively, so I put together a small workflow that gets it
> working using LanPaint KSampler + Differential Diffusion. Nothing fancy — just in case it's
> useful to anyone running into the same limitation.
> the few settings that made a difference (mainly keeping the resolution small and bumping
> LanPaint's NumSteps to 10).

Verlinkt, nicht beschrieben:
- Workflow: `https://drive.google.com/file/d/1GR4krxtDnP-9WZq2O0fowCUGUJWHfUlt/view?usp=sharing`
- Video: `https://youtu.be/iOSQzKyCYyw`

**Genau zwei belegte Einstellungen: kleine Auflösung, `LanPaint_NumSteps = 10`.**
Alles Weitere zu diesem Workflow ist unbestätigt.

### Die Kommentare — und warum sie wichtiger sind als der Post

Ein einziger substanzieller Kommentar (`Gremlation`, Score 8). Er ist überwiegend
**kritisch**, und der Autor hat auf keinen technischen Punkt geantwortet, sondern mit
„feel free to strip out the extra nodes" ausgewichen. Das stützt die Vermutung
„ungetestete Weiterreichung".

Die vier Punkte, die uns betreffen:

1. > You crop to mask then grow mask with blur. This is the wrong way around — if the blur
   > reaches the edges of the crop, you have a hard edge. You need to grow mask with blur then
   > crop afterwards.

   Echter Fehler im Reddit-Workflow. **Unser Detailer macht es bereits richtig herum**
   (`inpaint_pipeline.py:359-379`: erst `fill_mask_holes_2d`, dann `expand_mask`, dann
   `compute_crop_region`, dann `feather_mask`). Kein Handlungsbedarf, aber der Workflow
   ist damit nachweislich nicht sauber.

2. > It doesn't seem to take the original pixels into account much if at all — it just draws
   > what it likes indiscriminately without regard for the surrounding image. […] I'm not sure
   > what the benefit is of LanPaint?

   **Das ist der wichtigste Satz des ganzen Threads für uns**, und er deckt sich exakt mit
   dem, was der LanPaint-Quelltext tut (siehe §3). LanPaint erzeugt im maskierten Bereich
   bei `denoise=1` neuen Inhalt, der *nur* über die bekannte Umgebung und den Prompt
   konditioniert ist. Ein Glyph-Template, das **innerhalb** der Maske liegt, wird dabei
   zerstört. Für uns heißt das: LanPaint ersetzt `glyph_guidance="init"`, es ergänzt es nicht.

3. > I got much better results using Text Encode (Krea2) than writing a prompt.

   Unabhängige Achse. Der Detailer nutzt `self._encode(clip, prompt)`
   (`detailer.py:304`), also den generischen CLIPTextEncode-Pfad. Ob ein Krea2-spezifischer
   Text-Encoder hier besser ist: **unbestätigt**, aber notiert.

4. Zwei Node-Packs des Autors sind nicht auflösbar; `control after generate` steht auf `1`
   statt auf einem gültigen Wert. Der Workflow ist also nicht einmal sauber exportiert.

Nebenbefund aus dem Thread, unbestätigt aber erwähnenswert: `chuckaholic` behauptet,
Inpainting habe im Krea-2-Trainingslauf gesteckt, und benutzt statt LanPaint die
**Identity-Edit-LoRA**. Alternativer Workflow eines anderen Kommentators:
`https://civitai.com/models/2788964/krea2-t2i-i2i-inpaint-2pass-workflow-foxfuressence`.

### Die bessere Primärquelle: LanPaints eigenes Krea2-Beispiel

Der Reddit-Workflow ist nicht die beste verfügbare Quelle. **LanPaint liefert selbst ein
Krea2-Beispiel mit**, lokal unter
`ComfyUI/custom_nodes/LanPaint/example_workflows/Krea2_LanPaint_Inpaint.json`.
Das ist der Autor des Verfahrens, nicht ein Weiterreicher.

Die `LanPaint_KSampler`-Node darin (Subgraph „Text to Image (Krea-2 Turbo)", Node-ID 3):

```
widgets_values = [8, 1, 8, 1, 'euler', 'simple', 1, 5, 'Image First',
                  'LanPaint KSampler.', '🖼️ Image Inpainting', 'lanpaint_star_button']
```

Aufgelöst gegen `INPUT_TYPES` (`LanPaint/src/LanPaint/nodes.py:298-320`) plus das vom
Frontend eingeschobene `control_after_generate`:

| Feld | Wert |
|---|---|
| seed | 8 (überschrieben, hat einen Link) |
| control_after_generate | 1 |
| **steps** | **8** |
| **cfg** | **1** |
| **sampler_name** | **euler** |
| **scheduler** | **simple** |
| **denoise** | **1.0** |
| **LanPaint_NumSteps** | **5** |
| LanPaint_PromptMode | Image First |
| LanPaint_Info | „LanPaint KSampler." |
| Inpainting_mode | 🖼️ Image Inpainting |

Die Zuordnung ist selbstprüfend: `'euler'`/`'simple'` landen exakt auf
`sampler_name`/`scheduler`, `'Image First'` auf `LanPaint_PromptMode`,
`'LanPaint KSampler.'` auf dem Default-String von `LanPaint_Info`. Zusätzlich bestätigt
`Gremlation` im Reddit-Thread unabhängig, dass `control after generate` bei LanPaint als
`1` serialisiert wird — genau die Anomalie, die die Zuordnung braucht.

**Damit ist belegt: der Verfahrensautor fährt Krea 2 Turbo mit 8 Steps, cfg 1, euler/simple,
denoise 1.0, NumSteps 5.** Unsere 8 Steps und cfg 1 sind also nicht das Problem — sie sind
exakt die Referenzkonfiguration. Die Frage „greift LanPaint bei so wenigen Steps überhaupt"
ist damit beantwortet: ja, das ist der vorgesehene Betriebspunkt.

Weiterer Kontext aus demselben Workflow: die Bilder werden per `ImageScale` auf
**1024×1024** gebracht (Nodes 402 `nearest-exact`, 407 `area`), Modelle
`krea2_turbo_fp8_scaled` + `qwen3vl_4b_fp8_scaled` (type `krea2`) + `qwen_image_vae`,
Negativ-Conditioning über `ConditioningZeroOut`, und ein `LanPaint_MaskBlend` mit
`blend_overlap=9` am Ende.

**Widerspruch, den ich nicht auflösen kann:** die README schreibt zum selben Beispiel
„Krea2: InPaint (LanPaint K Sampler, **3** steps of thinking)" (`README.md:366`), das
mitgelieferte JSON sagt **5**. Der Reddit-Post sagt **10**. Die drei Zahlen widersprechen
sich; keine ist gegen unser Material gemessen.

### Der dritte Fund: RunComfy

`https://www.runcomfy.com/comfyui-workflows/krea-2-turbo-inpainting-in-comfyui-lanpaint-sam3`
— ein 27-Node-Graph „Krea 2 Turbo Inpainting | LanPaint + SAM3", der dieselbe Kombination
fährt und zusätzlich SAM3-Maskierung und Crop-and-Stitch enthält, also unserem Aufbau
strukturell am nächsten kommt. Node-Liste laut Seite:
`LanPaint_KSampler`, `SetLatentNoiseMask`, `DifferentialDiffusionAdvanced`,
`GrowMaskWithBlur`, `CropByMask`, `ImageCompositeMasked`, `SAM3Segment`,
`VAEEncode`/`VAEDecode`, `CLIPTextEncode`, `UNETLoader`, `VAELoader`, `CLIPLoader`.

**Die Seite nennt keine Zahlenwerte** — Steps, cfg, Sampler, Scheduler, denoise und alle
LanPaint-Parameter fehlen dort. Also nur als Strukturbeleg brauchbar, nicht als
Parameterquelle. Beachte: `DifferentialDiffusionAdvanced` ist eine KJNodes-Variante, nicht
die Kern-Node.

---

## 2 — Der Workflow als Node-Liste

Nach Belegstufe getrennt. **Der Reddit-Workflow selbst ist als Node-Graph nicht belegt** —
die Google-Drive-Datei habe ich nicht geöffnet, und der Post beschreibt ihn nicht.

### Belegt (LanPaints eigenes Krea2-Beispiel, lokal auf Platte)

```
UNETLoader(krea2_turbo_fp8_scaled)  ─┐
CLIPLoader(qwen3vl_4b_fp8_scaled, type=krea2) ─→ CLIPTextEncode ─→ positive
                                                 └→ ConditioningZeroOut ─→ negative
VAELoader(qwen_image_vae)
LoadImage(+Maske) ─→ ImageScale(nearest-exact, 1024×1024, center) ─→ VAEEncode ─┐
                  └→ ImageToMask(red) ────────────────────────────────────────┐ │
                                                        SetLatentNoiseMask ←──┴─┘
                                                                 ↓
      LanPaint_KSampler(steps=8, cfg=1, euler, simple, denoise=1.0,
                        NumSteps=5, PromptMode="Image First")
                                                                 ↓
                                                            VAEDecode
                                                                 ↓
                              LanPaint_MaskBlend(blend_overlap=9)
```
Quelle: `LanPaint/example_workflows/Krea2_LanPaint_Inpaint.json`.
Bemerkenswert: **hier ist gar keine `DifferentialDiffusion` drin.** Der Verfahrensautor
hält sie für Krea2 nicht für nötig.

### Belegt nur als Struktur, ohne Werte (RunComfy)

`SAM3Segment → GrowMaskWithBlur → CropByMask → VAEEncode → SetLatentNoiseMask →
LanPaint_KSampler` mit `DifferentialDiffusionAdvanced` am `model`-Eingang, danach
`VAEDecode → ImageCompositeMasked`. Alle Zahlenwerte **unbestätigt**.

### Belegt aus dem Reddit-Post selbst

Nur: `LanPaint_KSampler` + `DifferentialDiffusion`, `NumSteps = 10`, „kleine Auflösung".
Reihenfolge, Verbindungen, Steps, cfg, Sampler, Scheduler, denoise, Maskenwachstum,
Weichzeichnung: **alle unbestätigt.**

---

## 3 — LanPaint technisch

### Nodes und Eingänge

`NODE_CLASS_MAPPINGS` (`LanPaint/src/LanPaint/nodes.py`):
`LanPaint_KSampler`, `LanPaint_KSamplerAdvanced`, `LanPaint_SamplerCustom`,
`LanPaint_SamplerCustomAdvanced`, `LanPaint_MaskBlend`.

`LanPaint_KSampler` ist ein KSampler-Klon mit drei Zusatzfeldern
(`LanPaint_NumSteps`, `LanPaint_PromptMode`, `Inpainting_mode`) und **hartkodiert** den Rest
(`nodes.py:355-362`):

```python
model.LanPaint_StepSize = 0.2
model.LanPaint_Lambda   = 16.0
model.LanPaint_Beta     = 1.
model.LanPaint_NumSteps = LanPaint_NumSteps
model.LanPaint_Friction = 15.
model.LanPaint_EarlyStop = 1
model.LanPaint_cfg_BIG  = cfg           # "Image First"
```

Vom Autor empfohlene Werte (README „LanPaint KSampler (Advanced)"):

| Parameter | Bereich | Empfehlung |
|---|---|---|
| `Steps` | 0-100 | 20-50 |
| `LanPaint_NumSteps` | 0-20 | leicht 2-5, schwer 5-10; Default 5 |
| `LanPaint_Lambda` | 0.1-50 | 4.0-10.0 (Node setzt **16.0**) |
| `LanPaint_StepSize` | 0.1-1.0 | 0.1-0.5 |
| `LanPaint_Beta` | 0.1-2.0 | 1.0 |
| `LanPaint_Friction` | 0.0-100 | 10.0-20.0 |
| `LanPaint_EarlyStop` | 0-10 | 1-5 |

Die README empfiehlt `Lambda` 4-10, die einfache Node setzt 16 — **die einfache Node steht
außerhalb der eigenen Empfehlung.** Wer daran dreht, braucht `LanPaint_KSamplerAdvanced`.

### Was `NumSteps` genau ist — und was es kostet

Bestätigt: **innere Langevin-Iterationen je Sampling-Schritt.**
`lanpaint.py:79-108` ist die Schleife `for i in range(n_steps)`, darin genau ein Aufruf von
`langevin_dynamics`. In `run_damped` (`lanpaint.py:211-226`) wird `Coef_C(x_t)` **genau
einmal** pro Iteration aufgerufen, und `Coef_C` ruft `score(x_t)` → `score_model` →
`self.inner_model(...)`. Also **1 Modellauswertung je Denkschritt.**
Dazu kommt nach der Schleife ein weiterer Aufruf (`lanpaint.py:117-119`).

> **NFE je Sampling-Schritt = NumSteps + 1**

`LanPaint_EarlyStop=1` schaltet die Denkschritte für den letzten Sampling-Schritt ab
(`nodes.py:179-183`: `if total_steps - current_step <= LanPaint_early_stop: n_steps=0`).
Bei 8 Schritten denkt LanPaint also auf 7 von 8.

Für **unsere 8 Sampling-Schritte**:

| NumSteps | NFE gesamt | Faktor gegen heute (8 NFE) |
|---|---|---|
| 0 (aus) | 8 | 1,0× |
| 2 | 7×3 + 1 = **22** | **2,8×** |
| 5 (Autor, Krea2-Beispiel) | 7×6 + 1 = **43** | **5,4×** |
| 10 (Reddit) | 7×11 + 1 = **78** | **9,8×** |

Der Faktor gilt für den **Sampler-Anteil**, nicht für den ganzen Durchgang. Nach
`JOURNAL.md` sind das 33 s Fixkosten + 1,6 s je Region; der Sampler ist nur ein Teil dieser
1,6 s (dazu kommen Glyph-Rendering, VAE-Encode/Decode, Stitch). Konservativ, **ungemessen**:
liegt der Sampler bei ~0,6-1,0 s je Region, kostet NumSteps=5 rund +3-5 s je Region und
NumSteps=10 rund +6-9 s. Bei 4-8 Regionen sind das **+15-70 s je Durchgang**; der volle
Zwölfer-Zensus ginge von ~13 min auf grob 25-40 min. Das ist tragbar, aber es ist ein
anderer Betrieb — und ich kann es ohne GPU nicht messen. Der NFE-Faktor selbst ist dagegen
aus dem Quelltext hart belegt.

Wichtig: cfg 1 ist hier **kein** Nachteil. Bei cfg 1 überspringt
`sampling_function_LanPaint` (`nodes.py:86-89`) den Uncond-Zweig genau wie der Kernpfad —
der Faktor oben gilt gegen unsere heutige Basis, nicht zusätzlich.

### Verhalten bei cfg > 1

- **LanPaint:** im Modus „Image First" ist `cfg_BIG = cfg`, es gibt also keinen zweiten
  Guidance-Pfad und keine Zusatzkosten. Bei „Prompt First" wird `cfg_BIG = -0.5` gesetzt,
  was die README ausdrücklich als qualitätskostend beschreibt. **Der einzige belegte
  cfg-Hinweis des Autors ist eine Warnung in die andere Richtung:** bei destillierten
  Modellen soll man die Guidance *niedrig* halten (README, Features: „Warning: LanPaint has
  degraded performance on distillation models […] Please use low flux guidance (1.0-2.0)").
  Krea 2 **Turbo** ist destilliert. Also: cfg 1-1.5 ist im Sinne des Autors, höher nicht.
- **Differential Diffusion:** ist cfg-agnostisch. Der Patch verändert nur die
  Rauschmaske je Schritt, er fasst die Guidance nicht an. **Kein Nutzen aus cfg > 1.**

Kurz: die eröffnete cfg-Achse ist eine **eigene, unabhängige** Änderung. Weder DiffDiff
noch LanPaint profitieren davon, und sie mit einer davon zusammen zu messen wäre genau der
Fehler aus „Band + Maske" (9→7).

### Der Strukturbruch — Node oder aufrufbar?

**Aufrufbar. LanPaint ist nicht node-gebunden.** Das ist der entscheidende Befund für die
Machbarkeit.

`LanPaint_KSampler.sample` endet mit (`nodes.py:364-365`):

```python
with override_sample_function():
    return nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                                 positive, negative, latent_image, denoise=denoise)
```

`override_sample_function` (`nodes.py:230-246`) ist ein Contextmanager, der **Klassenmethoden
global monkeypatcht** und danach zurücksetzt:

```python
@contextmanager
def override_sample_function():
    original_outer_sample = comfy.samplers.CFGGuider.outer_sample
    comfy.samplers.CFGGuider.outer_sample = CFGGuider_LanPaint.outer_sample
    original_predict_noise = comfy.samplers.CFGGuider.predict_noise
    comfy.samplers.CFGGuider.predict_noise = CFGGuider_LanPaint.predict_noise
    original_sample = comfy.samplers.KSAMPLER.sample
    comfy.samplers.KSAMPLER.sample = KSAMPLER.sample
    try:
        yield
    finally:
        ...  # alles zurück
```

Gepatcht werden `comfy.samplers.CFGGuider` und `comfy.samplers.KSAMPLER` — **genau die
beiden Klassen, die `inpaint_slot` bereits benutzt** (`inpaint_pipeline.py:464`
`comfy.samplers.sampler_object(sampler_name)` liefert eine `comfy.samplers.KSAMPLER`,
`:486` `CFGGuider(round_model)`, `:527` `guider.sample(...)`). Ein `with`-Block um den
`guider.sample`-Aufruf genügt mechanisch.

Bedingungen, die dabei erfüllt sein müssen:

1. Auf dem ModelPatcher müssen **neun** Attribute gesetzt sein, sonst `AttributeError`.
   `KSAMPLER.sample` liest sie direkt (`nodes.py:219-234`):
   `LanPaint_NumSteps`, `LanPaint_Friction`, `LanPaint_Lambda`, `LanPaint_Beta`,
   `LanPaint_StepSize`, `LanPaint_EarlyStop`, `LanPaint_cfg_BIG` (hart), sowie
   `LanPaint_InnerThreshold`/`LanPaint_InnerPatience` (über `getattr` mit Default, also
   optional).
2. `model.model_options["video_inpainting"] = False`.
3. Der Sampler muss in LanPaints `KSAMPLER_NAMES` stehen (`nodes.py:291-296`).
   **`er_sde` steht drin** — unser gepinnter Sampler ist also kompatibel. `euler` ebenso,
   und die Node-Tooltip empfiehlt „Recommended: euler".
4. `denoise` sollte 1.0 sein. LanPaints Kernversprechen ist „no partial denoising"
   (README, Features: „Generates 100% new content (no blending or smoothing) without
   relying on partial denoising"). Mit `glyph_denoise=0.55` würde man das Verfahren gegen
   seine eigene Prämisse fahren.
5. `CFGGuider_LanPaint.predict_noise` gibt ein **Tupel** `(out, out_BIG)` zurück statt eines
   Tensors (`nodes.py:131-132`). Innerhalb des `with`-Blocks gilt das für *jede*
   `CFGGuider`-Instanz im Prozess. Der Patch ist global und nicht thread-sicher; ComfyUI
   führt einen Prompt zur Zeit aus, also praktisch tragbar, aber es ist ein Prozess-weiter
   Eingriff und gehört so eng wie möglich um den einen Aufruf gelegt.

**Bewertung: LanPaint ist kein Zweizeiler, aber auch kein Umbau.** Realistisch ~15 Zeilen in
`inpaint_slot` plus ein Widget. Der teure Teil ist nicht die Mechanik, sondern die
Folgeänderung an der Fachlogik (Punkt 4 unten).

---

## 4 — Der Eingriff in den Detailer

### Ersetzt `DifferentialDiffusion` die Handarbeit in `_split_noise_mask`/`_hot_zone`?

**Nein — sie ergänzen sich, und zwar sauber. Es muss nichts raus.**

Die beiden arbeiten auf verschiedenen Ebenen:

- `_hot_zone` (`detailer.py:449-464`) und `_split_noise_mask` (`detailer.py:466-481`)
  **erzeugen** eine Maske: 1.0 wo die neue Schrift steht, `glyph_surface_restyle` (0.35)
  auf der übrigen Fläche, 0 außerhalb der Region. Nach
  `feather_mask` (`inpaint_pipeline.py:398-399`, Gauß mit `mask_blend_pixels=16`) ist daraus
  ein weiches Feld: Plateau 1.0 → Rampe → Plateau 0.35 → Rampe → 0.
- `DifferentialDiffusion` **interpretiert** eine solche Maske. Sie erzeugt selbst keine.

Das heißt: **der Detailer baut bereits genau den Verlauf, den DiffDiff als Eingang will.**
Ein „echter Verlauf" muss nicht erst gebaut werden — er existiert. Ich würde für den ersten
Schritt **nichts an der Maske ändern**, weil sonst zwei Dinge auf einmal gemessen werden.

Was sich ändert, ist die **Bedeutung** von `glyph_surface_restyle`:

| | heute (Kernpfad) | mit DiffDiff |
|---|---|---|
| Maskenwert 0.35 heißt | *Amplitude*: in **jedem** Schritt wird zu 65 % das verrauschte Original zurückgeblendet | *Startzeit*: die Fläche ist bis ~65 % des Zeitplans **eingefroren** und danach **voll** frei |

Kernpfad, `comfy/samplers.py:632-641`:
```python
latent_mask = 1. - denoise_mask
x = x * denoise_mask + scale_latent_inpaint(...) * latent_mask   # Amplitudenmischung, jeder Schritt
```
DiffDiff, `comfy_extras/nodes_differential_diffusion.py:51-54`:
```python
threshold = (current_ts - ts_to) / (ts_from - ts_to)     # läuft linear 1 → 0
binary_mask = (denoise_mask >= threshold)                 # pro Schritt 0 oder 1, nie dazwischen
```

Der springende Punkt: **DiffDiff schafft die Amplitudenmischung ab.** Kein Pixel ist je
teils neu und teils altes Original — es ist zu jedem Zeitpunkt entweder eingefroren oder
frei. Das ist der Mechanismus, den man haben will, wenn zwei verschiedene Bilder
übereinanderliegen sollen und nicht ineinander.

Nachweis, dass DiffDiff mit unserem gekappten Zeitplan zurechtkommt: `forward` liest
`sigma_from = step_sigmas[0]` und `sigma_to = step_sigmas[-1]` aus dem **tatsächlich
übergebenen** Sigma-Vektor. `inpaint_slot` kappt bei `denoise<1`
(`inpaint_pipeline.py:496-500`: `total_steps = int(steps/denoise)`, `sigmas[-(steps+1):]`).
DiffDiff normiert also auf den gekappten Bereich — der Zeitplan läuft weiterhin sauber
1 → 0 über unsere 8 Schritte. **`glyph_denoise=0.55` bleibt gültig, kein Konflikt.**

### An welcher Stelle wird der `model`-Patch angewandt?

`nodes/utils/inpaint_pipeline.py`, Zeile **477**, in der Runden-Schleife:

```python
        round_model = model                      # ← hier
        ...
        guider = CFGGuider(round_model)          # :486
```

`DifferentialDiffusion.execute` ist ein reiner ModelPatcher-Eingriff
(`nodes_differential_diffusion.py:33-36`): `model.clone()` +
`set_model_denoise_mask_function(...)`, und das schreibt lediglich
`model_options["denoise_mask_function"]` (`comfy/model_patcher.py:613-614`). Der Kernsampler
ruft diese Funktion an der Stelle auf, an der er sonst die rohe Maske benutzt
(`comfy/samplers.py:635-636`) — **also greift der Patch im bestehenden `inpaint_slot`-Pfad
ohne jede strukturelle Änderung.** Kein neuer Sampler, kein Contextmanager, keine
Laufzeitkosten (die Funktion ist ein Vergleich pro Schritt).

Am saubersten ohne die Mathematik zu duplizieren:

```python
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion

round_model = model
if diff_diff:
    round_model = round_model.clone()
    round_model.set_model_denoise_mask_function(
        lambda sigma, dm, extra_options:
            DifferentialDiffusion.forward(sigma, dm, extra_options, strength=1.0))
```

`strength` auf 1.0 lassen: unter 1 mischt DiffDiff das binäre Ergebnis wieder mit der
weichen Maske (`:57-59`) und holt sich damit genau die Amplitudenmischung zurück, deretwegen
man den Patch einbaut.

### Welche Maske geht rein?

Die vorhandene. `_split_noise_mask` liefert sie bereits über den `noise_mask_2d`-Weg
(`detailer.py:646-657` → `inpaint_pipeline.py:387-399`), und dieser Weg tut schon das
Richtige: die Rauschmaske wird mit **demselben Crop und derselben Skalierung** verarbeitet
wie die Blend-Maske, und danach geweichzeichnet.

Der Journal-Eintrag „Rauschmaske nicht weichzeichnen → verworfen" bleibt damit gültig und
wird unter DiffDiff sogar **wichtiger**: die Weichzeichnung ist unter DiffDiff kein
Kompromiss mehr, sondern die eigentliche Nutzinformation — sie wird zur gestaffelten
Startzeit vom Rand zur Mitte. Also: Weichzeichner behalten.

### Die Falle, die die Reihenfolge erzwingt

`LanPaint/src/LanPaint/nodes.py:168-174`:

```python
if denoise_mask is not None:
    if "denoise_mask_function" in model_options:
        denoise_mask = model_options["denoise_mask_function"](sigma, denoise_mask, extra_options=...)
    denoise_mask = (denoise_mask > 0.5).float()      # ← harte Schwelle
    latent_mask = 1 - denoise_mask
```

**LanPaint binarisiert die Rauschmaske bei 0.5.** Unser Flächenwert
`glyph_surface_restyle = 0.35` liegt darunter und würde zu 0 → `latent_mask = 1` →
LanPaint behandelt die Fläche als **zu erhaltende bekannte Region**. Der gesamte
Flächen-Restyle wäre still abgeschaltet, ohne Fehlermeldung, ohne sichtbaren Hinweis.
Das deckt sich mit der README: „LanPaint requires binary masks […] any mask with smoothing
or gradients will automatically be converted to a binary mask."

Steht dagegen DiffDiff davor, liefert die Maskenfunktion bereits 0/1, und `> 0.5` ist ein
No-Op. **DiffDiff ist damit keine Alternative zu LanPaint, sondern seine Voraussetzung in
diesem Codebestand.** Wer LanPaint zuerst einbaut, misst unbemerkt eine Konfiguration ohne
Flächenbehandlung.

(Nebenwirkung derselben Zeile: unter LanPaint ist `DifferentialDiffusion.strength` wirkungslos,
weil das gemischte Ergebnis anschließend ohnehin bei 0.5 geschnitten wird.)

### Der zweite Konflikt: LanPaint gegen `glyph_guidance="init"`

LanPaint ersetzt im freien Bereich den Inhalt vollständig (`lanpaint.py:60`, `:120`). Die
Bindung an die Vorlage `y = latent_image` wirkt über `score_y` **nur in der bekannten
Region** (`lanpaint.py:139-141`). Die typografische Vorlage liegt aber definitionsgemäß
**in** der heißen Zone, also im freien Bereich.

Damit fällt bei LanPaint mit `denoise=1` die Glyph-Vorlage weg — genau das, was
`Gremlation` im Thread beobachtet hat („it just draws what it likes"). Der Detailer erreicht
aber ausgerechnet über diese Vorlage seine einzige bisher stabile Kennzahl: **Zieltext in
12 von 12 Durchgängen.** Ein Wechsel auf LanPaint setzt diese 12/12 aufs Spiel und tauscht
sie gegen „Krea 2 schreibt das Wort aus dem Prompt".

Das ist kein Argument gegen LanPaint — es ist ein Argument dafür, es **nicht als ersten
Schritt** und **nicht zusammen mit etwas anderem** zu messen.

### Auflösung — schon „klein" genug?

Der Detailer benutzt `target_width`/`target_height` mit `auto_resolution=True` als
**Pixelbudget**, nicht als Kantenlänge (`_resolve_target`, `detailer.py:339-362`):
`width = sqrt(budget·aspect)`, `height = sqrt(budget/aspect)`, Seitenverhältnis auf
`AUTO_RESOLUTION_MAX_ASPECT = 4.0` begrenzt, danach `_clamp_target` mit `max_upscale=8.0`.

Bei den Node-Defaults (1024 × 1024, die die Suite mangels `--set` übernimmt) ist das Budget
**1 048 576 px ≈ 1 MP**, unabhängig von der Form:

| Regionform | Sampling-Auflösung | Budget |
|---|---|---|
| quadratisch (1:1) | 1024 × 1024 | 1,0 MP |
| Schild 2:1 | 1448 × 720 | 1,04 MP |
| Schild 550×95 (5,8:1 → gekappt auf 4:1) | 2048 × 512 | 1,05 MP |

**LanPaints eigenes Krea2-Beispiel skaliert auf 1024 × 1024 = 1,05 MP** (Nodes 402/407).
Unser Budget ist also *exakt* die Referenzgröße des Verfahrensautors. Nach dem einzigen
belegbaren Maßstab ist die Auflösung damit **bereits „klein"** — hier ist nichts zu tun.

Zwei Vorbehalte:
- Der Reddit-Hinweis „keeping the resolution small" ist unquantifiziert. Was der Autor als
  klein ansieht, steht nirgends. **Unbestätigt.**
- Die 4:1-Form (2048 × 512) hat zwar dasselbe Budget, aber eine doppelt so lange Kante wie
  die Referenz. Ob Krea 2 als DiT bei 4:1 gleich gut liegt, ist ungemessen — das ist aber
  eine vorbestehende Eigenschaft unseres Aufbaus und hat mit LanPaint nichts zu tun. Nicht
  im selben Schritt anfassen.

---

## 5 — Reihenfolge, und der kleinstmögliche erste Bauschritt

### Reihenfolge

**DiffDiff allein zuerst. Dann messen. LanPaint erst danach, und nur wenn DiffDiff steht.**

Begründung aus der Sache, nicht aus Vorsicht:

1. **Zwangsläufige Abhängigkeit.** LanPaint binarisiert bei 0.5 und würde unsere 0.35-Fläche
   still abschalten (`LanPaint/…/nodes.py:172`). DiffDiff davor macht die Binarisierung zum
   No-Op. Die Reihenfolge ist nicht Geschmackssache, sie ist im Code festgelegt.
2. **Sehr unterschiedliche Eingriffstiefe.** DiffDiff: `model.clone()` +
   `set_model_denoise_mask_function`, greift im bestehenden Sampelpfad, null Laufzeitkosten,
   rückgängig durch ein Widget. LanPaint: globaler Monkeypatch zweier Comfy-Klassen,
   neun Pflichtattribute, 2,8-9,8× NFE, **und** der Zwang, `glyph_denoise` auf 1.0 zu
   ziehen, was die Glyph-Vorlage entwertet. Das sind in Wahrheit drei Änderungen in einer.
3. **Die Projektregel greift genau hier.** „Band + Maske" zusammen: 9→7, getrennt +1 und −3.
   DiffDiff und LanPaint zusammen wäre derselbe Fehler in größer — und die cfg-Achse wäre
   ein dritter.

Was allein schon eine messbare Verbesserung bringen könnte: **DiffDiff.** Es ist die
einzige der beiden Änderungen, die die vorhandene, gemessene Fachlogik (Glyph-Vorlage,
`glyph_denoise` 0.55, Zweistufenmaske) unangetastet lässt und trotzdem den Mechanismus
verändert, über den die Stufen wirken.

### Der erste Bauschritt — genau eine Änderung

**Ein optionaler Differential-Diffusion-Patch in `inpaint_slot`, per Widget abschaltbar,
Default aus.**

- **Codestelle:** `nodes/utils/inpaint_pipeline.py:477` (`round_model = model`), direkt vor
  `CFGGuider(round_model)` in Zeile 486. Neuer Parameter `diff_diff: bool = False` in der
  Signatur ab Zeile 321.
- **Durchreichen:** `nodes/signs/detailer.py:654-680` (`inpaint_slot(...)`-Aufruf) plus ein
  `BOOLEAN`-Widget `diff_diff` in `INPUT_TYPES`, damit `suite.py --set diff_diff=True` es
  ohne neuen Bau umlegen kann.
- **Nicht anfassen:** `_split_noise_mask`, `_hot_zone`, `HOT_ZONE_MARGIN`,
  `glyph_surface_restyle`, `glyph_denoise`, Steps, cfg, Sampler, Auflösung. **Nichts davon
  muss raus** — die beiden Verfahren arbeiten auf verschiedenen Ebenen.
- **Messung:** `suite.py` ohne `--fast`, einmal mit `diff_diff=False` (muss die Baseline
  4/12 byte-nah reproduzieren — sonst ist der Patch nicht sauber gegated) und einmal mit
  `diff_diff=True`.

**Was es messbar verbessern soll:** den Übergang zwischen heißer Zone und Fläche. Heute
wird dort in *jedem* der 8 Schritte anteilig das verrauschte Original zurückgeblendet; nach
der Änderung nie. Erwartung, in dieser Rangfolge:
1. weniger Geisterschrift im Rampenbereich um die neue Schrift herum (`synth B`, 0,8, ist
   der einzige offene Ghosting-Punkt);
2. weniger Bruchstücke der eigenen Wörter am Regionenrand (`CHULUN T`, `FRAENK`, `EIGELT`)
   — diese 18 von 46 Slop-Einträgen sitzen genau dort, wo die Amplitudenmischung wirkt;
3. möglicherweise ein breiteres nutzbares `glyph_denoise`-Fenster, weil die Geisterkopie ab
   0,65 nicht mehr durch Rückblendung gestützt wird.

Punkt 3 ist die schwächste der drei Erwartungen — bei höherem denoise ist ebenso plausibel,
dass die Geisterkopie daher kommt, dass Prompt **und** Vorlage dasselbe Wort setzen und die
Vorlagenbindung nachlässt. Das würde DiffDiff nicht heilen. **Nicht darauf bauen.**

Falls DiffDiff nichts bringt (±0), ist der zweite Schritt trotzdem nicht LanPaint, sondern
ein Sweep von `glyph_surface_restyle` **unter** DiffDiff — der Wert 0.35 wurde als Amplitude
bestimmt und bedeutet unter DiffDiff eine Startzeit. Ein Amplitudenoptimum ist kein
Startzeitoptimum.

### Was schiefgehen kann, und woran man es im Bild erkennt

| Risiko | Mechanismus | Erkennungsmerkmal im Bild |
|---|---|---|
| **Fläche wird zu spät freigegeben** | 0.35 heißt jetzt „letzte ~35 % des Zeitplans", bei 8 Schritten ~2-3 Schritte. Zu wenig, um Material zu erzeugen. | Die Fläche um die Schrift bleibt exakt das Original — Schrift sitzt sichtbar *aufgeklebt* auf unverändertem Untergrund, harte Kante am Rand der heißen Zone. Genau das Regal-Symptom aus dem verworfenen Abschnitt 1 („gestochen scharfe Schrift auf Bokeh-Flaschen"). Gegenmittel: `glyph_surface_restyle` hoch, **im nächsten Schritt**, nicht in diesem. |
| **Fläche wird zu früh freigegeben** | Ist der Wert zu hoch, denoist die Fläche fast von Anfang an voll. | Untergrunddrift über A→B→C: Schildfarbe, Schrauben, Abnutzung wandern von Durchgang zu Durchgang. Der bekannte Effekt von `glyph_surface_restyle=1.0`. |
| **Naht am Regionenrand** | Die Gauß-Rampe zum Rand wird zur Startzeit-Rampe. Ist `mask_blend_pixels=16` zu schmal, springen benachbarte Pixel um mehrere Schritte. | Ringförmige Stufe/Halo genau auf der Regionenkontur, oft als Helligkeitssprung. Vgl. LanPaint-Issue #80 („glowing / broken mask boundary"). |
| **Patch greift gar nicht** | Widget nicht durchgereicht, oder `model` wird woanders geklont und der Patch geht verloren. | Ergebnis byte-identisch zur Baseline. Deshalb ist der `diff_diff=False`-Kontrolllauf Teil der Messung, nicht Beiwerk. |
| **Falsche Schlussfolgerung** | — | `JOURNAL.md`, Falle 6: **Node-Code wirkt erst nach ComfyUI-Neustart.** Ohne Neustart misst man zweimal die Baseline und schreibt „kein Effekt" ins Journal. |

Zusätzliche Risiken, die **erst bei LanPaint** auftreten und hier nur zur Warnung stehen:
`glyph_denoise` muss auf 1.0 → Glyph-Vorlage weg → Zieltext-Trefferquote (heute 12/12) ist
das erste, was man prüfen muss, nicht der Slop-Zensus. Und der Monkeypatch ist prozessweit:
läuft im selben ComfyUI ein anderer Graph, bekommt der ein `predict_noise`, das ein Tupel
zurückgibt.

---

## Anhang — Quellen

Lokal (Primärquellen, verifiziert gelesen):
- `ComfyUI/comfy_extras/nodes_differential_diffusion.py` (74 Z.)
- `ComfyUI/comfy/samplers.py:628-641` (`KSamplerX0Inpaint`), `comfy/model_patcher.py:613-614`
- `ComfyUI/custom_nodes/LanPaint/src/LanPaint/nodes.py` (Klassen, `override_sample_function`, Binarisierung)
- `ComfyUI/custom_nodes/LanPaint/src/LanPaint/lanpaint.py` (Langevin-Schleife, NFE-Zählung)
- `ComfyUI/custom_nodes/LanPaint/README.md`
- `ComfyUI/custom_nodes/LanPaint/example_workflows/Krea2_LanPaint_Inpaint.json`
- `comfyui-FVMtools/nodes/signs/detailer.py`, `nodes/utils/inpaint_pipeline.py`,
  `nodes/utils/mask_utils.py`, `nodes/signs/options.py`, `tests/live/JOURNAL.md`

Web:
- Reddit-Post (über Browser): <https://www.reddit.com/r/comfyui/comments/1v93i6n/krea2_inpainting_workflow/>
- <https://github.com/scraed/LanPaint>
- <https://www.runcomfy.com/comfyui-workflows/krea-2-turbo-inpainting-in-comfyui-lanpaint-sam3> (Struktur, keine Werte)
- LanPaint-Paper (TMLR): <https://arxiv.org/abs/2502.03491>
- Nicht ausgewertet: Google-Drive-Workflow und YouTube-Video des Reddit-Autors
