# fonts/

Optional, **bundled** TrueType/OpenType files for the glyph-guidance stage of the
sign-repair nodes (`nodes/utils/glyph.py`).

Nothing in here is required. The glyph renderer works out of the box:

* **Windows system fonts are picked up automatically.** `C:\Windows\Fonts` and the
  per-user store `%LOCALAPPDATA%\Microsoft\Windows\Fonts` are scanned on every call to
  `discover_fonts()`, so anything you install through Explorer ("Install for all users"
  or a plain double-click → *Install*) shows up in the node's font dropdown after a
  browser refresh — no ComfyUI restart, no code change.
* **This directory is scanned first**, so a file dropped here wins over a system font
  with the same name. That is the point of the directory: it makes a workflow
  reproducible across machines that do not have the same fonts installed.
* If no font is found at all the renderer falls back to Pillow's built-in default face,
  which is readable but plain.

## How to use it

1. Drop `.ttf` or `.otf` files straight into this directory (sub-directories are
   scanned too).
2. Refresh the ComfyUI browser tab. The font dropdown re-scans live; it is never
   cached at import time.
3. Pick the family by its display name (the file stem), or leave the field on a free-text
   hint and let `resolve_font()` match it — it understands descriptions such as
   `"bold condensed sans"`, `"serif"`, `"handwritten"` or a family name like
   `"Helvetica"`.

## What to put in here

Signage covers a narrow set of typographic registers. Four faces are enough to hit
almost every real-world sign. Suggested licence-free families (all SIL Open Font
License or similar — **download them yourself**, this repo ships no font binaries):

| Role | Why it matters for signs | Licence-free candidates |
|------|--------------------------|-------------------------|
| **Grotesque / neutral sans** | The default for shopfronts, wayfinding, transport signage. | Inter, Roboto, Open Sans, Archivo, Work Sans |
| **Condensed / narrow sans** | Long words on a narrow fascia; awnings, market stalls, price boards. | Archivo Narrow, Oswald, Barlow Condensed, Roboto Condensed |
| **Serif** | Traditional shops, pubs, hotels, plaques, museum labels. | EB Garamond, Playfair Display, Libre Baskerville, Source Serif |
| **Monospace** | Departure boards, tickets, receipts, technical/industrial labels. | JetBrains Mono, IBM Plex Mono, Space Mono, Inconsolata |

Two optional extras that pay off if you generate a lot of street scenes:

* a **display / poster** face (Bebas Neue, Alfa Slab One) for big single-word fascias,
* a **script / handwriting** face (Caveat, Dancing Script) for chalkboards and café signs.

## Licensing

Only add fonts you are allowed to redistribute. The SIL Open Font License permits
bundling; most commercial desktop licences do **not**. If in doubt, install the font on
your machine instead — the system-font scan will find it and nothing gets committed.

`.gitignore` does not exclude this directory, so anything you drop here *will* be picked
up by `git status`. Keep binaries out of the repo unless you have checked the licence.
