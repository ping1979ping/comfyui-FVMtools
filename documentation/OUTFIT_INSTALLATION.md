# FVM Outfit Generator -- Installation & Quick Setup

## Prerequisites

- **ComfyUI** installed and working
- **Python 3.10** or later
- No additional pip packages required (the outfit system uses only the Python standard library)

## Installation

### Option 1: Git clone (recommended)

```bash
cd /path/to/ComfyUI/custom_nodes/
git clone https://github.com/your-repo/comfyui-FVMtools.git
```

### Option 2: Manual download

1. Download and extract the `comfyui-FVMtools` repository.
2. Place the entire folder into `ComfyUI/custom_nodes/` so the structure looks like:
   ```
   ComfyUI/
     custom_nodes/
       comfyui-FVMtools/
         __init__.py
         nodes/
         core/
         outfit_lists/
         ...
   ```

### First start

1. Restart ComfyUI after installing the node pack.
2. In the ComfyUI browser interface, right-click on the canvas.
3. Navigate to **FVM Tools > Fashion > Outfit Generator**.
4. If the node appears in the menu, installation is successful.

## Quick Pipeline Setup

The Outfit Generator works best as part of a 3-node chain:

```
[Color Palette Generator]  --(palette_string)-->  [Prompt Color Replace]  --(prompt)-->  [CLIP Text Encode]
                                                          ^
                                              (outfit_prompt)
                                                          |
                                                  [Outfit Generator]
```

### Step-by-step

1. **Add a Color Palette Generator** node (FVM Tools > Color > Color Palette Generator).
   - Set a seed and choose a style preset.
   - This generates named colors like `"navy-blue, soft-pink, charcoal-gray, gold, cream"`.

2. **Add an Outfit Generator** node (FVM Tools > Fashion > Outfit Generator).
   - Select an `outfit_set` (e.g., `general_female`).
   - Set a seed.
   - Enable the slots you want (top, bottom, footwear are on by default).

3. **Add a Prompt Color Replace** node (FVM Tools > Color > Prompt Color Replace).
   - Connect Outfit Generator's `outfit_prompt` output to Prompt Color Replace's `prompt` input.
   - Connect Color Palette Generator's `palette_string` output to Prompt Color Replace's `palette_string` input.

4. **Connect to CLIP Text Encode**.
   - Connect Prompt Color Replace's `prompt` output to your CLIP Text Encode node.

5. **Queue the prompt.** The output will be a fully resolved clothing description like:
   ```
   wearing navy blue silk blouse, charcoal gray wool trousers, cream leather heels
   ```

## Troubleshooting

### Node does not appear in the menu

- **Check the console** for import errors when ComfyUI starts. Look for lines mentioning `comfyui-FVMtools`.
- **Verify the directory structure.** The `__init__.py` file must be at the root of `comfyui-FVMtools/`, not nested in a subdirectory.
- **Restart ComfyUI.** A full restart is required after first installation.

### Outfit set dropdown is empty

- Verify the `outfit_lists/` directory exists inside `comfyui-FVMtools/` and contains at least one subdirectory (e.g., `general_female/`).
- If using `outfit_config.ini` with a `custom_lists_path`, verify that the path exists and contains set directories.

### Garment lists not loading / empty output

- Check that the set directory contains `.txt` files (`top.txt`, `bottom.txt`, `footwear.txt` at minimum).
- Verify the file format: each garment line must have exactly 4 pipe-separated fields.
  ```
  garment name | 0.75 | 0.0-0.5 | cotton,jersey
  ```
- Lines with fewer than 4 fields or invalid numbers are silently skipped.

### Color tags not being replaced

- Make sure the `outfit_prompt` output is connected to a **Prompt Color Replace** node, not directly to CLIP.
- Verify a `palette_string` is connected to Prompt Color Replace (from Color Palette Generator or typed manually).

### Changes to .txt files not taking effect

- Outfit lists are loaded on every execution -- no restart is needed.
- If changes seem ignored, verify you saved the file and are editing the correct set (check the `outfit_set` dropdown).
- After adding a completely new set directory, the dropdown updates on the next ComfyUI restart.
