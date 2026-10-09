# /enhance-node — ComfyUI Node Enhancer & Refactorer

Modify, extend, or refactor an existing ComfyUI node or infrastructure component in comfyui-FVMtools.

## Input

$ARGUMENTS

## Workflow

Follow these steps strictly in order. Use **Explore agents** for Step 1, then implement sequentially.

### Step 1: Analyse (READ-ONLY)

Read and understand everything that will be affected:

1. **Target node(s)**: Read the node file(s) in `nodes/` that will be modified
2. **Dependencies**: Trace imports — check `nodes/utils/`, `core/`, `parsing/`
3. **Pipeline flow**: If touching inpainting: read `nodes/utils/inpaint_pipeline.py`
4. **Current tests**: Read `tests/unit/test_<node>.py` to understand coverage
5. **Current docs**: Read relevant sections in `README.md` and `documentation/*.md`
6. **Config**: Check `outfit_config.ini` if config changes are needed

Identify:
- Which files need modification vs. new files
- Which existing utilities can be reused (don't reinvent)
- What the current INPUT_TYPES, RETURN_TYPES, execute() signature looks like
- Whether changes require ComfyUI restart or just browser refresh

### Step 2: Create/Extend Utility Modules

If new shared logic is needed:

**Location rules:**
- Face/matching/mask utilities → `nodes/utils/<module>.py`
- Config/data loading → `core/<module>.py`
- Color/palette logic → `core/<module>.py`

**Patterns to follow:**
- Pure functions where possible (testable without ComfyUI)
- Use try/except import pattern for ComfyUI + pytest compatibility:
  ```python
  try:
      from ...core.config import get_model_path
  except ImportError:
      from core.config import get_model_path
  ```
- Console logging with `[FVMTools]` prefix
- Graceful degradation: return None/defaults when data unavailable

**Config pattern** (when adding configurable paths/settings):
```python
from core.config import get_config, get_model_path
# get_config() returns ConfigParser with mtime-cache (no restart needed)
# get_model_path(key) reads from [models] section
```

**Multi-value input pattern** (for pipe-separated fields like "0.5|0.3"):
```python
def _parse_progression(value_str, count, default, cast_fn=float):
    if not value_str or not value_str.strip():
        return [default] * count
    try:
        parts = [cast_fn(v.strip()) for v in value_str.strip().split("|") if v.strip()]
    except (ValueError, TypeError):
        return [default] * count
    if not parts:
        return [default] * count
    while len(parts) < count:
        parts.append(parts[-1])  # pad with last value
    return parts[:count]  # truncate excess
```

### Step 3: Modify the Node

**Adding new inputs to INPUT_TYPES:**
```python
"field_name": ("TYPE", {
    "default": value,
    "tooltip": "Short description.\n"
               "More detail on second line.\n\n"
               "Examples:\n"
               "  value1 — explanation\n"
               "  value2 — explanation\n\n"
               "If empty: fallback behavior."
}),
```

**Tooltip conventions:**
- First line: what it does (short)
- Second paragraph: detailed explanation
- Examples section with indented values
- Mention fallback/default behavior
- Use `\n` for line breaks, `\n\n` for paragraphs

**Modifying execute():**
- Add new parameters with defaults for backward compatibility
- Keep the existing signature order, append new params
- Update `**kwargs` passthrough if present
- Always test that the node works WITHOUT the new feature (fallback)

**DESCRIPTION string:**
- Update to mention new capabilities
- Keep concise — details go in tooltips and docs

### Step 4: Update Pipeline/Internals

If modifying `inpaint_pipeline.py`, `matcher.py`, `masker.py`, or similar:

- Add new parameters with defaults to function signatures
- Keep backward compatibility (old callers still work)
- Add console output for debugging when new features are active:
  ```python
  if new_feature_active:
      print(f"[FVMTools] feature info: {details}")
  ```
- Ensure 0-VRAM-overhead approach: reuse existing model outputs (BiSeNet label maps, face embeddings) where possible

### Step 5: Add Tooltips to ALL Changed Fields

Every input field that was added or modified MUST have a tooltip with:
1. One-line description
2. Valid range/format
3. At least one example
4. Fallback behavior when empty/default

### Step 6: Update Tests

**For new utility modules** — create `tests/unit/test_<module>.py`:
```python
import pytest
# conftest.py handles ComfyUI mocking automatically
from nodes.utils.new_module import new_function

class TestNewFunction:
    def test_basic(self):
        ...
    def test_edge_cases(self):
        ...
    def test_fallback(self):
        ...
```

**For modified nodes** — update existing test file:
- Add tests for new inputs/parameters
- Test backward compatibility (node works without new params)
- Test edge cases (empty strings, None values, extreme numbers)

**Test fixtures** available in `conftest.py`:
- `dummy_image` — [1, 512, 512, 3] float32
- `dummy_image_batch` — [4, 512, 512, 3] float32
- `dummy_mask` — [1, 512, 512] float32
- `dummy_latent` — {"samples": [1, 4, 64, 64]}
- `small_image` — [1, 64, 64, 3] float32

### Step 7: Run Tests — Auto-Fix Loop

```bash
D:/AI/ComfyUI/ComfyUI/venv/Scripts/python.exe -m pytest tests/unit/ -v
```

**IMPORTANT:** Always use the venv Python, never system Python.

**Loop until all tests pass (max 5 iterations):**
1. Run tests
2. If failures: fix source code (NOT the test, unless test is wrong)
3. Re-run
4. Repeat

### Step 8: Update README.md

Update the relevant node reference table in README:
- Add new input rows to the table
- Update the node description if capabilities changed
- Add link to new guide doc if created in Step 9
- Update "Features at a Glance" section if it's a significant feature

### Step 9: Create/Update Guide Documentation (if complex)

For features that need explanation beyond a tooltip:
- Create `documentation/<FEATURE>_GUIDE.md`
- Include: concept explanation, ASCII diagrams, examples, tuning tips, technical details
- Link from README's "Detailed Guides" table
- Use bilingual approach: English for technical terms, structure in whatever language is natural

### Step 10: Summary

Report:
- Files created/modified (table format)
- Test results (pass/fail count)
- What needs ComfyUI restart vs. browser refresh
- Console output to watch for (new log messages)

## Project Conventions

- Package: `comfyui-FVMtools`
- Node files: `nodes/<name>.py`
- Node utilities: `nodes/utils/<module>.py`
- Core/shared logic: `core/<module>.py`
- Tests: `tests/unit/test_<name>.py`
- Docs: `documentation/<NAME>_GUIDE.md`
- Config: `outfit_config.ini` (INI format, [section] keys)
- Web: `web/js/<name>.js`
- Category prefix: `FVM Tools/`
- All nodes use V1 schema
- **Python/pytest**: `D:/AI/ComfyUI/ComfyUI/venv/Scripts/python.exe -m pytest`
- **NEVER add `__init__.py` to `tests/` directories**
- Root `conftest.py` handles sys.path + ComfyUI mocking
- Root `__init__.py` wraps imports in `try/except ImportError: pass`
- Tooltip format: short desc + `\n\n` + details + examples
- Console prefix: `[FVMTools]` or `[PersonSelectorMulti]` etc.
- Config changes take effect without restart (mtime-cache in core/config.py)

## Common Enhancement Patterns

### Adding a new input to an existing node
1. Add to `INPUT_TYPES` with tooltip
2. Add parameter to `execute()` with default
3. Pass through to internal functions
4. Update README table
5. Test backward compatibility

### Adding a configurable setting
1. Add to `outfit_config.ini` with comments + examples
2. Read via `core/config.py` `get_config()` or `get_model_path()`
3. Use as fallback (last in search order)
4. Update README "Custom Model Paths" or equivalent section

### Adding a new utility module
1. Create in `nodes/utils/` or `core/`
2. Use try/except import pattern
3. Write unit tests
4. Import in the consuming node(s)

### Adding pipe-separated multi-value input
1. Add STRING input with tooltip showing format
2. Use `_parse_progression()` pattern to parse
3. Pad with last value if too few, truncate if too many
4. Fallback to global value when empty
