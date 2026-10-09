# /build-node — ComfyUI Node Builder & Tester

Build a new ComfyUI node for the comfyui-FVMtools package with automatic testing.

## Input

$ARGUMENTS

## Workflow

Follow these steps strictly in order. Do NOT skip steps.

### Step 1: Understand the Spec

Parse the input specification. Identify:
- Node class name (e.g. `FVM_PromptColorReplace`)
- Display name
- Category (default: `FVM Tools/...`)
- All inputs (required + optional) with types, defaults, ranges
- All outputs with types
- Core logic / algorithm
- Shared modules needed (in `core/`)
- Web extensions needed (in `web/js/`)

### Step 2: Create Shared Modules (if needed)

If the node needs shared utility modules (e.g. `core/color_database.py`):
1. Create `core/__init__.py` if it doesn't exist
2. Create each module file in `core/`
3. Keep modules focused — one responsibility per file

### Step 3: Create the Node

Create the node file in `nodes/` following project conventions:
- One file per node (or thematic group)
- Class with `CATEGORY`, `INPUT_TYPES`, `RETURN_TYPES`, `RETURN_NAMES`, `FUNCTION`
- `DESCRIPTION` string for UI tooltip
- `execute()` method — keep logic in separate functions for testability
- Always return tuple: `return (result,)`
- Tensor formats: IMAGE=[B,H,W,C], MASK=[B,H,W], LATENT dict with "samples" key
- All float32, IMAGE/MASK range 0-1

### Step 4: Register in `__init__.py`

Add import and entries to `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in the root `__init__.py`.
**IMPORTANT:** All imports in `__init__.py` MUST be inside a `try/except ImportError: pass` block
(pytest loads this file outside ComfyUI context where relative imports fail).

### Step 5: Create Web Extension (if needed)

If UI widgets are specified, create JS file in `web/js/`.
Use ComfyUI extension pattern:
```javascript
import { app } from "../../scripts/app.js";
app.registerExtension({ name: "FVM.NodeName", async nodeCreated(node) { ... } });
```

### Step 6: Write Unit Tests

Create test file in `tests/unit/test_<node_name>.py`:
- Use fixtures from `conftest.py` (dummy_image, dummy_mask, etc.)
- Test categories:
  1. **Schema**: INPUT_TYPES structure, RETURN_TYPES, FUNCTION exists
  2. **Basic execution**: Call execute() with valid inputs, check tuple return
  3. **Edge cases**: Empty inputs, extreme values, single pixel, large batch
  4. **Output validation**: Correct shapes, dtypes, value ranges
  5. **Logic specifics**: Test the core algorithm with known inputs/outputs
- For `core/` modules: write separate test files in `tests/unit/`

### Step 7: Run Tests — Auto-Improve Loop

Run `pytest tests/unit/ -v` from the project root.

**Loop until all tests pass (max 5 iterations):**
1. Run tests
2. If failures: read the error, fix the source code (NOT the test, unless the test itself is wrong)
3. Re-run tests
4. Repeat

If stuck after 5 iterations, stop and report the remaining failures.

### Step 8: Summary

Report:
- Files created/modified
- Test results (pass/fail count)
- Any known issues or TODOs
- What needs a ComfyUI restart vs. hot-reload

## Project Conventions

- Package: `comfyui-FVMtools`
- Node files: `nodes/<name>.py`
- Shared logic: `core/<module>.py` or `nodes/utils/<module>.py`
- Tests: `tests/unit/test_<name>.py`
- Web: `web/js/<name>.js`
- Category prefix: `FVM Tools/`
- All nodes use V1 schema (class with INPUT_TYPES dict)
- **Python/pytest**: Always use ComfyUI venv: `D:/AI/ComfyUI/ComfyUI/venv/Scripts/python.exe -m pytest`
- System Python does NOT have the right dependencies — always use the venv
- **NEVER add `__init__.py` to `tests/` directories** — breaks pytest's package resolution
- Root `conftest.py` handles sys.path + ComfyUI mocking; `tests/conftest.py` has tensor fixtures
- Root `__init__.py` must wrap imports in `try/except ImportError: pass`
