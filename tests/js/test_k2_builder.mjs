/**
 * Headless-Test der reinen Editor-Logik aus fvm_k2_builder.js.
 *
 * Lädt die Datei, entfernt die ComfyUI-Imports, stubbt app/api und testet die
 * Funktionen, die den gemeldeten Sprung verursacht haben.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.join(HERE, "..", "..", "web", "js", "fvm_k2_builder.js");

let code = fs.readFileSync(SRC, "utf8");
code = code.replace(/^import .*$/gm, "");
code = code.replace(/app\.registerExtension\(\{[\s\S]*\}\);\s*$/m, "");
code += `
globalThis.__t = { rescaleRect, rescaleLayout, applyDrag, hitTest, normalizeBox,
                   readLayout, writeLayout, uniqueName, nextBoxId };
`;

const app = { api: { addEventListener() {} }, registerExtension() {} };
const api = { fetchApi: async () => ({ json: async () => ({}) }) };
// eslint-disable-next-line no-new-func
new Function("app", "api", code)(app, api);
const T = globalThis.__t;

let passed = 0;
let failed = 0;
function check(name, condition, detail) {
    if (condition) { passed += 1; return; }
    failed += 1;
    console.log(`FAIL  ${name}${detail ? " — " + detail : ""}`);
}
function near(a, b, eps = 1e-9) { return Math.abs(a - b) <= eps; }

// ── rescaleRect: Parität mit core/k2/layout.py ───────────────────────────
{
    const r = T.rescaleRect({ x: 0.35, y: 0.35, w: 0.3, h: 0.3 }, [1024, 1024], [1920, 1080]);
    check("square stays square on 16:9", near(r.w * 1920, r.h * 1080, 0.5),
        `${(r.w * 1920).toFixed(1)} vs ${(r.h * 1080).toFixed(1)}`);

    const same = T.rescaleRect({ x: 0.1, y: 0.2, w: 0.3, h: 0.4 }, [512, 512], [1024, 1024]);
    check("uniform scale is a noop",
        near(same.x, 0.1) && near(same.y, 0.2) && near(same.w, 0.3) && near(same.h, 0.4));

    const wide = T.rescaleRect({ x: 0.3, y: 0.3, w: 0.2, h: 0.2 }, [1024, 1024], [1920, 1080]);
    const back = T.rescaleRect(wide, [1920, 1080], [1024, 1024]);
    check("round trip is lossless", near(back.w, 0.2, 1e-9) && near(back.h, 0.2, 1e-9),
        `w=${back.w} h=${back.h}`);

    const big = T.rescaleRect({ x: 0, y: 0, w: 1, h: 1 }, [1024, 1024], [512, 2048]);
    check("oversized box shrinks uniformly", near(big.w * 512, big.h * 2048, 0.5));
    check("stays inside canvas", big.w <= 1 + 1e-9 && big.h <= 1 + 1e-9);

    const centre = T.rescaleRect({ x: 0.4, y: 0.4, w: 0.2, h: 0.2 }, [1024, 1024], [1536, 640]);
    check("centre stays relative",
        near(centre.x + centre.w / 2, 0.5, 0.01) && near(centre.y + centre.h / 2, 0.5, 0.01));
}

// ── applyDrag ────────────────────────────────────────────────────────────
{
    const start = { x: 0.2, y: 0.2, w: 0.3, h: 0.4 };
    const moved = T.applyDrag(start, "move", 0.1, -0.05);
    check("move shifts without resizing",
        near(moved.x, 0.3) && near(moved.y, 0.15) && near(moved.w, 0.3) && near(moved.h, 0.4));

    const clamped = T.applyDrag(start, "move", 0.9, 0);
    check("move clamps to canvas", near(clamped.x + clamped.w, 1) && near(clamped.w, 0.3));

    const east = T.applyDrag(start, "e", 0.1, 0);
    check("east handle only moves the right edge",
        near(east.x, 0.2) && near(east.w, 0.4) && near(east.h, 0.4));

    const north = T.applyDrag(start, "n", 0, 0.1);
    check("north handle anchors the bottom",
        near(north.y, 0.3) && near(north.h, 0.3) && near(north.x, 0.2));

    const corner = T.applyDrag(start, "se", 0.1, 0.1);
    check("se handle grows both", near(corner.w, 0.4) && near(corner.h, 0.5));

    const flipped = T.applyDrag(start, "w", 0.5, 0);
    check("crossing an edge flips cleanly", flipped.w > 0 && flipped.x >= 0.2 - 1e-9);

    const drawn = T.applyDrag({ x: 0.5, y: 0.5, w: 0, h: 0 }, "draw", -0.2, -0.3);
    check("draw handles negative direction",
        near(drawn.x, 0.3) && near(drawn.y, 0.2) && near(drawn.w, 0.2) && near(drawn.h, 0.3));
}

// ── hitTest ──────────────────────────────────────────────────────────────
{
    const data = { boxes: [T.normalizeBox({ id: "a", rect: { x: 0.2, y: 0.2, w: 0.4, h: 0.4 } })] };
    const W = 800;
    const H = 800;
    check("interior hit = move", T.hitTest(data, 0.4, 0.4, W, H)?.mode === "move");
    check("corner hit = nw", T.hitTest(data, 0.2, 0.2, W, H)?.mode === "nw");
    check("edge hit = e", T.hitTest(data, 0.6, 0.4, W, H)?.mode === "e");
    check("outside = miss", T.hitTest(data, 0.9, 0.9, W, H) === null);

    // Topmost box wins so overlapping regions stay selectable.
    const two = {
        boxes: [
            T.normalizeBox({ id: "a", rect: { x: 0.1, y: 0.1, w: 0.5, h: 0.5 } }),
            T.normalizeBox({ id: "b", rect: { x: 0.3, y: 0.3, w: 0.5, h: 0.5 } }),
        ],
    };
    check("later box wins in overlap", T.hitTest(two, 0.45, 0.45, W, H)?.box.id === "b");
}

// ── Der gemeldete Bug: Layoutwechsel mitten im Ziehen ────────────────────
{
    // Vorher: pointerNorm() maß bei jedem Move neu. Änderte sich die
    // Canvasgröße durch einen Panel-Rebuild, sprang die Box.
    const rectAtDown = { left: 100, top: 100, width: 600, height: 600 };
    const rectAfterRelayout = { left: 140, top: 100, width: 520, height: 520 };

    const normWith = (rect, clientX, clientY) => ({
        x: (clientX - rect.left) / rect.width,
        y: (clientY - rect.top) / rect.height,
    });

    const start = { x: 0.2, y: 0.2, w: 0.3, h: 0.3 };
    const origin = normWith(rectAtDown, 250, 250);

    // Zeiger bewegt sich um exakt 60 px nach rechts.
    const cached = normWith(rectAtDown, 310, 250);
    const stale = normWith(rectAfterRelayout, 310, 250);

    const withCache = T.applyDrag(start, "move", cached.x - origin.x, cached.y - origin.y);
    const withoutCache = T.applyDrag(start, "move", stale.x - origin.x, stale.y - origin.y);

    check("cached rect keeps the drag true to the cursor",
        near(withCache.x, 0.2 + 60 / 600, 1e-9),
        `x=${withCache.x}`);
    // Gemessen: 0.023 normalisiert ≈ 14 px auf einer 600-px-Fläche, und das pro
    // Move-Event — genau das sichtbare Springen.
    const jump = Math.abs(withoutCache.x - withCache.x);
    check("re-measured rect would jump (regression guard)", jump > 0.01,
        `cached=${withCache.x.toFixed(4)} stale=${withoutCache.x.toFixed(4)} jump=${jump.toFixed(4)}`);
}

// ── normalizeBox ─────────────────────────────────────────────────────────
{
    const flipped = T.normalizeBox({ id: "x", rect: { x: 0.6, y: 0.6, w: -0.2, h: -0.3 } });
    check("negative size is normalized",
        near(flipped.rect.x, 0.4) && near(flipped.rect.w, 0.2)
        && near(flipped.rect.y, 0.3) && near(flipped.rect.h, 0.3));

    const out = T.normalizeBox({ id: "x", rect: { x: 0.9, y: 0.9, w: 0.5, h: 0.5 } });
    check("clamped into canvas",
        out.rect.x + out.rect.w <= 1 + 1e-9 && out.rect.y + out.rect.h <= 1 + 1e-9);

    const defaults = T.normalizeBox({ id: "x", rect: { x: 0, y: 0, w: 0.5, h: 0.5 } });
    check("defaults filled in",
        defaults.enabled === true && defaults.role === "auto" && Array.isArray(defaults.loras));
}

// ── rescaleLayout ────────────────────────────────────────────────────────
{
    const layout = {
        canvas: { width: 1024, height: 1024 },
        boxes: [T.normalizeBox({ id: "a", rect: { x: 0.1, y: 0.1, w: 0.3, h: 0.3 } })],
    };
    const changed = T.rescaleLayout(layout, 1920, 1080);
    check("rescaleLayout reports a change", changed === true);
    check("canvas updated", layout.canvas.width === 1920 && layout.canvas.height === 1080);
    check("box shape preserved",
        near(layout.boxes[0].rect.w * 1920, layout.boxes[0].rect.h * 1080, 0.5));
    check("same size is a noop", T.rescaleLayout(layout, 1920, 1080) === false);
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
