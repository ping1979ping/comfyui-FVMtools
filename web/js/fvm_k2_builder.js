/**
 * FVM K2 Lab — Region Builder
 *
 * A detached editor window for Krea 2 regional layouts. One node holds
 * everything a region needs: its box, its prompt, its identity prompt and any
 * number of LoRAs bound to it.
 *
 *   ┌ node ────────────────┐        ┌ editor window (draggable, resizable) ──┐
 *   │ width / height       │        │ ┌ canvas ────┐ ┌ box list ───────────┐ │
 *   │ global_prompt        │  edit  │ │  backdrop  │ │ Anna  subject  100  │ │
 *   │ [ Edit layout ]      │ ─────► │ │  + boxes   │ │ Bea   subject   99  │ │
 *   │ ┌ mini preview ────┐ │        │ └────────────┘ ├─────────────────────┤ │
 *   │ └──────────────────┘ │        │                │ prompt / identity   │ │
 *   └──────────────────────┘        │                │ LoRAs (n per box)   │ │
 *                                   └────────────────┴─────────────────────┘ │
 *
 * Boxes are stored normalized (0..1). When the canvas aspect changes they are
 * rescaled shape-preserving — see rescaleRect(), which mirrors
 * core/k2/layout.py::rescale_rect exactly. Keep both in sync.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "FVM_K2_RegionBuilder";
const LAYOUT_WIDGET = "layout";
const LAYOUT_VERSION = 1;

const BOX_COLORS = [
    "#e0564f", "#4f8fe0", "#5fc27e", "#e0b64f",
    "#a765e0", "#4fd0d0", "#e07fb4", "#8ea34f",
];

const ROLES = ["auto", "subject", "background"];
const ROUTINGS = ["standard", "character_identity"];

// Catppuccin Mocha — same palette as the other FVMtools dialogs.
const C = {
    bg: "#1e1e2e", bgDeep: "#181825", bgDeepest: "#11111b",
    surface: "#313244", border: "#45475a",
    text: "#cdd6f4", muted: "#a6adc8", faint: "#6c7086",
    green: "#a6e3a1", red: "#f38ba8", yellow: "#f9e2af", blue: "#89b4fa",
};

const FONT = "Consolas, 'Courier New', monospace";

/** Newest sampled image, kept for the "latest render" backdrop. */
let lastResultImage = null;
app.api?.addEventListener?.("executed", (e) => {
    const images = e?.detail?.output?.images;
    if (Array.isArray(images) && images.length) lastResultImage = images[images.length - 1];
});

function viewUrl(image) {
    return "/view?" + new URLSearchParams({
        filename: image.filename,
        subfolder: image.subfolder || "",
        type: image.type || "output",
    });
}

// ─── Layout helpers ──────────────────────────────────────────────────────

function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

/**
 * Hide a widget but keep it serializing.
 * Deliberately NOT type = "converted-widget": that magic string makes ComfyUI
 * draw an input socket for it.
 */
function hideWidget(node, name) {
    const widget = findWidget(node, name);
    if (!widget || widget.__fvmHidden) return widget;
    widget.__fvmHidden = true;
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
    if (widget.options) widget.options.serialize = true;
    return widget;
}

function emptyLayout(width, height) {
    return {
        version: LAYOUT_VERSION,
        canvas: { width: width || 1024, height: height || 1024 },
        boxes: [],
        global_loras: [],
    };
}

function readLayout(node) {
    const widget = findWidget(node, LAYOUT_WIDGET);
    const width = findWidget(node, "width")?.value || 1024;
    const height = findWidget(node, "height")?.value || 1024;
    let data = null;
    try {
        data = JSON.parse(widget?.value || "{}");
    } catch (e) {
        console.error("[FVMTools] K2 builder: layout is not valid JSON, starting empty:", e);
    }
    if (!data || typeof data !== "object" || Array.isArray(data)) data = emptyLayout(width, height);
    if (!Array.isArray(data.boxes)) data.boxes = [];
    if (!Array.isArray(data.global_loras)) data.global_loras = [];
    if (!data.canvas || typeof data.canvas !== "object") data.canvas = { width, height };
    data.version = LAYOUT_VERSION;
    for (const box of data.boxes) normalizeBox(box);
    return data;
}

function writeLayout(node, data, silent) {
    const widget = findWidget(node, LAYOUT_WIDGET);
    if (!widget) return;
    widget.value = JSON.stringify(data, null, 2);
    // ComfyUI's change tracker samples on mouseup; canvas drags preventDefault,
    // so nudge it explicitly or the workflow stays "unmodified". Never during a
    // drag though — that would cut the gesture short.
    if (!silent) {
        try {
            window.dispatchEvent(new MouseEvent("mouseup", { bubbles: true }));
        } catch (e) { /* ignore */ }
    }
    node.setDirtyCanvas?.(true, true);
}

function normalizeBox(box) {
    const rect = box.rect || box;
    let x = Number(rect.x) || 0;
    let y = Number(rect.y) || 0;
    let w = Number(rect.w ?? rect.width) || 0;
    let h = Number(rect.h ?? rect.height) || 0;
    if (w < 0) { x += w; w = -w; }
    if (h < 0) { y += h; h = -h; }
    x = Math.min(Math.max(x, 0), 1);
    y = Math.min(Math.max(y, 0), 1);
    w = Math.min(Math.max(w, 0), 1 - x);
    h = Math.min(Math.max(h, 0), 1 - y);
    box.rect = { x, y, w, h };
    if (!Array.isArray(box.loras)) box.loras = [];
    if (typeof box.enabled !== "boolean") box.enabled = true;
    if (!ROLES.includes(box.role)) box.role = "auto";
    return box;
}

/**
 * Mirror of core/k2/layout.py::rescale_rect — keep both in sync.
 *
 * Compensates only the ASPECT change, split symmetrically across both axes so
 * the conversion is reversible. A plain fraction editor (like the KJ builder)
 * skips this and silently turns a square box into a wide rectangle.
 */
function rescaleRect(rect, oldSize, newSize) {
    const [ow, oh] = oldSize;
    const [nw, nh] = newSize;
    if (Math.min(ow, oh, nw, nh) <= 0 || rect.w <= 0 || rect.h <= 0) return rect;

    const ratio = (nw / nh) / (ow / oh);
    if (ratio === 1) return rect;

    const root = Math.sqrt(ratio);
    let w = rect.w / root;
    let h = rect.h * root;

    const contain = Math.min(1, w > 1 ? 1 / w : 1, h > 1 ? 1 / h : 1);
    w *= contain;
    h *= contain;

    const cx = rect.x + rect.w / 2;
    const cy = rect.y + rect.h / 2;
    let x = cx - w / 2;
    let y = cy - h / 2;
    x = Math.min(Math.max(x, 0), Math.max(0, 1 - w));
    y = Math.min(Math.max(y, 0), Math.max(0, 1 - h));
    return { x, y, w, h };
}

function rescaleLayout(data, newWidth, newHeight) {
    const old = [
        Number(data.canvas?.width) || newWidth,
        Number(data.canvas?.height) || newHeight,
    ];
    if (old[0] === newWidth && old[1] === newHeight) return false;
    for (const box of data.boxes) {
        box.rect = rescaleRect(box.rect, old, [newWidth, newHeight]);
    }
    data.canvas = { width: newWidth, height: newHeight };
    return true;
}

function nextBoxId(data) {
    let index = data.boxes.length + 1;
    const used = new Set(data.boxes.map((b) => b.id));
    while (used.has(`box-${index}`)) index += 1;
    return `box-${index}`;
}

function uniqueName(data, base) {
    const used = new Set(data.boxes.map((b) => (b.name || "").toLowerCase()));
    if (!used.has(base.toLowerCase())) return base;
    let index = 2;
    while (used.has(`${base} ${index}`.toLowerCase())) index += 1;
    return `${base} ${index}`;
}

function boxColor(index) {
    return BOX_COLORS[index % BOX_COLORS.length];
}

// ─── LoRA inventory ──────────────────────────────────────────────────────

let loraCache = null;
async function fetchLoras(force = false) {
    if (loraCache && !force) return loraCache;
    try {
        const resp = await api.fetchApi("/fvmtools/loras");
        const data = await resp.json();
        loraCache = ["None", ...(data.loras || [])];
    } catch (e) {
        console.error("[FVMTools] K2 builder: could not list LoRAs:", e);
        loraCache = ["None"];
    }
    return loraCache;
}

async function fetchRecentOutputs(limit = 24) {
    try {
        const resp = await api.fetchApi(`/fvmtools/k2/recent-outputs?limit=${limit}`);
        const data = await resp.json();
        return data.images || [];
    } catch (e) {
        console.error("[FVMTools] K2 builder: could not list outputs:", e);
        return [];
    }
}

// ─── Small DOM helpers ───────────────────────────────────────────────────

function el(tag, style, props) {
    const node = document.createElement(tag);
    if (style) Object.assign(node.style, style);
    if (props) Object.assign(node, props);
    return node;
}

function button(label, title, onClick, extra) {
    const b = el("button", Object.assign({
        background: C.surface, color: C.text, border: `1px solid ${C.border}`,
        borderRadius: "4px", padding: "3px 8px", cursor: "pointer",
        font: `12px ${FONT}`, whiteSpace: "nowrap",
    }, extra || {}));
    b.textContent = label;
    if (title) b.title = title;
    b.addEventListener("click", onClick);
    return b;
}

function labelled(text, control, title) {
    const wrap = el("label", {
        display: "flex", alignItems: "center", gap: "6px",
        font: `11px ${FONT}`, color: C.muted, marginBottom: "4px",
    });
    const span = el("span", { minWidth: "62px", flex: "0 0 auto" });
    span.textContent = text;
    if (title) wrap.title = title;
    wrap.append(span, control);
    return wrap;
}

function textInput(value, onChange, title, extra) {
    const input = el("input", Object.assign({
        flex: "1 1 auto", minWidth: "0", background: C.bgDeepest, color: C.text,
        border: `1px solid ${C.border}`, borderRadius: "4px", padding: "3px 6px",
        font: `12px ${FONT}`,
    }, extra || {}));
    input.value = value ?? "";
    if (title) input.title = title;
    input.addEventListener("input", () => onChange(input.value));
    return input;
}

function numberInput(value, onChange, title, extra) {
    const input = textInput(value, (v) => onChange(parseFloat(v)), title, extra);
    input.type = "number";
    return input;
}

function selectInput(options, value, onChange, title) {
    const select = el("select", {
        flex: "1 1 auto", minWidth: "0", background: C.bgDeepest, color: C.text,
        border: `1px solid ${C.border}`, borderRadius: "4px", padding: "3px 4px",
        font: `12px ${FONT}`,
    });
    for (const option of options) {
        const item = el("option");
        item.value = option;
        item.textContent = option;
        select.append(item);
    }
    select.value = value;
    if (title) select.title = title;
    select.addEventListener("change", () => onChange(select.value));
    return select;
}

function textArea(value, onChange, title, rows) {
    const area = el("textarea", {
        width: "100%", boxSizing: "border-box", background: C.bgDeepest,
        color: C.text, border: `1px solid ${C.border}`, borderRadius: "4px",
        padding: "4px 6px", font: `12px ${FONT}`, resize: "vertical",
        minHeight: `${(rows || 2) * 18}px`,
    });
    area.value = value ?? "";
    if (title) area.title = title;
    area.addEventListener("input", () => onChange(area.value));
    return area;
}

// ─── Canvas painting (shared by node preview and editor) ─────────────────

function paintLayout(canvas, data, options) {
    const opts = options || {};
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || canvas.width;
    const cssH = canvas.clientHeight || canvas.height;
    if (cssW <= 0 || cssH <= 0) return;
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
        canvas.width = Math.round(cssW * dpr);
        canvas.height = Math.round(cssH * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    ctx.fillStyle = C.bgDeepest;
    ctx.fillRect(0, 0, cssW, cssH);

    if (opts.backdrop && opts.backdrop.complete && opts.backdrop.naturalWidth) {
        ctx.globalAlpha = opts.backdropAlpha ?? 0.75;
        ctx.drawImage(opts.backdrop, 0, 0, cssW, cssH);
        ctx.globalAlpha = 1;
    }

    // Thirds guide — helps place subjects without a backdrop.
    if (opts.guides !== false) {
        ctx.strokeStyle = "rgba(205,214,244,0.10)";
        ctx.lineWidth = 1;
        for (let i = 1; i < 3; i += 1) {
            ctx.beginPath();
            ctx.moveTo(Math.round((cssW * i) / 3) + 0.5, 0);
            ctx.lineTo(Math.round((cssW * i) / 3) + 0.5, cssH);
            ctx.moveTo(0, Math.round((cssH * i) / 3) + 0.5);
            ctx.lineTo(cssW, Math.round((cssH * i) / 3) + 0.5);
            ctx.stroke();
        }
    }

    data.boxes.forEach((box, index) => {
        const color = boxColor(index);
        const r = box.rect;
        const x = r.x * cssW;
        const y = r.y * cssH;
        const w = r.w * cssW;
        const h = r.h * cssH;
        const selected = opts.selectedId && box.id === opts.selectedId;

        ctx.globalAlpha = box.enabled === false ? 0.25 : 1;
        ctx.fillStyle = color + "26";
        ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = color;
        ctx.lineWidth = selected ? 2.5 : 1.5;
        ctx.setLineDash(box.role === "background" ? [6, 4] : []);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);

        if (opts.labels !== false && h > 14) {
            const label = `${index + 1} ${box.name || ""}`.trim();
            ctx.font = `11px ${FONT}`;
            const tw = ctx.measureText(label).width + 8;
            ctx.fillStyle = color;
            ctx.fillRect(x, y, Math.min(tw, w), 15);
            ctx.fillStyle = "#11111b";
            ctx.fillText(label, x + 4, y + 11);
            if (box.loras?.length) {
                const badge = `${box.loras.length}L`;
                const bw = ctx.measureText(badge).width + 8;
                ctx.fillStyle = color;
                ctx.fillRect(x + w - Math.min(bw, w), y, Math.min(bw, w), 15);
                ctx.fillStyle = "#11111b";
                ctx.fillText(badge, x + w - Math.min(bw, w) + 4, y + 11);
            }
        }

        if (selected && opts.handles) {
            ctx.fillStyle = color;
            for (const [hx, hy] of [
                [x, y], [x + w / 2, y], [x + w, y],
                [x, y + h / 2], [x + w, y + h / 2],
                [x, y + h], [x + w / 2, y + h], [x + w, y + h],
            ]) {
                ctx.fillRect(hx - 4, hy - 4, 8, 8);
            }
        }
        ctx.globalAlpha = 1;
    });

    if (!data.boxes.length && opts.emptyHint !== false) {
        ctx.fillStyle = C.faint;
        ctx.font = `12px ${FONT}`;
        ctx.textAlign = "center";
        ctx.fillText("no regions — drag on the canvas to add one", cssW / 2, cssH / 2);
        ctx.textAlign = "left";
    }
}

// ─── Editor window (detached, draggable, resizable) ──────────────────────

const HANDLE = 8;

function hitTest(data, nx, ny, aspectW, aspectH) {
    // Handle radius in normalized units so it stays grabbable on any canvas.
    const rx = HANDLE / Math.max(aspectW, 1);
    const ry = HANDLE / Math.max(aspectH, 1);
    for (let i = data.boxes.length - 1; i >= 0; i -= 1) {
        const box = data.boxes[i];
        const { x, y, w, h } = box.rect;
        const x2 = x + w;
        const y2 = y + h;
        const near = (a, b, r) => Math.abs(a - b) <= r;
        const inX = nx >= x - rx && nx <= x2 + rx;
        const inY = ny >= y - ry && ny <= y2 + ry;
        if (!inX || !inY) continue;
        const hrx = Math.min(rx, w / 3);
        const hry = Math.min(ry, h / 3);
        if (near(nx, x, hrx) && near(ny, y, hry)) return { box, mode: "nw" };
        if (near(nx, x2, hrx) && near(ny, y, hry)) return { box, mode: "ne" };
        if (near(nx, x, hrx) && near(ny, y2, hry)) return { box, mode: "sw" };
        if (near(nx, x2, hrx) && near(ny, y2, hry)) return { box, mode: "se" };
        if (near(ny, y, hry) && nx >= x && nx <= x2) return { box, mode: "n" };
        if (near(ny, y2, hry) && nx >= x && nx <= x2) return { box, mode: "s" };
        if (near(nx, x, hrx) && ny >= y && ny <= y2) return { box, mode: "w" };
        if (near(nx, x2, hrx) && ny >= y && ny <= y2) return { box, mode: "e" };
        if (nx >= x && nx <= x2 && ny >= y && ny <= y2) return { box, mode: "move" };
    }
    return null;
}

const CURSORS = {
    move: "move", n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize",
    nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize",
};

function applyDrag(start, mode, dx, dy) {
    const clamp = (v) => Math.min(Math.max(v, 0), 1);
    let { x, y, w, h } = start;
    if (mode === "move") {
        return { x: clamp(Math.min(x + dx, 1 - w)), y: clamp(Math.min(y + dy, 1 - h)), w, h };
    }
    if (mode === "draw") {
        const ax = clamp(x);
        const ay = clamp(y);
        const bx = clamp(x + dx);
        const by = clamp(y + dy);
        return { x: Math.min(ax, bx), y: Math.min(ay, by), w: Math.abs(bx - ax), h: Math.abs(by - ay) };
    }
    let l = x;
    let t = y;
    let r = x + w;
    let b = y + h;
    if (mode.includes("w")) l = clamp(l + dx);
    if (mode.includes("e")) r = clamp(r + dx);
    if (mode.includes("n")) t = clamp(t + dy);
    if (mode.includes("s")) b = clamp(b + dy);
    if (r < l) [l, r] = [r, l];
    if (b < t) [t, b] = [b, t];
    return { x: l, y: t, w: r - l, h: b - t };
}

function createEditor() {
    let node = null;
    let data = emptyLayout(1024, 1024);
    let selectedId = null;
    let backdrop = null;
    let backdropAlpha = 0.75;
    let drag = null;

    const win = el("div", {
        position: "fixed", left: "120px", top: "90px", width: "1080px", height: "660px",
        minWidth: "640px", minHeight: "420px", background: C.bg, color: C.text,
        border: `1px solid ${C.border}`, borderRadius: "10px", zIndex: "10000",
        boxShadow: "0 12px 40px rgba(0,0,0,0.55)", display: "none",
        flexDirection: "column", overflow: "hidden", font: `12px ${FONT}`,
    });

    // Header ─────────────────────────────────────────────────────────────
    const head = el("div", {
        display: "flex", alignItems: "center", gap: "8px", padding: "7px 10px",
        background: C.bgDeep, borderBottom: `1px solid ${C.border}`, cursor: "move",
        flex: "0 0 auto",
    });
    const title = el("div", { fontWeight: "bold", flex: "1 1 auto", color: C.blue });
    title.textContent = "K2 Region Builder";
    const dimsLabel = el("div", { color: C.faint, font: `11px ${FONT}` });

    const backdropSelect = selectInput(
        ["none"], "none", (value) => applyBackdrop(value),
        "Backdrop image behind the canvas. Pick a recent render so boxes can be "
        + "placed against a real composition."
    );
    backdropSelect.style.maxWidth = "230px";
    backdropSelect.addEventListener("mousedown", (e) => e.stopPropagation());

    const refreshBtn = button("Pull latest", "Reload the list of recent renders and "
        + "show the newest one behind the canvas.", async () => {
        await refreshBackdrops(true);
    });
    const dimBtn = button("Dim", "Cycle backdrop brightness so box outlines stay readable.",
        () => {
            backdropAlpha = backdropAlpha >= 0.95 ? 0.35 : backdropAlpha + 0.2;
            draw();
        });
    const closeBtn = button("✕", "Close the editor (Esc)", () => close(),
        { padding: "3px 9px", color: C.red });

    head.append(title, dimsLabel, backdropSelect, refreshBtn, dimBtn, closeBtn);

    // Body ───────────────────────────────────────────────────────────────
    const body = el("div", { display: "flex", flex: "1 1 auto", minHeight: "0" });

    // 2/3 canvas, 1/3 inputs. The canvas is absolutely positioned inside its
    // pane: resizing it must never feed back into the flex layout, otherwise
    // the ResizeObserver and the canvas chase each other and the whole window
    // jitters while you click.
    const canvasPane = el("div", {
        flex: "2 1 0", minWidth: "0", position: "relative", overflow: "hidden",
        background: C.bgDeepest,
    });
    const canvas = el("canvas", {
        position: "absolute", left: "50%", top: "50%",
        transform: "translate(-50%, -50%)",
        background: C.bgDeepest, cursor: "crosshair", touchAction: "none",
    });
    canvas.title = "Drag on empty space to draw a new region. Drag a box to move it, "
        + "its edges or corners to resize. Del removes the selected region.";
    canvasPane.append(canvas);

    const side = el("div", {
        flex: "1 1 0", minWidth: "280px", display: "flex", flexDirection: "column",
        borderLeft: `1px solid ${C.border}`, background: C.bg, minHeight: "0",
    });

    const listHead = el("div", {
        display: "flex", alignItems: "center", gap: "6px", padding: "8px 10px",
        borderBottom: `1px solid ${C.border}`, flex: "0 0 auto",
    });
    const listTitle = el("div", { flex: "1 1 auto", color: C.muted });
    listTitle.textContent = "Regions";
    listHead.append(
        listTitle,
        button("+ Region", "Add a region in the centre of the canvas.", () => addBox()),
        button("Fit", "Spread all regions evenly across the canvas width.", () => fitBoxes()),
    );

    // overflowY: "scroll" (not "auto") — a scrollbar that appears and vanishes
    // changes the pane width and would shift the canvas under the cursor.
    const list = el("div", {
        flex: "0 0 auto", height: "170px", overflowY: "scroll", padding: "6px 8px",
        borderBottom: `1px solid ${C.border}`,
    });
    const detail = el("div", {
        flex: "1 1 auto", overflowY: "scroll", padding: "8px 10px", minHeight: "0",
    });

    side.append(listHead, list, detail);
    body.append(canvasPane, side);

    // Footer ─────────────────────────────────────────────────────────────
    const foot = el("div", {
        display: "flex", alignItems: "center", gap: "8px", padding: "6px 10px",
        borderTop: `1px solid ${C.border}`, background: C.bgDeep, flex: "0 0 auto",
    });
    const status = el("div", { flex: "1 1 auto", color: C.faint, font: `11px ${FONT}` });
    foot.append(
        status,
        button("Copy JSON", "Copy the layout JSON to the clipboard.", () => {
            navigator.clipboard?.writeText(JSON.stringify(data, null, 2));
            setStatus("layout copied", C.green);
        }),
        button("Paste JSON", "Replace the layout with JSON from the clipboard.", async () => {
            try {
                const text = await navigator.clipboard.readText();
                const parsed = JSON.parse(text);
                if (!parsed || !Array.isArray(parsed.boxes)) throw new Error("no 'boxes' array");
                data = parsed;
                data.boxes.forEach(normalizeBox);
                selectedId = data.boxes[0]?.id || null;
                commit();
                setStatus("layout pasted", C.green);
            } catch (e) {
                setStatus(`paste failed: ${e.message}`, C.red);
            }
        }),
        button("Close", "Close the editor (Esc)", () => close()),
    );

    const grip = el("div", {
        position: "absolute", right: "0", bottom: "0", width: "16px", height: "16px",
        cursor: "nwse-resize", background:
            `linear-gradient(135deg, transparent 50%, ${C.border} 50%)`,
    });
    grip.title = "Resize the editor window";

    win.append(head, body, foot, grip);
    document.body.append(win);

    // Keep the graph out of the dialog. Even with focus behaving correctly,
    // keystrokes and wheel events that bubble out of here would drive LiteGraph
    // (delete node, pan, zoom) instead of the field under the cursor.
    // Bubble phase only. Stopping in the capture phase would keep the event
    // from ever reaching the input it was meant for — that kills typing.
    for (const type of ["keydown", "keyup", "keypress"]) {
        win.addEventListener(type, (e) => e.stopPropagation(), false);
    }
    win.addEventListener("wheel", (e) => e.stopPropagation(), { passive: false });
    win.addEventListener("contextmenu", (e) => e.stopPropagation());
    // Do NOT preventDefault here — that would stop inputs from taking focus.
    win.addEventListener("pointerdown", (e) => e.stopPropagation());
    win.dataset.captureWheel = "true";

    // ─── Window drag / resize ────────────────────────────────────────────
    head.addEventListener("pointerdown", (e) => {
        if (e.target.tagName === "BUTTON" || e.target.tagName === "SELECT") return;
        const startX = e.clientX;
        const startY = e.clientY;
        const left = win.offsetLeft;
        const top = win.offsetTop;
        const move = (ev) => {
            win.style.left = `${left + ev.clientX - startX}px`;
            win.style.top = `${Math.max(0, top + ev.clientY - startY)}px`;
        };
        const up = () => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
        e.preventDefault();
    });

    grip.addEventListener("pointerdown", (e) => {
        const startX = e.clientX;
        const startY = e.clientY;
        const w0 = win.offsetWidth;
        const h0 = win.offsetHeight;
        const move = (ev) => {
            win.style.width = `${Math.max(640, w0 + ev.clientX - startX)}px`;
            win.style.height = `${Math.max(420, h0 + ev.clientY - startY)}px`;
            fitCanvas();
        };
        const up = () => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
        e.preventDefault();
        e.stopPropagation();
    });

    function onKey(e) {
        if (win.style.display === "none") return;
        if (e.key === "Escape") { close(); return; }
        if ((e.key === "Delete" || e.key === "Backspace") && selectedId) {
            const tag = document.activeElement?.tagName;
            if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
            removeBox(selectedId);
            e.preventDefault();
        }
    }
    document.addEventListener("keydown", onKey, true);

    // ─── Canvas sizing keeps the real aspect ratio ───────────────────────
    let lastFit = "";

    function fitCanvas(force) {
        // Never resize mid-drag: the pointer maths is relative to the rect
        // captured on pointerdown, so a resize here would drag the box away
        // from the cursor.
        if (drag && !force) return;
        const width = findWidget(node, "width")?.value || data.canvas.width || 1024;
        const height = findWidget(node, "height")?.value || data.canvas.height || 1024;
        const availW = Math.max(80, canvasPane.clientWidth - 20);
        const availH = Math.max(80, canvasPane.clientHeight - 20);
        const aspect = width / height;
        let w = availW;
        let h = w / aspect;
        if (h > availH) { h = availH; w = h * aspect; }
        w = Math.round(w);
        h = Math.round(h);

        const signature = `${w}x${h}x${width}x${height}`;
        if (signature === lastFit) { draw(); return; }
        lastFit = signature;

        canvas.style.width = `${w}px`;
        canvas.style.height = `${h}px`;
        dimsLabel.textContent = `${width} × ${height}`;
        draw();
    }

    function draw() {
        paintLayout(canvas, data, {
            selectedId, handles: true, backdrop, backdropAlpha,
        });
    }

    function setStatus(text, color) {
        status.textContent = text;
        status.style.color = color || C.faint;
    }

    /**
     * Save without touching the side panel.
     *
     * Typing must never rebuild the detail panel: replacing the DOM would tear
     * the focused input out from under the caret. The field loses focus after
     * the first character and every following keystroke reaches ComfyUI as a
     * shortcut instead.
     */
    let flushTimer = 0;

    function persist() {
        if (!node) return;
        const width = findWidget(node, "width")?.value || 1024;
        const height = findWidget(node, "height")?.value || 1024;
        data.canvas = { width, height };
        // The value is written immediately so nothing can be lost, but the
        // synthetic mouseup that pokes ComfyUI's change tracker is debounced —
        // firing it on every keystroke would interfere with the UI.
        writeLayout(node, data, true);
        node.__fvmK2Refresh?.();
        draw();
        if (flushTimer) clearTimeout(flushTimer);
        flushTimer = setTimeout(() => {
            flushTimer = 0;
            if (drag) return;
            try {
                window.dispatchEvent(new MouseEvent("mouseup", { bubbles: true }));
            } catch (e) { /* ignore */ }
        }, 400);
    }

    /** Save AND rebuild the panel — only for structural changes. */
    function commit() {
        persist();
        if (!drag) {
            renderList();
            renderDetail();
        }
    }

    // ─── Box operations ──────────────────────────────────────────────────
    function addBox(rect, silent) {
        const id = nextBoxId(data);
        const index = data.boxes.length;
        const box = normalizeBox({
            id,
            name: uniqueName(data, `Region ${index + 1}`),
            rect: rect || { x: 0.1 + 0.05 * index, y: 0.1, w: 0.35, h: 0.8 },
            prompt: "",
            identity_prompt: "",
            negative_prompt: "",
            role: "auto",
            priority: 100 - index,
            enabled: true,
            loras: [],
        });
        data.boxes.push(box);
        selectedId = id;
        // While drawing, committing would rebuild the panel and fire the
        // synthetic mouseup mid-drag. endDrag() commits once at the end.
        if (!silent) commit();
        return box;
    }

    function removeBox(id) {
        const index = data.boxes.findIndex((b) => b.id === id);
        if (index === -1) return;
        data.boxes.splice(index, 1);
        selectedId = data.boxes[Math.max(0, index - 1)]?.id || null;
        commit();
    }

    function fitBoxes() {
        const count = data.boxes.length;
        if (!count) return;
        const gap = 0.02;
        const width = (1 - gap * (count + 1)) / count;
        data.boxes.forEach((box, index) => {
            box.rect = { x: gap + index * (width + gap), y: 0.06, w: width, h: 0.88 };
        });
        commit();
        setStatus(`${count} region(s) spread evenly`, C.green);
    }

    function selected() {
        return data.boxes.find((b) => b.id === selectedId) || null;
    }

    // ─── Canvas interaction ──────────────────────────────────────────────
    /**
     * Normalized pointer position. During a drag the rect captured on
     * pointerdown is reused: re-measuring per move would let any layout change
     * (panel rebuild, scrollbar, window resize) shift the box under the cursor.
     */
    function pointerNorm(e, cachedRect) {
        const rect = cachedRect || canvas.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left) / Math.max(rect.width, 1),
            y: (e.clientY - rect.top) / Math.max(rect.height, 1),
        };
    }

    canvas.addEventListener("pointerdown", (e) => {
        const rect = canvas.getBoundingClientRect();
        const p = pointerNorm(e, rect);
        const hit = hitTest(data, p.x, p.y, rect.width, rect.height);
        canvas.setPointerCapture(e.pointerId);
        if (hit) {
            const changedSelection = selectedId !== hit.box.id;
            selectedId = hit.box.id;
            drag = { mode: hit.mode, box: hit.box, start: { ...hit.box.rect }, origin: p, rect };
            // Only rebuild the side panel when the selection actually changed —
            // rebuilding on every click is what made the window jump.
            if (changedSelection) {
                renderList();
                renderDetail();
            }
        } else {
            const box = addBox({ x: p.x, y: p.y, w: 0, h: 0 }, true);
            drag = { mode: "draw", box, start: { ...box.rect }, origin: p, rect };
        }
        draw();
        e.preventDefault();
    });

    canvas.addEventListener("pointermove", (e) => {
        if (!drag) {
            const rect = canvas.getBoundingClientRect();
            const p = pointerNorm(e, rect);
            const hit = hitTest(data, p.x, p.y, rect.width, rect.height);
            canvas.style.cursor = hit ? CURSORS[hit.mode] || "move" : "crosshair";
            return;
        }
        const p = pointerNorm(e, drag.rect);
        drag.box.rect = applyDrag(drag.start, drag.mode, p.x - drag.origin.x, p.y - drag.origin.y);
        draw();
        e.preventDefault();
    });

    function endDrag(e) {
        if (!drag) return;
        const box = drag.box;
        const tiny = box.rect.w < 0.01 || box.rect.h < 0.01;
        drag = null;
        if (tiny) {
            removeBox(box.id);
            setStatus("region discarded — too small", C.yellow);
            return;
        }
        normalizeBox(box);
        commit();
        if (e) e.preventDefault();
    }
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);

    // ─── Region list ─────────────────────────────────────────────────────
    function renderList() {
        list.textContent = "";
        if (!data.boxes.length) {
            const hint = el("div", { color: C.faint, padding: "6px 2px" });
            hint.textContent = "No regions yet — drag on the canvas or press + Region.";
            list.append(hint);
            return;
        }
        data.boxes.forEach((box, index) => {
            const row = el("div", {
                display: "flex", alignItems: "center", gap: "6px", padding: "3px 4px",
                borderRadius: "4px", cursor: "pointer", marginBottom: "2px",
                background: box.id === selectedId ? C.surface : "transparent",
                opacity: box.enabled === false ? "0.5" : "1",
            });
            const swatch = el("div", {
                width: "10px", height: "10px", borderRadius: "2px",
                background: boxColor(index), flex: "0 0 auto",
            });
            const name = el("div", {
                flex: "1 1 auto", overflow: "hidden", textOverflow: "ellipsis",
                whiteSpace: "nowrap",
            });
            name.textContent = `${index + 1}. ${box.name || "(unnamed)"}`;
            const meta = el("div", { color: C.faint, font: `11px ${FONT}` });
            meta.textContent = `${box.role}${box.loras?.length ? ` · ${box.loras.length}L` : ""}`;
            row.addEventListener("click", () => {
                selectedId = box.id;
                renderList();
                renderDetail();
                draw();
            });
            const up = button("↑", "Move earlier (higher priority in the compiled prompt)", (e) => {
                e.stopPropagation();
                if (index === 0) return;
                data.boxes.splice(index - 1, 0, data.boxes.splice(index, 1)[0]);
                commit();
            }, { padding: "0 5px" });
            const del = button("✕", "Delete this region", (e) => {
                e.stopPropagation();
                removeBox(box.id);
            }, { padding: "0 5px", color: C.red });
            row.append(swatch, name, meta, up, del);
            list.append(row);
        });
    }

    // ─── Region detail + LoRAs ───────────────────────────────────────────
    function renderDetail() {
        detail.textContent = "";
        const box = selected();
        if (!box) {
            const hint = el("div", { color: C.faint });
            hint.textContent = "Select a region to edit its prompt and LoRAs.";
            detail.append(hint);
            return;
        }
        const index = data.boxes.indexOf(box);

        const head2 = el("div", {
            display: "flex", alignItems: "center", gap: "6px", marginBottom: "6px",
        });
        const swatch = el("div", {
            width: "12px", height: "12px", borderRadius: "3px", background: boxColor(index),
        });
        const heading = el("div", { fontWeight: "bold", flex: "1 1 auto" });
        heading.textContent = `Region ${index + 1}`;
        const enable = el("input", {});
        enable.type = "checkbox";
        enable.checked = box.enabled !== false;
        enable.title = "Include this region. Unchecked keeps it in the layout but "
            + "excludes it from prompt compilation and LoRA routing.";
        enable.addEventListener("change", () => {
            box.enabled = enable.checked;
            persist();
            renderList();
        });
        const enableLabel = el("label", { display: "flex", alignItems: "center", gap: "4px", color: C.muted });
        enableLabel.append(enable, document.createTextNode("enabled"));
        head2.append(swatch, heading, enableLabel);

        // All text fields use persist() + a debounced write. renderList() is
        // safe to call — it lives in a different container than the focused
        // field — but renderDetail() must not run while typing.
        const nameRow = labelled("Name", textInput(box.name, (v) => {
            box.name = v;
            persist();
            renderList();
        }, "Human readable label. It appears in the generated spatial instructions "
           + "('Anna is to the left of Bea'), so keep it short and unique."),
            "Region label — must be unique.");

        const roleRow = labelled("Role", selectInput(ROLES, box.role, (v) => {
            box.role = v;
            persist();
            renderList();
        }, "subject: full outside penalty, competes with other subjects.\n"
           + "background: softer penalty, may feather beyond its box.\n"
           + "auto: boxes covering >=70% of canvas width become background."),
            "How hard the region is bound.");

        const prioRow = labelled("Priority", numberInput(box.priority, (v) => {
            box.priority = Number.isFinite(v) ? v : 100;
            persist();
        }, "Higher compiles first and claims an ambiguous detected face first. "
           + "It is NOT a strength and NOT an image z-index."),
            "Compile order.");

        const promptLabel = el("div", { color: C.muted, font: `11px ${FONT}`, margin: "6px 0 3px" });
        promptLabel.textContent = "Prompt";
        const prompt = textArea(box.prompt, (v) => { box.prompt = v; persist(); },
            "What should appear inside this box. An empty prompt disables the region.", 3);

        const identityLabel = el("div", { color: C.muted, font: `11px ${FONT}`, margin: "6px 0 3px" });
        identityLabel.textContent = "Identity prompt";
        const identity = textArea(box.identity_prompt, (v) => { box.identity_prompt = v; persist(); },
            "Face/identity description. It is attached to the region clause as an "
            + "attribute (', with …'), protected from the projector delta, and "
            + "preferred by K2 Regional Face Detail.", 2);

        const negLabel = el("div", { color: C.muted, font: `11px ${FONT}`, margin: "6px 0 3px" });
        negLabel.textContent = "Negative (stored only)";
        const negative = textArea(box.negative_prompt, (v) => { box.negative_prompt = v; persist(); },
            "Region-local negative text. Krea 2 Turbo runs CFG-free and has no "
            + "separate regional negative branch — this is kept for tooling only.", 2);

        const loraHead = el("div", {
            display: "flex", alignItems: "center", gap: "6px", margin: "10px 0 4px",
            borderTop: `1px solid ${C.border}`, paddingTop: "8px",
        });
        const loraTitle = el("div", { flex: "1 1 auto", color: C.muted });
        loraTitle.textContent = `LoRAs for this region (${box.loras.length})`;
        loraHead.append(loraTitle, button("+ LoRA",
            "Attach another LoRA to this region. Its delta is gated to this box only.",
            () => { box.loras.push({ name: "None", strength: 1.0, routing: "standard", trigger: "", enabled: true }); commit(); }));

        detail.append(head2, nameRow, roleRow, prioRow, promptLabel, prompt,
            identityLabel, identity, negLabel, negative, loraHead);

        box.loras.forEach((entry, slot) => {
            detail.append(renderLoraRow(box, entry, slot));
        });
    }

    function renderLoraRow(box, entry, slot) {
        const wrap = el("div", {
            border: `1px solid ${C.border}`, borderRadius: "5px", padding: "6px",
            marginBottom: "6px", background: C.bgDeep,
        });
        const top = el("div", { display: "flex", alignItems: "center", gap: "6px", marginBottom: "4px" });
        const enable = el("input", {});
        enable.type = "checkbox";
        enable.checked = entry.enabled !== false;
        enable.title = "Include this LoRA.";
        enable.addEventListener("change", () => {
            entry.enabled = enable.checked;
            persist();
            renderList();
        });

        const select = selectInput(loraCache || ["None"], entry.name || "None", (v) => {
            entry.name = v;
            persist();
            renderList();
        }, "Krea 2 LoRA applied only inside this region. Non-Krea architectures are "
           + "rejected at compose time instead of silently doing nothing.");
        select.style.flex = "1 1 auto";

        const remove = button("✕", "Remove this LoRA from the region", () => {
            box.loras.splice(slot, 1);
            commit();
        }, { padding: "0 6px", color: C.red });
        top.append(enable, select, remove);

        const strengthRow = labelled("Strength", numberInput(entry.strength ?? 1.0, (v) => {
            entry.strength = Number.isFinite(v) ? v : 1.0;
            persist();
        }, "Delta multiplier from -4 to 4. 0 disables the assignment, negative values "
           + "invert the learned delta.", { step: "0.05" }), "LoRA strength.");

        const routingRow = labelled("Routing", selectInput(ROUTINGS, entry.routing || "standard", (v) => {
            entry.routing = v;
            persist();
        }, "standard: gate text-fusion and local main-stream deltas to this box.\n"
           + "character_identity: same isolation plus an explicit identity anchor "
           + "built from the trigger phrase — use it for face/person LoRAs."),
            "Routing mode.");

        const triggerRow = labelled("Trigger", textInput(entry.trigger || "", (v) => {
            entry.trigger = v;
            persist();
        }, "Activation phrase learned during LoRA training. Required for "
           + "character_identity routing."), "Trigger phrase.");

        wrap.append(top, strengthRow, routingRow, triggerRow);
        return wrap;
    }

    // ─── Backdrop ────────────────────────────────────────────────────────
    let recentImages = [];

    async function refreshBackdrops(pickNewest) {
        recentImages = await fetchRecentOutputs(24);
        backdropSelect.textContent = "";
        const none = el("option");
        none.value = "none";
        none.textContent = "backdrop: none";
        backdropSelect.append(none);
        recentImages.forEach((image, index) => {
            const option = el("option");
            option.value = String(index);
            option.textContent = `${index === 0 ? "latest · " : ""}${image.subfolder ? image.subfolder + "/" : ""}${image.filename}`;
            backdropSelect.append(option);
        });
        if (pickNewest && recentImages.length) {
            backdropSelect.value = "0";
            applyBackdrop("0");
            setStatus(`backdrop: ${recentImages[0].filename}`, C.green);
        } else if (!recentImages.length) {
            setStatus("no rendered images found yet", C.yellow);
        }
    }

    function applyBackdrop(value) {
        if (value === "none") {
            backdrop = null;
            draw();
            return;
        }
        const image = recentImages[parseInt(value, 10)];
        if (!image) return;
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => { backdrop = img; draw(); };
        img.onerror = () => setStatus("could not load backdrop", C.red);
        img.src = viewUrl(image);
    }

    // ─── Open / close ────────────────────────────────────────────────────
    function open(hostNode) {
        node = hostNode;
        data = readLayout(node);
        selectedId = data.boxes[0]?.id || null;
        win.style.display = "flex";
        fetchLoras().then(() => renderDetail());
        refreshBackdrops(false);
        renderList();
        renderDetail();
        requestAnimationFrame(() => fitCanvas());
        setStatus(`${data.boxes.length} region(s)`, C.faint);
    }

    function close() {
        if (node) commit();
        win.style.display = "none";
        node = null;
    }

    // Only react to a real pane size change, coalesced into one frame — an
    // unguarded observer plus fitCanvas() is a feedback loop.
    let paneSize = "";
    let fitFrame = 0;
    const observer = new ResizeObserver(() => {
        if (win.style.display === "none" || drag) return;
        const signature = `${canvasPane.clientWidth}x${canvasPane.clientHeight}`;
        if (signature === paneSize) return;
        paneSize = signature;
        if (fitFrame) return;
        fitFrame = requestAnimationFrame(() => { fitFrame = 0; fitCanvas(); });
    });
    observer.observe(canvasPane);

    return { open, close, refresh: () => { if (node) { data = readLayout(node); renderList(); renderDetail(); fitCanvas(); } } };
}

let editor = null;
function getEditor() {
    if (!editor) editor = createEditor();
    return editor;
}

// Exposed for the headless test in tests/js/test_k2_builder.mjs.
if (typeof globalThis !== "undefined") {
    globalThis.__fvmK2Internals = {
        rescaleRect, rescaleLayout, applyDrag, hitTest, normalizeBox,
        readLayout, writeLayout, uniqueName, nextBoxId, createEditor,
    };
}

// ─── Extension ───────────────────────────────────────────────────────────

app.registerExtension({
    name: "FVMTools.K2.RegionBuilder",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;

            hideWidget(node, LAYOUT_WIDGET);

            // Mini preview inside the node.
            const host = el("div", {
                width: "100%", background: "#1a1a25", border: "1px solid #2a2a3a",
                borderRadius: "6px", padding: "4px", boxSizing: "border-box",
            });
            const preview = el("canvas", {
                width: "100%", height: "150px", display: "block", cursor: "pointer",
                borderRadius: "4px",
            });
            preview.title = "Region layout preview — click to open the editor window.";
            preview.addEventListener("click", () => getEditor().open(node));
            host.append(preview);

            const editButton = node.addWidget("button", "Edit layout ⤢", null, () => {
                getEditor().open(node);
            });
            editButton.tooltip = "Open the detached region editor: draw boxes, write "
                + "per-box prompts and attach LoRAs per box.";

            const domWidget = node.addDOMWidget("fvm_k2_preview", "div", host, {
                serialize: false,
                getHeight: () => 168,
            });
            // Frontend >=1.44 freezes widget.width at its initial value.
            try {
                Object.defineProperty(domWidget, "width", {
                    configurable: true, get() { return undefined; }, set() {},
                });
            } catch (e) { /* ignore */ }

            const redraw = () => {
                const data = readLayout(node);
                const width = findWidget(node, "width")?.value || 1024;
                const height = findWidget(node, "height")?.value || 1024;
                const availW = Math.max(40, host.clientWidth - 8);
                const aspect = width / height;
                let w = availW;
                let h = w / aspect;
                if (h > 150) { h = 150; w = h * aspect; }
                preview.style.width = `${Math.round(w)}px`;
                preview.style.height = `${Math.round(h)}px`;
                preview.style.margin = "0 auto";
                paintLayout(preview, data, { labels: true, guides: false });
            };
            node.__fvmK2Refresh = redraw;

            // Aspect compensation: rescale boxes when the canvas format changes.
            for (const name of ["width", "height"]) {
                const widget = findWidget(node, name);
                if (!widget) continue;
                const previous = widget.callback;
                widget.callback = function () {
                    const result2 = previous ? previous.apply(this, arguments) : undefined;
                    const data = readLayout(node);
                    const w = findWidget(node, "width")?.value || 1024;
                    const h = findWidget(node, "height")?.value || 1024;
                    if (rescaleLayout(data, w, h)) writeLayout(node, data);
                    redraw();
                    getEditor().refresh?.();
                    return result2;
                };
            }

            const observer = new ResizeObserver(() => redraw());
            observer.observe(host);
            setTimeout(redraw, 0);

            return result;
        };

        // ComfyUI restores widgets_values AFTER onNodeCreated and then calls
        // onConfigure — replay the layout one tick later or the node stays empty
        // after a workflow reload.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            hideWidget(this, LAYOUT_WIDGET);
            setTimeout(() => this.__fvmK2Refresh?.(), 0);
            return result;
        };
    },
});
