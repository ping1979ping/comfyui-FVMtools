import { app } from "../../../scripts/app.js";

// ═══════════════════════════════════════════════════════════════════════════
// Section 1: Canvas drawing utilities (ported from rgthree utils_canvas.js)
// ═══════════════════════════════════════════════════════════════════════════

function isLowQuality() {
    return (app.canvas.ds?.scale || 1) <= 0.5;
}

function measureText(ctx, str) {
    return ctx.measureText(str).width;
}

function fitString(ctx, str, maxWidth) {
    let width = measureText(ctx, str);
    const ellipsis = "\u2026";
    const ellipsisWidth = measureText(ctx, ellipsis);
    if (width <= maxWidth || width <= ellipsisWidth) return str;
    let min = 0, max = str.length;
    while (min <= max) {
        const guess = Math.floor((min + max) / 2);
        const w = measureText(ctx, str.substring(0, guess));
        if (w === maxWidth - ellipsisWidth) return str.substring(0, guess) + ellipsis;
        if (w < maxWidth - ellipsisWidth) min = guess + 1;
        else max = guess - 1;
    }
    return str.substring(0, max) + ellipsis;
}

function drawRoundedRectangle(ctx, options) {
    const lowQ = isLowQuality();
    ctx.save();
    ctx.strokeStyle = options.colorStroke || LiteGraph.WIDGET_OUTLINE_COLOR;
    ctx.fillStyle = options.colorBackground || LiteGraph.WIDGET_BGCOLOR;
    ctx.beginPath();
    ctx.roundRect(
        ...options.pos, ...options.size,
        lowQ ? [0] : options.borderRadius ? [options.borderRadius] : [options.size[1] * 0.5]
    );
    ctx.fill();
    if (!lowQ) ctx.stroke();
    ctx.restore();
}

function drawTogglePart(ctx, options) {
    const lowQ = isLowQuality();
    ctx.save();
    const { posX, posY, height, value } = options;
    const toggleRadius = height * 0.36;
    const toggleBgWidth = height * 1.5;
    if (!lowQ) {
        ctx.beginPath();
        ctx.roundRect(posX + 4, posY + 4, toggleBgWidth - 8, height - 8, [height * 0.5]);
        ctx.globalAlpha = app.canvas.editor_alpha * 0.25;
        ctx.fillStyle = "rgba(255,255,255,0.45)";
        ctx.fill();
        ctx.globalAlpha = app.canvas.editor_alpha;
    }
    ctx.fillStyle = value === true ? "#89B" : "#888";
    const toggleX = lowQ || value === false
        ? posX + height * 0.5
        : value === true
            ? posX + height
            : posX + height * 0.75;
    ctx.beginPath();
    ctx.arc(toggleX, posY + height * 0.5, toggleRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    return [posX, toggleBgWidth];
}

function drawNumberWidgetPart(ctx, options) {
    const arrowWidth = 9, arrowHeight = 10, innerMargin = 3, numberWidth = 32;
    const xBoundsArrowLess = [0, 0], xBoundsNumber = [0, 0], xBoundsArrowMore = [0, 0];
    ctx.save();
    let posX = options.posX;
    const { posY, height, value, textColor } = options;
    const midY = posY + height / 2;
    if (options.direction === -1) {
        posX = posX - arrowWidth - innerMargin - numberWidth - innerMargin - arrowWidth;
    }
    ctx.fill(new Path2D(`M ${posX} ${midY} l ${arrowWidth} ${arrowHeight / 2} l 0 -${arrowHeight} L ${posX} ${midY} z`));
    xBoundsArrowLess[0] = posX;
    xBoundsArrowLess[1] = arrowWidth;
    posX += arrowWidth + innerMargin;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const oldColor = ctx.fillStyle;
    if (textColor) ctx.fillStyle = textColor;
    ctx.fillText(fitString(ctx, value.toFixed(2), numberWidth), posX + numberWidth / 2, midY);
    ctx.fillStyle = oldColor;
    xBoundsNumber[0] = posX;
    xBoundsNumber[1] = numberWidth;
    posX += numberWidth + innerMargin;
    ctx.fill(new Path2D(`M ${posX} ${midY - arrowHeight / 2} l ${arrowWidth} ${arrowHeight / 2} l -${arrowWidth} ${arrowHeight / 2} v -${arrowHeight} z`));
    xBoundsArrowMore[0] = posX;
    xBoundsArrowMore[1] = arrowWidth;
    ctx.restore();
    return [xBoundsArrowLess, xBoundsNumber, xBoundsArrowMore];
}
drawNumberWidgetPart.WIDTH_TOTAL = 9 + 3 + 32 + 3 + 9;

function drawWidgetButton(ctx, options, text, pressed) {
    const lowQ = isLowQuality();
    ctx.save();
    ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
    ctx.fillStyle = pressed ? "#555" : LiteGraph.WIDGET_BGCOLOR;
    ctx.beginPath();
    ctx.roundRect(options.pos[0], options.pos[1], options.size[0], options.size[1],
        lowQ ? [0] : [options.size[1] * 0.5]);
    ctx.fill();
    if (!lowQ) ctx.stroke();
    if (!lowQ) {
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        ctx.fillText(text, options.pos[0] + options.size[0] / 2, options.pos[1] + options.size[1] / 2);
    }
    ctx.restore();
}


// ═══════════════════════════════════════════════════════════════════════════
// Section 2: Base widget class (ported from rgthree utils_widgets.js)
// ═══════════════════════════════════════════════════════════════════════════

class FvmBaseWidget {
    constructor(name) {
        this.type = "custom";
        this.options = {};
        this.y = 0;
        this.last_y = 0;
        this.mouseDowned = null;
        this.isMouseDownedAndOver = false;
        this.hitAreas = {};
        this.downedHitAreasForMove = [];
        this.downedHitAreasForClick = [];
        this.name = name;
    }

    serializeValue(node, index) { return this.value; }

    clickWasWithinBounds(pos, bounds) {
        const xStart = bounds[0];
        const xEnd = xStart + (bounds.length > 2 ? bounds[2] : bounds[1]);
        const clickedX = pos[0] >= xStart && pos[0] <= xEnd;
        if (bounds.length === 2) return clickedX;
        return clickedX && pos[1] >= bounds[1] && pos[1] <= bounds[1] + bounds[3];
    }

    mouse(event, pos, node) {
        if (event.type === "pointerdown") {
            this.mouseDowned = [...pos];
            this.isMouseDownedAndOver = true;
            this.downedHitAreasForMove.length = 0;
            this.downedHitAreasForClick.length = 0;
            let anyHandled = false;
            for (const part of Object.values(this.hitAreas)) {
                if (this.clickWasWithinBounds(pos, part.bounds)) {
                    if (part.onMove) this.downedHitAreasForMove.push(part);
                    if (part.onClick) this.downedHitAreasForClick.push(part);
                    if (part.onDown) {
                        const handled = part.onDown.apply(this, [event, pos, node, part]);
                        anyHandled = anyHandled || handled === true;
                    }
                    part.wasMouseClickedAndIsOver = true;
                }
            }
            return this.onMouseDown(event, pos, node) ?? anyHandled;
        }
        if (event.type === "pointerup") {
            if (!this.mouseDowned) return true;
            this.downedHitAreasForMove.length = 0;
            const wasOver = this.isMouseDownedAndOver;
            this.cancelMouseDown();
            let anyHandled = false;
            for (const part of Object.values(this.hitAreas)) {
                if (part.onUp && this.clickWasWithinBounds(pos, part.bounds)) {
                    const handled = part.onUp.apply(this, [event, pos, node, part]);
                    anyHandled = anyHandled || handled === true;
                }
                part.wasMouseClickedAndIsOver = false;
            }
            for (const part of this.downedHitAreasForClick) {
                if (this.clickWasWithinBounds(pos, part.bounds)) {
                    const handled = part.onClick.apply(this, [event, pos, node, part]);
                    anyHandled = anyHandled || handled === true;
                }
            }
            this.downedHitAreasForClick.length = 0;
            if (wasOver) {
                const handled = this.onMouseClick(event, pos, node);
                anyHandled = anyHandled || handled === true;
            }
            return this.onMouseUp(event, pos, node) ?? anyHandled;
        }
        if (event.type === "pointermove") {
            this.isMouseDownedAndOver = !!this.mouseDowned;
            if (this.mouseDowned &&
                (pos[0] < 15 || pos[0] > node.size[0] - 15 ||
                 pos[1] < this.last_y || pos[1] > this.last_y + LiteGraph.NODE_WIDGET_HEIGHT)) {
                this.isMouseDownedAndOver = false;
            }
            for (const part of Object.values(this.hitAreas)) {
                if (this.downedHitAreasForMove.includes(part)) {
                    part.onMove.apply(this, [event, pos, node, part]);
                }
                if (this.downedHitAreasForClick.includes(part)) {
                    part.wasMouseClickedAndIsOver = this.clickWasWithinBounds(pos, part.bounds);
                }
            }
            return this.onMouseMove(event, pos, node) ?? true;
        }
        return false;
    }

    cancelMouseDown() {
        this.mouseDowned = null;
        this.isMouseDownedAndOver = false;
        this.downedHitAreasForMove.length = 0;
    }

    onMouseDown(event, pos, node) { return; }
    onMouseUp(event, pos, node) { return; }
    onMouseClick(event, pos, node) { return; }
    onMouseMove(event, pos, node) { return; }
}


// ═══════════════════════════════════════════════════════════════════════════
// Section 3: Array utilities
// ═══════════════════════════════════════════════════════════════════════════

function moveArrayItem(arr, item, newIndex) {
    const oldIndex = arr.indexOf(item);
    if (oldIndex < 0 || newIndex < 0 || newIndex >= arr.length) return;
    arr.splice(oldIndex, 1);
    arr.splice(newIndex, 0, item);
}

function removeArrayItem(arr, item) {
    const idx = arr.indexOf(item);
    if (idx >= 0) arr.splice(idx, 1);
}


// ═══════════════════════════════════════════════════════════════════════════
// Section 4: LoRA list fetching & chooser
// ═══════════════════════════════════════════════════════════════════════════

let _loraListCache = null;

async function fetchLoraList(force = false) {
    if (_loraListCache && !force) return _loraListCache;
    try {
        const resp = await fetch("/fvmtools/loras");
        const data = await resp.json();
        _loraListCache = data.loras || [];
    } catch (e) {
        console.warn("[FVMTools] Failed to fetch lora list:", e);
        _loraListCache = [];
    }
    return _loraListCache;
}

function showLoraChooser(event, callback) {
    fetchLoraList().then(loras => {
        const items = ["None", ...loras];
        new LiteGraph.ContextMenu(items, {
            event,
            title: "Choose LoRA",
            scale: Math.max(1, app.canvas.ds?.scale ?? 1),
            className: "dark",
            callback,
        });
    });
}


// ═══════════════════════════════════════════════════════════════════════════
// Section 5: LoRA info dialog
// ═══════════════════════════════════════════════════════════════════════════

let _infoDialogStyleInjected = false;

function injectInfoDialogStyle() {
    if (_infoDialogStyleInjected) return;
    _infoDialogStyleInjected = true;
    const style = document.createElement("style");
    style.textContent = `
        .fvm-lora-info-dialog {
            position: fixed; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            background: #1a1a2e; color: #e0e0e0;
            border: 1px solid #444; border-radius: 8px;
            padding: 0; min-width: 420px; max-width: 600px;
            z-index: 10000; font-family: Arial, sans-serif;
            box-shadow: 0 8px 32px rgba(0,0,0,0.6);
        }
        .fvm-lora-info-dialog::backdrop {
            background: rgba(0,0,0,0.5);
        }
        .fvm-lora-info-header {
            padding: 12px 16px; background: #16213e;
            border-bottom: 1px solid #444; border-radius: 8px 8px 0 0;
            display: flex; justify-content: space-between; align-items: center;
        }
        .fvm-lora-info-header h3 {
            margin: 0; font-size: 14px; color: #89B;
        }
        .fvm-lora-info-close {
            background: none; border: none; color: #888; cursor: pointer;
            font-size: 18px; padding: 0 4px; line-height: 1;
        }
        .fvm-lora-info-close:hover { color: #fff; }
        .fvm-lora-info-body {
            padding: 16px; max-height: 500px; overflow-y: auto;
        }
        .fvm-lora-info-body table {
            width: 100%; border-collapse: collapse;
        }
        .fvm-lora-info-body td {
            padding: 6px 8px; border-bottom: 1px solid #333;
            font-size: 13px; vertical-align: top;
        }
        .fvm-lora-info-body td:first-child {
            color: #89B; white-space: nowrap; width: 110px;
        }
        .fvm-lora-info-body a {
            color: #6af; text-decoration: none;
        }
        .fvm-lora-info-body a:hover { text-decoration: underline; }
        .fvm-lora-info-tag {
            display: inline-block; padding: 2px 8px; border-radius: 4px;
            background: #2a2a4a; color: #aaa; font-size: 11px; margin-right: 4px;
        }
        .fvm-lora-info-words {
            display: flex; flex-wrap: wrap; gap: 4px;
        }
        .fvm-lora-info-word {
            display: inline-block; padding: 2px 8px; border-radius: 4px;
            background: #2a3a2a; color: #8c8; font-size: 12px; cursor: pointer;
        }
        .fvm-lora-info-word:hover { background: #3a4a3a; }
        .fvm-lora-info-loading {
            text-align: center; padding: 20px; color: #888;
        }
        .fvm-lora-info-error {
            color: #f88; padding: 8px;
        }
    `;
    document.head.appendChild(style);
}

async function showLoraInfoDialog(loraName) {
    if (!loraName || loraName === "None") return;
    injectInfoDialogStyle();

    // Create dialog
    const dialog = document.createElement("dialog");
    dialog.className = "fvm-lora-info-dialog";
    dialog.innerHTML = `
        <div class="fvm-lora-info-header">
            <h3>LORA INFO</h3>
            <button class="fvm-lora-info-close">\u00d7</button>
        </div>
        <div class="fvm-lora-info-body">
            <div class="fvm-lora-info-loading">Loading info for ${loraName}...</div>
        </div>
    `;
    document.body.appendChild(dialog);
    dialog.showModal();

    dialog.querySelector(".fvm-lora-info-close").onclick = () => {
        dialog.close();
        dialog.remove();
    };
    dialog.addEventListener("click", (e) => {
        if (e.target === dialog) { dialog.close(); dialog.remove(); }
    });

    // Fetch info
    const body = dialog.querySelector(".fvm-lora-info-body");
    try {
        const resp = await fetch(`/fvmtools/lora-info?file=${encodeURIComponent(loraName)}`);
        if (!resp.ok && resp.status === 404) {
            body.innerHTML = `
                <table>
                    <tr><td>File</td><td>${loraName}</td></tr>
                    <tr><td>Status</td><td class="fvm-lora-info-error">LoRA file not found. Check that the file exists in your configured loras folder.</td></tr>
                </table>
            `;
            return;
        }
        const data = await resp.json();

        if (data.error && !data.name) {
            body.innerHTML = `
                <table>
                    <tr><td>File</td><td>${loraName}</td></tr>
                    <tr><td>SHA256</td><td style="font-size:11px;word-break:break-all">${data.sha256 || "unknown"}</td></tr>
                    <tr><td>Status</td><td class="fvm-lora-info-error">${data.error}</td></tr>
                </table>
            `;
            return;
        }

        // Update dialog title with actual name
        const headerTitle = dialog.querySelector(".fvm-lora-info-header h3");
        if (headerTitle && data.name) {
            headerTitle.textContent = data.name;
        }

        const triggerHtml = (data.triggerWords || []).length > 0
            ? `<div class="fvm-lora-info-words">${data.triggerWords.map(w =>
                `<span class="fvm-lora-info-word" title="Click to copy">${w}</span>`
              ).join("")}</div>`
            : "<span style='color:#888'>None</span>";

        const civitaiHtml = data.civitaiUrl
            ? `<a href="${data.civitaiUrl}" target="_blank">${data.civitaiUrl}</a>`
            : "<span style='color:#888'>Not found on CivitAI</span>";

        body.innerHTML = `
            <table>
                <tr><td>Name</td><td><strong>${data.name || loraName}</strong></td></tr>
                ${data.version ? `<tr><td>Version</td><td>${data.version}</td></tr>` : ""}
                <tr><td>Type</td><td>
                    ${data.type ? `<span class="fvm-lora-info-tag">${data.type}</span>` : ""}
                    ${data.baseModel ? `<span class="fvm-lora-info-tag">${data.baseModel}</span>` : ""}
                </td></tr>
                <tr><td>File</td><td style="font-size:12px;word-break:break-all">${loraName}</td></tr>
                <tr><td>SHA256</td><td style="font-size:11px;word-break:break-all;color:#888">${data.sha256 || "unknown"}</td></tr>
                <tr><td>Trigger Words</td><td>${triggerHtml}</td></tr>
                <tr><td>CivitAI</td><td>${civitaiHtml}</td></tr>
                ${data.error ? `<tr><td>Note</td><td style="color:#fa8">${data.error}</td></tr>` : ""}
                ${data.source ? `<tr><td>Source</td><td style="color:#888;font-size:11px">${data.source === "sidecar" ? "Local metadata file" : "CivitAI API"}</td></tr>` : ""}
            </table>
        `;

        // Click trigger word to copy
        body.querySelectorAll(".fvm-lora-info-word").forEach(el => {
            el.onclick = () => {
                navigator.clipboard.writeText(el.textContent).then(() => {
                    el.style.background = "#4a5a4a";
                    setTimeout(() => { el.style.background = ""; }, 300);
                });
            };
        });
    } catch (e) {
        body.innerHTML = `<div class="fvm-lora-info-error">Failed to fetch info: ${e.message}</div>`;
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// Section 6: PowerLoraHeaderWidget
// ═══════════════════════════════════════════════════════════════════════════

class PowerLoraHeaderWidget extends FvmBaseWidget {
    constructor() {
        super("_power_lora_header");
        this.value = { type: "PowerLoraHeaderWidget" };
        this.type = "custom";
        this.options = { serialize: false };
        this.hitAreas = {
            toggle: { bounds: [0, 0], onDown: this.onToggleDown },
        };
    }

    draw(ctx, node, w, posY, height) {
        if (!this._hasLoraWidgets(node)) return;

        const margin = 10, innerMargin = margin * 0.33;
        const lowQ = isLowQuality();
        const allState = this._allLorasState(node);
        posY += 2;
        const midY = posY + height * 0.5;
        let posX = 10;

        ctx.save();
        this.hitAreas.toggle.bounds = drawTogglePart(ctx, { posX, posY, height, value: allState });

        if (!lowQ) {
            posX += this.hitAreas.toggle.bounds[1] + innerMargin;
            ctx.globalAlpha = app.canvas.editor_alpha * 0.55;
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText("Toggle All", posX, midY);

            const rposX = node.size[0] - margin - innerMargin - innerMargin;
            ctx.textAlign = "center";
            ctx.fillText("Strength", rposX - drawNumberWidgetPart.WIDTH_TOTAL / 2, midY);
        }
        ctx.restore();
    }

    onToggleDown(event, pos, node) {
        const widgets = this._getLoraWidgets(node);
        const allOn = widgets.every(w => w.value?.on);
        widgets.forEach(w => { if (w.value) w.value.on = !allOn; });
        this.cancelMouseDown();
        return true;
    }

    _hasLoraWidgets(node) {
        return node.widgets?.some(w => w.name?.startsWith("ref_lora_") || w.name === "gen_lora");
    }

    _getLoraWidgets(node) {
        return (node.widgets || []).filter(w => w.name?.startsWith("ref_lora_") || w.name === "gen_lora");
    }

    _allLorasState(node) {
        const widgets = this._getLoraWidgets(node);
        if (!widgets.length) return false;
        const allOn = widgets.every(w => w.value?.on === true);
        const allOff = widgets.every(w => w.value?.on === false);
        return allOn ? true : allOff ? false : null;
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// Section 7: RefLoraWidget
// ═══════════════════════════════════════════════════════════════════════════

const DEFAULT_LORA_VALUE = { on: true, lora: null, strength: 1.0 };

class RefLoraWidget extends FvmBaseWidget {
    constructor(name) {
        super(name);
        this.type = "custom";
        this.haveMouseMovedStrength = false;
        this._value = { ...DEFAULT_LORA_VALUE };
        this.hitAreas = {
            toggle:      { bounds: [0, 0], onDown: this.onToggleDown },
            lora:        { bounds: [0, 0], onClick: this.onLoraClick },
            strengthDec: { bounds: [0, 0], onClick: this.onStrengthDecClick },
            strengthVal: { bounds: [0, 0], onClick: this.onStrengthValClick },
            strengthInc: { bounds: [0, 0], onClick: this.onStrengthIncClick },
            strengthAny: { bounds: [0, 0], onMove: this.onStrengthDrag },
        };
    }

    get value() { return this._value; }
    set value(v) {
        if (typeof v === "object" && v !== null) {
            this._value = v;
        } else {
            this._value = { ...DEFAULT_LORA_VALUE };
        }
    }

    draw(ctx, node, w, posY, height) {
        const margin = 10, innerMargin = margin * 0.33;
        const midY = posY + height * 0.5;
        let posX = margin;

        // Background pill
        drawRoundedRectangle(ctx, {
            pos: [posX, posY],
            size: [node.size[0] - margin * 2, height],
        });

        // Toggle
        this.hitAreas.toggle.bounds = drawTogglePart(ctx, {
            posX, posY, height, value: this.value.on,
        });
        posX += this.hitAreas.toggle.bounds[1] + innerMargin;

        if (isLowQuality()) return;

        ctx.save();
        if (!this.value.on) {
            ctx.globalAlpha = app.canvas.editor_alpha * 0.4;
        }
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;

        // Strength (right-aligned)
        const rposX = node.size[0] - margin - innerMargin - innerMargin;
        const strengthVal = this.value.strength ?? 1.0;
        const [leftArr, textB, rightArr] = drawNumberWidgetPart(ctx, {
            posX: rposX, posY, height, value: strengthVal, direction: -1,
        });
        this.hitAreas.strengthDec.bounds = leftArr;
        this.hitAreas.strengthVal.bounds = textB;
        this.hitAreas.strengthInc.bounds = rightArr;
        this.hitAreas.strengthAny.bounds = [leftArr[0], rightArr[0] + rightArr[1] - leftArr[0]];

        // LoRA name (fills remaining space)
        const loraX = posX;
        const loraWidth = leftArr[0] - innerMargin - posX;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        const label = String(this.value.lora || "None");
        ctx.fillText(fitString(ctx, label, loraWidth), loraX, midY);
        this.hitAreas.lora.bounds = [loraX, loraWidth];

        ctx.globalAlpha = app.canvas.editor_alpha;
        ctx.restore();
    }

    serializeValue(node, index) {
        return { ...this.value };
    }

    // ── Event handlers ──

    onToggleDown(event, pos, node) {
        this.value.on = !this.value.on;
        this.cancelMouseDown();
        return true;
    }

    onLoraClick(event, pos, node) {
        showLoraChooser(event, (value) => {
            if (typeof value === "string") {
                this.value.lora = value === "None" ? null : value;
            }
            node.setDirtyCanvas(true, true);
        });
        this.cancelMouseDown();
    }

    onStrengthDecClick(event, pos, node) {
        this.stepStrength(-1);
    }

    onStrengthIncClick(event, pos, node) {
        this.stepStrength(1);
    }

    onStrengthDrag(event, pos, node) {
        if (event.deltaX) {
            this.haveMouseMovedStrength = true;
            this.value.strength = (this.value.strength ?? 1) + event.deltaX * 0.05;
        }
    }

    onStrengthValClick(event, pos, node) {
        if (this.haveMouseMovedStrength) return;
        const canvas = app.canvas;
        canvas.prompt("Value", this.value.strength, (v) => {
            this.value.strength = Number(v);
        }, event);
    }

    onMouseUp(event, pos, node) {
        super.onMouseUp(event, pos, node);
        this.haveMouseMovedStrength = false;
    }

    stepStrength(direction) {
        const step = 0.05;
        const strength = (this.value.strength ?? 1) + step * direction;
        this.value.strength = Math.round(strength * 100) / 100;
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// Section 8: Section separators
// ═══════════════════════════════════════════════════════════════════════════

function createSeparator(label) {
    return {
        name: "_sep_" + label.replace(/\s/g, "_"),
        type: "custom",
        value: label,
        options: { serialize: false },
        computeSize: () => [0, 24],
        draw(ctx, n, width, posY) {
            ctx.save();
            ctx.font = "bold 11px Arial";
            ctx.fillStyle = "#999";
            ctx.textAlign = "center";
            const cy = posY + 15;
            ctx.fillText(this.value, width / 2, cy);
            const tw = ctx.measureText(this.value).width;
            const cx = width / 2;
            ctx.strokeStyle = "#555";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(15, cy - 4);
            ctx.lineTo(Math.max(15, cx - tw / 2 - 10), cy - 4);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(Math.min(width - 15, cx + tw / 2 + 10), cy - 4);
            ctx.lineTo(width - 15, cy - 4);
            ctx.stroke();
            ctx.restore();
        },
    };
}


// ═══════════════════════════════════════════════════════════════════════════
// Section 9: Node registration
// ═══════════════════════════════════════════════════════════════════════════

// Store last canvas mouse event for context menu positioning
let _lastCanvasMouseEvent = null;

app.registerExtension({
    name: "FVMTools.PersonDetailerPower",

    async beforeRegisterNodeDef(nodeType, nodeData, appRef) {
        if (nodeData.name !== "PersonDetailerPower") return;

        // ── onNodeCreated: setup widgets ──
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            try {
                // Prefetch lora list
                fetchLoraList();

                // Insert section separators
                const SEPS = [
                    { label: "\u2500\u2500 References \u2500\u2500", before: "ref_prompt_1" },
                    { label: "\u2500\u2500 Generic \u2500\u2500", before: "generic_catch_unprocessed" },
                    { label: "\u2500\u2500 Inpaint \u2500\u2500", before: "mask_blend_pixels" },
                    { label: "\u2500\u2500 Detail Daemon \u2500\u2500", before: "detail_daemon_enabled" },
                    { label: "\u2500\u2500 Sampler \u2500\u2500", before: "seed" },
                ];
                for (let i = SEPS.length - 1; i >= 0; i--) {
                    const sep = SEPS[i];
                    const idx = this.widgets.findIndex(w => w.name === sep.before);
                    if (idx >= 0) {
                        this.widgets.splice(idx, 0, createSeparator(sep.label));
                    }
                }

                // Insert header widget before References separator
                const refSepIdx = this.widgets.findIndex(w => w.value === "\u2500\u2500 References \u2500\u2500");
                if (refSepIdx >= 0) {
                    this.widgets.splice(refSepIdx + 1, 0, new PowerLoraHeaderWidget());
                }

                // Insert RefLoraWidget before each ref_prompt_N
                for (let i = 1; i <= 5; i++) {
                    const promptIdx = this.widgets.findIndex(w => w.name === `ref_prompt_${i}`);
                    if (promptIdx >= 0) {
                        const loraWidget = new RefLoraWidget(`ref_lora_${i}`);
                        loraWidget.value = { on: i === 1, lora: null, strength: 1.0 };
                        this.widgets.splice(promptIdx, 0, loraWidget);
                    }
                }

                // Insert generic LoRA widget before gen_prompt
                const genPromptIdx = this.widgets.findIndex(w => w.name === "gen_prompt");
                if (genPromptIdx >= 0) {
                    const genLoraWidget = new RefLoraWidget("gen_lora");
                    genLoraWidget.value = { on: false, lora: null, strength: 1.0 };
                    this.widgets.splice(genPromptIdx, 0, genLoraWidget);
                }

                // Recompute node size
                const computed = this.computeSize();
                this.size[0] = Math.max(this.size[0], computed[0], 340);
                this.size[1] = Math.max(this.size[1], computed[1]);
            } catch (e) {
                console.error("[FVMTools] PersonDetailerPower widget setup error:", e);
            }

            return result;
        };

        // ── getSlotInPosition: detect clicks on lora widgets ──
        const origGetSlotInPosition = nodeType.prototype.getSlotInPosition;
        nodeType.prototype.getSlotInPosition = function (canvasX, canvasY) {
            const slot = origGetSlotInPosition?.apply(this, arguments);
            if (!slot) {
                let lastWidget = null;
                for (const widget of this.widgets) {
                    if (!widget.last_y) return;
                    if (canvasY > this.pos[1] + widget.last_y) {
                        lastWidget = widget;
                        continue;
                    }
                    break;
                }
                if (lastWidget?.name?.startsWith("ref_lora_") || lastWidget?.name === "gen_lora") {
                    return { widget: lastWidget, output: { type: "LORA WIDGET" } };
                }
            }
            return slot;
        };

        // ── getSlotMenuOptions: right-click context menu ──
        const origGetSlotMenuOptions = nodeType.prototype.getSlotMenuOptions;
        nodeType.prototype.getSlotMenuOptions = function (slot) {
            const widgetName = slot?.widget?.name;
            if (widgetName?.startsWith("ref_lora_") || widgetName === "gen_lora") {
                const widget = slot.widget;
                const index = this.widgets.indexOf(widget);

                // Check if adjacent widgets are also lora widgets (for move up/down)
                const canMoveUp = this.widgets[index - 1]?.name?.startsWith("ref_lora_") ||
                                  this.widgets[index - 1]?.name === "gen_lora";
                const canMoveDown = this.widgets[index + 1]?.name?.startsWith("ref_lora_") ||
                                    this.widgets[index + 1]?.name === "gen_lora";

                const menuItems = [
                    {
                        content: "\u2139\ufe0f Show Info",
                        callback: () => {
                            showLoraInfoDialog(widget.value.lora);
                        },
                    },
                    null, // separator
                    {
                        content: `${widget.value.on ? "\u26ab" : "\ud83d\udfe2"} Toggle ${widget.value.on ? "Off" : "On"}`,
                        callback: () => {
                            widget.value.on = !widget.value.on;
                        },
                    },
                    {
                        content: "\u2b06\ufe0f Move Up",
                        disabled: !canMoveUp,
                        callback: () => {
                            // Swap lora values with the widget above
                            const above = this.widgets[index - 1];
                            const tempVal = { ...widget.value };
                            widget.value = { ...above.value };
                            above.value = tempVal;
                        },
                    },
                    {
                        content: "\u2b07\ufe0f Move Down",
                        disabled: !canMoveDown,
                        callback: () => {
                            // Swap lora values with the widget below
                            const below = this.widgets[index + 1];
                            const tempVal = { ...widget.value };
                            widget.value = { ...below.value };
                            below.value = tempVal;
                        },
                    },
                    {
                        content: "\ud83d\uddd1\ufe0f Remove",
                        callback: () => {
                            widget.value = { on: false, lora: null, strength: 1.0 };
                        },
                    },
                ];

                new LiteGraph.ContextMenu(menuItems, {
                    title: "LORA WIDGET",
                    event: _lastCanvasMouseEvent || event,
                });
                return undefined; // prevent default menu
            }
            return origGetSlotMenuOptions?.apply(this, arguments);
        };

        // ── Track mouse events for context menu positioning ──
        const origOnMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (e) {
            _lastCanvasMouseEvent = e;
            return origOnMouseDown?.apply(this, arguments);
        };

        // ── Execution info display ──
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted?.apply(this, arguments);
            if (message?.text?.length > 0) {
                this._pdInfo = message.text[0];
                this.setDirtyCanvas(true);
            }
            return r;
        };

        const onDrawFG = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const r = onDrawFG?.apply(this, arguments);
            if (this._pdInfo) {
                ctx.save();
                ctx.font = "11px Arial";
                ctx.textAlign = "left";

                const parts = this._pdInfo.split(" | ");
                const lineH = 14;
                let y = this.size[1] - 6;

                for (let i = parts.length - 1; i >= 0; i--) {
                    const part = parts[i].trim();
                    if (part.includes("no input") || part.includes("no ref") || part.includes("skip")) {
                        ctx.fillStyle = "#f88";
                    } else if (part.startsWith("Generic")) {
                        ctx.fillStyle = "#ff8";
                    } else if (part.includes("aux(0)") || part.includes("0 faces")) {
                        ctx.fillStyle = "#888";
                    } else {
                        ctx.fillStyle = "#8f8";
                    }
                    ctx.fillText(part, 10, y);
                    y -= lineH;
                }
                ctx.restore();
            }
            return r;
        };
    },
});
