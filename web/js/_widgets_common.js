// Shared canvas-widget primitives for FVMtools nodes.
// Originally lived inside person_detailer_power.js — extracted here so
// inpaint_options.js (and future row-widget nodes) can reuse them without
// duplicating ~250 lines of canvas drawing + base-widget code.

import { app } from "../../../scripts/app.js";


// ─── Canvas drawing utilities (ported from rgthree utils_canvas.js) ─────────

export function isLowQuality() {
    return (app.canvas.ds?.scale || 1) <= 0.5;
}

export function measureText(ctx, str) {
    return ctx.measureText(str).width;
}

export function fitString(ctx, str, maxWidth) {
    let width = measureText(ctx, str);
    const ellipsis = "…";
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

export function drawInfoIcon(ctx, x, y, size, treatment = "GRAYED") {
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(x, y, size, size, [size * 0.15]);
    if (treatment === "GRAYED") {
        ctx.fillStyle = "#555";
        ctx.strokeStyle = "#888";
    } else {
        ctx.fillStyle = "#2f82ec";
        ctx.strokeStyle = "#2f82ec";
    }
    if (treatment === "FILLED") {
        ctx.fill();
    } else {
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }
    const iColor = treatment === "FILLED" ? "#FFF" : treatment === "OUTLINED" ? "#2f82ec" : "#999";
    ctx.fillStyle = iColor;
    ctx.font = `bold ${Math.round(size * 0.65)}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("i", x + size / 2, y + size / 2 + 0.5);
    ctx.restore();
}

export function drawRoundedRectangle(ctx, options) {
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

export function drawTogglePart(ctx, options) {
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

export function drawNumberWidgetPart(ctx, options) {
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
    const display = options.formatValue
        ? options.formatValue(value)
        : (typeof value === "number" ? value.toFixed(2) : String(value));
    ctx.fillText(fitString(ctx, display, numberWidth), posX + numberWidth / 2, midY);
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

export function drawWidgetButton(ctx, options, text, pressed) {
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


// ─── Base widget class with hit-area routing (port of rgthree utils_widgets.js) ─

export class FvmBaseWidget {
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


// ─── Section separator widget (small text divider in node widget list) ─────

export function createSeparator(label) {
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
