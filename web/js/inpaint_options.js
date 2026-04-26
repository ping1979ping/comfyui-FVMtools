// InpaintOptions UI: collapse per-reference (ref_index / mask_type / rounds /
// detail_daemon) from four vertical widgets into a single horizontal Canvas
// row per slot. Native Python widgets stay registered (so execute() and the
// workflow JSON keep working) but get hidden — the row widget mirrors and
// writes back their values.

import { app } from "../../../scripts/app.js";
import {
    isLowQuality,
    fitString,
    drawRoundedRectangle,
    drawTogglePart,
    drawNumberWidgetPart,
    FvmBaseWidget,
    createSeparator,
} from "./_widgets_common.js";


// ─── Column layout (shared between header and row) ────────────────────────

const MARGIN = 10;
const INNER = 4;
const STEPPER_W = drawNumberWidgetPart.WIDTH_TOTAL;   // 56

function computeColumns(node, height) {
    // Right → left so we can measure once and place from the right edge.
    const toggleBgW = height * 1.5;
    const toggleX = node.size[0] - MARGIN - INNER - toggleBgW;
    const roundsRightX = toggleX - INNER;
    const roundsLeftX = roundsRightX - STEPPER_W;
    const refRightX = MARGIN + INNER + STEPPER_W;
    const refLeftX = MARGIN + INNER;
    const comboLeftX = refRightX + INNER * 2;
    const comboRightX = roundsLeftX - INNER * 2;
    return {
        refLeftX, refRightX,
        comboLeftX, comboWidth: Math.max(40, comboRightX - comboLeftX),
        roundsLeftX, roundsRightX,
        toggleX, toggleBgW,
    };
}


// ─── Header widget (column abbreviations) ─────────────────────────────────

function createHeaderWidget() {
    return {
        name: "_inpaint_options_header",
        type: "custom",
        value: "header",
        options: { serialize: false },
        computeSize: () => [0, 16],
        draw(ctx, node, width, posY) {
            if (isLowQuality()) return;
            const col = computeColumns(node, LiteGraph.NODE_WIDGET_HEIGHT);
            ctx.save();
            ctx.font = "bold 10px Arial";
            ctx.fillStyle = "#999";
            ctx.textBaseline = "top";

            // Pick label or abbrev based on available width.
            const fits = (text, width) => ctx.measureText(text).width <= width - 4;
            const refLabel  = fits("Ref",  col.refRightX - col.refLeftX)        ? "Ref"  : "R";
            const typeLabel = fits("Type", col.comboWidth)                       ? "Type" : "T";
            const rndLabel  = fits("Rnd",  STEPPER_W)                            ? "Rnd"  : "#";
            const ddLabel   = fits("DD",   col.toggleBgW)                        ? "DD"   : "D";

            ctx.textAlign = "left";
            ctx.fillText(refLabel,  col.refLeftX, posY + 2);
            ctx.fillText(typeLabel, col.comboLeftX, posY + 2);

            ctx.textAlign = "center";
            ctx.fillText(rndLabel, col.roundsLeftX + STEPPER_W / 2, posY + 2);
            ctx.fillText(ddLabel,  col.toggleX + col.toggleBgW / 2, posY + 2);

            ctx.restore();
        },
    };
}


// ─── Row widget ────────────────────────────────────────────────────────────

class InpaintRowWidget extends FvmBaseWidget {
    /**
     * @param name              widget name (unique inside node)
     * @param refIndexWidget    hidden native INT widget for ref_index (1-10), or null for "generic"
     * @param maskWidget        hidden native COMBO widget for mask_type
     * @param roundsWidget      hidden native INT widget for rounds
     * @param daemonWidget      hidden native BOOLEAN widget for detail_daemon
     * @param fixedLabel        label string used when refIndexWidget is null (e.g. "Gen")
     */
    constructor(name, refIndexWidget, maskWidget, roundsWidget, daemonWidget, fixedLabel = null) {
        super(name);
        this.type = "custom";
        this.refIndexWidget = refIndexWidget;
        this.maskWidget = maskWidget;
        this.roundsWidget = roundsWidget;
        this.daemonWidget = daemonWidget;
        this.fixedLabel = fixedLabel;
        this.options = { serialize: false }; // values live on the native widgets
        this.hitAreas = {
            refDec:    { bounds: [0, 0], onClick: this.onRefDecClick },
            refVal:    { bounds: [0, 0], onClick: this.onRefValClick },
            refInc:    { bounds: [0, 0], onClick: this.onRefIncClick },
            mask:      { bounds: [0, 0], onClick: this.onMaskClick },
            roundsDec: { bounds: [0, 0], onClick: this.onRoundsDecClick },
            roundsVal: { bounds: [0, 0], onClick: this.onRoundsValClick },
            roundsInc: { bounds: [0, 0], onClick: this.onRoundsIncClick },
            daemon:    { bounds: [0, 0], onDown:  this.onDaemonDown },
        };
    }

    computeSize() {
        return [0, LiteGraph.NODE_WIDGET_HEIGHT];
    }

    serializeValue() { return undefined; }

    draw(ctx, node, w, posY, height) {
        const midY = posY + height * 0.5;
        const lowQ = isLowQuality();
        const col = computeColumns(node, height);

        // Background pill.
        drawRoundedRectangle(ctx, {
            pos: [MARGIN, posY],
            size: [node.size[0] - MARGIN * 2, height],
        });

        ctx.save();
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        ctx.textBaseline = "middle";
        ctx.font = "11px Arial";

        // ── Left: ref-index stepper (or fixed label for generic) ──
        if (this.refIndexWidget) {
            const refValue = Number(this.refIndexWidget.value ?? 1);
            const [decB, valB, incB] = drawNumberWidgetPart(ctx, {
                posX: col.refRightX,
                posY,
                height,
                value: refValue,
                direction: -1,
                formatValue: (v) => String(Math.round(v)),
            });
            this.hitAreas.refDec.bounds = decB;
            this.hitAreas.refVal.bounds = valB;
            this.hitAreas.refInc.bounds = incB;
        } else if (!lowQ) {
            ctx.textAlign = "left";
            ctx.globalAlpha = app.canvas.editor_alpha * 0.7;
            ctx.fillText(this.fixedLabel || "?", col.refLeftX, midY);
            ctx.globalAlpha = app.canvas.editor_alpha;
            // Disable hit areas when there's no widget.
            this.hitAreas.refDec.bounds = [-1, 0];
            this.hitAreas.refVal.bounds = [-1, 0];
            this.hitAreas.refInc.bounds = [-1, 0];
        }

        // ── Mask combo ──
        const maskValue = String(this.maskWidget?.value ?? "?");
        if (!lowQ) {
            ctx.textAlign = "left";
            const display = fitString(ctx, maskValue + " ▾", col.comboWidth);
            ctx.fillText(display, col.comboLeftX, midY);
            // Underline hint.
            ctx.save();
            ctx.strokeStyle = "#666";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(col.comboLeftX, posY + height - 3);
            ctx.lineTo(col.comboLeftX + col.comboWidth, posY + height - 3);
            ctx.stroke();
            ctx.restore();
        }
        this.hitAreas.mask.bounds = [col.comboLeftX, col.comboWidth];

        // ── Rounds stepper ──
        const roundsValue = Number(this.roundsWidget?.value ?? 1);
        const [rDec, rVal, rInc] = drawNumberWidgetPart(ctx, {
            posX: col.roundsRightX,
            posY,
            height,
            value: roundsValue,
            direction: -1,
            formatValue: (v) => String(Math.round(v)),
        });
        this.hitAreas.roundsDec.bounds = rDec;
        this.hitAreas.roundsVal.bounds = rVal;
        this.hitAreas.roundsInc.bounds = rInc;

        // ── Detail-daemon toggle ──
        this.hitAreas.daemon.bounds = drawTogglePart(ctx, {
            posX: col.toggleX,
            posY,
            height,
            value: !!this.daemonWidget?.value,
        });

        ctx.restore();
    }

    // ── Hit-area handlers ──

    onRefDecClick(event, pos, node) { this._stepRef(-1, node); }
    onRefIncClick(event, pos, node) { this._stepRef(+1, node); }

    onRefValClick(event, pos, node) {
        if (!this.refIndexWidget) return;
        const current = Number(this.refIndexWidget.value ?? 1);
        app.canvas.prompt("Reference (1-10)", current, (v) => {
            const n = Math.max(1, Math.min(10, parseInt(v, 10) || 1));
            this._setRef(n, node);
        }, event);
    }

    onMaskClick(event, pos, node) {
        const items = this.maskWidget?.options?.values
            || ["face", "head", "body", "hair", "facial_skin", "eyes", "mouth", "neck", "accessories", "aux"];
        new LiteGraph.ContextMenu(items, {
            event,
            scale: Math.max(1, app.canvas.ds?.scale ?? 1),
            className: "dark",
            callback: (selected) => {
                if (typeof selected !== "string") return;
                this._setMask(selected, node);
            },
        });
        this.cancelMouseDown();
    }

    onRoundsDecClick(event, pos, node) { this._stepRounds(-1, node); }
    onRoundsIncClick(event, pos, node) { this._stepRounds(+1, node); }

    onRoundsValClick(event, pos, node) {
        const current = Number(this.roundsWidget?.value ?? 1);
        app.canvas.prompt("Rounds (1-10)", current, (v) => {
            const n = Math.max(1, Math.min(10, parseInt(v, 10) || 1));
            this._setRounds(n, node);
        }, event);
    }

    onDaemonDown(event, pos, node) {
        this._setDaemon(!this.daemonWidget?.value, node);
        this.cancelMouseDown();
        return true;
    }

    // ── Native-widget write helpers ──

    _stepRef(delta, node) {
        if (!this.refIndexWidget) return;
        const current = Number(this.refIndexWidget.value ?? 1);
        this._setRef(Math.max(1, Math.min(10, current + delta)), node);
    }

    _setRef(value, node) {
        if (!this.refIndexWidget) return;
        this.refIndexWidget.value = value;
        this.refIndexWidget.callback?.(value);
        node.setDirtyCanvas(true, true);
    }

    _setMask(value, node) {
        if (!this.maskWidget) return;
        this.maskWidget.value = value;
        this.maskWidget.callback?.(value);
        node.setDirtyCanvas(true, true);
    }

    _stepRounds(delta, node) {
        const current = Number(this.roundsWidget?.value ?? 1);
        this._setRounds(Math.max(1, Math.min(10, current + delta)), node);
    }

    _setRounds(value, node) {
        if (!this.roundsWidget) return;
        this.roundsWidget.value = value;
        this.roundsWidget.callback?.(value);
        node.setDirtyCanvas(true, true);
    }

    _setDaemon(value, node) {
        if (!this.daemonWidget) return;
        this.daemonWidget.value = !!value;
        this.daemonWidget.callback?.(!!value);
        node.setDirtyCanvas(true, true);
    }
}


// ─── Helper: hide a native widget completely (no draw, no height) ─────────

function hideWidget(widget) {
    if (!widget) return;
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};                      // suppress native rendering (toggle/combo/int)
    widget.type = "hidden";
    widget.hidden = true;
    if (Array.isArray(widget.linkedWidgets)) {
        widget.linkedWidgets.forEach(hideWidget);
    }
}


// ─── Node registration ────────────────────────────────────────────────────

app.registerExtension({
    name: "FVMTools.InpaintOptions",

    async beforeRegisterNodeDef(nodeType, nodeData, appRef) {
        if (nodeData.name !== "InpaintOptions") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            try {
                const findW = (name) => this.widgets.find(w => w.name === name);

                // Build the 5 reference rows.
                const refRows = [];
                for (let i = 1; i <= 5; i++) {
                    const refIndexW = findW(`reference_${i}_ref_index`);
                    const maskW     = findW(`reference_${i}_mask_type`);
                    const roundsW   = findW(`reference_${i}_rounds`);
                    const daemonW   = findW(`reference_${i}_detail_daemon`);
                    if (!maskW || !roundsW || !daemonW) continue;

                    hideWidget(refIndexW);
                    hideWidget(maskW);
                    hideWidget(roundsW);
                    hideWidget(daemonW);

                    refRows.push({
                        widget: new InpaintRowWidget(
                            `reference_${i}_row`,
                            refIndexW, maskW, roundsW, daemonW,
                        ),
                        anchor: refIndexW || maskW,
                    });
                }

                // Generic row (no ref_index — just a fixed "Gen" label).
                const genMaskW   = findW("generic_mask_type");
                const genRoundsW = findW("generic_rounds");
                const genDaemonW = findW("generic_detail_daemon");
                let genericRow = null;
                if (genMaskW && genRoundsW && genDaemonW) {
                    hideWidget(genMaskW);
                    hideWidget(genRoundsW);
                    hideWidget(genDaemonW);
                    genericRow = {
                        widget: new InpaintRowWidget(
                            "generic_row",
                            null, genMaskW, genRoundsW, genDaemonW, "Gen",
                        ),
                        anchor: genMaskW,
                    };
                }

                // Insert references separator + header + ref rows in front of
                // the first hidden anchor. Going backwards keeps the index
                // stable while we splice multiple items at the same position.
                if (refRows.length) {
                    const firstAnchorIdx = this.widgets.indexOf(refRows[0].anchor);
                    for (let i = refRows.length - 1; i >= 0; i--) {
                        this.widgets.splice(firstAnchorIdx, 0, refRows[i].widget);
                    }
                    this.widgets.splice(firstAnchorIdx, 0, createHeaderWidget());
                    this.widgets.splice(firstAnchorIdx, 0, createSeparator("── References ──"));
                }

                // Generic row goes in directly before its anchor — no separator.
                if (genericRow) {
                    const anchorIdx = this.widgets.indexOf(genericRow.anchor);
                    if (anchorIdx >= 0) {
                        this.widgets.splice(anchorIdx, 0, genericRow.widget);
                    }
                }

                // Recompute node size so the new layout fits.
                const computed = this.computeSize();
                this.size[0] = Math.max(this.size[0], computed[0], 360);
                this.size[1] = Math.max(this.size[1], computed[1]);
            } catch (e) {
                console.error("[FVMTools] InpaintOptions row widget setup error:", e);
            }

            return result;
        };
    },
});
