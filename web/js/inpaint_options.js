// InpaintOptions UI: collapse per-reference (mask_type / rounds / detail_daemon)
// from three vertical widgets into a single horizontal Canvas row per ref.
// Native Python widgets stay registered (so execute() and workflow JSON keep
// working), but get hidden — the row widget mirrors and writes back their values.

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


// ─── Row widget ────────────────────────────────────────────────────────────

class InpaintRowWidget extends FvmBaseWidget {
    /**
     * @param name              widget name (unique inside node)
     * @param label             human label drawn at the left ("Ref 1", "Generic")
     * @param maskWidget        hidden native COMBO widget for mask_type
     * @param roundsWidget      hidden native INT widget for rounds
     * @param daemonWidget      hidden native BOOLEAN widget for detail_daemon
     */
    constructor(name, label, maskWidget, roundsWidget, daemonWidget) {
        super(name);
        this.type = "custom";
        this.label = label;
        this.maskWidget = maskWidget;
        this.roundsWidget = roundsWidget;
        this.daemonWidget = daemonWidget;
        this.options = { serialize: false }; // values live on the native widgets
        this.hitAreas = {
            mask:       { bounds: [0, 0], onClick: this.onMaskClick },
            roundsDec:  { bounds: [0, 0], onClick: this.onRoundsDecClick },
            roundsVal:  { bounds: [0, 0], onClick: this.onRoundsValClick },
            roundsInc:  { bounds: [0, 0], onClick: this.onRoundsIncClick },
            daemon:     { bounds: [0, 0], onDown:  this.onDaemonDown },
        };
    }

    computeSize() {
        return [0, LiteGraph.NODE_WIDGET_HEIGHT];
    }

    serializeValue() { return undefined; }

    draw(ctx, node, w, posY, height) {
        const margin = 10, innerMargin = 4;
        const midY = posY + height * 0.5;

        // Background pill across the row.
        drawRoundedRectangle(ctx, {
            pos: [margin, posY],
            size: [node.size[0] - margin * 2, height],
        });

        const lowQ = isLowQuality();

        let posX = margin + innerMargin;

        ctx.save();
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        ctx.textBaseline = "middle";
        ctx.font = "11px Arial";

        // Left label ("Ref 1", "Generic"...).
        const labelWidth = lowQ ? 0 : 56;
        if (!lowQ) {
            ctx.textAlign = "left";
            ctx.globalAlpha = app.canvas.editor_alpha * 0.7;
            ctx.fillText(this.label, posX, midY);
            ctx.globalAlpha = app.canvas.editor_alpha;
        }
        posX += labelWidth + innerMargin;

        // Right-anchored: toggle (rightmost), rounds stepper, then mask combo flex-fills.
        const togglePadding = innerMargin;
        const toggleHeight = height;
        const toggleBgWidth = toggleHeight * 1.5;
        const toggleX = node.size[0] - margin - innerMargin - toggleBgWidth;

        const roundsTotal = drawNumberWidgetPart.WIDTH_TOTAL;
        const roundsRightX = toggleX - togglePadding;          // right edge of stepper
        const roundsLeftX = roundsRightX - roundsTotal;        // left edge of stepper

        // Mask combo area = remaining space between label and rounds.
        const comboX = posX;
        const comboRight = roundsLeftX - innerMargin * 2;
        const comboWidth = Math.max(40, comboRight - comboX);

        // Mask combo display (clickable area shows current value).
        const maskValue = String(this.maskWidget?.value ?? "?");
        if (!lowQ) {
            ctx.textAlign = "left";
            const triangle = " ▾";
            const display = fitString(ctx, maskValue + triangle, comboWidth);
            ctx.fillText(display, comboX, midY);
        }
        // Light underline to hint at clickability.
        if (!lowQ) {
            ctx.save();
            ctx.strokeStyle = "#666";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(comboX, posY + height - 3);
            ctx.lineTo(comboX + comboWidth, posY + height - 3);
            ctx.stroke();
            ctx.restore();
        }
        this.hitAreas.mask.bounds = [comboX, comboWidth];

        // Rounds stepper (use integer formatter, not the default 2-decimal float).
        const roundsValue = Number(this.roundsWidget?.value ?? 1);
        const [decB, valB, incB] = drawNumberWidgetPart(ctx, {
            posX: roundsRightX,
            posY,
            height,
            value: roundsValue,
            direction: -1,
            formatValue: (v) => String(Math.round(v)),
        });
        this.hitAreas.roundsDec.bounds = decB;
        this.hitAreas.roundsVal.bounds = valB;
        this.hitAreas.roundsInc.bounds = incB;

        // Detail-daemon toggle.
        this.hitAreas.daemon.bounds = drawTogglePart(ctx, {
            posX: toggleX,
            posY,
            height,
            value: !!this.daemonWidget?.value,
        });

        ctx.restore();
    }

    // ── Hit-area handlers ──────────────────────────────────────────────────

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
        app.canvas.prompt("Rounds", current, (v) => {
            const n = Math.max(1, Math.min(10, parseInt(v, 10) || 1));
            this._setRounds(n, node);
        }, event);
    }

    onDaemonDown(event, pos, node) {
        this._setDaemon(!this.daemonWidget?.value, node);
        this.cancelMouseDown();
        return true;
    }

    // ── Native-widget write helpers ────────────────────────────────────────

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


// ─── Helpers: hide a native widget without removing it from the workflow ───

function hideWidget(widget) {
    if (!widget) return;
    widget.computeSize = () => [0, -4];
    widget.type = "hidden";
    if (Array.isArray(widget.linkedWidgets)) {
        widget.linkedWidgets.forEach(hideWidget);
    }
}


// ─── Node registration ────────────────────────────────────────────────────

const REFS = ["reference_1", "reference_2", "reference_3", "reference_4", "reference_5", "generic"];

const REF_LABELS = {
    reference_1: "Ref 1",
    reference_2: "Ref 2",
    reference_3: "Ref 3",
    reference_4: "Ref 4",
    reference_5: "Ref 5",
    generic:     "Generic",
};

app.registerExtension({
    name: "FVMTools.InpaintOptions",

    async beforeRegisterNodeDef(nodeType, nodeData, appRef) {
        if (nodeData.name !== "InpaintOptions") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            try {
                // Build one row widget per slot — using references to the
                // already-registered native widgets so values still serialize
                // and execute() reads them as before.
                const rows = [];
                for (const prefix of REFS) {
                    const maskW   = this.widgets.find(w => w.name === `${prefix}_mask_type`);
                    const roundsW = this.widgets.find(w => w.name === `${prefix}_rounds`);
                    const daemonW = this.widgets.find(w => w.name === `${prefix}_detail_daemon`);
                    if (!maskW || !roundsW || !daemonW) continue;

                    hideWidget(maskW);
                    hideWidget(roundsW);
                    hideWidget(daemonW);

                    rows.push({
                        prefix,
                        widget: new InpaintRowWidget(
                            `${prefix}_row`,
                            REF_LABELS[prefix] || prefix,
                            maskW, roundsW, daemonW,
                        ),
                        anchor: maskW,   // insert before the first hidden native
                    });
                }

                if (!rows.length) return result;

                // Insert references separator + ref rows in front of the first
                // ref's mask_type widget. Going backwards keeps anchor indices stable.
                const refRows = rows.filter(r => r.prefix !== "generic");
                const genericRow = rows.find(r => r.prefix === "generic");

                if (refRows.length) {
                    const firstAnchor = refRows[0].anchor;
                    const firstAnchorIdx = this.widgets.indexOf(firstAnchor);

                    // Insert ref rows in order, then a leading separator above them.
                    for (let i = refRows.length - 1; i >= 0; i--) {
                        this.widgets.splice(firstAnchorIdx, 0, refRows[i].widget);
                    }
                    this.widgets.splice(firstAnchorIdx, 0, createSeparator("── References ──"));
                }

                if (genericRow) {
                    const anchorIdx = this.widgets.indexOf(genericRow.anchor);
                    if (anchorIdx >= 0) {
                        this.widgets.splice(anchorIdx, 0, genericRow.widget);
                        this.widgets.splice(anchorIdx, 0, createSeparator("── Generic ──"));
                    }
                }

                // Recompute size so the new layout fits.
                const computed = this.computeSize();
                this.size[0] = Math.max(this.size[0], computed[0], 320);
                this.size[1] = Math.max(this.size[1], computed[1]);
            } catch (e) {
                console.error("[FVMTools] InpaintOptions row widget setup error:", e);
            }

            return result;
        };
    },
});
