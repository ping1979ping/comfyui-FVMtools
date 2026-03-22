import { app } from "../../../scripts/app.js";

// Static section separators only (no slot separators)
const STATIC_SEPS = [
    { label: "── References ──", before: "reference_1_enabled" },
    { label: "── Inpaint ──", before: "mask_blend_pixels" },
    { label: "── Detail Daemon ──", before: "detail_daemon_enabled" },
    { label: "── Sampler ──", before: "seed" },
];

app.registerExtension({
    name: "FVMTools.PersonDetailer",

    async beforeRegisterNodeDef(nodeType, nodeData, appRef) {
        if (nodeData.name !== "PersonDetailer") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            try {
                // Add static separators (backwards to avoid index shift)
                for (let i = STATIC_SEPS.length - 1; i >= 0; i--) {
                    const sep = STATIC_SEPS[i];
                    const idx = this.widgets.findIndex(w => w.name === sep.before);
                    if (idx < 0) continue;
                    this.widgets.splice(idx, 0, {
                        name: "_sep_" + sep.before,
                        type: "custom",
                        value: sep.label,
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
                    });
                }
            } catch (e) {
                console.error("[FVMTools] PersonDetailer separator setup error:", e);
            }

            return result;
        };

        // Execution info
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            if (message?.text?.length > 0) {
                this._pdInfo = message.text[0];
                this.setDirtyCanvas(true);
            }
            return r;
        };

        const onDrawFG = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const r = onDrawFG ? onDrawFG.apply(this, arguments) : undefined;
            if (this._pdInfo) {
                ctx.save();
                ctx.font = "11px Arial";
                ctx.fillStyle = "#8f8";
                ctx.textAlign = "left";
                ctx.fillText(this._pdInfo, 10, this.size[1] - 6);
                ctx.restore();
            }
            return r;
        };
    },
});
