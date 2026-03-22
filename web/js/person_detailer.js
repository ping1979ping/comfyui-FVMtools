import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "FVMTools.PersonDetailer",

    async nodeCreated(node) {
        if (node.comfyClass !== "PersonDetailer") return;

        // ── Robust hide/show using the "converted-widget" pattern ──
        // This is the pattern used by ComfyUI-Easy-Use and other mature extensions.
        // It fully removes the widget from layout calculations.
        function hideWidget(widget) {
            if (!widget || widget._fvm_hidden) return;
            widget._fvm_hidden = true;
            widget._fvm_origType = widget.type;
            widget._fvm_origComputeSize = widget.computeSize;
            widget._fvm_origSerialize = widget.serializeValue;
            widget.type = "converted-widget";
            widget.computeSize = () => [0, -4];
            widget.serializeValue = () => widget.value; // still serialize the value
        }

        function showWidget(widget) {
            if (!widget || !widget._fvm_hidden) return;
            widget._fvm_hidden = false;
            widget.type = widget._fvm_origType;
            widget.computeSize = widget._fvm_origComputeSize;
            widget.serializeValue = widget._fvm_origSerialize;
        }

        // ── Add separator labels via custom draw widgets ──
        // Insert separators BACKWARDS to avoid index shifting issues.
        const separators = [
            { label: "── Generic (Unmatched) ──", before: "generic_enabled" },
            { label: "── Reference 5 ──", before: "reference_5_enabled" },
            { label: "── Reference 4 ──", before: "reference_4_enabled" },
            { label: "── Reference 3 ──", before: "reference_3_enabled" },
            { label: "── Reference 2 ──", before: "reference_2_enabled" },
            { label: "── Reference 1 ──", before: "reference_1_enabled" },
            { label: "── Inpaint ──", before: "mask_blend_pixels" },
            { label: "── Detail Daemon ──", before: "detail_daemon_enabled" },
            { label: "── Sampler ──", before: "seed" },
        ];

        for (const sep of separators) {
            const idx = node.widgets.findIndex(w => w.name === sep.before);
            if (idx < 0) continue;

            const sepWidget = {
                name: "_sep_" + sep.before,
                type: "custom",
                value: sep.label,
                options: { serialize: false },
                computeSize: () => [0, 24],
                _fvm_separator: true,
                draw(ctx, nodeRef, width, posY) {
                    ctx.save();
                    ctx.font = "bold 11px Arial";
                    ctx.fillStyle = "#999";
                    ctx.textAlign = "center";
                    const cy = posY + 15;
                    ctx.fillText(this.value, width / 2, cy);
                    // Lines on each side
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
            node.widgets.splice(idx, 0, sepWidget);
        }

        // ── Slot toggle logic ──
        const slotConfigs = [
            { prefix: "reference_1_", sepName: "_sep_reference_1_enabled" },
            { prefix: "reference_2_", sepName: "_sep_reference_2_enabled" },
            { prefix: "reference_3_", sepName: "_sep_reference_3_enabled" },
            { prefix: "reference_4_", sepName: "_sep_reference_4_enabled" },
            { prefix: "reference_5_", sepName: "_sep_reference_5_enabled" },
            { prefix: "generic_", sepName: "_sep_generic_enabled" },
        ];

        function updateAllSlots() {
            for (const cfg of slotConfigs) {
                const toggle = node.widgets.find(w => w.name === cfg.prefix + "enabled");
                const lora = node.widgets.find(w => w.name === cfg.prefix + "lora");
                const prompt = node.widgets.find(w => w.name === cfg.prefix + "prompt");
                if (!toggle) continue;

                if (toggle.value) {
                    showWidget(lora);
                    showWidget(prompt);
                } else {
                    hideWidget(lora);
                    hideWidget(prompt);
                }
            }

            // Recalculate node height
            const sz = node.computeSize();
            node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
            node.graph?.setDirtyCanvas(true, true);
        }

        // Attach toggle callbacks
        for (const cfg of slotConfigs) {
            const toggle = node.widgets.find(w => w.name === cfg.prefix + "enabled");
            if (!toggle) continue;
            const origCb = toggle.callback;
            toggle.callback = function(value) {
                if (origCb) origCb.call(this, value);
                updateAllSlots();
            };
        }

        // Initial visibility — use requestAnimationFrame for reliable timing
        requestAnimationFrame(() => {
            updateAllSlots();
            // Second pass after ComfyUI finishes its own layout
            requestAnimationFrame(updateAllSlots);
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "PersonDetailer") return;

        // Display execution info
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
            if (!this._pdInfo) return r;

            ctx.save();
            ctx.font = "11px Arial";
            ctx.fillStyle = "#8f8";
            ctx.textAlign = "left";
            ctx.fillText(this._pdInfo, 10, this.size[1] - 6);
            ctx.restore();
            return r;
        };
    },
});
