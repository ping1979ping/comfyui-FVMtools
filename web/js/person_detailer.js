import { app } from "../../../scripts/app.js";

// Section separator names and the widget that starts each section
const SECTIONS = [
    { label: "── Sampler ──", before: "seed" },
    { label: "── Detail Daemon ──", before: "detail_daemon_enabled" },
    { label: "── Inpaint ──", before: "mask_blend_pixels" },
    { label: "── Reference 1 ──", before: "reference_1_enabled" },
    { label: "── Reference 2 ──", before: "reference_2_enabled" },
    { label: "── Reference 3 ──", before: "reference_3_enabled" },
    { label: "── Reference 4 ──", before: "reference_4_enabled" },
    { label: "── Reference 5 ──", before: "reference_5_enabled" },
    { label: "── Generic (Unmatched) ──", before: "generic_enabled" },
];

app.registerExtension({
    name: "FVMTools.PersonDetailer",

    async nodeCreated(node) {
        if (node.comfyClass !== "PersonDetailer") return;

        // --- Add section separator widgets ---
        // We insert separator widgets before the target widgets.
        // ComfyUI renders widgets in array order, so we insert at the right index.
        for (const section of SECTIONS) {
            const targetIdx = node.widgets.findIndex(w => w.name === section.before);
            if (targetIdx < 0) continue;

            const separator = {
                name: "_sep_" + section.before,
                type: "custom",
                value: section.label,
                hidden: false,
                options: { serialize: false },
                computeSize: () => [0, 20],
                draw: function(ctx, node, width, posY, h) {
                    ctx.save();
                    ctx.font = "bold 11px Arial";
                    ctx.fillStyle = "#888";
                    ctx.textAlign = "center";
                    ctx.fillText(this.value, width / 2, posY + 14);
                    // Draw line
                    ctx.strokeStyle = "#555";
                    ctx.lineWidth = 1;
                    const textW = ctx.measureText(this.value).width;
                    const cx = width / 2;
                    if (cx - textW/2 - 10 > 15) {
                        ctx.beginPath();
                        ctx.moveTo(15, posY + 10);
                        ctx.lineTo(cx - textW/2 - 8, posY + 10);
                        ctx.stroke();
                    }
                    if (cx + textW/2 + 10 < width - 15) {
                        ctx.beginPath();
                        ctx.moveTo(cx + textW/2 + 8, posY + 10);
                        ctx.lineTo(width - 15, posY + 10);
                        ctx.stroke();
                    }
                    ctx.restore();
                },
            };
            node.widgets.splice(targetIdx, 0, separator);
        }

        // --- Helper: hide/show a widget ---
        function setWidgetVisible(widget, visible) {
            if (!widget) return;
            if (!visible) {
                widget.hidden = true;
                if (!widget._origType) widget._origType = widget.type;
                if (!widget._origComputeSize) widget._origComputeSize = widget.computeSize;
                widget.type = "hidden";
                widget.computeSize = () => [0, -4];
            } else {
                widget.hidden = false;
                if (widget._origType) widget.type = widget._origType;
                if (widget._origComputeSize) widget.computeSize = widget._origComputeSize;
            }
        }

        // --- Toggle slot visibility ---
        const slotPrefixes = [
            "reference_1_", "reference_2_", "reference_3_",
            "reference_4_", "reference_5_", "generic_"
        ];

        function updateAllSlots() {
            for (const prefix of slotPrefixes) {
                const toggle = node.widgets.find(w => w.name === prefix + "enabled");
                const lora = node.widgets.find(w => w.name === prefix + "lora");
                const prompt = node.widgets.find(w => w.name === prefix + "prompt");
                if (!toggle) continue;
                setWidgetVisible(lora, toggle.value);
                setWidgetVisible(prompt, toggle.value);
            }
            // Force full size recalculation
            const newH = node.computeSize()[1];
            node.setSize([node.size[0], newH]);
            node.graph?.setDirtyCanvas(true, true);
        }

        // Attach callbacks
        for (const prefix of slotPrefixes) {
            const toggle = node.widgets.find(w => w.name === prefix + "enabled");
            if (!toggle) continue;
            const origCb = toggle.callback;
            toggle.callback = (value) => {
                if (origCb) origCb(value);
                updateAllSlots();
            };
        }

        // Initial state — delay to let ComfyUI finish layout
        setTimeout(updateAllSlots, 100);
        // Second pass for loaded workflows
        setTimeout(updateAllSlots, 500);
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "PersonDetailer") return;

        // Display execution info and preview
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            if (message && message.text && message.text.length > 0) {
                this._pdInfo = message.text[0];
                this.setDirtyCanvas(true);
            }
            // Show preview images inline
            if (message && message.images) {
                this.imgs = message.images.map(img => {
                    const url = `/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${encodeURIComponent(img.subfolder || "")}`;
                    const imgEl = new Image();
                    imgEl.src = url;
                    return imgEl;
                });
                this.setSizeForImage?.();
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
