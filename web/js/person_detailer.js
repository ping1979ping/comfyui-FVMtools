import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "FVMTools.PersonDetailer",

    async nodeCreated(node) {
        if (node.comfyClass !== "PersonDetailer") return;

        // ── Hide/Show helpers ──
        function hideWidget(widget) {
            if (!widget || widget._fvm_hidden) return;
            widget._fvm_hidden = true;
            widget._fvm_origType = widget.type;
            widget._fvm_origComputeSize = widget.computeSize;
            widget.type = "converted-widget";
            widget.computeSize = () => [0, -4];
            if (widget.inputEl) widget.inputEl.style.display = "none";
        }

        function showWidget(widget) {
            if (!widget || !widget._fvm_hidden) return;
            widget._fvm_hidden = false;
            widget.type = widget._fvm_origType;
            widget.computeSize = widget._fvm_origComputeSize;
            if (widget.inputEl) widget.inputEl.style.display = "";
        }

        // ── Add static separators for non-slot sections (insert backwards) ──
        const staticSeps = [
            { label: "── Inpaint ──", before: "mask_blend_pixels" },
            { label: "── Detail Daemon ──", before: "detail_daemon_enabled" },
            { label: "── Sampler ──", before: "seed" },
        ];
        for (const sep of staticSeps) {
            const idx = node.widgets.findIndex(w => w.name === sep.before);
            if (idx < 0) continue;
            node.widgets.splice(idx, 0, {
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

        // ── Slot config: replace enabled widget draw with separator+toggle ──
        const slotDefs = [
            { prefix: "reference_1_", label: "Reference 1" },
            { prefix: "reference_2_", label: "Reference 2" },
            { prefix: "reference_3_", label: "Reference 3" },
            { prefix: "reference_4_", label: "Reference 4" },
            { prefix: "reference_5_", label: "Reference 5" },
            { prefix: "generic_", label: "Generic (Unmatched)" },
        ];

        for (const def of slotDefs) {
            const enabledW = node.widgets.find(w => w.name === def.prefix + "enabled");
            if (!enabledW) continue;

            // Override the enabled widget to render as a separator line with toggle
            enabledW._fvm_label = def.label;
            enabledW._fvm_origDraw = enabledW.draw;
            enabledW.type = "custom";
            enabledW.computeSize = () => [0, 28];

            enabledW.draw = function(ctx, nodeRef, width, posY, height) {
                const enabled = this.value;
                const cy = posY + 16;

                ctx.save();

                // Toggle circle
                const toggleX = 22;
                const toggleR = 6;
                ctx.beginPath();
                ctx.arc(toggleX, cy - 2, toggleR, 0, Math.PI * 2);
                if (enabled) {
                    ctx.fillStyle = "#4CAF50";
                    ctx.fill();
                } else {
                    ctx.strokeStyle = "#666";
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }

                // Label text
                ctx.font = "bold 11px Arial";
                ctx.fillStyle = enabled ? "#ccc" : "#666";
                ctx.textAlign = "left";
                ctx.fillText("── " + this._fvm_label + " ──", toggleX + toggleR + 8, cy);

                // Right-side line
                const labelText = "── " + this._fvm_label + " ──";
                const textEnd = toggleX + toggleR + 8 + ctx.measureText(labelText).width + 8;
                if (textEnd < width - 15) {
                    ctx.strokeStyle = "#555";
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(textEnd, cy - 4);
                    ctx.lineTo(width - 15, cy - 4);
                    ctx.stroke();
                }

                ctx.restore();
            };

            // Handle click on the toggle area
            enabledW.mouse = function(event, pos, nodeRef) {
                if (event.type === "pointerdown") {
                    // Toggle on click anywhere on the widget
                    this.value = !this.value;
                    updateAllSlots();
                    return true;
                }
                return false;
            };
        }

        // ── Slot visibility ──
        function updateAllSlots() {
            for (const def of slotDefs) {
                const enabledW = node.widgets.find(w => w.name === def.prefix + "enabled");
                const loraW = node.widgets.find(w => w.name === def.prefix + "lora");
                const strengthW = node.widgets.find(w => w.name === def.prefix + "lora_strength");
                const promptW = node.widgets.find(w => w.name === def.prefix + "prompt");
                if (!enabledW) continue;

                if (enabledW.value) {
                    showWidget(loraW);
                    showWidget(strengthW);
                    showWidget(promptW);
                } else {
                    hideWidget(loraW);
                    hideWidget(strengthW);
                    hideWidget(promptW);
                }
            }

            const sz = node.computeSize();
            node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
            node.graph?.setDirtyCanvas(true, true);
        }

        // Initial state
        requestAnimationFrame(() => {
            updateAllSlots();
            requestAnimationFrame(updateAllSlots);
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "PersonDetailer") return;

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
