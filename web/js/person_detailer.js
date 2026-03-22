import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "FVMTools.PersonDetailer",

    async nodeCreated(node) {
        if (node.comfyClass !== "PersonDetailer") return;

        // ── Hide/Show helpers ──
        // For canvas widgets: converted-widget pattern
        // For DOM widgets (textarea): also hide the DOM element
        function hideWidget(widget) {
            if (!widget || widget._fvm_hidden) return;
            widget._fvm_hidden = true;
            widget._fvm_origType = widget.type;
            widget._fvm_origComputeSize = widget.computeSize;
            widget._fvm_origSerialize = widget.serializeValue;
            widget.type = "converted-widget";
            widget.computeSize = () => [0, -4];
            widget.serializeValue = () => widget.value;
            if (widget.inputEl) widget.inputEl.style.display = "none";
        }

        function showWidget(widget) {
            if (!widget || !widget._fvm_hidden) return;
            widget._fvm_hidden = false;
            widget.type = widget._fvm_origType;
            widget.computeSize = widget._fvm_origComputeSize;
            widget.serializeValue = widget._fvm_origSerialize;
            if (widget.inputEl) widget.inputEl.style.display = "";
        }

        // ── Merge enabled + lora onto one line ──
        // Hide the original enabled widget and draw a toggle on the lora widget
        const slotConfigs = [
            "reference_1_", "reference_2_", "reference_3_",
            "reference_4_", "reference_5_", "generic_"
        ];

        for (const prefix of slotConfigs) {
            const enabledWidget = node.widgets.find(w => w.name === prefix + "enabled");
            const loraWidget = node.widgets.find(w => w.name === prefix + "lora");
            const strengthWidget = node.widgets.find(w => w.name === prefix + "lora_strength");
            if (!enabledWidget || !loraWidget) continue;

            // Hide the standalone enabled widget — we'll draw its toggle on the lora row
            hideWidget(enabledWidget);

            // Store reference so lora widget can access it
            loraWidget._fvm_enabledWidget = enabledWidget;

            // Override lora widget draw to add toggle circle
            const origDraw = loraWidget.draw?.bind(loraWidget);
            loraWidget._fvm_origDraw = origDraw;

            loraWidget.draw = function(ctx, nodeRef, width, posY, height) {
                const enabled = this._fvm_enabledWidget.value;

                // Draw toggle circle on the left
                const toggleX = 15;
                const toggleY = posY + height / 2;
                const toggleR = 6;

                ctx.save();
                ctx.beginPath();
                ctx.arc(toggleX, toggleY, toggleR, 0, Math.PI * 2);
                if (enabled) {
                    ctx.fillStyle = "#4CAF50";
                    ctx.fill();
                } else {
                    ctx.strokeStyle = "#666";
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                    // Dim the whole row when disabled
                    ctx.globalAlpha = 0.4;
                }
                ctx.restore();

                // Store toggle hit area for click detection
                this._fvm_toggleBounds = { x: toggleX - toggleR - 4, y: posY, w: toggleR * 2 + 8, h: height };
            };

            // Handle clicks on the toggle area
            const origMouse = loraWidget.mouse?.bind(loraWidget);
            loraWidget.mouse = function(event, pos, nodeRef) {
                if (this._fvm_toggleBounds && event.type === "pointerdown") {
                    const [mx, my] = pos;
                    const b = this._fvm_toggleBounds;
                    if (mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h) {
                        this._fvm_enabledWidget.value = !this._fvm_enabledWidget.value;
                        updateAllSlots();
                        return true;
                    }
                }
                if (origMouse) return origMouse(event, pos, nodeRef);
                return false;
            };
        }

        // ── Add section separators (insert backwards to avoid index shift) ──
        const separators = [
            { label: "── Generic (Unmatched) ──", before: "generic_lora" },
            { label: "── Reference 5 ──", before: "reference_5_lora" },
            { label: "── Reference 4 ──", before: "reference_4_lora" },
            { label: "── Reference 3 ──", before: "reference_3_lora" },
            { label: "── Reference 2 ──", before: "reference_2_lora" },
            { label: "── Reference 1 ──", before: "reference_1_lora" },
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
                draw(ctx, nodeRef, width, posY) {
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
            node.widgets.splice(idx, 0, sepWidget);
        }

        // ── Slot visibility toggle ──
        function updateAllSlots() {
            for (const prefix of slotConfigs) {
                const enabledWidget = node.widgets.find(w => w.name === prefix + "enabled");
                const promptWidget = node.widgets.find(w => w.name === prefix + "prompt");
                const strengthWidget = node.widgets.find(w => w.name === prefix + "lora_strength");
                const loraWidget = node.widgets.find(w => w.name === prefix + "lora");
                if (!enabledWidget) continue;

                // Prompt and strength only visible when enabled
                if (enabledWidget.value) {
                    showWidget(promptWidget);
                    showWidget(strengthWidget);
                } else {
                    hideWidget(promptWidget);
                    hideWidget(strengthWidget);
                }

                // Force lora widget to redraw (for toggle state change)
                if (loraWidget) loraWidget._fvm_needsRedraw = true;
            }

            const sz = node.computeSize();
            node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
            node.graph?.setDirtyCanvas(true, true);
        }

        // No separate toggle callbacks needed — toggle is on the lora widget now
        // But we still need initial state
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
