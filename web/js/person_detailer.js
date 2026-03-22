import { app } from "../../../scripts/app.js";

// Slot definitions
const SLOT_DEFS = [
    { prefix: "reference_1_", label: "Reference 1" },
    { prefix: "reference_2_", label: "Reference 2" },
    { prefix: "reference_3_", label: "Reference 3" },
    { prefix: "reference_4_", label: "Reference 4" },
    { prefix: "reference_5_", label: "Reference 5" },
    { prefix: "generic_", label: "Generic (Unmatched)" },
];

// Static section separators
const STATIC_SEPS = [
    { label: "── Inpaint ──", before: "mask_blend_pixels" },
    { label: "── Detail Daemon ──", before: "detail_daemon_enabled" },
    { label: "── Sampler ──", before: "seed" },
];

function drawSeparatorLine(ctx, label, width, posY) {
    ctx.save();
    ctx.font = "bold 11px Arial";
    ctx.fillStyle = "#999";
    ctx.textAlign = "center";
    const cy = posY + 15;
    ctx.fillText(label, width / 2, cy);
    const tw = ctx.measureText(label).width;
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
}

// Hide/show helpers
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

function updateSlotVisibility(node) {
    for (const def of SLOT_DEFS) {
        const enabledW = node.widgets.find(w => w.name === def.prefix + "enabled");
        const loraW = node.widgets.find(w => w.name === def.prefix + "lora");
        const strengthW = node.widgets.find(w => w.name === def.prefix + "lora_strength");
        const promptW = node.widgets.find(w => w.name === def.prefix + "prompt");
        if (!enabledW) continue;

        const enabled = enabledW.value;
        if (enabled) {
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

app.registerExtension({
    name: "FVMTools.PersonDetailer",

    async beforeRegisterNodeDef(nodeType, nodeData, appRef) {
        if (nodeData.name !== "PersonDetailer") return;

        // Hook into onNodeCreated to replace widgets AFTER they exist
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;

            // --- Add static separators (backwards to avoid index shift) ---
            for (let i = STATIC_SEPS.length - 1; i >= 0; i--) {
                const sep = STATIC_SEPS[i];
                const idx = node.widgets.findIndex(w => w.name === sep.before);
                if (idx < 0) continue;
                node.widgets.splice(idx, 0, {
                    name: "_sep_" + sep.before,
                    type: "custom",
                    value: sep.label,
                    options: { serialize: false },
                    computeSize: () => [0, 24],
                    draw(ctx, n, width, posY) {
                        drawSeparatorLine(ctx, this.value, width, posY);
                    },
                });
            }

            // --- Replace enabled BOOLEAN widgets with custom toggle+separator ---
            // Process in reverse to avoid index shifting
            for (let si = SLOT_DEFS.length - 1; si >= 0; si--) {
                const def = SLOT_DEFS[si];
                const idx = node.widgets.findIndex(w => w.name === def.prefix + "enabled");
                if (idx < 0) continue;

                // Save current value and remove original BOOLEAN widget
                const origValue = node.widgets[idx].value;
                node.widgets[idx].onRemove?.();
                node.widgets.splice(idx, 1);

                // Create custom replacement widget
                const toggleWidget = {
                    name: def.prefix + "enabled",
                    type: "custom",
                    value: origValue,
                    options: { serialize: true },
                    computeSize: () => [0, 28],
                    _fvm_label: def.label,

                    draw(ctx, nodeRef, width, posY, height) {
                        const enabled = this.value;
                        const cy = posY + 16;

                        ctx.save();

                        // Toggle circle
                        const toggleX = 22;
                        const toggleR = 7;
                        ctx.beginPath();
                        ctx.arc(toggleX, cy - 1, toggleR, 0, Math.PI * 2);
                        if (enabled) {
                            ctx.fillStyle = "#4CAF50";
                            ctx.fill();
                            // Checkmark
                            ctx.strokeStyle = "#fff";
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            ctx.moveTo(toggleX - 3, cy - 1);
                            ctx.lineTo(toggleX - 0.5, cy + 2);
                            ctx.lineTo(toggleX + 4, cy - 4);
                            ctx.stroke();
                        } else {
                            ctx.fillStyle = "#333";
                            ctx.fill();
                            ctx.strokeStyle = "#666";
                            ctx.lineWidth = 1.5;
                            ctx.beginPath();
                            ctx.arc(toggleX, cy - 1, toggleR, 0, Math.PI * 2);
                            ctx.stroke();
                        }

                        // Label
                        const labelX = toggleX + toggleR + 10;
                        ctx.font = "bold 11px Arial";
                        ctx.fillStyle = enabled ? "#ccc" : "#777";
                        ctx.textAlign = "left";
                        ctx.fillText(this._fvm_label, labelX, cy + 1);

                        // Line after label
                        const textEnd = labelX + ctx.measureText(this._fvm_label).width + 10;
                        ctx.strokeStyle = "#555";
                        ctx.lineWidth = 1;
                        if (textEnd < width - 15) {
                            ctx.beginPath();
                            ctx.moveTo(textEnd, cy - 3);
                            ctx.lineTo(width - 15, cy - 3);
                            ctx.stroke();
                        }

                        ctx.restore();
                    },

                    mouse(event, pos, nodeRef) {
                        if (event.type === "pointerdown") {
                            this.value = !this.value;
                            updateSlotVisibility(nodeRef);
                            return true;
                        }
                        return false;
                    },

                    serializeValue() {
                        return this.value;
                    },
                };

                // Insert at same position
                node.widgets.splice(idx, 0, toggleWidget);
            }

            // Set initial visibility
            requestAnimationFrame(() => {
                updateSlotVisibility(node);
                requestAnimationFrame(() => updateSlotVisibility(node));
            });

            return result;
        };

        // --- Execution info display ---
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
