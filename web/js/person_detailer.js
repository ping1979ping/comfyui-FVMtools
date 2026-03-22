import { app } from "../../../scripts/app.js";

const SLOT_DEFS = [
    { prefix: "reference_1_", label: "Reference 1" },
    { prefix: "reference_2_", label: "Reference 2" },
    { prefix: "reference_3_", label: "Reference 3" },
    { prefix: "reference_4_", label: "Reference 4" },
    { prefix: "reference_5_", label: "Reference 5" },
    { prefix: "generic_", label: "Generic (Unmatched)" },
];

const STATIC_SEPS = [
    { label: "── Inpaint ──", before: "mask_blend_pixels" },
    { label: "── Detail Daemon ──", before: "detail_daemon_enabled" },
    { label: "── Sampler ──", before: "seed" },
];

function drawSepLine(ctx, label, width, posY) {
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

// ── Splice-based hide/show ──
// The only reliable way to hide widgets in this ComfyUI version.
// Stores removed widgets in node._fvm_stash keyed by widget name.

function spliceHide(node, widgetName) {
    if (!node._fvm_stash) node._fvm_stash = {};
    if (node._fvm_stash[widgetName]) return; // already hidden

    const idx = node.widgets.findIndex(w => w.name === widgetName);
    if (idx < 0) return;

    const widget = node.widgets[idx];
    // For DOM widgets (textarea), also hide the element
    if (widget.inputEl) widget.inputEl.style.display = "none";

    node._fvm_stash[widgetName] = { widget, afterName: null };

    // Remember which widget it was after (for re-insertion)
    if (idx > 0) {
        node._fvm_stash[widgetName].afterName = node.widgets[idx - 1].name;
    }

    node.widgets.splice(idx, 1);
}

function spliceShow(node, widgetName) {
    if (!node._fvm_stash?.[widgetName]) return; // not hidden

    const { widget, afterName } = node._fvm_stash[widgetName];
    delete node._fvm_stash[widgetName];

    // For DOM widgets, show element
    if (widget.inputEl) widget.inputEl.style.display = "";

    // Find insertion point: after the widget it was previously after
    let insertIdx = node.widgets.length;
    if (afterName) {
        const afterIdx = node.widgets.findIndex(w => w.name === afterName);
        if (afterIdx >= 0) insertIdx = afterIdx + 1;
    }

    node.widgets.splice(insertIdx, 0, widget);
}

function updateSlots(node) {
    // Collect which widgets to hide per slot
    for (const def of SLOT_DEFS) {
        const enabledW = node.widgets.find(w => w.name === def.prefix + "enabled")
                      || node._fvm_stash?.[def.prefix + "enabled"]?.widget;
        if (!enabledW) continue;

        const names = [
            def.prefix + "lora_strength",
            def.prefix + "prompt",
            def.prefix + "catch_unprocessed",
        ];

        if (enabledW.value) {
            for (const n of names) spliceShow(node, n);
        } else {
            for (const n of names) spliceHide(node, n);
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

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;

            try {
            // --- Static separators (backwards) ---
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
                    draw(ctx, n, width, posY) { drawSepLine(ctx, this.value, width, posY); },
                });
            }

            // --- Replace enabled BOOLEAN widgets (backwards) ---
            for (let si = SLOT_DEFS.length - 1; si >= 0; si--) {
                const def = SLOT_DEFS[si];
                const enIdx = node.widgets.findIndex(w => w.name === def.prefix + "enabled");
                if (enIdx < 0) continue;

                const enabledValue = node.widgets[enIdx].value;
                node.widgets[enIdx].onRemove?.();
                node.widgets.splice(enIdx, 1);

                const toggle = {
                    name: def.prefix + "enabled",
                    type: "custom",
                    value: enabledValue,
                    options: { serialize: true },
                    computeSize: () => [0, 28],
                    _label: def.label,

                    draw(ctx, nodeRef, width, posY) {
                        const on = this.value;
                        const cy = posY + 16;
                        ctx.save();

                        const cx = 22, r = 7;
                        ctx.beginPath();
                        ctx.arc(cx, cy - 1, r, 0, Math.PI * 2);
                        if (on) {
                            ctx.fillStyle = "#4CAF50";
                            ctx.fill();
                            ctx.strokeStyle = "#fff";
                            ctx.lineWidth = 2;
                            ctx.beginPath();
                            ctx.moveTo(cx - 3, cy - 1);
                            ctx.lineTo(cx - 0.5, cy + 2);
                            ctx.lineTo(cx + 4, cy - 4);
                            ctx.stroke();
                        } else {
                            ctx.fillStyle = "#333";
                            ctx.fill();
                            ctx.strokeStyle = "#666";
                            ctx.lineWidth = 1.5;
                            ctx.beginPath();
                            ctx.arc(cx, cy - 1, r, 0, Math.PI * 2);
                            ctx.stroke();
                        }

                        const lx = cx + r + 10;
                        ctx.font = "bold 11px Arial";
                        ctx.fillStyle = on ? "#ccc" : "#777";
                        ctx.textAlign = "left";
                        ctx.fillText(this._label, lx, cy + 1);

                        const te = lx + ctx.measureText(this._label).width + 10;
                        if (te < width - 15) {
                            ctx.strokeStyle = "#555";
                            ctx.lineWidth = 1;
                            ctx.beginPath();
                            ctx.moveTo(te, cy - 3);
                            ctx.lineTo(width - 15, cy - 3);
                            ctx.stroke();
                        }
                        ctx.restore();
                    },

                    mouse(event, pos, nodeRef) {
                        if (event.type === "pointerdown") {
                            this.value = !this.value;
                            updateSlots(nodeRef);
                            return true;
                        }
                        return false;
                    },

                    serializeValue() { return this.value; },
                };

                node.widgets.splice(enIdx, 0, toggle);
            }

            // Initial state
            requestAnimationFrame(() => {
                updateSlots(node);
                requestAnimationFrame(() => updateSlots(node));
            });

            } catch (e) {
                console.error("[FVMTools] PersonDetailer widget setup error:", e);
            }

            return result;
        };

        // --- Execution info display ---
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

        // --- Execution info ---
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            if (message?.text?.length > 0) {
                this._pdInfo = message.text[0];
                this.setDirtyCanvas(true);
            }
            return r;
        };
    },
});
