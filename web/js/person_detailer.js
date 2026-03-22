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

// Hide/show for remaining standard widgets (prompt textarea)
function hideWidget(w) {
    if (!w || w._fvm_hidden) return;
    w._fvm_hidden = true;
    w._fvm_origType = w.type;
    w._fvm_origCS = w.computeSize;
    w.type = "converted-widget";
    w.computeSize = () => [0, -4];
    if (w.inputEl) w.inputEl.style.display = "none";
}

function showWidget(w) {
    if (!w || !w._fvm_hidden) return;
    w._fvm_hidden = false;
    w.type = w._fvm_origType;
    w.computeSize = w._fvm_origCS;
    if (w.inputEl) w.inputEl.style.display = "";
}

function updateSlots(node) {
    for (const def of SLOT_DEFS) {
        const enabledW = node.widgets.find(w => w.name === def.prefix + "enabled");
        const promptW = node.widgets.find(w => w.name === def.prefix + "prompt");
        if (!enabledW) continue;

        // LoRA combo is NEVER hidden (always visible, just dimmed via draw override)
        // Only prompt textarea toggles visibility
        if (enabledW.value) {
            showWidget(promptW);
        } else {
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

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const node = this;

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

            // --- Process slots (backwards to avoid index shift) ---
            for (let si = SLOT_DEFS.length - 1; si >= 0; si--) {
                const def = SLOT_DEFS[si];

                // 1. Remove lora_strength widget, store value on a custom object
                const strIdx = node.widgets.findIndex(w => w.name === def.prefix + "lora_strength");
                let strengthValue = 1.0;
                if (strIdx >= 0) {
                    strengthValue = node.widgets[strIdx].value;
                    node.widgets[strIdx].onRemove?.();
                    node.widgets.splice(strIdx, 1);
                }

                // Create a hidden proxy widget to hold strength value for serialization
                const strengthProxy = {
                    name: def.prefix + "lora_strength",
                    type: "custom",
                    value: strengthValue,
                    options: { serialize: true },
                    computeSize: () => [0, -4], // zero height, invisible
                    draw() {}, // nothing to draw
                    serializeValue() { return this.value; },
                };

                // 2. Remove enabled BOOLEAN widget
                const enIdx = node.widgets.findIndex(w => w.name === def.prefix + "enabled");
                if (enIdx < 0) continue;
                const enabledValue = node.widgets[enIdx].value;
                node.widgets[enIdx].onRemove?.();
                node.widgets.splice(enIdx, 1);

                // 3. Create custom toggle+separator widget
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

                // Insert toggle at the position where enabled was
                node.widgets.splice(enIdx, 0, toggle);

                // Insert strength proxy right after toggle (hidden, just for serialization)
                // Find the lora widget position and insert proxy after it
                const loraIdx = node.widgets.findIndex(w => w.name === def.prefix + "lora");
                if (loraIdx >= 0) {
                    node.widgets.splice(loraIdx + 1, 0, strengthProxy);
                } else {
                    // Fallback: insert after toggle
                    node.widgets.splice(enIdx + 1, 0, strengthProxy);
                }

                // 4. Augment lora combo widget to draw strength on same row
                const loraW = node.widgets.find(w => w.name === def.prefix + "lora");
                if (loraW) {
                    loraW._fvm_strProxy = strengthProxy;

                    const origDraw = loraW.draw;
                    loraW.draw = function(ctx, nodeRef, width, posY, height) {
                        if (origDraw) origDraw.call(this, ctx, nodeRef, width, posY, height);

                        const sp = this._fvm_strProxy;
                        if (!sp) return;

                        const val = sp.value.toFixed(2);
                        const sw = 55, sx = width - sw - 5;
                        const sy = posY, sh = height || 20;

                        ctx.save();
                        ctx.fillStyle = "#2a2a2a";
                        ctx.strokeStyle = "#555";
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.roundRect(sx, sy + 1, sw, sh - 2, 4);
                        ctx.fill();
                        ctx.stroke();

                        ctx.font = "12px Arial";
                        ctx.fillStyle = "#ddd";
                        ctx.textAlign = "center";
                        ctx.textBaseline = "middle";
                        ctx.fillText(val, sx + sw / 2, sy + sh / 2);

                        ctx.font = "8px Arial";
                        ctx.fillStyle = "#888";
                        ctx.fillText("◀", sx + 7, sy + sh / 2);
                        ctx.fillText("▶", sx + sw - 7, sy + sh / 2);
                        ctx.restore();

                        this._fvm_strBounds = { x: sx, y: sy, w: sw, h: sh };
                    };

                    const origMouse = loraW.mouse;
                    loraW.mouse = function(event, pos, nodeRef) {
                        if (this._fvm_strBounds && this._fvm_strProxy && event.type === "pointerdown") {
                            const [mx, my] = pos;
                            const b = this._fvm_strBounds;
                            if (mx >= b.x && mx <= b.x + b.w && my >= b.y && my <= b.y + b.h) {
                                const sp = this._fvm_strProxy;
                                const half = b.x + b.w / 2;
                                if (mx < half) {
                                    sp.value = Math.max(0, +(sp.value - 0.05).toFixed(2));
                                } else {
                                    sp.value = Math.min(2, +(sp.value + 0.05).toFixed(2));
                                }
                                nodeRef.graph?.setDirtyCanvas(true, true);
                                return true;
                            }
                        }
                        if (origMouse) return origMouse.call(this, event, pos, nodeRef);
                        return false;
                    };
                }
            }

            // Initial state
            requestAnimationFrame(() => {
                updateSlots(node);
                requestAnimationFrame(() => updateSlots(node));
            });

            return result;
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
