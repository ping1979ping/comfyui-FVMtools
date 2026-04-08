import { app } from "../../../scripts/app.js";

// Helper: draw "label: value" with value in bold
function drawLabelValue(ctx, x, y, label, value) {
    ctx.save();
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";

    ctx.font = "12px Arial";
    const labelText = label + ": ";
    const valueText = String(value);
    const valueWidth = ctx.measureText(valueText).width;

    ctx.font = "bold 12px Arial";
    ctx.fillStyle = "#fff";
    ctx.fillText(valueText, x, y);

    ctx.font = "12px Arial";
    ctx.fillStyle = "#aaa";
    ctx.fillText(labelText, x - valueWidth, y);

    ctx.restore();
}

app.registerExtension({
    name: "FVMTools.PersonSelector",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        // --- PersonSelector (single) ---
        if (nodeData.name === "PersonSelector") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
                if (message && message.text && message.text.length > 0) {
                    const parts = message.text[0].split("|");
                    this._psValues = {
                        similarity: parts[0],
                        face_count: parts[1],
                        matched_face_index: parseInt(parts[2]) >= 0 ? parts[2] : "-",
                    };
                    this.outputs[0]["label"] = " ";
                    this.outputs[4]["label"] = " ";
                    this.outputs[5]["label"] = " ";
                    this.setDirtyCanvas(true);
                }
                return r;
            };

            const onDrawFG = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                const r = onDrawFG ? onDrawFG.apply(this, arguments) : undefined;
                if (!this.outputs || !this._psValues) return r;

                const entries = [
                    { idx: 0, label: "similarity", value: this._psValues.similarity },
                    { idx: 4, label: "face_count", value: this._psValues.face_count },
                    { idx: 5, label: "matched_idx", value: this._psValues.matched_face_index },
                ];
                for (const e of entries) {
                    if (!this.outputs[e.idx]) continue;
                    const y = this.getConnectionPos(false, e.idx)[1] - this.pos[1];
                    drawLabelValue(ctx, this.size[0] - 14, y, e.label, e.value);
                }
                return r;
            };
        }

        // --- PersonSelectorMulti ---
        // Output order (PERSON_DATA prepended, all legacy indices +1):
        //   0: person_data, 1: face_masks, 2: head_masks, 3: body_masks,
        //   4: combined_face, 5: combined_head, 6: combined_body,
        //   7: preview, 8: similarities, 9: matches, 10: matched_count, 11: face_count, 12: report
        if (nodeData.name === "PersonSelectorMulti") {

            const PSM_SEPS = [
                { label: "── Matching ──",        before: "auto_threshold" },
                { label: "── Mask Refinement ──", before: "mask_fill_holes" },
                { label: "── Aux / YOLO ──",      before: "aux_mask_type" },
                { label: "── Appearance ──",      before: "match_weights" },
            ];

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                for (let i = this.inputs.length - 1; i >= 0; i--) {
                    const name = this.inputs[i].name;
                    if (name.startsWith("reference_") && name !== "reference_1") {
                        this.removeInput(i);
                    }
                }
                this.addInput("reference_2", "IMAGE");

                // Static section separators (same pattern as person_detailer.js)
                try {
                    for (let i = PSM_SEPS.length - 1; i >= 0; i--) {
                        const sep = PSM_SEPS[i];
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
                    console.error("[FVMTools] PersonSelectorMulti separator setup error:", e);
                }

                // Wire aux_model → fetch class names and stash them on the node so
                // they can be drawn directly under the aux_label widget. The new Vue
                // frontend doesn't honor mutated widget tooltips, so we render text
                // on the canvas instead — always visible, no hover needed.
                const auxModelW = this.widgets?.find(w => w.name === "aux_model");
                const auxLabelW = this.widgets?.find(w => w.name === "aux_label");
                if (auxModelW) {
                    const node = this;
                    node._fvmAuxClasses = "";
                    const updateClasses = async (modelName) => {
                        if (!modelName || modelName === "none") {
                            node._fvmAuxClasses = "";
                            node.setDirtyCanvas(true, true);
                            return;
                        }
                        try {
                            const resp = await fetch(`/fvmtools/yolo-classes?model=${encodeURIComponent(modelName)}`);
                            const data = await resp.json();
                            const cls = (data.classes || []);
                            node._fvmAuxClasses = cls.length
                                ? "classes: " + cls.join(", ")
                                : "(no class metadata)";
                            console.log(`[FVMTools] ${modelName} classes:`, cls);
                            // Best-effort tooltip update for frontends that DO honor mutations
                            if (auxLabelW) {
                                auxLabelW.options = auxLabelW.options || {};
                                auxLabelW.options.tooltip =
                                    "Comma-separated substring filter. Available: " + cls.join(", ");
                                if (auxLabelW.inputEl) {
                                    auxLabelW.inputEl.title = auxLabelW.options.tooltip;
                                }
                            }
                            node.setDirtyCanvas(true, true);
                        } catch (e) {
                            console.warn("[FVMTools] yolo-classes fetch failed:", e);
                        }
                    };
                    const origCallback = auxModelW.callback;
                    auxModelW.callback = function (value) {
                        const ret = origCallback ? origCallback.apply(this, arguments) : undefined;
                        updateClasses(value);
                        return ret;
                    };
                    // Initial fetch for the default value
                    if (auxModelW.value && auxModelW.value !== "none") {
                        updateClasses(auxModelW.value);
                    }
                }

                return r;
            };

            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                if (onConnectionsChange) {
                    onConnectionsChange.apply(this, arguments);
                }
                if (!link_info || type !== 1) return;

                const getRefInputs = () => {
                    const refs = [];
                    for (let i = 0; i < this.inputs.length; i++) {
                        if (this.inputs[i].name.startsWith("reference_")) {
                            refs.push({ index: i, name: this.inputs[i].name, linked: !!this.inputs[i].link });
                        }
                    }
                    return refs;
                };

                if (connected) {
                    const refInputs = getRefInputs();
                    const lastRef = refInputs[refInputs.length - 1];
                    if (lastRef && lastRef.index === index) {
                        const nextNum = refInputs.length + 1;
                        if (nextNum <= 10) {
                            this.addInput(`reference_${nextNum}`, "IMAGE");
                        }
                    }
                } else {
                    const refInputs = getRefInputs();
                    let lastConnectedIdx = -1;
                    for (let i = refInputs.length - 1; i >= 0; i--) {
                        if (refInputs[i].linked) {
                            lastConnectedIdx = i;
                            break;
                        }
                    }
                    const keepCount = Math.max(2, lastConnectedIdx + 2);
                    for (let i = refInputs.length - 1; i >= keepCount; i--) {
                        this.removeInput(refInputs[i].index);
                    }
                    let num = 1;
                    for (let i = 0; i < this.inputs.length; i++) {
                        if (this.inputs[i].name.startsWith("reference_")) {
                            this.inputs[i].name = `reference_${num}`;
                            num++;
                        }
                    }
                }

                this.setDirtyCanvas(true, true);
            };

            const onExecutedMulti = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                const r = onExecutedMulti ? onExecutedMulti.apply(this, arguments) : undefined;
                if (message && message.text && message.text.length > 0) {
                    // Format: "m1|m2|f1|f2|72%, 85%" — all pipe-separated
                    // Numeric parts (matched + faces) come first, then similarities (contain %)
                    const segs = message.text[0].split("|");
                    let simIdx = segs.length;
                    for (let i = 0; i < segs.length; i++) {
                        if (segs[i].includes("%")) { simIdx = i; break; }
                    }
                    const nums = segs.slice(0, simIdx);
                    const half = Math.floor(nums.length / 2);
                    this._psmValues = {
                        matched_count: nums.slice(0, half).join("|") || "0",
                        face_count: nums.slice(half).join("|") || "0",
                        similarities: segs.slice(simIdx).join("|") || "",
                    };
                    if (this.outputs[11]) this.outputs[11]["label"] = " ";
                    if (this.outputs[12]) this.outputs[12]["label"] = " ";
                    if (this.outputs[9]) this.outputs[9]["label"] = " ";
                    this.setDirtyCanvas(true);
                }
                return r;
            };

            const onDrawFGMulti = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                const r = onDrawFGMulti ? onDrawFGMulti.apply(this, arguments) : undefined;

                // Draw YOLO class list above the auto_threshold widget,
                // left half of the node so it doesn't collide with right-side widgets.
                // Header "classes:" on its own line, followed by word-wrapped names.
                if (this._fvmAuxClasses) {
                    const anchorW = this.widgets?.find(w => w.name === "auto_threshold");
                    if (anchorW && anchorW.last_y != null) {
                        ctx.save();
                        ctx.font = "10px sans-serif";
                        ctx.fillStyle = "#9cf";
                        ctx.textBaseline = "bottom";
                        const maxW = this.size[0] * 0.5 - 10;
                        // Strip leading "classes: " so we can render it as a header line
                        const body = this._fvmAuxClasses.replace(/^classes:\s*/, "");
                        const words = body.split(" ");
                        const lines = ["classes:"];
                        let cur = "";
                        for (const w of words) {
                            const test = cur ? cur + " " + w : w;
                            if (ctx.measureText(test).width > maxW && cur) {
                                lines.push(cur);
                                cur = w;
                            } else {
                                cur = test;
                            }
                        }
                        if (cur) lines.push(cur);
                        const maxLines = Math.min(lines.length, 7);
                        const lineH = 12;
                        // Shift up by ~2 widget heights (default widget height = 20)
                        // so the text clears the widgets below it.
                        const bottomY = anchorW.last_y - 44;
                        for (let i = 0; i < maxLines; i++) {
                            const y = bottomY - (maxLines - 1 - i) * lineH;
                            ctx.fillText(lines[i], 10, y);
                        }
                        ctx.restore();
                    }
                }

                if (!this.outputs || !this._psmValues) return r;

                const entries = [
                    { idx: 9, label: "similarities", value: this._psmValues.similarities },
                    { idx: 11, label: "matched", value: this._psmValues.matched_count },
                    { idx: 12, label: "faces", value: this._psmValues.face_count },
                ];
                for (const e of entries) {
                    if (!this.outputs[e.idx]) continue;
                    const y = this.getConnectionPos(false, e.idx)[1] - this.pos[1];
                    drawLabelValue(ctx, this.size[0] - 14, y, e.label, e.value);
                }
                return r;
            };
        }
    },
});
