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
                        similarity: parseFloat(parts[0]).toFixed(4),
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
                    const parts = message.text[0].split("|");
                    this._psmValues = {
                        matched_count: parts[0],
                        face_count: parts[1],
                        similarities: parts.slice(2).join("|"),
                    };
                    // Output indices: person_data(0), face/head/body(1-3), combined(4-6),
                    // aux_masks(7), preview(8), similarities(9), matches(10), matched_count(11), face_count(12), report(13)
                    if (this.outputs[11]) this.outputs[11]["label"] = " ";  // matched_count
                    if (this.outputs[12]) this.outputs[12]["label"] = " ";  // face_count
                    if (this.outputs[9]) this.outputs[9]["label"] = " ";   // similarities
                    this.setDirtyCanvas(true);
                }
                return r;
            };

            const onDrawFGMulti = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                const r = onDrawFGMulti ? onDrawFGMulti.apply(this, arguments) : undefined;
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
