import { app } from "../../../scripts/app.js";

// YOLO class display widget for PersonDataRefiner — same pattern as PersonSelectorMulti.
// Shows available classes right under the aux_model dropdown, expands dynamically.

app.registerExtension({
    name: "FVMTools.PersonDataRefiner",

    async beforeRegisterNodeDef(nodeType, nodeData, appRef) {
        if (nodeData.name !== "PersonDataRefiner") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const auxModelW = this.widgets?.find(w => w.name === "aux_model");
            if (auxModelW) {
                const auxModelIdx = this.widgets.indexOf(auxModelW);
                const classWidget = {
                    name: "_aux_classes",
                    type: "custom",
                    value: "",
                    _lines: [],
                    options: { serialize: false },
                    computeSize(nodeWidth) {
                        const lineH = 13;
                        const lines = this._lines.length || 0;
                        return [0, lines > 0 ? lines * lineH + 6 : 20];
                    },
                    draw(ctx, node, width, posY, height) {
                        if (!this._lines.length) return;
                        ctx.save();
                        ctx.font = "10px sans-serif";
                        ctx.fillStyle = "#9cf";
                        ctx.textBaseline = "top";
                        const lineH = 13;
                        for (let i = 0; i < this._lines.length; i++) {
                            ctx.fillText(this._lines[i], 15, posY + 3 + i * lineH);
                        }
                        ctx.restore();
                    },
                };
                this.widgets.splice(auxModelIdx + 1, 0, classWidget);

                const node = this;
                const updateClasses = async (modelName) => {
                    if (!modelName || modelName === "none") {
                        classWidget._lines = [];
                        classWidget.value = "";
                        node.setDirtyCanvas(true, true);
                        return;
                    }
                    try {
                        const resp = await fetch(`/fvmtools/yolo-classes?model=${encodeURIComponent(modelName)}`);
                        const data = await resp.json();
                        const cls = (data.classes || []);
                        if (!cls.length) {
                            classWidget._lines = ["(no class metadata)"];
                            classWidget.value = "";
                            node.setDirtyCanvas(true, true);
                            return;
                        }
                        const maxW = (node.size?.[0] || 300) - 30;
                        const lines = ["classes:"];
                        let cur = "";
                        for (const c of cls) {
                            const test = cur ? cur + ", " + c : c;
                            if ((test.length * 6) > maxW && cur) {
                                lines.push(cur);
                                cur = c;
                            } else {
                                cur = test;
                            }
                        }
                        if (cur) lines.push(cur);
                        classWidget._lines = lines;
                        classWidget.value = cls.join(", ");
                        console.log(`[FVMTools] PersonDataRefiner ${modelName} classes:`, cls);
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
                if (auxModelW.value && auxModelW.value !== "none") {
                    updateClasses(auxModelW.value);
                }
            }

            return result;
        };
    },
});
