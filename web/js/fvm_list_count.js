import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// List / Batch Count: the node reports what it received (LIST / BATCH / SINGLE
// plus the socket type) and the count in one read-only text box on the node.
// Same widget route as pysssss ShowText, which works on this frontend.

const NODE_NAME = "FVM_ListCount";
const WIDGET_NAME = "fvm_list_count_text";

function showText(node, text) {
    let widget = node.widgets?.find((w) => w.name === WIDGET_NAME);
    if (!widget) {
        widget = ComfyWidgets["STRING"](node, WIDGET_NAME, ["STRING", { multiline: true }], app).widget;
        widget.serialize = false;
        if (widget.inputEl) {
            widget.inputEl.readOnly = true;
            widget.inputEl.style.opacity = 0.8;
        }
    }
    widget.value = text;
    if (widget.inputEl) widget.inputEl.value = text;

    requestAnimationFrame(() => {
        const size = node.computeSize();
        node.setSize([Math.max(size[0], 300, node.size[0]), Math.max(size[1], node.size[1])]);
        app.graph.setDirtyCanvas(true, false);
    });
}

app.registerExtension({
    name: "FVMtools.ListCount",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            const payload = message?.fvm_list_count?.[0];
            if (payload) showText(this, `${payload.kind}\ncount: ${payload.count}`);
            return result;
        };
    },
});
