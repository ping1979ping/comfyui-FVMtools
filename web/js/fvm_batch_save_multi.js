import { app } from "../../../scripts/app.js";

// Batch Save Multi: the Python side declares eight fixed slots (image_N,
// gate_N, subdir_N). The `slots` widget decides how many are shown — sockets
// beyond it are removed unless something is wired to them, and their folder
// widgets are hidden. Values stay serialized, so shrinking and growing again
// keeps what was typed.

const MAX_SLOTS = 8;
const NODE_NAME = "FVM_BatchSaveMulti";
const SOCKETS = [
    ["image", "IMAGE"],
    ["gate", "*"],
];

function slotCount(node) {
    const value = node.widgets?.find((w) => w.name === "slots")?.value;
    return Math.max(1, Math.min(MAX_SLOTS, Number(value) || 1));
}

function setWidgetVisible(widget, visible) {
    if (!widget) return;
    if (widget._fvmComputeSize === undefined) {
        widget._fvmComputeSize = widget.computeSize ?? null;
    }
    widget.hidden = !visible;
    if (visible) {
        if (widget._fvmComputeSize) widget.computeSize = widget._fvmComputeSize;
        else delete widget.computeSize;
    } else {
        widget.computeSize = () => [0, -4];
    }
}

function applySlots(node) {
    const count = slotCount(node);

    for (let n = MAX_SLOTS; n >= 1; n--) {
        for (const [prefix, type] of SOCKETS) {
            const name = `${prefix}_${n}`;
            const index = node.inputs?.findIndex((input) => input.name === name) ?? -1;
            if (n > count) {
                // Never cut a wire the user made — a linked socket stays.
                if (index >= 0 && node.inputs[index].link == null) node.removeInput(index);
            } else if (index < 0) {
                node.addInput(name, type);
            }
        }
    }

    for (let n = 1; n <= MAX_SLOTS; n++) {
        setWidgetVisible(node.widgets?.find((w) => w.name === `subdir_${n}`), n <= count);
    }

    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "FVMTools.BatchSaveMulti",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            try {
                const widget = this.widgets?.find((w) => w.name === "slots");
                if (widget) {
                    const callback = widget.callback;
                    const node = this;
                    widget.callback = function () {
                        const value = callback?.apply(this, arguments);
                        applySlots(node);
                        return value;
                    };
                }
                applySlots(this);
            } catch (error) {
                console.error("[FVMTools] BatchSaveMulti setup failed:", error);
            }
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            // After a workflow load the saved links exist; now trim the rest.
            setTimeout(() => applySlots(this), 0);
            return result;
        };
    },
});
