import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "FVMTools.PersonDetailer",

    async nodeCreated(node) {
        if (node.comfyClass !== "PersonDetailer") return;

        // Helper: hide/show a widget with proper size recalculation
        function setWidgetVisible(widget, visible) {
            if (!widget) return;
            widget.hidden = !visible;
            if (!visible) {
                if (!widget._origComputeSize) {
                    widget._origComputeSize = widget.computeSize;
                }
                widget.computeSize = () => [0, -4];
            } else if (widget._origComputeSize) {
                widget.computeSize = widget._origComputeSize;
            }
        }

        // For each slot, toggle lora+prompt visibility based on enabled
        const slotPrefixes = ["ref_1_", "ref_2_", "ref_3_", "ref_4_", "ref_5_", "generic_"];

        for (const prefix of slotPrefixes) {
            const toggle = node.widgets.find(w => w.name === prefix + "enabled");
            const lora = node.widgets.find(w => w.name === prefix + "lora");
            const prompt = node.widgets.find(w => w.name === prefix + "prompt");

            if (!toggle) continue;

            const updateVisibility = () => {
                setWidgetVisible(lora, toggle.value);
                setWidgetVisible(prompt, toggle.value);
                // Recalculate node size
                node.setSize([node.size[0], node.computeSize()[1]]);
            };

            // Store original callback if any
            const origCallback = toggle.callback;
            toggle.callback = (value) => {
                if (origCallback) origCallback(value);
                updateVisibility();
            };

            // Set initial state (after a small delay to ensure widgets are ready)
            setTimeout(updateVisibility, 50);
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "PersonDetailer") return;

        // Display execution info
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            if (message && message.text && message.text.length > 0) {
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
