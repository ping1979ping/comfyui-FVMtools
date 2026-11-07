import { app } from "../../../scripts/app.js";

// FVM_MOSS_SAM3Layout — "use suggested rules" button.
//
// In rules_mode auto / manual + auto the node grounds a whole vocabulary and
// returns the rules for the words that hit (ui.suggested_rules). After a run
// this button copies them into box_rules and switches rules_mode to manual,
// so the next run only grounds what is actually in the picture.

const NODE_NAME = "FVM_MOSS_SAM3Layout";
const LABEL = "use suggested rules";

function setWidget(node, name, value) {
    const w = node.widgets?.find(w => w.name === name);
    if (!w) return false;
    w.value = value;
    if (w.inputEl) w.inputEl.value = value;   // multiline DOM widget
    w.callback?.(value);
    return true;
}

function applySuggestion(node) {
    const text = node._fvmSuggestedRules;
    if (!text?.trim()) return;
    const rulesW = node.widgets?.find(w => w.name === "box_rules");
    if (!rulesW) return;

    // manual + auto suggestions start with the manual rules — only warn when
    // the current rules would really be lost.
    const current = String(rulesW.value ?? "").trim();
    if (current && !text.includes(current) && !confirm(
            "Replace box_rules with the suggested rules?\n\n" +
            "Your current rules are not part of the suggestion and will be lost.")) {
        return;
    }
    setWidget(node, "box_rules", text);
    setWidget(node, "rules_mode", "manual");
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "FVMTools.MossLayout",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            const text = message?.suggested_rules?.[0];
            if (typeof text !== "string") return result;
            this._fvmSuggestedRules = text;

            let button = this.widgets?.find(w => w._fvmSuggest);
            if (!button) {
                button = this.addWidget("button", LABEL, null, () => applySuggestion(this));
                button._fvmSuggest = true;
                button.serialize = false;              // not part of widgets_values
                if (button.options) button.options.serialize = false;
                const size = this.computeSize();
                this.setSize([Math.max(this.size[0], size[0]), Math.max(this.size[1], size[1])]);
            }
            const count = text.split("\n")
                .filter(l => l.split("#", 1)[0].trim()).length;
            button.name = count ? `${LABEL} (${count})` : `${LABEL} (none found)`;
            this.setDirtyCanvas(true, true);
            return result;
        };
    },
});
