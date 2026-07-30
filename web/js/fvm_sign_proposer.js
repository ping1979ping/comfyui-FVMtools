import { app } from "../../../scripts/app.js";

// FVM_SignTextProposer — reset button for the system prompt.
//
// The stock prompt is tuned against measured model behaviour (it has to stop the
// model transcribing the garbled original instead of replacing it), so editing it
// is easy to get wrong and hard to notice. The button pulls the current default
// from the backend rather than carrying a copy here — a duplicated string would
// drift out of sync with the tuning and silently reintroduce the bug.
//
// Also resets temperature, since that sits on a measured cliff: 0 transcriptions
// at 0.2, half of all runs at 0.25.

const NODE_NAME = "FVM_SignTextProposer";

async function fetchDefaults() {
    const res = await fetch("/fvmtools/sign-default-prompt");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    if (typeof data.system_prompt !== "string" || !data.system_prompt.length) {
        throw new Error("empty prompt returned");
    }
    return data;
}

app.registerExtension({
    name: "FVMTools.SignTextProposer",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const button = this.addWidget(
                "button", "↺ Reset system prompt", null, async () => {
                    const promptW = this.widgets?.find(w => w.name === "system_prompt");
                    if (!promptW) return;

                    const label = button.name;
                    button.name = "↺ Loading…";
                    this.setDirtyCanvas(true, true);

                    try {
                        const data = await fetchDefaults();

                        // Only warn when there is actual work to undo.
                        const changed = promptW.value !== data.system_prompt;
                        if (changed && !confirm(
                                "Replace the edited system prompt with the stock one?\n\n" +
                                "Your changes to this field will be lost.")) {
                            return;
                        }

                        promptW.value = data.system_prompt;
                        promptW.callback?.(promptW.value);

                        const tempW = this.widgets?.find(w => w.name === "temperature");
                        if (tempW && typeof data.temperature === "number") {
                            tempW.value = data.temperature;
                            tempW.callback?.(tempW.value);
                        }

                        button.name = changed ? "↺ Reset — done" : "↺ Already default";
                        setTimeout(() => {
                            button.name = label;
                            this.setDirtyCanvas(true, true);
                        }, 1600);
                    } catch (err) {
                        console.error("[FVMTools] system prompt reset failed:", err);
                        button.name = "↺ Failed — see console";
                        setTimeout(() => {
                            button.name = label;
                            this.setDirtyCanvas(true, true);
                        }, 2500);
                    } finally {
                        this.setDirtyCanvas(true, true);
                    }
                });

            // Park the button directly above the prompt it resets.
            const promptIdx = this.widgets?.findIndex(w => w.name === "system_prompt");
            if (promptIdx > -1) {
                const btnIdx = this.widgets.indexOf(button);
                if (btnIdx > -1 && btnIdx !== promptIdx) {
                    this.widgets.splice(btnIdx, 1);
                    const target = this.widgets.findIndex(w => w.name === "system_prompt");
                    this.widgets.splice(target, 0, button);
                }
            }

            return result;
        };
    },
});
