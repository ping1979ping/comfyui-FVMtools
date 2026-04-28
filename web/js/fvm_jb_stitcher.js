/**
 * FVM_JB_Stitcher — dynamic optional input slots.
 *
 * Backed by a static set of optional `input_N` slots (N = 1..MAX_INPUTS) on
 * the Python side. The JS layer hides the unused slots by default and shows
 * exactly one trailing empty slot at all times — same UX as person_selector.js,
 * but for STRING inputs.
 *
 * Behavior:
 *   - On node creation: hide all slots except `input_1` (visible, optional).
 *   - When a slot gets a connection: if it's the last visible one, reveal the
 *     next one (up to MAX_INPUTS).
 *   - When a slot is disconnected: collapse trailing empty slots, keeping
 *     exactly one trailing empty slot after the last connected input.
 *
 * Pattern adapted from web/js/person_selector.js dynamic-reference-slot logic.
 */
import { app } from "../../scripts/app.js";

const MAX_INPUTS = 24;
const NODE_NAME = "FVM_JB_Stitcher";

function inputSlotsByName(node) {
    const slots = [];
    for (let i = 0; i < node.inputs.length; i++) {
        const inp = node.inputs[i];
        if (inp && typeof inp.name === "string" && inp.name.startsWith("input_")) {
            slots.push({ index: i, name: inp.name, linked: !!inp.link });
        }
    }
    return slots;
}

function visibleCount(node) {
    return inputSlotsByName(node).length;
}

function ensureTrailingEmpty(node) {
    const slots = inputSlotsByName(node);
    let lastConnected = -1;
    for (let i = 0; i < slots.length; i++) {
        if (slots[i].linked) lastConnected = i;
    }
    // Want at least lastConnected+1 trailing empty slot, capped at MAX_INPUTS.
    const targetCount = Math.min(MAX_INPUTS, Math.max(1, lastConnected + 2));
    const current = slots.length;

    if (current < targetCount) {
        for (let n = current + 1; n <= targetCount; n++) {
            node.addInput(`input_${n}`, "STRING");
        }
    } else if (current > targetCount) {
        for (let i = current - 1; i >= targetCount; i--) {
            node.removeInput(slots[i].index);
        }
    }
    // Renumber visible slots so they always read input_1, input_2, ...
    let n = 1;
    for (let i = 0; i < node.inputs.length; i++) {
        const inp = node.inputs[i];
        if (inp && typeof inp.name === "string" && inp.name.startsWith("input_")) {
            inp.name = `input_${n}`;
            n++;
        }
    }
}

app.registerExtension({
    name: "FVMTools.JB.Stitcher",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // Strip every `input_N` slot the Python side declared, then add
            // back exactly one (`input_1`) so the user starts with a clean
            // single-slot layout.
            for (let i = this.inputs.length - 1; i >= 0; i--) {
                const inp = this.inputs[i];
                if (inp && typeof inp.name === "string" && inp.name.startsWith("input_")) {
                    this.removeInput(i);
                }
            }
            this.addInput("input_1", "STRING");
            this.setDirtyCanvas(true, true);
            return r;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (slotType, index, connected, linkInfo, ioSlot) {
            const r = onConnectionsChange
                ? onConnectionsChange.apply(this, arguments)
                : undefined;

            // slotType 1 = INPUT
            if (slotType !== 1) return r;
            const slot = this.inputs[index];
            if (!slot || !slot.name || !slot.name.startsWith("input_")) return r;

            ensureTrailingEmpty(this);
            this.setDirtyCanvas(true, true);
            return r;
        };
    },
});
