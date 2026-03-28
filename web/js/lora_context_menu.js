import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Nodes that have LoRA widgets
const LORA_NODES = ["PersonDetailer", "PersonDetailerControlNet"];

// LoRA widget name patterns
const LORA_WIDGET_NAMES = [
    "reference_1_lora", "reference_2_lora", "reference_3_lora",
    "reference_4_lora", "reference_5_lora", "generic_lora",
];

// Find the prompt widget associated with a lora widget
function findPromptWidget(node, loraWidgetName) {
    const prefix = loraWidgetName.replace("_lora", "_prompt");
    return node.widgets?.find(w => w.name === prefix);
}

// Cache for CivitAI info
const _infoCache = {};

async function fetchLoraInfo(loraName) {
    if (_infoCache[loraName]) return _infoCache[loraName];

    try {
        const resp = await api.fetchApi(
            `/fvmtools/lora-info?file=${encodeURIComponent(loraName)}`
        );
        if (!resp.ok) {
            console.warn(`[FVMTools] LoRA info request failed: ${resp.status}`);
            return null;
        }
        const data = await resp.json();
        _infoCache[loraName] = data;
        return data;
    } catch (e) {
        console.error("[FVMTools] LoRA info fetch failed:", e);
        return null;
    }
}

function addLoraMenuEntries(node, options) {
    const loraEntries = [];

    for (const wName of LORA_WIDGET_NAMES) {
        const widget = node.widgets?.find(w => w.name === wName);
        if (!widget || widget.value === "None" || !widget.value) continue;

        const loraName = widget.value;
        const shortName = loraName.split(/[/\\]/).pop();
        const slotLabel = wName.replace("_lora", "").replace("reference_", "Ref ");

        // Separator before first lora entry
        if (loraEntries.length === 0) {
            loraEntries.push(null); // separator
        }

        // Open on CivitAI
        loraEntries.push({
            content: `${slotLabel} → Open on CivitAI`,
            callback: () => {
                fetchLoraInfo(loraName).then(info => {
                    if (info?.civitaiUrl) {
                        window.open(info.civitaiUrl, "_blank");
                    } else {
                        // Fallback: search on CivitAI by filename
                        const searchName = shortName.replace(/\.\w+$/, "");
                        window.open(
                            `https://civitai.com/search/models?sortBy=models_v9&query=${encodeURIComponent(searchName)}`,
                            "_blank"
                        );
                    }
                });
            },
        });

        // Copy Trigger Words
        loraEntries.push({
            content: `${slotLabel} → Copy Trigger Words`,
            callback: () => {
                fetchLoraInfo(loraName).then(info => {
                    if (!info?.triggerWords?.length) {
                        alert(`No trigger words found for:\n${shortName}`);
                        return;
                    }
                    const triggerStr = info.triggerWords.join(", ");
                    const promptWidget = findPromptWidget(node, wName);
                    if (promptWidget) {
                        const current = (promptWidget.value || "").trim();
                        promptWidget.value = current ? current + ", " + triggerStr : triggerStr;
                        promptWidget.callback?.(promptWidget.value);
                        node.setDirtyCanvas(true);
                    } else {
                        navigator.clipboard.writeText(triggerStr);
                        alert(`Trigger words copied to clipboard:\n${triggerStr}`);
                    }
                });
            },
        });

        // Show Info
        loraEntries.push({
            content: `${slotLabel} → Show Info`,
            callback: () => {
                fetchLoraInfo(loraName).then(info => {
                    if (!info) {
                        alert(`Could not fetch info for:\n${shortName}`);
                        return;
                    }
                    const lines = [
                        `Name: ${info.name || "?"}`,
                        `Version: ${info.version || "?"}`,
                        `Type: ${info.type || "?"}`,
                        `Base Model: ${info.baseModel || "?"}`,
                        `Trigger Words: ${info.triggerWords?.join(", ") || "none"}`,
                        `SHA256: ${info.sha256 || "?"}`,
                        `CivitAI: ${info.civitaiUrl || "not found"}`,
                    ];
                    if (info.error) lines.unshift(`Error: ${info.error}`);
                    alert(lines.join("\n"));
                });
            },
        });
    }

    if (loraEntries.length > 0) {
        options.push(...loraEntries);
    }
}

app.registerExtension({
    name: "FVMTools.LoraContextMenu",

    async beforeRegisterNodeDef(nodeType, nodeData, appRef) {
        if (!LORA_NODES.includes(nodeData.name)) return;

        const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            const result = origGetExtraMenuOptions?.apply(this, arguments);
            addLoraMenuEntries(this, options);
            return result;
        };
    },
});
