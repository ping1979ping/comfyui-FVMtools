import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "FVM.ColorPaletteGenerator",

    async nodeCreated(node) {
        if (node.comfyClass !== "FVM_ColorPaletteGenerator") return;

        // Helper: find output value by name
        function getOutputValue(outputName) {
            const idx = node.outputs?.findIndex(o => o.name === outputName);
            if (idx === undefined || idx < 0) return "";
            // Try to get the last executed value
            if (node.widgets_values && node.widgets_values[outputName]) {
                return node.widgets_values[outputName];
            }
            return "";
        }

        // Helper: get palette string from node title or last execution
        function getPaletteFromNode() {
            // Access via the graph execution results
            const paletteWidget = node.widgets?.find(w => w.name === "palette_string");
            if (paletteWidget && paletteWidget.value) return paletteWidget.value;
            // Fallback: reconstruct from output slots if connected
            return "";
        }

        // "Copy Tags: Numbered" button
        node.addWidget("button", "Copy Tags: Numbered", null, () => {
            const numColors = node.widgets?.find(w => w.name === "num_colors");
            const count = numColors ? numColors.value : 5;
            const tags = [];
            for (let i = 1; i <= count; i++) {
                tags.push(`#color${i}#`);
            }
            const text = tags.join(", ");
            navigator.clipboard.writeText(text).then(() => {
                app.ui?.dialog?.show?.(`Copied: ${text}`) || console.log(`Copied: ${text}`);
            });
        });

        // "Copy Tags: Semantic" button
        node.addWidget("button", "Copy Tags: Semantic", null, () => {
            const text = "#primary#, #secondary#, #accent#, #neutral#, #metallic#";
            navigator.clipboard.writeText(text).then(() => {
                app.ui?.dialog?.show?.(`Copied: ${text}`) || console.log(`Copied: ${text}`);
            });
        });

        // "Copy Example Prompt" button
        node.addWidget("button", "Copy Example Prompt", null, () => {
            const text = "wearing #primary# {miniskirt|dress} with #neutral# top, #accent# accessories, #metallic# jewelry";
            navigator.clipboard.writeText(text).then(() => {
                app.ui?.dialog?.show?.("Copied example prompt to clipboard") || console.log("Copied example prompt");
            });
        });
    }
});
