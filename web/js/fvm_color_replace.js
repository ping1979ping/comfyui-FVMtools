import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "FVM.PromptColorReplace",
    async nodeCreated(node) {
        if (node.comfyClass !== "FVM_PromptColorReplace") return;

        node.addWidget("button", "\ud83d\udccb Paste Example Prompt", null, () => {
            const promptWidget = node.widgets.find(w => w.name === "prompt");
            if (promptWidget) {
                promptWidget.value =
                    "wearing #primary# {miniskirt|skirt|shorts|bikini bottoms} with #neutral# bikini top, " +
                    "#accent# {open shirt|transparent dress|sarong|beach dress}, #metallic# jewelry, #secondary# accessories";
                app.graph.setDirtyCanvas(true, true);
            }
        });

        node.addWidget("button", "\ud83d\udccb Copy All Tags", null, () => {
            navigator.clipboard.writeText(
                "Numbered: #color1# #color2# #color3# #color4# #color5# #color6# #color7# #color8#\n" +
                "Semantic: #primary# #secondary# #accent# #neutral# #metallic#\n" +
                "Short:    #pri# #sec# #acc# #neu# #met#"
            );
        });
    },
});
