import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "FVM.OutfitGenerator",
    async nodeCreated(node) {
        if (node.comfyClass !== "FVM_OutfitGenerator") return;

        node.addWidget("button", "Copy Override Template", null, () => {
            navigator.clipboard.writeText(
                "headwear: auto\ntop: auto\nouterwear: auto\nbottom: auto\nfootwear: auto\naccessories: auto\nbag: auto"
            );
        });

        node.addWidget("button", "Copy List Path", null, () => {
            const setWidget = node.widgets.find(w => w.name === "outfit_set");
            const set = setWidget ? setWidget.value : "general_female";
            navigator.clipboard.writeText(`outfit_lists/${set}/`);
        });
    }
});
