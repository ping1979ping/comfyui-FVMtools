import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/* ── Outfit List Editor Modal ── */

function createEditorModal() {
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        display: "none", position: "fixed", inset: "0",
        background: "rgba(0,0,0,0.6)", zIndex: "10000",
        justifyContent: "center", alignItems: "center",
    });

    const dialog = document.createElement("div");
    Object.assign(dialog.style, {
        background: "#1e1e2e", color: "#cdd6f4", borderRadius: "10px",
        padding: "16px", width: "660px", maxHeight: "85vh",
        display: "flex", flexDirection: "column", gap: "10px",
        boxShadow: "0 8px 32px rgba(0,0,0,0.5)", fontFamily: "monospace",
    });

    // Header row
    const header = document.createElement("div");
    Object.assign(header.style, { display: "flex", alignItems: "center", gap: "10px" });

    const title = document.createElement("span");
    title.textContent = "Edit List";
    Object.assign(title.style, { fontWeight: "bold", fontSize: "14px", flex: "1" });

    const fileSelect = document.createElement("select");
    Object.assign(fileSelect.style, {
        background: "#313244", color: "#cdd6f4", border: "1px solid #45475a",
        borderRadius: "6px", padding: "4px 8px", fontSize: "13px", cursor: "pointer",
    });

    header.append(title, fileSelect);

    // Textarea
    const textarea = document.createElement("textarea");
    Object.assign(textarea.style, {
        background: "#181825", color: "#a6e3a1", border: "1px solid #45475a",
        borderRadius: "6px", padding: "10px", fontSize: "13px", lineHeight: "1.5",
        fontFamily: "Consolas, 'Courier New', monospace",
        flex: "1", minHeight: "400px", resize: "vertical",
        tabSize: "4", whiteSpace: "pre", overflowWrap: "normal", overflowX: "auto",
    });
    // Allow Tab key in textarea
    textarea.addEventListener("keydown", (e) => {
        if (e.key === "Tab") {
            e.preventDefault();
            const start = textarea.selectionStart;
            const end = textarea.selectionEnd;
            textarea.value = textarea.value.substring(0, start) + "\t" + textarea.value.substring(end);
            textarea.selectionStart = textarea.selectionEnd = start + 1;
        }
    });

    // Status bar
    const status = document.createElement("div");
    Object.assign(status.style, {
        fontSize: "12px", color: "#6c7086", minHeight: "18px",
    });

    // Button row
    const btnRow = document.createElement("div");
    Object.assign(btnRow.style, { display: "flex", gap: "8px", justifyContent: "flex-end" });

    const btnStyle = {
        padding: "6px 18px", borderRadius: "6px", border: "none",
        fontSize: "13px", cursor: "pointer", fontWeight: "bold",
    };

    const saveBtn = document.createElement("button");
    saveBtn.textContent = "Save";
    Object.assign(saveBtn.style, { ...btnStyle, background: "#a6e3a1", color: "#1e1e2e" });

    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Close";
    Object.assign(cancelBtn.style, { ...btnStyle, background: "#45475a", color: "#cdd6f4" });

    btnRow.append(saveBtn, cancelBtn);
    dialog.append(header, textarea, status, btnRow);
    overlay.append(dialog);
    document.body.append(overlay);

    // State
    let currentSet = "";
    let currentFile = "";
    let dirty = false;

    textarea.addEventListener("input", () => { dirty = true; });

    async function loadFile(set, file) {
        status.textContent = "Loading...";
        status.style.color = "#6c7086";
        try {
            const resp = await api.fetchApi(`/fvmtools/outfit-list?set=${encodeURIComponent(set)}&file=${encodeURIComponent(file)}`);
            const data = await resp.json();
            if (data.error) { status.textContent = data.error; status.style.color = "#f38ba8"; return; }
            textarea.value = data.content;
            currentSet = set;
            currentFile = file;
            dirty = false;
            status.textContent = data.path;
            status.style.color = "#6c7086";
        } catch (e) {
            status.textContent = "Error: " + e.message;
            status.style.color = "#f38ba8";
        }
    }

    async function saveFile() {
        if (!currentSet || !currentFile) return;
        status.textContent = "Saving...";
        status.style.color = "#f9e2af";
        try {
            const resp = await api.fetchApi("/fvmtools/outfit-list", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ set: currentSet, file: currentFile, content: textarea.value }),
            });
            const data = await resp.json();
            if (data.success) {
                dirty = false;
                status.textContent = "Saved! Changes active on next Queue.";
                status.style.color = "#a6e3a1";
            } else {
                status.textContent = "Save failed: " + (data.error || "unknown");
                status.style.color = "#f38ba8";
            }
        } catch (e) {
            status.textContent = "Error: " + e.message;
            status.style.color = "#f38ba8";
        }
    }

    function close() {
        if (dirty && !confirm("Unsaved changes — discard?")) return;
        overlay.style.display = "none";
    }

    saveBtn.addEventListener("click", saveFile);
    cancelBtn.addEventListener("click", close);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    // Ctrl+S to save
    textarea.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "s") {
            e.preventDefault();
            saveFile();
        }
    });

    fileSelect.addEventListener("change", async () => {
        if (dirty && !confirm("Unsaved changes — discard?")) {
            fileSelect.value = currentFile;
            return;
        }
        await loadFile(currentSet, fileSelect.value);
    });

    return {
        async open(outfitSet) {
            overlay.style.display = "flex";
            dirty = false;
            textarea.value = "";
            status.textContent = "Loading file list...";
            status.style.color = "#6c7086";

            try {
                const resp = await api.fetchApi(`/fvmtools/outfit-files?set=${encodeURIComponent(outfitSet)}`);
                const data = await resp.json();
                if (data.error) { status.textContent = data.error; status.style.color = "#f38ba8"; return; }

                fileSelect.innerHTML = "";
                for (const f of data.files) {
                    const opt = document.createElement("option");
                    opt.value = f;
                    opt.textContent = f + ".txt";
                    fileSelect.append(opt);
                }
                // Default to "top" if available, else first file
                const defaultFile = data.files.includes("top") ? "top" : data.files[0];
                if (defaultFile) {
                    fileSelect.value = defaultFile;
                    await loadFile(outfitSet, defaultFile);
                } else {
                    status.textContent = "No .txt files in this set";
                    status.style.color = "#f9e2af";
                }
            } catch (e) {
                status.textContent = "Error: " + e.message;
                status.style.color = "#f38ba8";
            }
        },
    };
}

let editorModal = null;

/* ── Node Extension ── */

app.registerExtension({
    name: "FVM.OutfitGenerator",
    async nodeCreated(node) {
        if (node.comfyClass !== "FVM_OutfitGenerator") return;

        node.addWidget("button", "Copy Override Template", null, () => {
            navigator.clipboard.writeText(
                "headwear: auto\ntop: auto\nouterwear: auto\nbottom: auto\nfootwear: auto\naccessories: auto\nbag: auto"
            );
        });

        node.addWidget("button", "Edit List", null, () => {
            if (!editorModal) editorModal = createEditorModal();
            const setWidget = node.widgets.find(w => w.name === "outfit_set");
            const set = setWidget ? setWidget.value : "general_female";
            editorModal.open(set);
        });
    }
});
