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

    const helpBtn = document.createElement("button");
    helpBtn.textContent = "?";
    helpBtn.title = "Show format reference";
    Object.assign(helpBtn.style, {
        background: "#313244", color: "#89b4fa", border: "1px solid #45475a",
        borderRadius: "50%", width: "24px", height: "24px", fontSize: "14px",
        cursor: "pointer", fontWeight: "bold", lineHeight: "1", padding: "0",
    });

    header.append(title, fileSelect, helpBtn);

    // Help panel (hidden by default)
    const helpPanel = document.createElement("div");
    helpPanel.style.display = "none";
    Object.assign(helpPanel.style, {
        background: "#11111b", border: "1px solid #45475a", borderRadius: "6px",
        padding: "12px", fontSize: "12px", lineHeight: "1.6", color: "#a6adc8",
        maxHeight: "280px", overflowY: "auto",
    });
    helpPanel.innerHTML = `
        <div style="color:#89b4fa;font-weight:bold;margin-bottom:8px">Garment List Format Reference</div>
        <div style="color:#f9e2af;font-family:monospace;margin-bottom:10px">
            garment_name | probability | formality_min-formality_max | fabric1,fabric2,...
        </div>

        <div style="color:#cba6f7;font-weight:bold;margin-top:8px">Probability</div>
        <div>Selection weight when multiple garments match. Higher = picked more often.<br>
        Not a percentage — values are relative. A garment with <b>0.9</b> is 3x more likely
        than one with <b>0.3</b>. Range: any positive number, typically 0.1–1.0.</div>

        <div style="color:#cba6f7;font-weight:bold;margin-top:8px">Formality (min-max)</div>
        <div>Range <b>0.0</b> (very casual) to <b>1.0</b> (very formal).<br>
        A garment is only eligible when the node's formality slider falls within its range.<br>
        <span style="color:#a6e3a1">0.0-0.3</span> = casual only &nbsp;
        <span style="color:#f9e2af">0.3-0.7</span> = mid-range &nbsp;
        <span style="color:#f38ba8">0.7-1.0</span> = formal only<br>
        Wide ranges like <b>0.0-0.8</b> = versatile, narrow like <b>0.8-1.0</b> = specialized.<br>
        Overlap is fine — multiple garments compete via probability.</div>

        <div style="color:#cba6f7;font-weight:bold;margin-top:8px">Fabrics</div>
        <div>Comma-separated list of compatible fabrics for this garment.<br>
        The engine picks one based on formality distance from <b>fabrics.txt</b>.<br>
        Each fabric in fabrics.txt has its own formality score — closer match = preferred.<br>
        Props without fabric (barefoot, candy cane) can leave this empty.</div>

        <div style="color:#cba6f7;font-weight:bold;margin-top:8px">#color# prefix</div>
        <div>Prefix garment names with <b>#color#</b> to insert a color placeholder:<br>
        <span style="color:#a6e3a1;font-family:monospace">#color# sports bra | 0.95 | 0.0-0.1 | spandex</span><br>
        The Color Replace node substitutes #color# with actual color names.</div>

        <div style="color:#cba6f7;font-weight:bold;margin-top:8px">Comments</div>
        <div>Lines starting with <b>#</b> (without a pipe <b>|</b>) are comments — ignored by the parser.</div>

        <div style="color:#cba6f7;font-weight:bold;margin-top:8px">Other file types</div>
        <div style="margin-top:2px">
        <b>fabrics.txt</b>: <span style="font-family:monospace">name | formality | family | weight</span><br>
        <b>prints.txt</b>: <span style="font-family:monospace">name | probability | slot1,slot2 | formality_min-max</span><br>
        <b>texts.txt</b>: <span style="font-family:monospace">"TEXT" | probability | slot1,slot2 | font description</span>
        </div>
    `;
    helpBtn.addEventListener("click", () => {
        helpPanel.style.display = helpPanel.style.display === "none" ? "block" : "none";
    });

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
    dialog.append(header, helpPanel, textarea, status, btnRow);
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
