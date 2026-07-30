/**
 * FVM_JB_OutfitBlock + FVM_JB_LocationBlock — Edit List button + modal.
 *
 * Same UX as the V1 outfit Edit List, generalised to two backends:
 *   - OutfitBlock  → /fvmtools/outfit-list   (set widget: "outfit_set")
 *   - LocationBlock→ /fvmtools/location-list (set widget: "location_set")
 *
 * One modal instance is shared between both node classes (lazy-built on
 * first open). The modal remembers which endpoint to talk to via the
 * config object passed to open().
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

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

    const textarea = document.createElement("textarea");
    Object.assign(textarea.style, {
        background: "#181825", color: "#a6e3a1", border: "1px solid #45475a",
        borderRadius: "6px", padding: "10px", fontSize: "13px", lineHeight: "1.5",
        fontFamily: "Consolas, 'Courier New', monospace",
        flex: "1", minHeight: "400px", resize: "vertical",
        tabSize: "4", whiteSpace: "pre", overflowWrap: "normal", overflowX: "auto",
    });
    textarea.addEventListener("keydown", (e) => {
        if (e.key === "Tab") {
            e.preventDefault();
            const start = textarea.selectionStart;
            const end = textarea.selectionEnd;
            textarea.value = textarea.value.substring(0, start) + "\t" + textarea.value.substring(end);
            textarea.selectionStart = textarea.selectionEnd = start + 1;
        }
    });

    const status = document.createElement("div");
    Object.assign(status.style, { fontSize: "12px", color: "#6c7086", minHeight: "18px" });

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

    let cfg = null;          // { filesEndpoint, listEndpoint, defaultFile }
    let currentSet = "";
    let currentFile = "";
    let dirty = false;

    textarea.addEventListener("input", () => { dirty = true; });

    async function loadFile(set, file) {
        status.textContent = "Loading...";
        status.style.color = "#6c7086";
        try {
            const url = `${cfg.listEndpoint}?set=${encodeURIComponent(set)}&file=${encodeURIComponent(file)}`;
            const resp = await api.fetchApi(url);
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
            const resp = await api.fetchApi(cfg.listEndpoint, {
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
        async open(setName, config) {
            cfg = config;
            overlay.style.display = "flex";
            dirty = false;
            textarea.value = "";
            status.textContent = "Loading file list...";
            status.style.color = "#6c7086";
            try {
                const resp = await api.fetchApi(`${cfg.filesEndpoint}?set=${encodeURIComponent(setName)}`);
                const data = await resp.json();
                if (data.error) { status.textContent = data.error; status.style.color = "#f38ba8"; return; }

                fileSelect.innerHTML = "";
                for (const f of data.files) {
                    const opt = document.createElement("option");
                    opt.value = f;
                    opt.textContent = f + ".txt";
                    fileSelect.append(opt);
                }
                const defaultFile = data.files.includes(cfg.defaultFile) ? cfg.defaultFile : data.files[0];
                if (defaultFile) {
                    fileSelect.value = defaultFile;
                    await loadFile(setName, defaultFile);
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

/* ── Override Editor Modal (Outfit Block) ──────────────────────────────
 *
 * Structured editor for the `overrides` widget: one row per slot
 * (auto / custom / exclude, garment, fabric, colour role, decoration)
 * plus a palette section that forces the colours behind the roles
 * (primary=navy blue, ...). Serialises to the same plain-text grammar
 * core/outfit_parser.py::parse_overrides understands, so hand-written
 * override text round-trips through the editor.
 */

const OV_SLOTS = ["headwear", "top", "outerwear", "bottom", "footwear",
                  "accessories", "bag"];
const OV_ROLES = ["primary", "secondary", "accent", "neutral", "metallic",
                  "tertiary"];
const OV_PALETTE_KEYS = [...OV_ROLES, "ambient_light", "shadow_tone"];

const OV_TIPS = {
    mode:
        "auto — the engine picks this slot from the set's list files " +
        "(default, writes nothing).\n" +
        "custom — force the garment below; the slot is activated even if " +
        "its probability roll would have skipped it.\n" +
        "exclude — the slot never appears, even when its enable toggle " +
        "is on.",
    fabric:
        "Optional fabric written in front of the garment, e.g. 'silk'.\n" +
        "Grammar note: in the text form the FIRST word before the pipe is " +
        "read as the fabric whenever two or more words are present — this " +
        "field just makes that explicit.\n" +
        "Leave empty to name no fabric.",
    garment:
        "The garment name exactly as it should appear in the prompt, " +
        "e.g. 'wrap dress' or 'band tee'.\n" +
        "Wildcards pick randomly per seed: '{silk|satin} blouse'.\n" +
        "Names that already contain a colour ('white tennis shoes') are " +
        "not given a second colour by the engine.",
    role:
        "Which palette colour this garment wears.\n" +
        "(default) keeps the slot's built-in role: top=primary, " +
        "bottom/outerwear=secondary, footwear=neutral, " +
        "accessories=metallic, headwear/bag=accent.\n" +
        "The actual colour behind each role comes from the colour mood — " +
        "or from the Palette section below if you force it there.",
    decoration:
        "Optional print/text on the garment, e.g. 'floral print' or " +
        "'thin stripes'.\n" +
        "'none' forces the garment plain even when print_probability " +
        "would have added a pattern. Empty = engine decides.",
    palette:
        "Force the actual colour behind a role for THIS node's palette. " +
        "Every garment using that role follows automatically.\n" +
        "Any colour phrase works: 'navy blue', 'washed-out sage', " +
        "'charcoal grey'.\n" +
        "Empty fields keep the colour generated by the colour mood / " +
        "harmony engine.",
    ambient:
        "Atmosphere phrases used by location fragments and #ambient_light# / " +
        "#shadow_tone# tokens. Usually left empty — the warmth slider " +
        "already tints them.",
};

function createOverrideModal() {
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        display: "none", position: "fixed", inset: "0",
        background: "rgba(0,0,0,0.6)", zIndex: "10000",
        justifyContent: "center", alignItems: "center",
    });

    const dialog = document.createElement("div");
    Object.assign(dialog.style, {
        background: "#1e1e2e", color: "#cdd6f4", borderRadius: "10px",
        padding: "16px", width: "780px", maxHeight: "90vh", overflowY: "auto",
        display: "flex", flexDirection: "column", gap: "12px",
        boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
        fontFamily: "system-ui, sans-serif", fontSize: "13px",
    });

    // ── Header ──
    const header = document.createElement("div");
    Object.assign(header.style, { display: "flex", alignItems: "center", gap: "10px" });
    const title = document.createElement("span");
    title.textContent = "Outfit Overrides";
    Object.assign(title.style, { fontWeight: "bold", fontSize: "15px", flex: "1" });
    const helpBtn = document.createElement("button");
    helpBtn.textContent = "?";
    helpBtn.title = "Show the full override syntax reference";
    Object.assign(helpBtn.style, {
        background: "#313244", color: "#89b4fa", border: "1px solid #45475a",
        borderRadius: "50%", width: "24px", height: "24px", fontSize: "14px",
        cursor: "pointer", fontWeight: "bold", lineHeight: "1", padding: "0",
    });
    header.append(title, helpBtn);

    // ── Help panel ──
    const helpPanel = document.createElement("div");
    helpPanel.style.display = "none";
    Object.assign(helpPanel.style, {
        background: "#11111b", border: "1px solid #45475a", borderRadius: "6px",
        padding: "12px", fontSize: "12px", lineHeight: "1.6", color: "#a6adc8",
    });
    helpPanel.innerHTML = `
        <div style="color:#89b4fa;font-weight:bold;margin-bottom:6px">How overrides work</div>
        <div>Everything here compiles into the <b>overrides</b> text widget on the node —
        one directive per line. You can always type that text by hand; this editor just
        writes it for you. Empty rows / fields write nothing and keep the engine's choice.</div>

        <div style="color:#cba6f7;font-weight:bold;margin-top:10px">Slot lines</div>
        <div style="font-family:monospace;color:#f9e2af">
        top: silk blouse<br>
        top: silk blouse | accent<br>
        top: silk blouse | accent | floral print<br>
        top: {silk|satin} blouse&nbsp;&nbsp;&nbsp;# wildcard, picked per seed<br>
        bag: exclude</div>
        <div>Fields after the garment: <b>colour role</b>, then <b>decoration</b>.
        A forced slot is always active, even when its enable toggle would have rolled it away.
        With two or more words the first word is read as the fabric.</div>

        <div style="color:#cba6f7;font-weight:bold;margin-top:10px">Palette line</div>
        <div style="font-family:monospace;color:#f9e2af">
        palette: primary=navy blue, secondary=cream, accent=burnt orange</div>
        <div>Forces the actual colours behind the roles for this node. Garments keep their
        roles (top=primary, bottom=secondary, …) — you swap what the role resolves to.
        Roles you don't list keep the colour generated by the colour mood.
        Also accepted: <b>ambient_light</b> and <b>shadow_tone</b> for the atmosphere phrases.</div>

        <div style="color:#cba6f7;font-weight:bold;margin-top:10px">Colour roles</div>
        <div><b>primary</b> tops · <b>secondary</b> bottoms &amp; outerwear ·
        <b>neutral</b> footwear · <b>accent</b> headwear &amp; bags ·
        <b>metallic</b> jewellery · <b>tertiary</b> spare.
        The engine skips colours on garments that name their own
        ("white tennis shoes") and on colourless entries ("bare feet").</div>
    `;
    helpBtn.addEventListener("click", () => {
        helpPanel.style.display = helpPanel.style.display === "none" ? "block" : "none";
    });

    const sectionTitle = (text, tip) => {
        const el = document.createElement("div");
        el.textContent = text;
        if (tip) el.title = tip;
        Object.assign(el.style, { fontWeight: "bold", color: "#89b4fa", marginTop: "2px" });
        return el;
    };

    const inputStyle = {
        background: "#181825", color: "#cdd6f4", border: "1px solid #45475a",
        borderRadius: "4px", padding: "3px 6px", fontSize: "12px",
    };

    // ── Slot rows ──
    const slotGrid = document.createElement("div");
    Object.assign(slotGrid.style, {
        display: "grid",
        gridTemplateColumns: "84px 82px 90px 1fr 100px 1fr",
        gap: "4px 6px", alignItems: "center",
    });
    const headRow = ["slot", "mode", "fabric", "garment", "colour role", "decoration"];
    const headTips = [null, OV_TIPS.mode, OV_TIPS.fabric, OV_TIPS.garment,
                      OV_TIPS.role, OV_TIPS.decoration];
    headRow.forEach((h, i) => {
        const el = document.createElement("div");
        el.textContent = h;
        if (headTips[i]) el.title = headTips[i];
        Object.assign(el.style, { color: "#6c7086", fontSize: "11px", cursor: headTips[i] ? "help" : "default" });
        slotGrid.append(el);
    });

    const slotRows = {};
    for (const slot of OV_SLOTS) {
        const label = document.createElement("div");
        label.textContent = slot;
        label.style.color = "#cdd6f4";

        const mode = document.createElement("select");
        mode.title = OV_TIPS.mode;
        for (const m of ["auto", "custom", "exclude"]) {
            const opt = document.createElement("option");
            opt.value = m; opt.textContent = m;
            mode.append(opt);
        }
        Object.assign(mode.style, { ...inputStyle, cursor: "pointer" });

        const fabric = document.createElement("input");
        fabric.placeholder = "silk";
        fabric.title = OV_TIPS.fabric;
        Object.assign(fabric.style, inputStyle);

        const garment = document.createElement("input");
        garment.placeholder = "wrap dress";
        garment.title = OV_TIPS.garment;
        Object.assign(garment.style, inputStyle);

        const role = document.createElement("select");
        role.title = OV_TIPS.role;
        const defOpt = document.createElement("option");
        defOpt.value = ""; defOpt.textContent = "(default)";
        role.append(defOpt);
        for (const r of OV_ROLES) {
            const opt = document.createElement("option");
            opt.value = r; opt.textContent = r;
            role.append(opt);
        }
        Object.assign(role.style, { ...inputStyle, cursor: "pointer" });

        const deco = document.createElement("input");
        deco.placeholder = "floral print / none";
        deco.title = OV_TIPS.decoration;
        Object.assign(deco.style, inputStyle);

        const syncEnabled = () => {
            const custom = mode.value === "custom";
            for (const el of [fabric, garment, role, deco]) {
                el.disabled = !custom;
                el.style.opacity = custom ? "1" : "0.35";
            }
        };
        mode.addEventListener("change", syncEnabled);
        syncEnabled();

        slotGrid.append(label, mode, fabric, garment, role, deco);
        slotRows[slot] = { mode, fabric, garment, role, deco, syncEnabled };
    }

    // ── Palette section ──
    const palGrid = document.createElement("div");
    Object.assign(palGrid.style, {
        display: "grid", gridTemplateColumns: "repeat(4, 90px 1fr)",
        gap: "4px 6px", alignItems: "center",
    });
    const palInputs = {};
    for (const key of OV_PALETTE_KEYS) {
        const label = document.createElement("div");
        label.textContent = key.replace("_", " ");
        Object.assign(label.style, { color: "#6c7086", fontSize: "11px", cursor: "help" });
        label.title = key.startsWith("ambient") || key.startsWith("shadow")
            ? OV_TIPS.ambient : OV_TIPS.palette;
        const input = document.createElement("input");
        input.placeholder = "(generated)";
        input.title = label.title;
        Object.assign(input.style, inputStyle);
        palGrid.append(label, input);
        palInputs[key] = input;
    }

    // ── Footer ──
    const status = document.createElement("div");
    Object.assign(status.style, { fontSize: "12px", color: "#6c7086", minHeight: "16px" });

    const btnRow = document.createElement("div");
    Object.assign(btnRow.style, { display: "flex", gap: "8px", justifyContent: "flex-end" });
    const btnStyle = {
        padding: "6px 18px", borderRadius: "6px", border: "none",
        fontSize: "13px", cursor: "pointer", fontWeight: "bold",
    };
    const applyBtn = document.createElement("button");
    applyBtn.textContent = "Apply";
    applyBtn.title = "Write these overrides into the node's overrides widget";
    Object.assign(applyBtn.style, { ...btnStyle, background: "#a6e3a1", color: "#1e1e2e" });
    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Cancel";
    Object.assign(cancelBtn.style, { ...btnStyle, background: "#45475a", color: "#cdd6f4" });
    btnRow.append(applyBtn, cancelBtn);

    dialog.append(
        header, helpPanel,
        sectionTitle("Slots", "Force, exclude or leave each slot on auto"),
        slotGrid,
        sectionTitle("Palette colours",
                     "Force the colour behind each role; empty = generated by the colour mood"),
        palGrid,
        status, btnRow,
    );
    overlay.append(dialog);
    document.body.append(overlay);

    let targetWidget = null;
    let targetNode = null;

    // Parse the widget text into the UI (mirror of parse_overrides).
    function loadFromText(text) {
        for (const slot of OV_SLOTS) {
            const r = slotRows[slot];
            r.mode.value = "auto";
            r.fabric.value = ""; r.garment.value = "";
            r.role.value = ""; r.deco.value = "";
            r.syncEnabled();
        }
        for (const key of OV_PALETTE_KEYS) palInputs[key].value = "";

        for (let line of (text || "").split("\n")) {
            line = line.trim();
            if (!line || line.startsWith("#") || !line.includes(":")) continue;
            const idx = line.indexOf(":");
            const slot = line.slice(0, idx).trim().toLowerCase();
            const spec = line.slice(idx + 1).trim();
            if (!spec) continue;

            if (["palette", "colors", "colours"].includes(slot)) {
                for (const pair of spec.split(",")) {
                    const eq = pair.indexOf("=");
                    if (eq < 0) continue;
                    const role = pair.slice(0, eq).trim().toLowerCase().replace(/#/g, "");
                    const value = pair.slice(eq + 1).trim();
                    if (palInputs[role] && value) palInputs[role].value = value;
                }
                continue;
            }
            const row = slotRows[slot];
            if (!row) continue;
            const low = spec.toLowerCase();
            if (low === "exclude") { row.mode.value = "exclude"; row.syncEnabled(); continue; }
            if (low === "auto")    { row.mode.value = "auto";    row.syncEnabled(); continue; }

            row.mode.value = "custom";
            const parts = spec.split("|").map(p => p.trim());
            const words = parts[0].split(/\s+/).filter(Boolean);
            if (words.length >= 2) {
                row.fabric.value = words[0];
                row.garment.value = words.slice(1).join(" ");
            } else {
                row.garment.value = parts[0];
            }
            if (parts[1]) row.role.value = parts[1].replace(/#/g, "").toLowerCase();
            if (parts[2]) row.deco.value = parts[2];
            row.syncEnabled();
        }
    }

    function serialize() {
        const lines = [];
        for (const slot of OV_SLOTS) {
            const r = slotRows[slot];
            if (r.mode.value === "exclude") {
                lines.push(`${slot}: exclude`);
                continue;
            }
            if (r.mode.value !== "custom") continue;
            const garment = r.garment.value.trim();
            if (!garment) continue;
            let spec = r.fabric.value.trim()
                ? `${r.fabric.value.trim()} ${garment}` : garment;
            const role = r.role.value;
            const deco = r.deco.value.trim();
            if (deco)      spec += ` | ${role} | ${deco}`;
            else if (role) spec += ` | ${role}`;
            lines.push(`${slot}: ${spec}`);
        }
        const pal = OV_PALETTE_KEYS
            .filter(k => palInputs[k].value.trim())
            .map(k => `${k}=${palInputs[k].value.trim()}`);
        if (pal.length) lines.push(`palette: ${pal.join(", ")}`);
        return lines.join("\n");
    }

    applyBtn.addEventListener("click", () => {
        if (!targetWidget) return;
        targetWidget.value = serialize();
        targetWidget.callback?.(targetWidget.value);
        targetNode?.graph?.setDirtyCanvas(true, true);
        overlay.style.display = "none";
    });
    cancelBtn.addEventListener("click", () => { overlay.style.display = "none"; });
    overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.style.display = "none"; });

    return {
        open(node, widget) {
            targetNode = node;
            targetWidget = widget;
            loadFromText(widget ? widget.value : "");
            status.textContent = "Empty fields keep the engine's choice. " +
                "Apply writes into the overrides widget.";
            overlay.style.display = "flex";
        },
    };
}

let modal = null;
let overrideModal = null;

const NODE_CONFIGS = {
    "FVM_JB_OutfitBlock": {
        setWidget:      "outfit_set",
        filesEndpoint:  "/fvmtools/outfit-files",
        listEndpoint:   "/fvmtools/outfit-list",
        defaultFile:    "top",
    },
    "FVM_JB_LocationBlock": {
        setWidget:      "location_set",
        filesEndpoint:  "/fvmtools/location-files",
        listEndpoint:   "/fvmtools/location-list",
        defaultFile:    "background",
    },
};

app.registerExtension({
    name: "FVMTools.JB.Blocks",
    async nodeCreated(node) {
        const cfg = NODE_CONFIGS[node.comfyClass];
        if (!cfg) return;

        node.addWidget("button", "Edit List", null, () => {
            if (!modal) modal = createEditorModal();
            const w = node.widgets.find(x => x.name === cfg.setWidget);
            const setName = w ? w.value : "";
            if (!setName) return;
            modal.open(setName, {
                filesEndpoint: cfg.filesEndpoint,
                listEndpoint:  cfg.listEndpoint,
                defaultFile:   cfg.defaultFile,
            });
        });

        if (node.comfyClass === "FVM_JB_OutfitBlock") {
            node.addWidget("button", "Edit Overrides", null, () => {
                if (!overrideModal) overrideModal = createOverrideModal();
                const w = node.widgets.find(x => x.name === "overrides");
                overrideModal.open(node, w);
            });
        }
    },
});
