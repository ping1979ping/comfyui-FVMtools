/**
 * FVM_JB_Builder — custom row widget + catalog browser + Edit Catalog modal.
 *
 * UX:
 *   ┌──────────────────────────────────────────────────────────────────┐
 *   │ [+ Add Row] [Insert From Catalog ▾] [Edit Catalog]               │
 *   ├──────────────────────────────────────────────────────────────────┤
 *   │ │ [hosiery        ] [                                ] [⋮]       │  ← indent 0
 *   │ │ │ [type         ] [sheer black stockings, ...      ] [⋮]       │  ← indent 1
 *   │ │ │ [opacity      ] [semi-sheer (20-30 denier)       ] [⋮]       │  ← indent 1
 *   │ │ │ [details      ] [smooth texture                  ] [⋮]       │  ← indent 1
 *   └──────────────────────────────────────────────────────────────────┘
 *
 * State:
 *   - rows array lives in JS memory.
 *   - On every change, serialised as JSON into the hidden `rows` widget
 *     (a STRING widget the Python side reads from). Survives workflow
 *     save/load like any other widget value.
 *
 * Context-menu (⋮) actions: Move Up, Move Down, Indent Right, Indent Left,
 * Delete, Duplicate.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "FVM_JB_Builder";
const INDENT_PX = 18;
const ROW_HEIGHT = 28;
const PADDING = 6;

// ─── catalog editor modal (lazy) ─────────────────────────────────────

function createCatalogModal() {
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        display: "none", position: "fixed", inset: "0",
        background: "rgba(0,0,0,0.6)", zIndex: "10000",
        justifyContent: "center", alignItems: "center",
    });

    const dialog = document.createElement("div");
    Object.assign(dialog.style, {
        background: "#1e1e2e", color: "#cdd6f4", borderRadius: "10px",
        padding: "16px", width: "720px", maxHeight: "85vh",
        display: "flex", flexDirection: "column", gap: "10px",
        boxShadow: "0 8px 32px rgba(0,0,0,0.5)", fontFamily: "monospace",
    });

    const header = document.createElement("div");
    Object.assign(header.style, { display: "flex", alignItems: "center", gap: "8px" });
    const title = document.createElement("span");
    title.textContent = "Edit Catalog";
    Object.assign(title.style, { fontWeight: "bold", fontSize: "14px", flex: "1" });
    const catSelect = document.createElement("select");
    const entrySelect = document.createElement("select");
    for (const sel of [catSelect, entrySelect]) {
        Object.assign(sel.style, {
            background: "#313244", color: "#cdd6f4", border: "1px solid #45475a",
            borderRadius: "6px", padding: "4px 8px", fontSize: "13px",
        });
    }
    const newBtn = document.createElement("button");
    newBtn.textContent = "+ New";
    Object.assign(newBtn.style, {
        background: "#313244", color: "#a6e3a1", border: "1px solid #45475a",
        borderRadius: "6px", padding: "4px 10px", fontSize: "13px", cursor: "pointer",
    });
    const delBtn = document.createElement("button");
    delBtn.textContent = "Delete";
    Object.assign(delBtn.style, {
        background: "#313244", color: "#f38ba8", border: "1px solid #45475a",
        borderRadius: "6px", padding: "4px 10px", fontSize: "13px", cursor: "pointer",
    });
    header.append(title, catSelect, entrySelect, newBtn, delBtn);

    const textarea = document.createElement("textarea");
    Object.assign(textarea.style, {
        background: "#181825", color: "#a6e3a1", border: "1px solid #45475a",
        borderRadius: "6px", padding: "10px", fontSize: "13px",
        fontFamily: "Consolas, 'Courier New', monospace",
        flex: "1", minHeight: "360px", resize: "vertical",
    });
    textarea.addEventListener("keydown", (e) => {
        if (e.key === "Tab") {
            e.preventDefault();
            const s = textarea.selectionStart, t = textarea.selectionEnd;
            textarea.value = textarea.value.substring(0, s) + "  " + textarea.value.substring(t);
            textarea.selectionStart = textarea.selectionEnd = s + 2;
        }
    });

    const status = document.createElement("div");
    Object.assign(status.style, { fontSize: "12px", color: "#6c7086", minHeight: "18px" });

    const btnRow = document.createElement("div");
    Object.assign(btnRow.style, { display: "flex", gap: "8px", justifyContent: "flex-end" });
    const saveBtn = document.createElement("button");
    saveBtn.textContent = "Save";
    Object.assign(saveBtn.style, {
        padding: "6px 18px", borderRadius: "6px", border: "none",
        background: "#a6e3a1", color: "#1e1e2e", fontSize: "13px",
        cursor: "pointer", fontWeight: "bold",
    });
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "Close";
    Object.assign(closeBtn.style, {
        padding: "6px 18px", borderRadius: "6px", border: "none",
        background: "#45475a", color: "#cdd6f4", fontSize: "13px",
        cursor: "pointer", fontWeight: "bold",
    });
    btnRow.append(saveBtn, closeBtn);

    dialog.append(header, textarea, status, btnRow);
    overlay.append(dialog);
    document.body.append(overlay);

    let dirty = false;
    let onChangedCb = null;
    textarea.addEventListener("input", () => { dirty = true; });

    async function loadCatalog() {
        const resp = await api.fetchApi("/fvmtools/jb-catalog");
        const data = await resp.json();
        return data.catalog || {};
    }

    async function refreshCategories() {
        const cat = await loadCatalog();
        catSelect.innerHTML = "";
        for (const name of Object.keys(cat).sort()) {
            const opt = document.createElement("option");
            opt.value = name; opt.textContent = name;
            catSelect.append(opt);
        }
        await refreshEntries(cat);
    }

    async function refreshEntries(cachedCat) {
        const cat = cachedCat || (await loadCatalog());
        const entries = cat[catSelect.value] || [];
        entrySelect.innerHTML = "";
        if (entries.length === 0) {
            const opt = document.createElement("option");
            opt.value = ""; opt.textContent = "(empty)";
            entrySelect.append(opt);
            textarea.value = "";
            status.textContent = "No entries in this category — click + New.";
            return;
        }
        for (const name of entries) {
            const opt = document.createElement("option");
            opt.value = name; opt.textContent = name;
            entrySelect.append(opt);
        }
        await loadEntry();
    }

    async function loadEntry() {
        if (!entrySelect.value) return;
        const url = `/fvmtools/jb-catalog-entry?category=${encodeURIComponent(catSelect.value)}&name=${encodeURIComponent(entrySelect.value)}`;
        const resp = await api.fetchApi(url);
        const data = await resp.json();
        if (data.error) { status.textContent = data.error; return; }
        textarea.value = JSON.stringify(data.data, null, 2);
        dirty = false;
        status.textContent = `${catSelect.value}/${entrySelect.value}.json`;
        status.style.color = "#6c7086";
    }

    async function saveEntry() {
        if (!entrySelect.value) return;
        let parsed;
        try {
            parsed = JSON.parse(textarea.value);
        } catch (e) {
            status.textContent = "Invalid JSON: " + e.message;
            status.style.color = "#f38ba8";
            return;
        }
        status.textContent = "Saving...";
        status.style.color = "#f9e2af";
        const resp = await api.fetchApi("/fvmtools/jb-catalog-entry", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                category: catSelect.value,
                name:     entrySelect.value,
                data:     parsed,
            }),
        });
        const data = await resp.json();
        if (data.success) {
            dirty = false;
            status.textContent = "Saved.";
            status.style.color = "#a6e3a1";
            if (onChangedCb) onChangedCb();
        } else {
            status.textContent = "Save failed: " + (data.error || "unknown");
            status.style.color = "#f38ba8";
        }
    }

    async function newEntry() {
        const name = prompt("New entry name (lowercase + underscores):");
        if (!name || !/^[a-z0-9_]+$/.test(name)) {
            if (name) alert("Use only lowercase letters, digits, and underscores.");
            return;
        }
        const resp = await api.fetchApi("/fvmtools/jb-catalog-entry", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                category: catSelect.value,
                name:     name,
                data:     { [name]: { description: "" } },
            }),
        });
        const data = await resp.json();
        if (data.success) {
            await refreshEntries();
            entrySelect.value = name;
            await loadEntry();
            if (onChangedCb) onChangedCb();
        } else {
            alert("Failed: " + (data.error || "unknown"));
        }
    }

    async function deleteEntry() {
        if (!entrySelect.value) return;
        if (!confirm(`Delete ${catSelect.value}/${entrySelect.value}.json?`)) return;
        const url = `/fvmtools/jb-catalog-entry?category=${encodeURIComponent(catSelect.value)}&name=${encodeURIComponent(entrySelect.value)}`;
        await api.fetchApi(url, { method: "DELETE" });
        await refreshEntries();
        if (onChangedCb) onChangedCb();
    }

    catSelect.addEventListener("change", () => refreshEntries());
    entrySelect.addEventListener("change", () => loadEntry());
    saveBtn.addEventListener("click", saveEntry);
    newBtn.addEventListener("click", newEntry);
    delBtn.addEventListener("click", deleteEntry);
    closeBtn.addEventListener("click", () => {
        if (dirty && !confirm("Unsaved changes — discard?")) return;
        overlay.style.display = "none";
    });
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) closeBtn.click();
    });

    return {
        async open(onChanged) {
            onChangedCb = onChanged || null;
            overlay.style.display = "flex";
            await refreshCategories();
        },
    };
}

let catalogModal = null;
let catalogCache = null;
async function getCatalog() {
    if (catalogCache) return catalogCache;
    const resp = await api.fetchApi("/fvmtools/jb-catalog");
    const data = await resp.json();
    catalogCache = data.catalog || {};
    return catalogCache;
}
function invalidateCatalog() { catalogCache = null; }

let wildcardsModal = null;
let wildcardsCache = null;
let syntaxInfoModal = null;
async function getWildcards() {
    if (wildcardsCache) return wildcardsCache;
    const resp = await api.fetchApi("/fvmtools/jb-wildcards");
    const data = await resp.json();
    wildcardsCache = Array.isArray(data.wildcards) ? data.wildcards : [];
    return wildcardsCache;
}
function invalidateWildcards() { wildcardsCache = null; }

// ─── wildcard autocomplete (attaches to a row's <input> value field) ─

/**
 * Watch an input element and surface a floating list of wildcard names
 * whenever the caret sits inside an unclosed ``__partial`` token. The
 * user navigates with ↑/↓, accepts with Enter/Tab (the partial expands
 * to ``__name__``), and dismisses with Escape or by clicking away.
 */
function attachWildcardAutocomplete(input) {
    let popup = null;
    let items = [];
    let activeIdx = 0;

    function getQuery() {
        const v = input.value;
        const caret = input.selectionStart ?? v.length;
        const left = v.substring(0, caret);
        const open = left.lastIndexOf("__");
        if (open < 0) return null;
        const after = left.substring(open + 2);
        // If there's a closing pair between the opener and the caret,
        // we're past the wildcard — no autocomplete.
        if (after.includes("__")) return null;
        // Only suggest while the partial is path-like (or empty).
        if (after && !/^[a-zA-Z0-9_\-/]*$/.test(after)) return null;
        return { start: open, query: after };
    }

    function close() {
        if (popup && popup.parentNode) popup.parentNode.removeChild(popup);
        popup = null;
        items = [];
        activeIdx = 0;
    }

    function render() {
        if (!popup) {
            popup = document.createElement("div");
            Object.assign(popup.style, {
                position: "fixed", background: "#1e1e2e", color: "#cdd6f4",
                border: "1px solid #45475a", borderRadius: "6px",
                padding: "4px 0", zIndex: "9999",
                boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
                maxHeight: "240px", overflowY: "auto", minWidth: "220px",
                fontFamily: "Consolas, monospace", fontSize: "12px",
            });
            document.body.append(popup);
        }
        const r = input.getBoundingClientRect();
        popup.style.left = r.left + "px";
        popup.style.top  = (r.bottom + 2) + "px";
        if (activeIdx < 0) activeIdx = 0;
        if (activeIdx >= items.length) activeIdx = items.length - 1;
        popup.innerHTML = "";
        items.forEach((name, i) => {
            const el = document.createElement("div");
            el.textContent = name;
            Object.assign(el.style, {
                padding: "4px 12px", cursor: "pointer", whiteSpace: "nowrap",
                background: i === activeIdx ? "#313244" : "transparent",
                color:      i === activeIdx ? "#a6e3a1" : "#cdd6f4",
            });
            // mousedown (not click) so the input doesn't blur first.
            el.addEventListener("mousedown", (ev) => {
                ev.preventDefault();
                accept(i);
            });
            popup.append(el);
        });
    }

    async function update() {
        const q = getQuery();
        if (!q) { close(); return; }
        const all = await getWildcards();
        const ql = q.query.toLowerCase();
        items = ql
            ? all.filter(n => n.toLowerCase().includes(ql)).slice(0, 100)
            : all.slice(0, 100);
        if (items.length === 0) { close(); return; }
        // Prefer prefix matches at the top of the list.
        items.sort((a, b) => {
            const ap = a.toLowerCase().startsWith(ql) ? 0 : 1;
            const bp = b.toLowerCase().startsWith(ql) ? 0 : 1;
            if (ap !== bp) return ap - bp;
            return a.localeCompare(b);
        });
        activeIdx = 0;
        render();
    }

    function accept(idx) {
        const q = getQuery();
        if (!q || idx < 0 || idx >= items.length) { close(); return; }
        const name = items[idx];
        const v = input.value;
        const caret = input.selectionStart ?? v.length;
        const before = v.substring(0, q.start);
        const after  = v.substring(caret);
        const ins = `__${name}__`;
        input.value = before + ins + after;
        const newCaret = (before + ins).length;
        input.selectionStart = input.selectionEnd = newCaret;
        input.dispatchEvent(new Event("input", { bubbles: true }));
        close();
    }

    input.addEventListener("input", update);
    input.addEventListener("keydown", (e) => {
        if (!popup || items.length === 0) return;
        if (e.key === "ArrowDown") {
            e.preventDefault();
            activeIdx = (activeIdx + 1) % items.length;
            render();
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            activeIdx = (activeIdx - 1 + items.length) % items.length;
            render();
        } else if (e.key === "Enter" || e.key === "Tab") {
            e.preventDefault();
            accept(activeIdx);
        } else if (e.key === "Escape") {
            e.preventDefault();
            close();
        }
    });
    // blur fires before click; defer close so a popup mousedown still wins.
    input.addEventListener("blur", () => setTimeout(close, 120));
}

// ─── advanced-prompt syntax reference modal (lazy) ──────────────────

function createSyntaxInfoModal() {
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        display: "none", position: "fixed", inset: "0",
        background: "rgba(0,0,0,0.6)", zIndex: "10000",
        justifyContent: "center", alignItems: "center",
    });

    const dialog = document.createElement("div");
    Object.assign(dialog.style, {
        background: "#1e1e2e", color: "#cdd6f4", borderRadius: "10px",
        padding: "16px", width: "820px", maxHeight: "85vh",
        display: "flex", flexDirection: "column", gap: "10px",
        boxShadow: "0 8px 32px rgba(0,0,0,0.5)", fontFamily: "monospace",
        overflow: "hidden",
    });

    const title = document.createElement("div");
    title.textContent = "Advanced Prompt Syntax — JB Builder Wildcards";
    Object.assign(title.style, { fontWeight: "bold", fontSize: "14px" });

    const scroll = document.createElement("div");
    Object.assign(scroll.style, { overflow: "auto", flex: "1", padding: "4px 0" });

    // [section_title, [[token, description, example], ...]]
    const sections = [
        ["Wildcards (files in /wildcards)", [
            ["__name__",          "random line from name.txt",                 "__color__ → red"],
            ["__cat/sub__",       "nested category (subdir)",                  "__outfit/top__"],
            ["__cat/*__",         "random .txt in dir",                        "__color/*__"],
            ["__cat/pre*__",      "any .txt with given prefix",                "__color/dark*__"],
            ["__name^v__",        "pick a line, bind it to variable v",        "__color^c__"],
            ["__name^v1^v2__",    "bind same pick to multiple variables",      "__color^a^b__"],
            ["__^v__",            "recall a value bound to v",                 "__^c__ → red"],
            ["__^pre*__",         "recall from any var whose name starts pre", "__^col*__"],
        ]],
        ["Brackets (inline alternatives)", [
            ["{a|b|c}",           "pick one",                                  "{red|blue} → blue"],
            ["{%2%a|%1%b}",       "weighted choices (%w% prefix)",             "{%5%hot|cold}"],
            ["{N$$a|b|c}",        "pick N (deck — no repeat)",                 "{2$$a|b|c|d} → a, c"],
            ["{N-M$$a|b|c}",      "pick a random count between N and M",       "{2-3$$a|b|c|d}"],
            ["{*$$a|b|c}",        "include all, joined",                       "{*$$a|b} → a, b"],
            ["{N$$sep$$...}",     "custom join separator",                     "{2$$ and $$a|b|c}"],
            ["{N??a|b}",          "roulette — with repeat (N may exceed pool)","{4??x|y} → x, y, y, x"],
            ["{a|b}^v",           "bind the bracket result to variable v",     "{red|blue}^c"],
        ]],
        ["Per-line weights inside .txt files", [
            ["%2.5%silk",         "scales this line's pick weight (default 1)", "biases the draw"],
            ["red # comment",     "everything after unescaped # is stripped",   "→ red"],
        ]],
        ["Comments & escaping", [
            ["##throwaway##",     "block — output stripped, side-effects keep", "##__a^v__## __^v__"],
            ["\\__name__",        "literal — wildcard is NOT resolved",         "\\__color__ → __color__"],
            ["\\{ ... \\}",       "literal braces — bracket not parsed",        "\\{a|b\\} → {a|b}"],
            ["\\% / \\#",         "literal % or # in line text",                "\\%50 off"],
        ]],
        ["Node inputs", [
            ["seed",              "same seed + same rows = identical output",   "0 … 2⁶⁴"],
            ["context_from_prompt_generator", "DICT input — variable bag from adaptiveprompts PromptGenerator", "(plain STRING won't connect)"],
        ]],
    ];

    for (const [head, rows] of sections) {
        const h = document.createElement("div");
        h.textContent = head;
        Object.assign(h.style, {
            margin: "10px 0 4px", fontSize: "12px",
            color: "#89b4fa", fontWeight: "bold",
            borderBottom: "1px solid #45475a", paddingBottom: "2px",
        });
        scroll.append(h);
        const table = document.createElement("table");
        Object.assign(table.style, {
            width: "100%", borderCollapse: "collapse", fontSize: "12px",
        });
        for (const [tok, desc, ex] of rows) {
            const tr = document.createElement("tr");
            const tdTok = document.createElement("td");
            tdTok.textContent = tok;
            Object.assign(tdTok.style, {
                width: "30%", padding: "3px 8px",
                color: "#a6e3a1", whiteSpace: "nowrap",
                verticalAlign: "top",
            });
            const tdDesc = document.createElement("td");
            tdDesc.textContent = desc;
            Object.assign(tdDesc.style, {
                padding: "3px 8px", color: "#cdd6f4",
                verticalAlign: "top",
            });
            const tdEx = document.createElement("td");
            tdEx.textContent = ex;
            Object.assign(tdEx.style, {
                width: "30%", padding: "3px 8px",
                color: "#f9e2af", whiteSpace: "nowrap",
                verticalAlign: "top", fontStyle: "italic",
            });
            tr.append(tdTok, tdDesc, tdEx);
            table.append(tr);
        }
        scroll.append(table);
    }

    const btnRow = document.createElement("div");
    Object.assign(btnRow.style, { display: "flex", justifyContent: "flex-end" });
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "Close";
    Object.assign(closeBtn.style, {
        padding: "6px 18px", borderRadius: "6px", border: "none",
        background: "#45475a", color: "#cdd6f4", fontSize: "13px",
        cursor: "pointer", fontWeight: "bold",
    });
    btnRow.append(closeBtn);

    dialog.append(title, scroll, btnRow);
    overlay.append(dialog);
    document.body.append(overlay);

    closeBtn.addEventListener("click", () => { overlay.style.display = "none"; });
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) overlay.style.display = "none";
    });

    return { open() { overlay.style.display = "flex"; } };
}

// ─── wildcards editor modal (lazy) ───────────────────────────────────

function createWildcardsModal() {
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        display: "none", position: "fixed", inset: "0",
        background: "rgba(0,0,0,0.6)", zIndex: "10000",
        justifyContent: "center", alignItems: "center",
    });

    const dialog = document.createElement("div");
    Object.assign(dialog.style, {
        background: "#1e1e2e", color: "#cdd6f4", borderRadius: "10px",
        padding: "16px", width: "820px", maxHeight: "85vh",
        display: "flex", flexDirection: "column", gap: "10px",
        boxShadow: "0 8px 32px rgba(0,0,0,0.5)", fontFamily: "monospace",
    });

    const header = document.createElement("div");
    Object.assign(header.style, { display: "flex", alignItems: "center", gap: "8px" });
    const title = document.createElement("span");
    title.textContent = "Edit Wildcards";
    Object.assign(title.style, { fontWeight: "bold", fontSize: "14px", flex: "1" });
    const newBtn = document.createElement("button");
    newBtn.textContent = "+ New";
    Object.assign(newBtn.style, {
        background: "#313244", color: "#a6e3a1", border: "1px solid #45475a",
        borderRadius: "6px", padding: "4px 10px", fontSize: "13px", cursor: "pointer",
    });
    const delBtn = document.createElement("button");
    delBtn.textContent = "Delete";
    Object.assign(delBtn.style, {
        background: "#313244", color: "#f38ba8", border: "1px solid #45475a",
        borderRadius: "6px", padding: "4px 10px", fontSize: "13px", cursor: "pointer",
    });
    header.append(title, newBtn, delBtn);

    const body = document.createElement("div");
    Object.assign(body.style, {
        display: "flex", gap: "10px", flex: "1", minHeight: "360px",
    });

    // Left pane — flat list of wildcard paths.
    const list = document.createElement("div");
    Object.assign(list.style, {
        flex: "0 0 240px", overflow: "auto",
        background: "#181825", border: "1px solid #45475a",
        borderRadius: "6px", padding: "4px", fontSize: "12px",
    });

    // Right pane — text contents, one option per line.
    const textarea = document.createElement("textarea");
    Object.assign(textarea.style, {
        background: "#181825", color: "#a6e3a1", border: "1px solid #45475a",
        borderRadius: "6px", padding: "10px", fontSize: "13px",
        fontFamily: "Consolas, 'Courier New', monospace",
        flex: "1", resize: "none", whiteSpace: "pre",
    });
    textarea.addEventListener("keydown", (e) => {
        if (e.key === "Tab") {
            e.preventDefault();
            const s = textarea.selectionStart, t = textarea.selectionEnd;
            textarea.value = textarea.value.substring(0, s) + "  " + textarea.value.substring(t);
            textarea.selectionStart = textarea.selectionEnd = s + 2;
        }
    });

    body.append(list, textarea);

    const status = document.createElement("div");
    Object.assign(status.style, { fontSize: "12px", color: "#6c7086", minHeight: "18px" });

    const btnRow = document.createElement("div");
    Object.assign(btnRow.style, { display: "flex", gap: "8px", justifyContent: "flex-end" });
    const saveBtn = document.createElement("button");
    saveBtn.textContent = "Save";
    Object.assign(saveBtn.style, {
        padding: "6px 18px", borderRadius: "6px", border: "none",
        background: "#a6e3a1", color: "#1e1e2e", fontSize: "13px",
        cursor: "pointer", fontWeight: "bold",
    });
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "Close";
    Object.assign(closeBtn.style, {
        padding: "6px 18px", borderRadius: "6px", border: "none",
        background: "#45475a", color: "#cdd6f4", fontSize: "13px",
        cursor: "pointer", fontWeight: "bold",
    });
    btnRow.append(saveBtn, closeBtn);

    dialog.append(header, body, status, btnRow);
    overlay.append(dialog);
    document.body.append(overlay);

    let dirty = false;
    let selected = null;
    let onChangedCb = null;
    textarea.addEventListener("input", () => { dirty = true; });

    async function refreshList() {
        const resp = await api.fetchApi("/fvmtools/jb-wildcards");
        const data = await resp.json();
        const names = Array.isArray(data.wildcards) ? data.wildcards : [];
        list.innerHTML = "";
        if (names.length === 0) {
            const empty = document.createElement("div");
            empty.textContent = "(no wildcards yet — click + New)";
            Object.assign(empty.style, { padding: "8px", color: "#6c7086" });
            list.append(empty);
            textarea.value = "";
            selected = null;
            status.textContent = "";
            return names;
        }
        for (const name of names) {
            const item = document.createElement("div");
            item.textContent = name;
            Object.assign(item.style, {
                padding: "4px 8px", cursor: "pointer", borderRadius: "4px",
                color: name === selected ? "#1e1e2e" : "#cdd6f4",
                background: name === selected ? "#89b4fa" : "transparent",
            });
            item.addEventListener("mouseenter", () => {
                if (name !== selected) item.style.background = "#313244";
            });
            item.addEventListener("mouseleave", () => {
                if (name !== selected) item.style.background = "transparent";
            });
            item.addEventListener("click", async () => {
                if (dirty && !confirm("Unsaved changes — discard?")) return;
                selected = name;
                await loadEntry();
                await refreshList();
            });
            list.append(item);
        }
        return names;
    }

    async function loadEntry() {
        if (!selected) return;
        const url = `/fvmtools/jb-wildcard?name=${encodeURIComponent(selected)}`;
        const resp = await api.fetchApi(url);
        const data = await resp.json();
        if (data.error) { status.textContent = data.error; return; }
        textarea.value = data.text || "";
        dirty = false;
        status.textContent = `${selected}.txt`;
        status.style.color = "#6c7086";
    }

    async function saveEntry() {
        if (!selected) return;
        status.textContent = "Saving...";
        status.style.color = "#f9e2af";
        const resp = await api.fetchApi("/fvmtools/jb-wildcard", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: selected, text: textarea.value }),
        });
        const data = await resp.json();
        if (data.success) {
            dirty = false;
            status.textContent = "Saved.";
            status.style.color = "#a6e3a1";
            invalidateWildcards();
            if (onChangedCb) onChangedCb();
        } else {
            status.textContent = "Save failed: " + (data.error || "unknown");
            status.style.color = "#f38ba8";
        }
    }

    async function newEntry() {
        const name = prompt(
            "New wildcard path (use slashes for subfolders):\n" +
            "  e.g. outfits/colors  or  scenes/lighting\n" +
            "Allowed: lowercase letters, digits, underscores, hyphens, slashes."
        );
        if (!name) return;
        const cleaned = name.trim().toLowerCase();
        if (!/^[a-z0-9_\-]+(\/[a-z0-9_\-]+)*$/.test(cleaned)) {
            alert("Invalid path. Use lowercase a-z, 0-9, _, -, separated by /.");
            return;
        }
        const resp = await api.fetchApi("/fvmtools/jb-wildcard", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: cleaned, text: "# one option per line\n" }),
        });
        const data = await resp.json();
        if (data.success) {
            invalidateWildcards();
            selected = cleaned;
            await refreshList();
            await loadEntry();
            if (onChangedCb) onChangedCb();
        } else {
            alert("Failed: " + (data.error || "unknown"));
        }
    }

    async function deleteEntry() {
        if (!selected) return;
        if (!confirm(`Delete ${selected}.txt?`)) return;
        const url = `/fvmtools/jb-wildcard?name=${encodeURIComponent(selected)}`;
        await api.fetchApi(url, { method: "DELETE" });
        invalidateWildcards();
        selected = null;
        textarea.value = "";
        status.textContent = "";
        await refreshList();
        if (onChangedCb) onChangedCb();
    }

    saveBtn.addEventListener("click", saveEntry);
    newBtn.addEventListener("click", newEntry);
    delBtn.addEventListener("click", deleteEntry);
    closeBtn.addEventListener("click", () => {
        if (dirty && !confirm("Unsaved changes — discard?")) return;
        overlay.style.display = "none";
        invalidateWildcards();
        if (onChangedCb) onChangedCb();
    });
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) closeBtn.click();
    });

    return {
        async open(onChanged) {
            onChangedCb = onChanged || null;
            overlay.style.display = "flex";
            const names = await refreshList();
            if (names && names.length && !selected) {
                selected = names[0];
                await loadEntry();
                await refreshList();
            }
        },
    };
}

// ─── save-as-template dialog ─────────────────────────────────────────

function openSaveTemplateDialog(catalog, payload) {
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        position: "fixed", inset: "0", background: "rgba(0,0,0,0.6)",
        zIndex: "10001", display: "flex",
        justifyContent: "center", alignItems: "center",
    });

    const dialog = document.createElement("div");
    Object.assign(dialog.style, {
        background: "#1e1e2e", color: "#cdd6f4", borderRadius: "10px",
        padding: "16px", width: "520px", maxHeight: "85vh",
        display: "flex", flexDirection: "column", gap: "10px",
        boxShadow: "0 8px 32px rgba(0,0,0,0.5)", fontFamily: "monospace",
    });

    const title = document.createElement("div");
    title.textContent = "Save as Template";
    Object.assign(title.style, { fontWeight: "bold", fontSize: "14px" });

    // Category row
    const catRow = document.createElement("div");
    Object.assign(catRow.style, { display: "flex", gap: "8px", alignItems: "center" });
    const catLabel = document.createElement("span");
    catLabel.textContent = "Category:";
    Object.assign(catLabel.style, { flex: "0 0 80px", fontSize: "13px" });
    const catSelect = document.createElement("select");
    Object.assign(catSelect.style, {
        flex: "1", background: "#313244", color: "#cdd6f4",
        border: "1px solid #45475a", borderRadius: "6px",
        padding: "4px 8px", fontSize: "13px",
    });
    const NEW_CAT = "__new__";
    for (const name of Object.keys(catalog).sort()) {
        const opt = document.createElement("option");
        opt.value = name; opt.textContent = name;
        catSelect.append(opt);
    }
    const newCatOpt = document.createElement("option");
    newCatOpt.value = NEW_CAT; newCatOpt.textContent = "+ new category…";
    catSelect.append(newCatOpt);
    const newCatIn = document.createElement("input");
    newCatIn.type = "text"; newCatIn.placeholder = "new category name";
    Object.assign(newCatIn.style, {
        flex: "1", display: "none",
        background: "#181825", color: "#cdd6f4",
        border: "1px solid #45475a", borderRadius: "6px",
        padding: "4px 8px", fontSize: "13px", fontFamily: "inherit",
    });
    catSelect.addEventListener("change", () => {
        newCatIn.style.display = catSelect.value === NEW_CAT ? "block" : "none";
    });
    catRow.append(catLabel, catSelect, newCatIn);

    // Name row
    const nameRow = document.createElement("div");
    Object.assign(nameRow.style, { display: "flex", gap: "8px", alignItems: "center" });
    const nameLabel = document.createElement("span");
    nameLabel.textContent = "Name:";
    Object.assign(nameLabel.style, { flex: "0 0 80px", fontSize: "13px" });
    const nameIn = document.createElement("input");
    nameIn.type = "text"; nameIn.placeholder = "lowercase_with_underscores";
    Object.assign(nameIn.style, {
        flex: "1", background: "#181825", color: "#cdd6f4",
        border: "1px solid #45475a", borderRadius: "6px",
        padding: "4px 8px", fontSize: "13px", fontFamily: "inherit",
    });
    nameRow.append(nameLabel, nameIn);

    // Preview
    const previewLabel = document.createElement("div");
    previewLabel.textContent = "Preview:";
    Object.assign(previewLabel.style, { fontSize: "12px", color: "#6c7086" });
    const preview = document.createElement("pre");
    Object.assign(preview.style, {
        background: "#181825", color: "#a6e3a1",
        border: "1px solid #45475a", borderRadius: "6px",
        padding: "10px", fontSize: "12px", margin: "0",
        maxHeight: "260px", overflow: "auto", whiteSpace: "pre-wrap",
    });
    preview.textContent = JSON.stringify(payload, null, 2);

    // Status + buttons
    const status = document.createElement("div");
    Object.assign(status.style, { fontSize: "12px", color: "#6c7086", minHeight: "18px" });

    const btnRow = document.createElement("div");
    Object.assign(btnRow.style, { display: "flex", gap: "8px", justifyContent: "flex-end" });
    const saveBtn = document.createElement("button");
    saveBtn.textContent = "Save";
    Object.assign(saveBtn.style, {
        padding: "6px 18px", borderRadius: "6px", border: "none",
        background: "#a6e3a1", color: "#1e1e2e", fontSize: "13px",
        cursor: "pointer", fontWeight: "bold",
    });
    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Cancel";
    Object.assign(cancelBtn.style, {
        padding: "6px 18px", borderRadius: "6px", border: "none",
        background: "#45475a", color: "#cdd6f4", fontSize: "13px",
        cursor: "pointer", fontWeight: "bold",
    });
    btnRow.append(cancelBtn, saveBtn);

    dialog.append(title, catRow, nameRow, previewLabel, preview, status, btnRow);
    overlay.append(dialog);
    document.body.append(overlay);

    function close() { if (overlay.parentNode) document.body.removeChild(overlay); }
    cancelBtn.addEventListener("click", close);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });

    const NAME_RE = /^[a-z0-9_]+$/;
    saveBtn.addEventListener("click", async () => {
        const category = catSelect.value === NEW_CAT
            ? newCatIn.value.trim().toLowerCase()
            : catSelect.value;
        const name = nameIn.value.trim().toLowerCase();
        if (!NAME_RE.test(category)) {
            status.textContent = "Invalid category — lowercase letters, digits, underscores only.";
            status.style.color = "#f38ba8"; return;
        }
        if (!NAME_RE.test(name)) {
            status.textContent = "Invalid name — lowercase letters, digits, underscores only.";
            status.style.color = "#f38ba8"; return;
        }
        const exists = (catalog[category] || []).includes(name);
        if (exists && !confirm(`${category}/${name}.json already exists — overwrite?`)) return;
        status.textContent = "Saving…";
        status.style.color = "#f9e2af";
        try {
            const resp = await api.fetchApi("/fvmtools/jb-catalog-entry", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ category, name, data: payload }),
            });
            const data = await resp.json();
            if (data.success) {
                invalidateCatalog();
                status.textContent = "Saved.";
                status.style.color = "#a6e3a1";
                setTimeout(close, 400);
            } else {
                status.textContent = "Save failed: " + (data.error || "unknown");
                status.style.color = "#f38ba8";
            }
        } catch (e) {
            status.textContent = "Save failed: " + e.message;
            status.style.color = "#f38ba8";
        }
    });

    nameIn.focus();
}

// ─── helpers ─────────────────────────────────────────────────────────

function rowsToHidden(rows) {
    return JSON.stringify(rows);
}

function hiddenToRows(text) {
    if (!text || !text.trim()) return [];
    try {
        const parsed = JSON.parse(text);
        if (Array.isArray(parsed)) return parsed.map(r => ({
            key:    r.key || "",
            value:  r.value == null ? "" : String(r.value),
            indent: Math.max(0, parseInt(r.indent ?? 0, 10) || 0),
        }));
    } catch (e) { /* ignore */ }
    return [];
}

// Inverse of dictToRows — build a nested dict from the row-list. Mirrors
// Python's core.jb.serialize.rows_to_dict so saved templates round-trip
// cleanly: a branch is an empty-value row followed by children at indent+1,
// every other row is a leaf with a string value (with JSON literals like
// numbers / booleans / null parsed back to their native type).
function rowsToDict(rows) {
    const root = {};
    const stack = [{ indent: -1, dict: root }];
    for (let i = 0; i < rows.length; i++) {
        const row = rows[i] || {};
        const key = String(row.key || "").trim();
        if (!key) continue;
        const indent = Math.max(0, parseInt(row.indent ?? 0, 10) || 0);
        const valueRaw = row.value == null ? "" : String(row.value);

        while (stack.length && stack[stack.length - 1].indent >= indent) {
            stack.pop();
        }
        if (!stack.length) stack.push({ indent: -1, dict: root });
        const parent = stack[stack.length - 1].dict;

        const next = rows[i + 1];
        const nextIndent = next ? (parseInt(next.indent ?? -1, 10) || 0) : -1;
        const isBranch = (!valueRaw) && next && nextIndent > indent;

        if (isBranch) {
            const child = {};
            parent[key] = child;
            stack.push({ indent, dict: child });
        } else {
            parent[key] = coerceLeaf(valueRaw);
        }
    }
    return root;
}

function coerceLeaf(s) {
    if (s == null) return "";
    const t = String(s).trim();
    if (!t) return "";
    if (t === "true")  return true;
    if (t === "false") return false;
    if (t === "null")  return null;
    if ("-0123456789[{".includes(t[0])) {
        try { return JSON.parse(t); } catch (_) { /* fall through */ }
    }
    return s;
}

// Flatten a catalog entry (a JSON object) into rows we can append.
function dictToRows(obj, indent = 0, parentKey = null) {
    const out = [];
    if (parentKey != null) {
        if (obj && typeof obj === "object" && !Array.isArray(obj)) {
            out.push({ key: parentKey, value: "", indent });
            for (const k of Object.keys(obj)) {
                out.push(...dictToRows(obj[k], indent + 1, k));
            }
            return out;
        }
        let valueStr;
        if (obj === null) valueStr = "null";
        else if (typeof obj === "boolean") valueStr = obj ? "true" : "false";
        else if (typeof obj === "number") valueStr = String(obj);
        else if (typeof obj === "string") valueStr = obj;
        else valueStr = JSON.stringify(obj);
        out.push({ key: parentKey, value: valueStr, indent });
        return out;
    }
    if (obj && typeof obj === "object" && !Array.isArray(obj)) {
        for (const k of Object.keys(obj)) {
            out.push(...dictToRows(obj[k], indent, k));
        }
    }
    return out;
}

// ─── row widget builder ──────────────────────────────────────────────

function buildRowWidget(node) {
    const rowsHidden = node.widgets.find(w => w.name === "rows");
    let rows = hiddenToRows(rowsHidden ? rowsHidden.value : "");

    // Host element holds the row stack + toolbar.
    const host = document.createElement("div");
    Object.assign(host.style, {
        display: "flex", flexDirection: "column", gap: "4px",
        background: "#1a1a25", border: "1px solid #2a2a3a",
        borderRadius: "6px", padding: "6px",
        fontFamily: "Consolas, 'Courier New', monospace",
        color: "#cdd6f4", fontSize: "12px",
    });

    const toolbar = document.createElement("div");
    Object.assign(toolbar.style, { display: "flex", gap: "6px", flexWrap: "wrap" });
    const addRowBtn = document.createElement("button");
    addRowBtn.textContent = "+ Row";
    const insertBtn = document.createElement("button");
    insertBtn.textContent = "Insert ▾";
    const editCatBtn = document.createElement("button");
    editCatBtn.textContent = "Edit Catalog";
    const editWildBtn = document.createElement("button");
    editWildBtn.textContent = "Edit Wildcards";
    editWildBtn.title = "Edit __wildcard__ files (no restart required)";
    const infoBtn = document.createElement("button");
    infoBtn.textContent = "AdvPmptInfo";
    infoBtn.title = "Wildcard / bracket / variable syntax reference";
    const saveTplBtn = document.createElement("button");
    saveTplBtn.textContent = "Save as Template";
    saveTplBtn.title = "Save current rows as a catalog entry";
    const clearBtn = document.createElement("button");
    clearBtn.textContent = "Clear";
    clearBtn.title = "Delete all rows";
    for (const b of [addRowBtn, insertBtn, editCatBtn, editWildBtn, infoBtn, saveTplBtn, clearBtn]) {
        Object.assign(b.style, {
            background: "#313244", color: "#cdd6f4", border: "1px solid #45475a",
            borderRadius: "4px", padding: "4px 10px", fontSize: "12px", cursor: "pointer",
        });
    }
    infoBtn.style.color = "#89b4fa";
    saveTplBtn.style.color = "#a6e3a1";
    clearBtn.style.color = "#f38ba8";
    clearBtn.style.borderColor = "#6e3636";
    clearBtn.style.marginLeft = "auto";
    toolbar.append(addRowBtn, insertBtn, editCatBtn, editWildBtn, infoBtn, saveTplBtn, clearBtn);

    const rowList = document.createElement("div");
    Object.assign(rowList.style, { display: "flex", flexDirection: "column", gap: "3px" });

    host.append(toolbar, rowList);

    // ── persistence ────────────────────────────────────────────────
    function commit() {
        if (rowsHidden) {
            rowsHidden.value = rowsToHidden(rows);
            if (typeof rowsHidden.callback === "function") rowsHidden.callback(rowsHidden.value);
        }
        node.setDirtyCanvas(true, true);
    }

    // ── render ─────────────────────────────────────────────────────
    function render() {
        rowList.innerHTML = "";
        rows.forEach((r, i) => rowList.append(makeRowEl(r, i)));
        // Don't pin a fixed height. ComfyUI sizes the DOM widget through
        // the ``getHeight`` callback (which reads ``host.scrollHeight``);
        // pinning here would clip the host when the toolbar wraps to two
        // rows on a narrow node, and prevent it from shrinking back when
        // the user widens the node and the toolbar un-wraps.
        host.style.height = "";
        host.style.minHeight = "50px";
        node.setDirtyCanvas(true, true);
    }

    // Force the node to re-fit when content shrinks. ComfyUI's DOM-widget
    // layout grows on its own via getHeight, but it does not actively
    // shrink node.size when the host gets shorter — so after deletions we
    // ask computeSize() for the new minimum and snap the node down to it.
    function shrinkNodeToContent() {
        try {
            const computed = node.computeSize();
            if (computed && Array.isArray(computed) && node.size) {
                node.size[0] = Math.max(node.size[0], computed[0]);
                node.size[1] = computed[1];
                node.setDirtyCanvas(true, true);
            }
        } catch (e) { /* ignore */ }
    }

    function makeRowEl(row, idx) {
        const wrap = document.createElement("div");
        wrap.dataset.rowIdx = String(idx);
        Object.assign(wrap.style, {
            display: "flex", alignItems: "center", gap: "4px",
            paddingLeft: (row.indent * INDENT_PX) + "px",
            transition: "background-color 0.1s",
        });

        // Indent bar
        const bar = document.createElement("div");
        Object.assign(bar.style, {
            width: "3px", height: "20px", background: "#585b70",
            borderRadius: "1px", flex: "0 0 auto",
        });
        wrap.append(bar);

        // Key input
        const keyIn = document.createElement("input");
        keyIn.type = "text"; keyIn.value = row.key; keyIn.placeholder = "key";
        Object.assign(keyIn.style, {
            flex: "0 0 130px", background: "#181825", color: "#89b4fa",
            border: "1px solid #45475a", borderRadius: "4px", padding: "3px 6px",
            fontSize: "12px", fontFamily: "inherit",
        });
        keyIn.addEventListener("input", () => { row.key = keyIn.value; commit(); });
        wrap.append(keyIn);

        // Value input
        const valIn = document.createElement("input");
        valIn.type = "text"; valIn.value = row.value; valIn.placeholder = "value (leave empty for branch)";
        Object.assign(valIn.style, {
            flex: "1 1 auto", background: "#181825", color: "#a6e3a1",
            border: "1px solid #45475a", borderRadius: "4px", padding: "3px 6px",
            fontSize: "12px", fontFamily: "inherit", minWidth: "60px",
        });
        valIn.addEventListener("input", () => { row.value = valIn.value; commit(); });
        attachWildcardAutocomplete(valIn);
        wrap.append(valIn);

        // ⋮ drag handle. Pointerdown starts tracking; if the pointer moves
        // more than DRAG_THRESHOLD pixels before release we enter drag mode
        // and reorder on drop. A clean click without drag opens the small
        // action menu (Indent, Duplicate, Delete).
        const menuBtn = document.createElement("button");
        menuBtn.textContent = "⋮";
        menuBtn.title = "Drag to move · click for actions";
        Object.assign(menuBtn.style, {
            flex: "0 0 24px", background: "#313244", color: "#cdd6f4",
            border: "1px solid #45475a", borderRadius: "4px", cursor: "grab",
            fontSize: "14px", padding: "0", height: "22px",
            userSelect: "none", touchAction: "none",
        });
        attachDragHandle(menuBtn, wrap, idx);
        wrap.append(menuBtn);

        return wrap;
    }

    // ── drag handle ────────────────────────────────────────────────
    const DRAG_THRESHOLD_PX = 4;

    // A "block" is a row plus all consecutive following rows whose indent is
    // strictly greater than the row's own indent — i.e. its descendants in
    // the indent-tree. Drag-and-drop moves the whole block as a unit so the
    // hierarchy stays intact.
    function getBlockEnd(startIdx) {
        const baseIndent = rows[startIdx]?.indent ?? 0;
        let end = startIdx + 1;
        while (end < rows.length && (rows[end].indent ?? 0) > baseIndent) end++;
        return end;
    }

    function attachDragHandle(handle, rowEl, idx) {
        let pointerStart = null;
        let dragging = false;
        let placeholder = null;
        let ghost = null;
        let startIndent = 0;
        let blockStart = idx;
        let blockEnd = idx + 1;
        let blockEls = [];      // rowList children inside the block (for ghost + visual fade)
        let blockIndents = [];  // captured original indents of every block row

        function onPointerMove(ev) {
            if (!pointerStart) return;
            const dx = ev.clientX - pointerStart.x;
            const dy = ev.clientY - pointerStart.y;
            if (!dragging && Math.hypot(dx, dy) >= DRAG_THRESHOLD_PX) {
                enterDrag();
            }
            if (!dragging) return;
            // Update ghost position
            ghost.style.transform = `translate(${ev.clientX - pointerStart.x}px, ${ev.clientY - pointerStart.y}px)`;
            // Compute drop target row by Y over the row list.
            const targetIdx = computeDropIndex(ev.clientY);
            placeholder.dataset.targetIdx = String(targetIdx);
            // Compute indent delta from horizontal drift.
            const indentDelta = Math.round((ev.clientX - pointerStart.x) / INDENT_PX);
            placeholder.dataset.indent = String(Math.max(0, startIndent + indentDelta));
            redrawPlaceholder();
        }

        function onPointerUp(ev) {
            window.removeEventListener("pointermove", onPointerMove);
            window.removeEventListener("pointerup", onPointerUp);
            handle.style.cursor = "grab";
            if (!dragging) {
                // Click without drag → open small actions menu.
                if (pointerStart) {
                    const dx = ev.clientX - pointerStart.x;
                    const dy = ev.clientY - pointerStart.y;
                    if (Math.hypot(dx, dy) < DRAG_THRESHOLD_PX) {
                        showRowActions(idx, handle);
                    }
                }
                pointerStart = null;
                return;
            }
            // Apply the drop.
            const targetIdx = parseInt(placeholder.dataset.targetIdx, 10);
            const newIndent = parseInt(placeholder.dataset.indent, 10);
            cleanupDrag();
            applyDrop(blockStart, blockEnd, targetIdx, newIndent);
            pointerStart = null;
        }

        function enterDrag() {
            dragging = true;
            handle.style.cursor = "grabbing";
            startIndent = rows[idx].indent || 0;

            // Capture the block (this row + descendants) so the entire
            // subtree moves together and we know which DOM rows to fade.
            blockStart = idx;
            blockEnd = getBlockEnd(idx);
            blockIndents = rows.slice(blockStart, blockEnd).map(r => r.indent || 0);
            const children = Array.from(rowList.children);
            blockEls = children.slice(blockStart, blockEnd);
            for (const el of blockEls) el.style.opacity = "0.35";

            // Ghost: visual stack of the whole block following the cursor.
            ghost = document.createElement("div");
            const headRect = rowEl.getBoundingClientRect();
            Object.assign(ghost.style, {
                position: "fixed", left: headRect.left + "px", top: headRect.top + "px",
                width: headRect.width + "px", opacity: "0.85",
                pointerEvents: "none", zIndex: "9998",
                background: "#313244", borderRadius: "4px",
                boxShadow: "0 4px 12px rgba(0,0,0,0.4)",
                display: "flex", flexDirection: "column", gap: "3px",
            });
            for (const el of blockEls) {
                const clone = el.cloneNode(true);
                clone.style.opacity = "1";
                ghost.append(clone);
            }
            document.body.append(ghost);

            // Placeholder: thin insertion line that snaps to drop target.
            placeholder = document.createElement("div");
            Object.assign(placeholder.style, {
                height: "2px", background: "#89b4fa",
                margin: "0 6px", borderRadius: "1px",
                pointerEvents: "none",
            });
            placeholder.dataset.targetIdx = String(blockStart);
            placeholder.dataset.indent = String(startIndent);
            redrawPlaceholder();
        }

        function redrawPlaceholder() {
            if (!placeholder) return;
            const targetIdx = parseInt(placeholder.dataset.targetIdx, 10);
            // Remove from current parent then reinsert at target position.
            if (placeholder.parentNode) placeholder.parentNode.removeChild(placeholder);
            const children = Array.from(rowList.children);
            const before = children[targetIdx];
            if (before) rowList.insertBefore(placeholder, before);
            else rowList.append(placeholder);
            const indent = parseInt(placeholder.dataset.indent, 10) || 0;
            placeholder.style.marginLeft = (indent * INDENT_PX + 6) + "px";
        }

        function computeDropIndex(clientY) {
            // Skip rows that belong to the dragged block — they're moving
            // with the cursor, so dropping "into" them is meaningless. The
            // valid drop slots are just before blockStart, just after
            // blockEnd, and around any non-block row.
            const children = Array.from(rowList.children).filter(c => c !== placeholder);
            for (let i = 0; i < children.length; i++) {
                if (i >= blockStart && i < blockEnd) continue;
                const r = children[i].getBoundingClientRect();
                if (clientY < r.top + r.height / 2) {
                    // Snap to the block boundary if we land on a row inside it.
                    return (i >= blockStart && i < blockEnd) ? blockStart : i;
                }
            }
            return children.length;
        }

        function cleanupDrag() {
            if (ghost && ghost.parentNode) ghost.parentNode.removeChild(ghost);
            if (placeholder && placeholder.parentNode) placeholder.parentNode.removeChild(placeholder);
            for (const el of blockEls) el.style.opacity = "";
            blockEls = [];
            ghost = null;
            placeholder = null;
            dragging = false;
        }

        function applyDrop(blkStart, blkEnd, toIdx, newIndent) {
            // No-op cases: dropping back inside or just at the block edges.
            if (toIdx >= blkStart && toIdx <= blkEnd) {
                // Only an indent change (lateral drift) is meaningful here.
                const indentDelta = Math.max(0, Math.min(8, newIndent)) - (rows[blkStart].indent || 0);
                if (indentDelta !== 0) {
                    for (let i = blkStart; i < blkEnd; i++) {
                        rows[i].indent = Math.max(0, Math.min(8, (rows[i].indent || 0) + indentDelta));
                    }
                    commit();
                    render();
                }
                return;
            }
            const block = rows.splice(blkStart, blkEnd - blkStart);
            const headOldIndent = blockIndents[0] || 0;
            const headNewIndent = Math.max(0, Math.min(8, newIndent));
            const indentDelta = headNewIndent - headOldIndent;
            // Apply the same indent shift to every block row so the subtree
            // structure is preserved (clamped to the [0,8] range).
            for (let i = 0; i < block.length; i++) {
                block[i].indent = Math.max(0, Math.min(8, (blockIndents[i] || 0) + indentDelta));
            }
            // Indices after blkEnd shifted left by block.length once we
            // removed the slice; targets ≤ blkStart are unaffected.
            const adjusted = (toIdx > blkEnd) ? toIdx - block.length : toIdx;
            rows.splice(adjusted, 0, ...block);
            commit();
            render();
        }

        handle.addEventListener("pointerdown", (e) => {
            e.preventDefault();
            pointerStart = { x: e.clientX, y: e.clientY };
            window.addEventListener("pointermove", onPointerMove);
            window.addEventListener("pointerup", onPointerUp);
        });
    }

    // ── row actions menu (opens on bare ⋮ click without drag) ──────
    function showRowActions(idx, anchor) {
        const menu = document.createElement("div");
        Object.assign(menu.style, {
            position: "fixed", background: "#1e1e2e", color: "#cdd6f4",
            border: "1px solid #45475a", borderRadius: "6px",
            padding: "4px 0", zIndex: "9999", boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
            fontFamily: "Consolas, monospace", fontSize: "12px", minWidth: "150px",
        });
        const r = anchor.getBoundingClientRect();
        menu.style.left = r.left + "px";
        menu.style.top  = (r.bottom + 2) + "px";
        const blockEnd = getBlockEnd(idx);
        const childCount = blockEnd - idx - 1;
        // Block-aware actions are the default. When the row has children
        // we also surface single-row variants so the user can edit just
        // the head without disturbing the subtree.
        const items = [
            ["Indent Right", () => indentBlock(idx, +1)],
            ["Indent Left",  () => indentBlock(idx, -1)],
            ["Duplicate",    () => duplicateBlock(idx)],
            ["Delete",       () => deleteBlock(idx)],
        ];
        if (childCount > 0) {
            const suffix = ` (+${childCount} sub-row${childCount === 1 ? "" : "s"})`;
            items[0][0] += suffix;
            items[1][0] += suffix;
            items[2][0] += suffix;
            items[3][0] += suffix;
            items.push(
                "sep",
                ["Indent Right (only this row)", () => indentRow(idx, +1)],
                ["Indent Left  (only this row)", () => indentRow(idx, -1)],
                ["Duplicate    (only this row)", () => duplicateRow(idx)],
                ["Delete       (only this row)", () => deleteRow(idx)],
            );
        }
        for (const entry of items) {
            if (entry === "sep") {
                const sep = document.createElement("div");
                Object.assign(sep.style, {
                    height: "1px", background: "#45475a", margin: "4px 0",
                });
                menu.append(sep);
                continue;
            }
            const [label, fn] = entry;
            const item = document.createElement("div");
            item.textContent = label;
            Object.assign(item.style, {
                padding: "5px 12px", cursor: "pointer", whiteSpace: "nowrap",
            });
            item.addEventListener("mouseenter", () => item.style.background = "#313244");
            item.addEventListener("mouseleave", () => item.style.background = "");
            item.addEventListener("click", () => {
                fn();
                if (menu.parentNode) document.body.removeChild(menu);
                render();
            });
            menu.append(item);
        }
        document.body.append(menu);
        const dismiss = (e) => {
            if (!menu.contains(e.target)) {
                if (menu.parentNode) document.body.removeChild(menu);
                document.removeEventListener("mousedown", dismiss);
            }
        };
        setTimeout(() => document.addEventListener("mousedown", dismiss), 10);
    }

    function moveRow(idx, dir) {
        const target = idx + dir;
        if (target < 0 || target >= rows.length) return;
        [rows[idx], rows[target]] = [rows[target], rows[idx]];
        commit();
    }
    function indentRow(idx, delta) {
        rows[idx].indent = Math.max(0, Math.min(8, (rows[idx].indent || 0) + delta));
        commit();
    }
    function duplicateRow(idx) {
        rows.splice(idx + 1, 0, { ...rows[idx] });
        commit();
    }
    function deleteRow(idx) {
        rows.splice(idx, 1);
        commit();
        // Defer until after the menu's render() runs so the DOM has settled.
        setTimeout(shrinkNodeToContent, 0);
    }

    // Block-aware variants — operate on a row plus its descendants
    // (consecutive following rows at strictly greater indent). Keeps the
    // hierarchy intact when the user nudges, duplicates, or deletes.
    function indentBlock(idx, delta) {
        const end = getBlockEnd(idx);
        const head = rows[idx].indent || 0;
        const newHead = Math.max(0, Math.min(8, head + delta));
        const realDelta = newHead - head;
        if (realDelta === 0) return;
        for (let i = idx; i < end; i++) {
            rows[i].indent = Math.max(0, Math.min(8, (rows[i].indent || 0) + realDelta));
        }
        commit();
    }
    function duplicateBlock(idx) {
        const end = getBlockEnd(idx);
        const block = rows.slice(idx, end).map(r => ({ ...r }));
        rows.splice(end, 0, ...block);
        commit();
    }
    function deleteBlock(idx) {
        const end = getBlockEnd(idx);
        rows.splice(idx, end - idx);
        commit();
        setTimeout(shrinkNodeToContent, 0);
    }
    function addRow() {
        const tail = rows[rows.length - 1];
        const indent = tail ? tail.indent : 0;
        rows.push({ key: "", value: "", indent });
        commit();
        render();
    }

    addRowBtn.addEventListener("click", addRow);

    // ── Clear all rows ─────────────────────────────────────────────
    clearBtn.addEventListener("click", () => {
        if (rows.length === 0) return;
        const n = rows.length;
        if (!confirm(`Delete all ${n} row${n === 1 ? "" : "s"}? This cannot be undone.`)) return;
        rows.length = 0;
        commit();
        render();
        setTimeout(shrinkNodeToContent, 0);
    });

    // ── Insert From Catalog dropdown ───────────────────────────────
    insertBtn.addEventListener("click", async (e) => {
        e.preventDefault();
        const cat = await getCatalog();
        const menu = document.createElement("div");
        Object.assign(menu.style, {
            position: "fixed", background: "#1e1e2e", color: "#cdd6f4",
            border: "1px solid #45475a", borderRadius: "6px",
            padding: "4px 0", zIndex: "9999",
            boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
            maxHeight: "70vh", overflowY: "auto", minWidth: "240px",
            fontFamily: "Consolas, monospace", fontSize: "12px",
        });
        const r = insertBtn.getBoundingClientRect();
        menu.style.left = r.left + "px";
        menu.style.top  = (r.bottom + 2) + "px";

        for (const category of Object.keys(cat).sort()) {
            const head = document.createElement("div");
            head.textContent = category + " /";
            Object.assign(head.style, {
                padding: "4px 12px", color: "#89b4fa", fontWeight: "bold",
                background: "#11111b",
            });
            menu.append(head);
            for (const name of cat[category]) {
                const item = document.createElement("div");
                item.textContent = "  " + name;
                Object.assign(item.style, {
                    padding: "4px 12px", cursor: "pointer", whiteSpace: "nowrap",
                });
                item.addEventListener("mouseenter", () => item.style.background = "#313244");
                item.addEventListener("mouseleave", () => item.style.background = "");
                item.addEventListener("click", async () => {
                    document.body.removeChild(menu);
                    const url = `/fvmtools/jb-catalog-entry?category=${encodeURIComponent(category)}&name=${encodeURIComponent(name)}`;
                    const resp = await api.fetchApi(url);
                    const data = await resp.json();
                    if (data.error) return;
                    const newRows = dictToRows(data.data);
                    rows.push(...newRows);
                    commit();
                    render();
                });
                menu.append(item);
            }
        }

        document.body.append(menu);
        const dismiss = (e2) => {
            if (!menu.contains(e2.target)) {
                if (menu.parentNode) document.body.removeChild(menu);
                document.removeEventListener("mousedown", dismiss);
            }
        };
        setTimeout(() => document.addEventListener("mousedown", dismiss), 10);
    });

    // ── Edit Catalog modal ─────────────────────────────────────────
    editCatBtn.addEventListener("click", () => {
        if (!catalogModal) catalogModal = createCatalogModal();
        catalogModal.open(() => invalidateCatalog());
    });

    // ── Edit Wildcards modal ───────────────────────────────────────
    editWildBtn.addEventListener("click", () => {
        if (!wildcardsModal) wildcardsModal = createWildcardsModal();
        wildcardsModal.open(() => invalidateWildcards());
    });

    // ── Advanced prompt syntax reference ───────────────────────────
    infoBtn.addEventListener("click", () => {
        if (!syntaxInfoModal) syntaxInfoModal = createSyntaxInfoModal();
        syntaxInfoModal.open();
    });

    // ── Save as Template ───────────────────────────────────────────
    saveTplBtn.addEventListener("click", async () => {
        const hasContent = rows.some(r => (r.key || "").trim());
        if (!hasContent) {
            alert("Nothing to save — add at least one row with a key first.");
            return;
        }
        const cat = await getCatalog();
        openSaveTemplateDialog(cat, rowsToDict(rows));
    });

    // initial render
    render();

    // Expose a re-hydrate hook so onConfigure can replay the saved value
    // into the JS state once ComfyUI restores widgets from `widgets_values`.
    // ComfyUI's lifecycle is: onNodeCreated → restore widget values →
    // onConfigure. We snapshot rows in onNodeCreated, so without this we'd
    // freeze the default `[]` and never see the saved content on reload.
    node.__fvmJbRefreshRows = () => {
        const restored = hiddenToRows(rowsHidden ? rowsHidden.value : "");
        rows.length = 0;
        rows.push(...restored);
        render();
        // Now that rows are visible the node's natural height has changed —
        // ask the layout to re-fit so we don't leave a gap or a clipped tail.
        setTimeout(() => {
            try {
                const computed = node.computeSize();
                if (computed && Array.isArray(computed) && node.size) {
                    node.size[1] = Math.max(node.size[1], computed[1]);
                    node.setDirtyCanvas(true, true);
                }
            } catch (e) { /* ignore */ }
        }, 0);
    };

    // Hide the raw `rows` STRING widget — the host element above is the UI.
    // IMPORTANT: do NOT set ``type = "converted-widget"``. That magic string
    // tells ComfyUI's frontend to draw an input socket for the widget on
    // the left side of the node, which would surface ``rows`` as a fake
    // second input next to the genuine ``context_from_prompt_generator``.
    // We just neutralise its size and draw — the widget keeps its real
    // STRING type so it serialises normally but never renders.
    if (rowsHidden) {
        rowsHidden.hidden = true;
        rowsHidden.computeSize = () => [0, -4];
        rowsHidden.draw = () => {};
        if (rowsHidden.options) {
            rowsHidden.options.hidden = true;
            rowsHidden.options.serialize = true;  // keep value in saved workflows
        }
    }

    return host;
}

// ─── Extension registration ──────────────────────────────────────────

app.registerExtension({
    name: "FVMTools.JB.Builder",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        // ComfyUI restores `widgets_values` AFTER onNodeCreated, then calls
        // onConfigure. Hook it so we replay the saved hidden-widget value
        // back into the visible row stack — otherwise reload of a saved
        // workflow shows an empty Builder.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            if (typeof this.__fvmJbRefreshRows === "function") {
                // Defer one tick so any other configure-time widget mutations
                // (e.g. value-coercion callbacks) settle before we re-read.
                setTimeout(() => this.__fvmJbRefreshRows(), 0);
            }
            return r;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const host = buildRowWidget(this);
            // Insert the host as a DOM widget. The getHeight callback makes
            // the widget (and therefore the node) auto-grow as the user adds
            // rows. ComfyUI calls this on every redraw; we return the host's
            // measured natural content height with a small safety margin.
            const node = this;
            this.addDOMWidget("jb_rows_host", "div", host, {
                serialize: false,
                getHeight: () => {
                    // scrollHeight includes children + padding even when no
                    // explicit min-height is set on the host.
                    const h = Math.max(60, host.scrollHeight + 8);
                    return h;
                },
            });

            // Re-flow the node whenever the host content size changes.
            // Triggers include: rows added/removed, value text wrapping,
            // and crucially the toolbar reflowing to one or two lines as
            // the user resizes the node horizontally. We snap the node
            // height to ``computeSize()`` each tick so we both grow AND
            // shrink in sync with the host's natural scrollHeight.
            try {
                let raf = 0;
                const reflow = () => {
                    if (raf) return;
                    raf = requestAnimationFrame(() => {
                        raf = 0;
                        if (typeof node.onResize === "function") {
                            node.onResize(node.size);
                        }
                        try {
                            const c = node.computeSize();
                            if (c && Array.isArray(c) && node.size) {
                                node.size[1] = c[1];
                            }
                        } catch (e) { /* ignore */ }
                        node.setDirtyCanvas(true, true);
                    });
                };
                const ro = new ResizeObserver(reflow);
                ro.observe(host);
                ro.observe(toolbar);
            } catch (e) { /* ResizeObserver not available — fall through */ }

            // Default to a reasonable initial node size.
            try {
                const computed = this.computeSize();
                const w = Math.max(this.size?.[0] || 0, 480);
                const h = Math.max(this.size?.[1] || 0, computed?.[1] || 260);
                this.size = [w, h];
            } catch (e) { /* ignore */ }

            return r;
        };
    },
});
