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
    for (const b of [addRowBtn, insertBtn, editCatBtn]) {
        Object.assign(b.style, {
            background: "#313244", color: "#cdd6f4", border: "1px solid #45475a",
            borderRadius: "4px", padding: "4px 10px", fontSize: "12px", cursor: "pointer",
        });
    }
    toolbar.append(addRowBtn, insertBtn, editCatBtn);

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
        // Resize the host so ComfyUI's DOM widget allocates the right height.
        const h = Math.max(50, 30 + rows.length * (ROW_HEIGHT + 3));
        host.style.minHeight = h + "px";
        node.setDirtyCanvas(true, true);
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

    function attachDragHandle(handle, rowEl, idx) {
        let pointerStart = null;
        let dragging = false;
        let placeholder = null;
        let ghost = null;
        let startIndent = 0;

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
            applyDrop(idx, targetIdx, newIndent);
            pointerStart = null;
        }

        function enterDrag() {
            dragging = true;
            handle.style.cursor = "grabbing";
            startIndent = rows[idx].indent || 0;

            // Ghost: visual clone that follows the cursor.
            ghost = rowEl.cloneNode(true);
            const r = rowEl.getBoundingClientRect();
            Object.assign(ghost.style, {
                position: "fixed", left: r.left + "px", top: r.top + "px",
                width: r.width + "px", opacity: "0.85",
                pointerEvents: "none", zIndex: "9998",
                background: "#313244", borderRadius: "4px",
                boxShadow: "0 4px 12px rgba(0,0,0,0.4)",
            });
            document.body.append(ghost);

            // Placeholder: thin insertion line that snaps to drop target.
            placeholder = document.createElement("div");
            Object.assign(placeholder.style, {
                height: "2px", background: "#89b4fa",
                margin: "0 6px", borderRadius: "1px",
                pointerEvents: "none",
            });
            placeholder.dataset.targetIdx = String(idx);
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
            const children = Array.from(rowList.children).filter(c => c !== placeholder && c.classList ? true : true);
            for (let i = 0; i < children.length; i++) {
                const r = children[i].getBoundingClientRect();
                if (clientY < r.top + r.height / 2) return i;
            }
            return children.length;
        }

        function cleanupDrag() {
            if (ghost && ghost.parentNode) ghost.parentNode.removeChild(ghost);
            if (placeholder && placeholder.parentNode) placeholder.parentNode.removeChild(placeholder);
            ghost = null;
            placeholder = null;
            dragging = false;
        }

        function applyDrop(fromIdx, toIdx, newIndent) {
            const moved = rows.splice(fromIdx, 1)[0];
            // After removing fromIdx, indices ≥ fromIdx shift down by 1.
            const adjusted = (toIdx > fromIdx) ? toIdx - 1 : toIdx;
            moved.indent = Math.max(0, Math.min(8, newIndent));
            rows.splice(adjusted, 0, moved);
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
        const items = [
            ["Indent Right", () => indentRow(idx, +1)],
            ["Indent Left",  () => indentRow(idx, -1)],
            ["Duplicate",    () => duplicateRow(idx)],
            ["Delete",       () => deleteRow(idx)],
        ];
        for (const [label, fn] of items) {
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
    }
    function addRow() {
        const tail = rows[rows.length - 1];
        const indent = tail ? tail.indent : 0;
        rows.push({ key: "", value: "", indent });
        commit();
        render();
    }

    addRowBtn.addEventListener("click", addRow);

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

    // initial render
    render();

    // Hide the raw `rows` STRING widget — the host element above is the UI.
    if (rowsHidden) {
        // Multiple belt-and-braces hide flags. Different ComfyUI versions
        // honour different ones; setting them all guarantees the widget
        // never draws AND claims zero vertical space in the node layout.
        rowsHidden.type = "converted-widget";
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

            // Re-flow the node when the host resizes (rows added/removed,
            // wide values overflowing, etc.).
            try {
                const ro = new ResizeObserver(() => {
                    if (typeof node.onResize === "function") {
                        node.onResize(node.size);
                    }
                    node.setDirtyCanvas(true, true);
                });
                ro.observe(host);
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
