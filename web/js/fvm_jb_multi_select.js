/**
 * FVM_JB_OutfitBlock + FVM_JB_LocationBlock — multi-select for the set and
 * colour-mood dropdowns.
 *
 * Each dropdown keeps its single choice. Next to it, Python declares a STRING
 * widget (`outfit_set_multi`, `location_set_multi`, `color_mood_multi`) that
 * this extension hides; a "Multi-Select …" button opens a checkbox tree and
 * writes the checked choices into it, one per line. With choices in it, the
 * block draws one per seed (core/jb/multi_pick.py) and ignores the dropdown —
 * so the dropdown and the button glow, and the button says how many.
 *
 * Apply with one checked choice just sets the dropdown; nothing is left in
 * the hidden widget then.
 */
import { app } from "../../scripts/app.js";

const CONFIGS = {
    FVM_JB_OutfitBlock: [
        { combo: "outfit_set", hidden: "outfit_set_multi", label: "Multi-Select Sets", noun: "sets" },
        { combo: "color_mood", hidden: "color_mood_multi", label: "Multi-Select Moods", noun: "moods" },
    ],
    FVM_JB_LocationBlock: [
        { combo: "location_set", hidden: "location_set_multi", label: "Multi-Select Sets", noun: "sets" },
        { combo: "color_mood", hidden: "color_mood_multi", label: "Multi-Select Moods", noun: "moods" },
    ],
};

const GLOW = "#ffb300";

function parseChoices(text) {
    if (typeof text !== "string") return [];
    const seen = new Set();
    for (const line of text.split(/\r?\n/)) {
        const v = line.trim();
        if (v && !v.startsWith("#")) seen.add(v);
    }
    return [...seen];
}

function comboValues(widget) {
    const v = widget?.options?.values;
    return (typeof v === "function" ? v() : v) || [];
}

// Same approach as the Builder's hidden `rows` widget: keep the real STRING
// type so it serialises, but never draw it. Not "converted-widget" — that
// would surface it as an input socket.
function hideWidget(widget) {
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
    if (widget.options) {
        widget.options.hidden = true;
        widget.options.serialize = true;
    }
}

// ─── Checkbox tree modal ─────────────────────────────────────────────

function createModal() {
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        display: "none", position: "fixed", inset: "0",
        background: "rgba(0,0,0,0.6)", zIndex: "10000",
        justifyContent: "center", alignItems: "center",
    });
    const dialog = document.createElement("div");
    Object.assign(dialog.style, {
        background: "#1e1e2e", color: "#cdd6f4", borderRadius: "10px",
        padding: "16px", width: "560px", maxHeight: "85vh",
        display: "flex", flexDirection: "column", gap: "10px",
        boxShadow: "0 8px 32px rgba(0,0,0,0.5)", fontFamily: "monospace",
    });

    const header = document.createElement("div");
    Object.assign(header.style, { display: "flex", alignItems: "center", gap: "10px" });
    const title = document.createElement("span");
    Object.assign(title.style, { fontWeight: "bold", fontSize: "14px", flex: "1" });
    const counter = document.createElement("span");
    Object.assign(counter.style, { fontSize: "13px", color: GLOW });
    header.append(title, counter);

    const filter = document.createElement("input");
    filter.placeholder = "Filter…";
    Object.assign(filter.style, {
        background: "#181825", color: "#cdd6f4", border: "1px solid #45475a",
        borderRadius: "6px", padding: "6px 8px", fontSize: "13px",
    });

    const list = document.createElement("div");
    Object.assign(list.style, {
        background: "#181825", border: "1px solid #45475a", borderRadius: "6px",
        padding: "6px", overflowY: "auto", minHeight: "260px", maxHeight: "60vh",
        fontSize: "13px", lineHeight: "1.7",
    });

    const note = document.createElement("div");
    Object.assign(note.style, { fontSize: "12px", color: "#a6adc8" });

    const footer = document.createElement("div");
    Object.assign(footer.style, { display: "flex", gap: "8px", alignItems: "center" });
    const mkBtn = (text, primary) => {
        const b = document.createElement("button");
        b.textContent = text;
        Object.assign(b.style, {
            background: primary ? "#89b4fa" : "#313244", color: primary ? "#1e1e2e" : "#cdd6f4",
            border: "1px solid #45475a", borderRadius: "6px", padding: "5px 12px",
            fontSize: "13px", cursor: "pointer", fontWeight: primary ? "bold" : "normal",
        });
        return b;
    };
    const btnAll = mkBtn("Select visible");
    const btnNone = mkBtn("Clear");
    const spacer = document.createElement("span");
    spacer.style.flex = "1";
    const btnCancel = mkBtn("Cancel");
    const btnApply = mkBtn("Apply", true);
    footer.append(btnAll, btnNone, spacer, btnCancel, btnApply);

    dialog.append(header, filter, list, note, footer);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    let values = [];
    let checked = new Set();
    let open = new Set();
    let onApply = null;

    const leavesUnder = (prefix) => values.filter(v => v.startsWith(prefix + "/"));

    function updateCounter() {
        counter.textContent = checked.size ? `${checked.size} checked` : "";
    }

    function checkbox(state, onChange) {
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = state === true;
        cb.indeterminate = state === "some";
        cb.style.marginRight = "6px";
        cb.style.cursor = "pointer";
        cb.addEventListener("click", (e) => e.stopPropagation());
        cb.addEventListener("change", () => onChange(cb.checked));
        return cb;
    }

    function leafRow(value, text, depth) {
        const row = document.createElement("label");
        Object.assign(row.style, {
            display: "flex", alignItems: "center", cursor: "pointer",
            paddingLeft: `${depth * 16 + 18}px`,
        });
        row.append(checkbox(checked.has(value), (on) => {
            on ? checked.add(value) : checked.delete(value);
            render();
        }), document.createTextNode(text));
        return row;
    }

    function folderRow(path, name, depth) {
        const leaves = leavesUnder(path);
        const n = leaves.filter(v => checked.has(v)).length;
        const state = n === 0 ? false : n === leaves.length ? true : "some";
        const row = document.createElement("div");
        Object.assign(row.style, {
            display: "flex", alignItems: "center", cursor: "pointer",
            paddingLeft: `${depth * 16}px`, color: n ? GLOW : "#cdd6f4",
        });
        const arrow = document.createElement("span");
        arrow.textContent = open.has(path) ? "▼" : "▶";
        Object.assign(arrow.style, { width: "18px", display: "inline-block", fontSize: "11px" });
        const label = document.createElement("span");
        label.textContent = `${name}  (${n}/${leaves.length})`;
        row.append(arrow, checkbox(state, (on) => {
            for (const v of leaves) on ? checked.add(v) : checked.delete(v);
            render();
        }), label);
        row.addEventListener("click", () => {
            open.has(path) ? open.delete(path) : open.add(path);
            render();
        });
        return row;
    }

    function renderTree(parent, prefix, depth) {
        const childFolders = new Set();
        const childLeaves = [];
        for (const v of values) {
            if (prefix && !v.startsWith(prefix + "/")) continue;
            const rest = prefix ? v.slice(prefix.length + 1) : v;
            const cut = rest.indexOf("/");
            if (cut < 0) childLeaves.push(v);
            else childFolders.add(rest.slice(0, cut));
        }
        for (const name of childFolders) {
            const path = prefix ? `${prefix}/${name}` : name;
            parent.appendChild(folderRow(path, name, depth));
            if (open.has(path)) renderTree(parent, path, depth + 1);
        }
        for (const v of childLeaves) {
            parent.appendChild(leafRow(v, prefix ? v.slice(prefix.length + 1) : v, depth));
        }
    }

    function visibleValues() {
        const q = filter.value.trim().toLowerCase();
        return q ? values.filter(v => v.toLowerCase().includes(q)) : values;
    }

    function render() {
        const scroll = list.scrollTop;
        list.replaceChildren();
        if (filter.value.trim()) {
            for (const v of visibleValues()) list.appendChild(leafRow(v, v, 0));
        } else {
            renderTree(list, "", 0);
        }
        list.scrollTop = scroll;
        updateCounter();
    }

    const close = () => { overlay.style.display = "none"; onApply = null; };

    filter.addEventListener("input", render);
    btnAll.addEventListener("click", () => { for (const v of visibleValues()) checked.add(v); render(); });
    btnNone.addEventListener("click", () => { checked.clear(); render(); });
    btnCancel.addEventListener("click", close);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    overlay.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
    btnApply.addEventListener("click", () => {
        const picked = values.filter(v => checked.has(v));
        const cb = onApply;
        close();
        cb?.(picked);
    });

    return {
        open({ heading, choices, preChecked, unknown, apply }) {
            values = choices;
            checked = new Set(preChecked);
            // Open every folder on the way to a checked choice, plus the top level.
            open = new Set(values.filter(v => v.includes("/")).map(v => v.split("/")[0]));
            for (const v of checked) {
                const parts = v.split("/");
                for (let i = 1; i < parts.length; i++) open.add(parts.slice(0, i).join("/"));
            }
            title.textContent = heading;
            note.textContent = (unknown.length
                ? `Not on disk any more, dropped on Apply: ${unknown.join(", ")}. ` : "")
                + "Two or more checked: the node draws one per seed. One: it just sets the dropdown.";
            filter.value = "";
            onApply = apply;
            render();
            overlay.style.display = "flex";
            filter.focus();
        },
    };
}

let modal = null;

// ─── Node wiring ─────────────────────────────────────────────────────

function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
    else ctx.rect(x, y, w, h);
}

// Widgets draw after onDrawForeground, so a frame drawn there sits under
// them. The decoration therefore wraps the widget's own drawWidget.
function strokeGlow(ctx, x, y, w, h) {
    ctx.save();
    ctx.strokeStyle = GLOW;
    ctx.shadowColor = GLOW;
    ctx.lineWidth = 2;
    for (const blur of [18, 6]) {
        ctx.shadowBlur = blur;
        roundRect(ctx, x, y, w, h, h / 2);
        ctx.stroke();
    }
    ctx.restore();
}

// The dropdown still shows its single value, which the node ignores while
// the multi-select is active — cover it with the count. The label is drawn
// again because the combo hides it when its value is long.
function overlayValue(ctx, widget, x, y, w, h, long, short) {
    const LG = globalThis.LiteGraph;
    const label = widget.label || widget.name;
    const left = x + 18;
    const right = x + w - 20;
    ctx.save();
    ctx.fillStyle = LG?.WIDGET_BGCOLOR ?? "#222";
    ctx.fillRect(left, y + 2, right - left, h - 4);
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    ctx.fillStyle = LG?.WIDGET_SECONDARY_TEXT_COLOR ?? "#999";
    ctx.fillText(label, left + 4, y + h / 2);
    const room = right - (left + 4 + ctx.measureText(label).width + 12);
    ctx.textAlign = "right";
    ctx.fillStyle = GLOW;
    ctx.fillText(ctx.measureText(long).width <= room ? long : short, right, y + h / 2);
    ctx.restore();
}

function decorate(widget, entry, withValue) {
    const base = widget.drawWidget;
    if (typeof base !== "function") return false;
    widget.drawWidget = function (ctx, opts) {
        base.call(this, ctx, opts);
        if (!entry.count) return;
        const margin = this.constructor?.margin ?? 15;
        const x = margin;
        const w = (opts?.width ?? 0) - 2 * margin;
        const h = this.height ?? 20;
        if (withValue) {
            overlayValue(ctx, this, x, this.y, w, h,
                `✦ ${entry.count} ${entry.cfg.noun} · random per seed`, `✦ ${entry.count} ${entry.cfg.noun}`);
        }
        strokeGlow(ctx, x, this.y, w, h);
    };
    return true;
}

// Older frontends without class widgets: a frame from onDrawForeground.
function drawGlowBehind(ctx, node, widget) {
    const y = widget?.last_y ?? widget?.y;
    if (y == null) return;
    const h = globalThis.LiteGraph?.NODE_WIDGET_HEIGHT ?? 20;
    strokeGlow(ctx, 12, y - 3, node.size[0] - 24, h + 6);
}

function setup(node, configs) {
    const entries = [];
    for (const cfg of configs) {
        const combo = node.widgets?.find(w => w.name === cfg.combo);
        const hidden = node.widgets?.find(w => w.name === cfg.hidden);
        if (!combo || !hidden) continue;
        hideWidget(hidden);
        const entry = { cfg, combo, hidden, button: null, count: 0 };
        entry.active = () => {
            const known = new Set(comboValues(combo));
            return parseChoices(hidden.value).filter(v => known.has(v));
        };
        entry.button = node.addWidget("button", cfg.label, null, () => openFor(node, entry));
        const a = decorate(combo, entry, true);
        const b = decorate(entry.button, entry, false);
        entry.needsFallback = !(a && b);
        entries.push(entry);
    }
    if (!entries.length) return;

    // FVMTools.JB.Blocks adds Edit List / Edit Overrides in its own
    // nodeCreated, before or after this one depending on load order. Keep
    // these buttons last so the layout doesn't change between sessions.
    // (Buttons serialise as null, so their order in widgets_values is moot.)
    setTimeout(() => {
        for (const e of entries) {
            const i = node.widgets.indexOf(e.button);
            if (i >= 0) node.widgets.push(...node.widgets.splice(i, 1));
        }
        node.setDirtyCanvas?.(true, true);
    }, 0);

    const refreshLabels = () => {
        for (const e of entries) {
            e.count = e.active().length;
            const text = e.count ? `✦ ${e.count} ${e.cfg.noun} drawn per seed — edit` : e.cfg.label;
            e.button.label = text;
            e.button.name = text;
        }
        node.setDirtyCanvas?.(true, true);
    };
    node.__fvmMultiRefresh = refreshLabels;

    // Workflows saved before these widgets existed restore a button's null
    // into them (widgets_values are positional) — that must read as "empty".
    const onConfigure = node.onConfigure;
    node.onConfigure = function () {
        const r = onConfigure?.apply(this, arguments);
        for (const e of entries) {
            if (typeof e.hidden.value !== "string") e.hidden.value = "";
        }
        setTimeout(refreshLabels, 0);
        return r;
    };

    if (entries.some(e => e.needsFallback)) {
        const onDrawForeground = node.onDrawForeground;
        node.onDrawForeground = function (ctx) {
            const r = onDrawForeground?.apply(this, arguments);
            if (this.flags?.collapsed) return r;
            for (const e of entries) {
                if (e.needsFallback && e.count) {
                    drawGlowBehind(ctx, this, e.combo);
                    drawGlowBehind(ctx, this, e.button);
                }
            }
            return r;
        };
    }

    for (const e of entries) {
        if (typeof e.hidden.value !== "string") e.hidden.value = "";
    }
    refreshLabels();
}

function openFor(node, entry) {
    if (!modal) modal = createModal();
    const choices = comboValues(entry.combo);
    const known = new Set(choices);
    const saved = parseChoices(entry.hidden.value);
    const preChecked = saved.length ? saved.filter(v => known.has(v)) : [entry.combo.value];
    modal.open({
        heading: `${entry.cfg.label} — ${entry.cfg.combo}`,
        choices,
        preChecked,
        unknown: saved.filter(v => !known.has(v)),
        apply: (picked) => {
            if (picked.length === 1) {
                entry.combo.value = picked[0];
                entry.combo.callback?.(picked[0]);
                entry.hidden.value = "";
            } else {
                entry.hidden.value = picked.join("\n");
            }
            node.__fvmMultiRefresh?.();
            node.graph?.change?.();
        },
    });
}

app.registerExtension({
    name: "FVMTools.JB.MultiSelect",
    async nodeCreated(node) {
        const configs = CONFIGS[node.comfyClass];
        if (configs) setup(node, configs);
    },
});
