/**
 * FVM_DatasetPromptList — editor toolbar for the prompt_text widget.
 *
 *   [Presets ▾] [Save Preset] [Search & Replace] [Preview] [Wildcards] [Syntax] [?]
 *   72 lines · close 26 · half 24 · full 22 · untagged 0
 *
 * - Presets ▾   load (replace) or append a list from dataset_presets/presets/
 * - Save Preset store the current text as a preset
 * - Search & Replace  swap text / __wildcards__ (plain or regex, undo),
 *               plus one-click unisex / female / male outfit slots
 * - Preview     resolve all prompts server-side with the current widgets
 * - Wildcards / Syntax / __ autocomplete are shared with the JB Builder.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    attachWildcardAutocomplete,
    createSyntaxInfoModal,
    createWildcardsModal,
} from "./fvm_jb_builder.js";
import { dismissOnOutside } from "./_widgets_common.js";

const NODE_NAME = "FVM_DatasetPromptList";
const TAG_RE = /^\[([a-zA-Z0-9_ ,\-]+)\]/;

let wildcardsModal = null;
let syntaxModal = null;

const BTN_STYLE = {
    background: "#313244", color: "#cdd6f4", border: "1px solid #45475a",
    borderRadius: "4px", padding: "3px 9px", fontSize: "12px", cursor: "pointer",
};

function button(label, title) {
    const b = document.createElement("button");
    b.textContent = label;
    b.title = title;
    Object.assign(b.style, BTN_STYLE);
    return b;
}

function countLines(text) {
    const c = { total: 0, close: 0, half: 0, full: 0, untagged: 0 };
    for (const raw of (text || "").split("\n")) {
        const s = raw.trim();
        if (!s || s.startsWith("#")) continue;
        c.total++;
        const m = s.match(TAG_RE);
        const tags = m ? m[1].split(",").map(t => t.trim().toLowerCase()) : [];
        const shot = ["close", "half", "full"].find(t => tags.includes(t));
        c[shot || "untagged"]++;
    }
    return c;
}

// ─── generic modal ──────────────────────────────────────────────────

function openModal(titleText, bodyEl, buttons = []) {
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        position: "fixed", inset: "0", background: "rgba(0,0,0,0.6)",
        zIndex: "10000", display: "flex", justifyContent: "center", alignItems: "center",
    });
    const dialog = document.createElement("div");
    Object.assign(dialog.style, {
        background: "#1e1e2e", color: "#cdd6f4", borderRadius: "10px",
        padding: "16px", width: "min(900px, 92vw)", maxHeight: "85vh",
        display: "flex", flexDirection: "column", gap: "10px",
        boxShadow: "0 8px 32px rgba(0,0,0,0.5)", fontFamily: "monospace",
    });
    const title = document.createElement("div");
    title.textContent = titleText;
    Object.assign(title.style, { fontWeight: "bold", fontSize: "14px" });
    const row = document.createElement("div");
    Object.assign(row.style, { display: "flex", gap: "8px", justifyContent: "flex-end" });
    const close = () => overlay.remove();
    for (const [label, fn] of [...buttons, ["Close", close]]) {
        const b = button(label, "");
        b.addEventListener("click", () => fn(close));
        row.append(b);
    }
    dialog.append(title, bodyEl, row);
    overlay.append(dialog);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    document.body.append(overlay);
    return close;
}

// ─── help ───────────────────────────────────────────────────────────

const HELP_HTML = `
<div style="line-height:1.55;font-size:12.5px;overflow:auto;max-height:65vh">
<b style="color:#89b4fa">What this node does</b><br>
Like <i>CR Prompt List</i>: every line is one prompt, downstream nodes run once per
prompt. On top: wildcards are rolled per image, so one line gives a new outfit,
background and light every time.<br><br>
<b style="color:#89b4fa">Line format</b><br>
<code style="color:#a6e3a1">[close] Left three-quarter portrait …, wearing __dataset/upper__, __dataset/setting__</code><br>
• <code>#</code> at line start = comment &nbsp;• empty lines are ignored<br>
• tag <code>[close]</code> head &amp; shoulders · <code>[half]</code> waist-up/seated ·
<code>[full]</code> head to feet · extra tags allowed: <code>[full, back]</code><br><br>
<b style="color:#89b4fa">Widgets</b><br>
• <b>shot_filter</b> only close / half / full lines<br>
• <b>start_index / max_rows</b> which lines (counted after the filter)<br>
• <b>variations</b> each line N times with new wildcards (20 × 10 = 200 images)<br>
• <b>order</b> rounds = every angle first, then again — safe to stop early<br>
• <b>seed</b> same seed = same prompts; change it for a fresh set<br>
• <b>prefix / suffix</b> trigger word in front, photo terms at the end<br><br>
<b style="color:#89b4fa">Outputs</b><br>
• <b>prompt</b> → text encoder &nbsp;• <b>caption</b> → training .txt (no prefix/suffix)<br>
• <b>shot</b> → file name / subfolder &nbsp;• <b>listing</b> → Show Text for a check<br><br>
<b style="color:#89b4fa">Built-in wildcard slots</b> (edit with “Wildcards”)<br>
<code>__dataset/upper__</code> top (+ jacket) for close/half ·
<code>__dataset/outfit__</code> full outfit ·
<code>__dataset/setting__</code> background + matching light (mix) ·
<code>__dataset/setting_studio__ / _indoor__ / _outdoor__</code> ·
<code>__dataset/expression__</code> · <code>__dataset/arms__</code> ·
<code>__dataset/stance__</code> · <code>__dataset/photo__</code><br>Gendered outfits: <code>__dataset/female/upper__</code>, <code>__dataset/female/outfit__</code>, <code>__dataset/male/…__</code> — switch all slots at once with <b>Search &amp; Replace → Outfit slots</b>.<br><br>
Full guide: <code>documentation/WILDCARDS_GUIDE.md</code>
</div>`;

// ─── search & replace ───────────────────────────────────────────────

// Outfit slots that differ per gender: unisex __dataset/upper__, female
// __dataset/female/upper__, male __dataset/male/upper__ (same for outfit).
const GENDER_SLOT_RE = /__dataset\/(?:female\/|male\/)?(upper|outfit)__/g;

function escapeRegExp(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function buildMatcher(find, useRegex, caseSensitive) {
    if (!find) return null;
    const flags = "g" + (caseSensitive ? "" : "i");
    try {
        return new RegExp(useRegex ? find : escapeRegExp(find), flags);
    } catch (e) {
        return e;   // invalid regex — caller shows the message
    }
}

function openSearchReplace(getText, setText) {
    let undoText = null;
    const body = document.createElement("div");
    Object.assign(body.style, { display: "flex", flexDirection: "column", gap: "8px", fontSize: "12.5px" });

    const field = (label, placeholder) => {
        const wrap = document.createElement("label");
        Object.assign(wrap.style, { display: "flex", alignItems: "center", gap: "8px" });
        const span = document.createElement("span");
        span.textContent = label;
        span.style.width = "70px";
        const input = document.createElement("input");
        input.placeholder = placeholder;
        Object.assign(input.style, {
            flex: "1", background: "#11111b", color: "#cdd6f4",
            border: "1px solid #45475a", borderRadius: "4px", padding: "5px",
            fontFamily: "Consolas, monospace",
        });
        attachWildcardAutocomplete(input);
        wrap.append(span, input);
        return { wrap, input };
    };
    const find = field("Find", "text, expression or __wildcard__ (type __ for suggestions)");
    const repl = field("Replace", "new text or __wildcard__ — empty deletes");

    const opts = document.createElement("div");
    Object.assign(opts.style, { display: "flex", gap: "14px", paddingLeft: "78px" });
    const check = (label, title) => {
        const l = document.createElement("label");
        l.title = title;
        const c = document.createElement("input");
        c.type = "checkbox";
        l.append(c, " " + label);
        opts.append(l);
        return c;
    };
    const caseBox = check("Case sensitive", "Match upper/lower case exactly");
    const regexBox = check("Regex", "Find is a regular expression; use $1 in Replace for groups");

    const status = document.createElement("div");
    Object.assign(status.style, { paddingLeft: "78px", color: "#a6adc8", minHeight: "16px" });

    const quick = document.createElement("div");
    Object.assign(quick.style, {
        display: "flex", gap: "6px", alignItems: "center", flexWrap: "wrap",
        borderTop: "1px solid #313244", paddingTop: "8px",
    });
    const qLabel = document.createElement("span");
    qLabel.textContent = "Outfit slots →";
    qLabel.title = "Swap every __dataset/upper__ and __dataset/outfit__ slot in one click";
    quick.append(qLabel);

    function applyText(next, message) {
        const before = getText();
        if (next === before) { status.textContent = "nothing changed"; return; }
        undoText = before;
        setText(next);
        status.style.color = "#a6e3a1";
        status.textContent = message;
        refreshCount();
    }

    for (const [label, prefix] of [["Unisex", ""], ["Female", "female/"], ["Male", "male/"]]) {
        const b = button(label, `Use __dataset/${prefix}upper__ and __dataset/${prefix}outfit__`);
        b.addEventListener("click", () => {
            const text = getText();
            const n = (text.match(GENDER_SLOT_RE) || []).length;
            applyText(text.replace(GENDER_SLOT_RE, `__dataset/${prefix}$1__`),
                `${n} outfit slot(s) set to ${label.toLowerCase()}`);
        });
        quick.append(b);
    }

    function refreshCount() {
        const m = buildMatcher(find.input.value, regexBox.checked, caseBox.checked);
        status.style.color = "#a6adc8";
        if (!m) { status.textContent = ""; return; }
        if (m instanceof Error) { status.style.color = "#f38ba8"; status.textContent = m.message; return; }
        const n = (getText().match(m) || []).length;
        status.textContent = `${n} match${n === 1 ? "" : "es"}`;
    }
    for (const el of [find.input, caseBox, regexBox]) el.addEventListener("input", refreshCount);

    body.append(find.wrap, repl.wrap, opts, status, quick);
    openModal("Search & Replace", body, [
        ["Undo", () => {
            if (undoText === null) { status.textContent = "nothing to undo"; return; }
            setText(undoText);
            undoText = null;
            status.textContent = "undone";
            refreshCount();
        }],
        ["Replace All", () => {
            const m = buildMatcher(find.input.value, regexBox.checked, caseBox.checked);
            if (!m || m instanceof Error) return;
            const text = getText();
            const n = (text.match(m) || []).length;
            // Plain mode inserts the replacement literally ($ has no meaning).
            const next = regexBox.checked
                ? text.replace(m, repl.input.value)
                : text.replace(m, () => repl.input.value);
            applyText(next, `replaced ${n} match${n === 1 ? "" : "es"}`);
        }],
    ]);
    find.input.focus();
}

// ─── toolbar ────────────────────────────────────────────────────────

function buildToolbar(node) {
    const w = (name) => node.widgets?.find(x => x.name === name);
    const textW = w("prompt_text");
    // Classic canvas: the widget owns its <textarea> (inputEl). Nodes 2.0
    // (Vue) renders its own textarea inside [data-node-id] instead, so look
    // it up at use time — it may not exist yet when the node is created.
    const findTextEl = () => {
        const el = textW?.inputEl || textW?.element;
        if (el instanceof HTMLTextAreaElement) return el;
        return document.querySelector(`[data-node-id="${node.id}"] textarea`);
    };

    const host = document.createElement("div");
    Object.assign(host.style, {
        display: "flex", flexDirection: "column", gap: "4px", padding: "2px 0",
        fontFamily: "Consolas, monospace", fontSize: "12px", color: "#cdd6f4",
    });
    const bar = document.createElement("div");
    Object.assign(bar.style, { display: "flex", gap: "5px", flexWrap: "wrap" });
    const presetsBtn = button("Presets ▾", "Load or append a saved prompt list");
    const saveBtn = button("Save Preset", "Save the current text as a preset");
    const replaceBtn = button("Search & Replace", "Find and swap expressions, words or __wildcards__; one-click unisex / female / male outfit slots");
    const previewBtn = button("Preview", "Resolve all prompts with the current settings — no image generation");
    const wildBtn = button("Wildcards", "Edit wildcard files (clothes, backgrounds, light …) — no restart needed");
    const syntaxBtn = button("Syntax", "Wildcard / bracket / variable syntax reference");
    const helpBtn = button("?", "How this node works");
    previewBtn.style.color = "#a6e3a1";
    syntaxBtn.style.color = helpBtn.style.color = "#89b4fa";
    bar.append(presetsBtn, saveBtn, replaceBtn, previewBtn, wildBtn, syntaxBtn, helpBtn);

    const stats = document.createElement("div");
    stats.style.color = "#a6adc8";
    host.append(bar, stats);

    const getText = () => (textW ? String(textW.value ?? "") : "");
    const setText = (t) => {
        if (!textW) return;
        textW.value = t;
        textW.callback?.(t);
        const el = findTextEl();
        if (el && el.value !== t) {
            el.value = t;
            // Vue's v-model only listens to input events on its textarea.
            el.dispatchEvent(new Event("input", { bubbles: true }));
        }
        refreshStats();
        node.setDirtyCanvas(true, true);
    };
    function refreshStats() {
        const c = countLines(getText());
        stats.textContent = `${c.total} lines · close ${c.close} · half ${c.half} · full ${c.full}`
            + (c.untagged ? ` · untagged ${c.untagged}` : "");
    }
    // Autocomplete + live stats for every text field of this node. Called
    // once per element: directly for the classic textarea, and from the
    // focusin delegate below for Vue-rendered fields.
    const attached = new WeakSet();
    function attach(el) {
        if (!el || attached.has(el)) return;
        attached.add(el);
        attachWildcardAutocomplete(el);
        el.addEventListener("input", () => setTimeout(refreshStats, 0));
    }
    node.__fvmDatasetAttach = attach;
    const classic = textW?.inputEl;
    if (classic instanceof HTMLTextAreaElement) attach(classic);
    refreshStats();

    // Presets dropdown
    let closePresetMenu = null;
    presetsBtn.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        // A second click on the button toggles the open menu closed.
        if (closePresetMenu) { closePresetMenu(); return; }
        const resp = await api.fetchApi("/fvmtools/dataset-presets");
        const names = resp.ok ? (await resp.json()).presets : [];
        const menu = document.createElement("div");
        const r = presetsBtn.getBoundingClientRect();
        Object.assign(menu.style, {
            position: "fixed", left: r.left + "px", top: (r.bottom + 2) + "px",
            background: "#1e1e2e", border: "1px solid #45475a", borderRadius: "6px",
            padding: "4px 0", zIndex: "9999", minWidth: "280px",
            boxShadow: "0 4px 12px rgba(0,0,0,0.5)", fontFamily: "monospace", fontSize: "12px",
        });
        if (!names.length) menu.textContent = "  no presets found";
        for (const name of names) {
            const item = document.createElement("div");
            Object.assign(item.style, { display: "flex", alignItems: "center", gap: "6px", padding: "3px 10px" });
            const label = document.createElement("span");
            label.textContent = name;
            label.style.flex = "1";
            const load = button("Load", "Replace the text with this preset");
            const add = button("+ Append", "Add this preset below the current text");
            for (const [b, append] of [[load, false], [add, true]]) {
                b.addEventListener("click", async () => {
                    closePresetMenu?.();
                    const res = await api.fetchApi(`/fvmtools/dataset-preset?name=${encodeURIComponent(name)}`);
                    if (!res.ok) return;
                    const t = (await res.json()).text;
                    setText(append ? getText().replace(/\s*$/, "\n\n") + t : t);
                });
            }
            item.append(label, load, add);
            menu.append(item);
        }
        document.body.append(menu);
        closePresetMenu = dismissOnOutside(menu, () => { closePresetMenu = null; }, presetsBtn);
    });

    // Save preset
    saveBtn.addEventListener("click", () => {
        const body = document.createElement("div");
        body.innerHTML = `<div style="margin-bottom:6px">Preset name (letters, digits, _ and -). An existing name is overwritten.</div>`;
        const input = document.createElement("input");
        Object.assign(input.style, {
            width: "100%", background: "#11111b", color: "#cdd6f4",
            border: "1px solid #45475a", borderRadius: "4px", padding: "5px",
        });
        input.value = "my_character_set";
        const msg = document.createElement("div");
        msg.style.marginTop = "6px";
        body.append(input, msg);
        openModal("Save Preset", body, [["Save", async (close) => {
            const res = await api.fetchApi("/fvmtools/dataset-preset", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: input.value.trim(), text: getText() }),
            });
            if (res.ok) close();
            else { msg.style.color = "#f38ba8"; msg.textContent = (await res.json()).error || "save failed"; }
        }]]);
        input.focus();
        input.select();
    });

    // Preview
    async function runPreview(out) {
        const val = (n, d) => (w(n) ? w(n).value : d);
        const res = await api.fetchApi("/fvmtools/dataset-preview", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: getText(), shot_filter: val("shot_filter", "all"),
                start_index: val("start_index", 0), max_rows: val("max_rows", 1000),
                variations: val("variations", 1), order: val("order", "rounds"),
                seed: val("seed", 0), prefix: val("prefix", ""), suffix: val("suffix", ""),
            }),
        });
        const data = await res.json();
        out.value = res.ok ? data.listing : `Error: ${data.error}`;
    }
    previewBtn.addEventListener("click", async () => {
        const out = document.createElement("textarea");
        out.readOnly = true;
        Object.assign(out.style, {
            width: "100%", height: "60vh", background: "#11111b", color: "#cdd6f4",
            border: "1px solid #45475a", borderRadius: "6px", padding: "8px",
            fontFamily: "Consolas, monospace", fontSize: "12px", whiteSpace: "pre-wrap",
        });
        out.value = "resolving …";
        openModal("Preview — resolved prompts (#line.variation [shot])", out, [
            ["New Seed", async () => {
                const s = w("seed");
                if (s) { s.value = Math.floor(Math.random() * 2 ** 32); s.callback?.(s.value); }
                node.setDirtyCanvas(true, true);
                await runPreview(out);
            }],
        ]);
        await runPreview(out);
    });

    replaceBtn.addEventListener("click", () => openSearchReplace(getText, setText));
    wildBtn.addEventListener("click", () => {
        if (!wildcardsModal) wildcardsModal = createWildcardsModal();
        wildcardsModal.open();
    });
    syntaxBtn.addEventListener("click", () => {
        if (!syntaxModal) syntaxModal = createSyntaxInfoModal();
        syntaxModal.open();
    });
    helpBtn.addEventListener("click", () => {
        const body = document.createElement("div");
        body.innerHTML = HELP_HTML;
        openModal("FVM · Dataset Prompt List — Help", body);
    });

    return { host, refreshStats };
}

// Nodes 2.0: widgets are Vue components with their own inputs. Attach to a
// text field of a Dataset Prompt List the first time it gets focus.
document.addEventListener("focusin", (e) => {
    const t = e.target;
    const isText = t instanceof HTMLTextAreaElement
        || (t instanceof HTMLInputElement && (t.type === "text" || t.type === ""));
    if (!isText) return;
    const host = t.closest?.("[data-node-id]");
    if (!host) return;
    const id = host.dataset.nodeId;
    const graphs = [app.canvas?.graph, app.graph].filter(Boolean);
    let node = null;
    for (const g of graphs) {
        node = g.getNodeById?.(id) ?? g.getNodeById?.(Number(id));
        if (node) break;
    }
    if (node && (node.comfyClass === NODE_NAME || node.type === NODE_NAME)) {
        node.__fvmDatasetAttach?.(t);
    }
}, true);

app.registerExtension({
    name: "FVMTools.DatasetPromptList",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const { host, refreshStats } = buildToolbar(this);
            this.__fvmDatasetRefresh = refreshStats;
            this.addDOMWidget("dataset_toolbar", "div", host, {
                serialize: false,
                getHeight: () => Math.max(56, host.scrollHeight + 6),  // toolbar may wrap
            });
            this.size = [Math.max(this.size?.[0] || 0, 560), Math.max(this.size?.[1] || 0, 520)];
            return r;
        };

        // Saved workflows restore widget values after creation.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            setTimeout(() => this.__fvmDatasetRefresh?.(), 0);
            return r;
        };
    },
});
