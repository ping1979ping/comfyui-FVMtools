/**
 * FVM_Ideogram_BoxJitter — default jitter block + per-box override slots.
 *
 *   ┌─ Default (alle Boxen) ─────────────────────────────────────┐
 *   │  pos ±[0.06]  size ±[0.12]  aspect ±[0.08]  min [0.03]      │
 *   └─────────────────────────────────────────────────────────────┘
 *   [ + Box-Regel ]
 *     boxes:[ logo          ]  pos±[0.12] size±[0.03] aspect±[0.00] ✕
 *
 * State is serialised as JSON into the hidden `jitter_rules` STRING widget:
 *   { default:{pos,size,aspect,min}, overrides:[{boxes,pos,size,aspect}, ...] }
 */
import { app } from "../../scripts/app.js";

const NODE_NAME = "FVM_Ideogram_BoxJitter";
const DEF = { pos: 0.06, size: 0.12, aspect: 0.08, min: 0.03 };

function parseRules(text) {
    try {
        const o = JSON.parse(text);
        if (o && typeof o === "object" && o.default) {
            return {
                default: { ...DEF, ...o.default },
                overrides: Array.isArray(o.overrides) ? o.overrides.map(r => ({
                    boxes: String(r.boxes ?? ""),
                    pos: numOr(r.pos, DEF.pos), size: numOr(r.size, DEF.size),
                    aspect: numOr(r.aspect, DEF.aspect),
                })) : [],
            };
        }
    } catch (e) { /* ignore */ }
    return { default: { ...DEF }, overrides: [] };
}

function numOr(v, fallback) {
    const n = parseFloat(v);
    return Number.isFinite(n) ? n : fallback;
}

function clamp(n, lo, hi) { return Math.max(lo, Math.min(hi, n)); }

// Natural content height from the children — host gets `h-full` from the
// frontend so its own scrollHeight echoes the dragged node height. (Same fix
// as the JB Builder widget.)
function measureContentHeight(host) {
    const cs = getComputedStyle(host);
    const padV = (parseFloat(cs.paddingTop) || 0) + (parseFloat(cs.paddingBottom) || 0);
    const borderV = (parseFloat(cs.borderTopWidth) || 0) + (parseFloat(cs.borderBottomWidth) || 0);
    const gap = parseFloat(cs.rowGap || cs.gap) || 0;
    const kids = host.children;
    let inner = padV + borderV + gap * Math.max(0, kids.length - 1);
    for (const k of kids) inner += k.offsetHeight;
    return Math.max(50, Math.ceil(inner) + 2);
}

function labeledNum(labelText, value, onCommit, { lo = 0, hi = 1, step = 0.01 } = {}) {
    const wrap = document.createElement("label");
    Object.assign(wrap.style, {
        display: "inline-flex", alignItems: "center", gap: "3px",
        color: "#a6adc8", fontSize: "11px",
    });
    const lbl = document.createElement("span"); lbl.textContent = labelText;
    const inp = document.createElement("input");
    inp.type = "number"; inp.value = String(value); inp.step = String(step);
    inp.min = String(lo); inp.max = String(hi);
    Object.assign(inp.style, {
        width: "52px", background: "#181825", color: "#cdd6f4",
        border: "1px solid #45475a", borderRadius: "4px", padding: "2px 4px",
        fontSize: "11px", fontFamily: "inherit",
    });
    inp.addEventListener("input", () => {
        const n = clamp(numOr(inp.value, value), lo, hi);
        onCommit(n);
    });
    wrap.append(lbl, inp);
    return wrap;
}

function buildWidget(node) {
    const hidden = node.widgets.find(w => w.name === "jitter_rules");
    let rules = parseRules(hidden ? hidden.value : "");

    const host = document.createElement("div");
    Object.assign(host.style, {
        display: "flex", flexDirection: "column", gap: "6px",
        background: "#1a1a25", border: "1px solid #2a2a3a",
        borderRadius: "6px", padding: "6px",
        fontFamily: "Consolas, 'Courier New', monospace",
        color: "#cdd6f4", fontSize: "12px",
    });

    // ── Default block ──
    const defBox = document.createElement("div");
    Object.assign(defBox.style, {
        display: "flex", flexWrap: "wrap", alignItems: "center", gap: "8px",
        background: "#181825", border: "1px solid #313244",
        borderRadius: "5px", padding: "6px 8px",
    });
    const defTitle = document.createElement("div");
    defTitle.textContent = "Default (alle Boxen)";
    Object.assign(defTitle.style, { color: "#89b4fa", flex: "0 0 100%", marginBottom: "2px" });

    function commit() {
        if (hidden) {
            hidden.value = JSON.stringify(rules);
            if (typeof hidden.callback === "function") hidden.callback(hidden.value);
        }
        node.setDirtyCanvas(true, true);
    }

    defBox.append(defTitle);
    defBox.append(
        labeledNum("pos ±", rules.default.pos, v => { rules.default.pos = v; commit(); }),
        labeledNum("size ±", rules.default.size, v => { rules.default.size = v; commit(); }),
        labeledNum("aspect ±", rules.default.aspect, v => { rules.default.aspect = v; commit(); }),
        labeledNum("min", rules.default.min, v => { rules.default.min = v; commit(); }, { lo: 0, hi: 0.5 }),
    );

    // ── Override list + add button ──
    const ovList = document.createElement("div");
    Object.assign(ovList.style, { display: "flex", flexDirection: "column", gap: "4px" });

    const addBtn = document.createElement("button");
    addBtn.textContent = "+ Box-Regel";
    Object.assign(addBtn.style, {
        background: "#313244", color: "#a6e3a1", border: "1px solid #45475a",
        borderRadius: "4px", padding: "4px 10px", fontSize: "12px",
        cursor: "pointer", alignSelf: "flex-start",
    });
    addBtn.addEventListener("click", () => {
        rules.overrides.push({ boxes: "", pos: rules.default.pos, size: rules.default.size, aspect: rules.default.aspect });
        renderOverrides(); commit();
    });

    function makeOverrideRow(ov, idx) {
        const row = document.createElement("div");
        Object.assign(row.style, { display: "flex", flexWrap: "wrap", alignItems: "center", gap: "6px" });

        const namesIn = document.createElement("input");
        namesIn.type = "text"; namesIn.value = ov.boxes; namesIn.placeholder = "box-name(n), Komma getrennt";
        Object.assign(namesIn.style, {
            flex: "1 1 150px", minWidth: "110px", background: "#181825", color: "#f9e2af",
            border: "1px solid #45475a", borderRadius: "4px", padding: "3px 6px",
            fontSize: "12px", fontFamily: "inherit",
        });
        namesIn.addEventListener("input", () => { ov.boxes = namesIn.value; commit(); });

        const del = document.createElement("button");
        del.textContent = "✕"; del.title = "Regel löschen";
        Object.assign(del.style, {
            flex: "0 0 24px", height: "22px", padding: "0",
            background: "#313244", color: "#f38ba8", border: "1px solid #6e3636",
            borderRadius: "4px", cursor: "pointer", fontSize: "12px",
        });
        del.addEventListener("click", () => {
            rules.overrides.splice(idx, 1); renderOverrides(); commit();
        });

        row.append(
            namesIn,
            labeledNum("pos ±", ov.pos, v => { ov.pos = v; commit(); }),
            labeledNum("size ±", ov.size, v => { ov.size = v; commit(); }),
            labeledNum("aspect ±", ov.aspect, v => { ov.aspect = v; commit(); }),
            del,
        );
        return row;
    }

    function renderOverrides() {
        ovList.innerHTML = "";
        rules.overrides.forEach((ov, i) => ovList.append(makeOverrideRow(ov, i)));
        host.style.height = "";
        node.setDirtyCanvas(true, true);
    }

    host.append(defBox, ovList, addBtn);
    renderOverrides();

    // Replay saved state after ComfyUI restores widget values (onConfigure).
    node.__fvmIjRefreshRules = () => {
        rules = parseRules(hidden ? hidden.value : "");
        // rebuild default inputs by recreating the block contents
        defBox.innerHTML = "";
        defBox.append(defTitle);
        defBox.append(
            labeledNum("pos ±", rules.default.pos, v => { rules.default.pos = v; commit(); }),
            labeledNum("size ±", rules.default.size, v => { rules.default.size = v; commit(); }),
            labeledNum("aspect ±", rules.default.aspect, v => { rules.default.aspect = v; commit(); }),
            labeledNum("min", rules.default.min, v => { rules.default.min = v; commit(); }, { lo: 0, hi: 0.5 }),
        );
        renderOverrides();
    };

    return host;
}

app.registerExtension({
    name: "FVMTools.Ideogram.BoxJitter",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            if (typeof this.__fvmIjRefreshRules === "function") {
                setTimeout(() => this.__fvmIjRefreshRules(), 0);
            }
            return r;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;
            const host = buildWidget(this);
            const domWidget = this.addDOMWidget("ij_rules_host", "div", host, {
                serialize: false,
                getHeight: () => measureContentHeight(host),
            });

            // Width tracking — force widget.width undefined so the layout uses
            // node.width (frontend ≥1.44 reads widget.width ?? node.width).
            try {
                Object.defineProperty(domWidget, "width", {
                    configurable: true, get() { return undefined; }, set() {},
                });
            } catch (e) { /* ignore */ }

            // Snap node height to content; observe children (content changes)
            // and host (node-resize) with a diff-guard against oscillation.
            try {
                let raf = 0;
                const reflow = () => {
                    if (raf) return;
                    raf = requestAnimationFrame(() => {
                        raf = 0;
                        try {
                            const c = node.computeSize();
                            if (c && Array.isArray(c) && node.size &&
                                Math.abs(node.size[1] - c[1]) > 0.5) {
                                node.size[1] = c[1];
                                node.setDirtyCanvas(true, true);
                            }
                        } catch (e) { /* ignore */ }
                    });
                };
                const ro = new ResizeObserver(reflow);
                for (const child of host.children) ro.observe(child);
                ro.observe(host);
            } catch (e) { /* ignore */ }

            // Hide the raw jitter_rules STRING widget — the host above is the UI.
            const hidden = this.widgets.find(w => w.name === "jitter_rules");
            if (hidden) {
                hidden.hidden = true;
                hidden.computeSize = () => [0, -4];
                hidden.draw = () => {};
                if (hidden.options) { hidden.options.hidden = true; hidden.options.serialize = true; }
            }

            try {
                const computed = this.computeSize();
                const w = Math.max(this.size?.[0] || 0, 420);
                const h = Math.max(this.size?.[1] || 0, computed?.[1] || 180);
                this.size = [w, h];
            } catch (e) { /* ignore */ }

            return r;
        };
    },
});
