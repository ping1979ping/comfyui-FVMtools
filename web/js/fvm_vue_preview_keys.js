/**
 * Nodes 2.0 image preview: arrow keys must not also move the canvas.
 *
 * The Vue image preview (frontend 1.52) handles ← / → to page through a
 * batch or list and calls preventDefault() — but not stopPropagation(). The
 * key then bubbles on to the global keybinding handler on `window`, which
 * runs its own arrow-key command (Comfy.Canvas.MoveSelectedNodes.*), so the
 * picture changes AND the canvas shifts.
 *
 * Fix: on the way up, stop an arrow key at `document` when a widget inside a
 * Vue node already consumed it (defaultPrevented). Text fields are left
 * alone, and arrows nobody inside a node handled still reach the keybindings.
 */
import { app } from "../../scripts/app.js";

const ARROWS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"]);

function isTextField(el) {
    return el instanceof HTMLTextAreaElement
        || el instanceof HTMLInputElement
        || el?.isContentEditable;
}

document.addEventListener("keydown", (e) => {
    if (!ARROWS.has(e.key) || !e.defaultPrevented) return;
    const t = e.target;
    if (!(t instanceof Element) || isTextField(t)) return;
    if (!t.closest("[data-node-id]")) return;
    e.stopPropagation();
});

app.registerExtension({ name: "FVMTools.VuePreviewArrowKeys" });
