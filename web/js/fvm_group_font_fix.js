/**
 * FVMtools.GroupFontSizeFix — make the per-group title font size actually work.
 *
 * Frontend bugs (seen in comfyui_frontend_package 1.45.20, still present in 1.52.7):
 *  1. The group context-menu option "Font size" writes `group.font_size`, but
 *     LGraphGroup.draw() renders the title with the global
 *     `LiteGraph.GROUP_TEXT_SIZE` and never reads `font_size`.
 *  2. The title bar height is the global `LiteGraph.NODE_TITLE_HEIGHT` — the
 *     title bar, its hit area, the selection outline and "fit group to nodes"
 *     (resizeTo) all ignore the font size, so a big title overflows its bar.
 *  3. serialize()/configure() drop `font_size`, so the size is lost on reload
 *     although the workflow schema allows it.
 *
 * Fix: titleHeight scales with font_size, draw() swaps both globals for the
 * duration of the call, and font_size is round-tripped through the workflow.
 * When the font size is changed interactively the group grows upwards by the
 * title delta, so the title bar does not cover the nodes inside.
 * Harmless once upstream reads font_size itself.
 */
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "FVMtools.GroupFontSizeFix",
    async setup() {
        const LG = window.LiteGraph;
        const proto = LG?.LGraphGroup?.prototype;
        if (!proto?.draw || proto.draw.__fvmGroupFontFix) return;

        // While draw() runs the globals are swapped; keep the real defaults here.
        let baseText = null;
        let baseTitle = null;
        const defaultText = () => baseText ?? LG.GROUP_TEXT_SIZE;
        const defaultTitle = () => baseTitle ?? LG.NODE_TITLE_HEIGHT;

        const fontOf = (group) => {
            const fs = Number(group.font_size);
            return Number.isFinite(fs) && fs > 0 ? fs : null;
        };

        const titleHeightOf = (group) => {
            const fs = fontOf(group);
            const text = defaultText();
            const title = defaultTitle();
            if (!fs || fs === text || !(text > 0)) return title;
            return Math.max(Math.round((title * fs) / text), Math.round(fs * 1.2));
        };

        Object.defineProperty(proto, "titleHeight", {
            configurable: true,
            get() {
                return titleHeightOf(this);
            },
        });

        const origDraw = proto.draw;
        proto.draw = function (canvas, ctx) {
            const savedText = LG.GROUP_TEXT_SIZE;
            const savedTitle = LG.NODE_TITLE_HEIGHT;
            const titleHeight = titleHeightOf(this);

            // Font size changed since the last frame: grow upwards so the
            // enlarged title bar keeps clear of the contained nodes.
            const last = this.__fvmTitleHeight;
            if (last !== undefined && last !== titleHeight && !this.pinned) {
                const delta = titleHeight - last;
                this._pos[1] -= delta;
                this._size[1] += delta;
            }
            this.__fvmTitleHeight = titleHeight;

            const fs = fontOf(this);
            baseText = savedText;
            baseTitle = savedTitle;
            if (fs) LG.GROUP_TEXT_SIZE = fs;
            LG.NODE_TITLE_HEIGHT = titleHeight;
            try {
                return origDraw.call(this, canvas, ctx);
            } finally {
                LG.GROUP_TEXT_SIZE = savedText;
                LG.NODE_TITLE_HEIGHT = savedTitle;
                baseText = null;
                baseTitle = null;
            }
        };
        proto.draw.__fvmGroupFontFix = true;

        const origSerialize = proto.serialize;
        proto.serialize = function () {
            const data = origSerialize.call(this);
            const fs = fontOf(this);
            if (fs && fs !== LG.GROUP_TEXT_SIZE) data.font_size = fs;
            return data;
        };

        const origConfigure = proto.configure;
        proto.configure = function (data) {
            origConfigure.call(this, data);
            const fs = Number(data?.font_size);
            if (Number.isFinite(fs) && fs > 0) this.font_size = fs;
        };
    },
});
