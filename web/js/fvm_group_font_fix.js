/**
 * FVMtools.GroupFontSizeFix — restore per-group title font size.
 *
 * Frontend regression (seen in comfyui_frontend_package 1.45.20): the group
 * context-menu option "Font size" still writes `group.font_size`, but
 * LGraphGroup.draw() renders the title with the global constant
 * `LiteGraph.GROUP_TEXT_SIZE` and never reads `font_size`.
 *
 * This wraps draw() and swaps the global to the group's own font_size for the
 * duration of the call. Harmless once upstream fixes draw() to read font_size.
 */
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "FVMtools.GroupFontSizeFix",
    async setup() {
        const LGraphGroup = window.LiteGraph?.LGraphGroup;
        const proto = LGraphGroup?.prototype;
        if (!proto?.draw || proto.draw.__fvmGroupFontFix) return;

        const origDraw = proto.draw;
        proto.draw = function (canvas, ctx) {
            const saved = LiteGraph.GROUP_TEXT_SIZE;
            const fs = Number(this.font_size);
            if (Number.isFinite(fs) && fs > 0) LiteGraph.GROUP_TEXT_SIZE = fs;
            try {
                return origDraw.call(this, canvas, ctx);
            } finally {
                LiteGraph.GROUP_TEXT_SIZE = saved;
            }
        };
        proto.draw.__fvmGroupFontFix = true;
    },
});
