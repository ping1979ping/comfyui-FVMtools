import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Batch Load Image: a progress bar showing how far through the folder we are,
// live counts fetched from disk (so the numbers are there before the first run),
// and a button to forget the progress and start over.

const BAR_HEIGHT = 22;

function widgetValue(node, name) {
    return node.widgets?.find((w) => w.name === name)?.value ?? "";
}

async function refreshStatus(node) {
    const directory = String(widgetValue(node, "directory") || "").trim();
    if (!directory) {
        node._fvmBatch = null;
        node.setDirtyCanvas(true);
        return;
    }
    const params = new URLSearchParams({
        directory,
        tracker: String(widgetValue(node, "tracker") || "default"),
        sort_by: String(widgetValue(node, "sort_by") || "name"),
        include_subdirs: widgetValue(node, "include_subdirs") ? "1" : "0",
        exclude: [widgetValue(node, "pass_subdir"), widgetValue(node, "fail_subdir")]
            .map((v) => String(v || "").trim())
            .filter(Boolean)
            .join("|"),
    });
    try {
        const response = await api.fetchApi(`/fvmtools/batch/status?${params}`);
        const data = await response.json();
        if (!data.ok) {
            node._fvmBatch = { error: data.error || "not found" };
        } else {
            const doneCount = data.total - data.remaining;
            node._fvmBatch = {
                position: doneCount,
                total: data.total,
                remaining: data.remaining,
                fraction: data.total ? doneCount / data.total : 0,
                filename: "",
            };
        }
    } catch (error) {
        node._fvmBatch = { error: String(error) };
    }
    node.setDirtyCanvas(true);
}

app.registerExtension({
    name: "FVMTools.BatchLoadImage",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "FVM_BatchLoadImage") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            try {
                // The bar draws itself into a zero-input widget slot, which keeps
                // it inside the node's layout instead of floating over it.
                this.addCustomWidget({
                    name: "_fvm_batch_bar",
                    type: "custom",
                    value: "",
                    options: { serialize: false },
                    // One widget row. The newer frontend lays custom widgets
                    // out at row height, so bar and caption share that row —
                    // text drawn below it used to spill into the Reset button.
                    computeSize: (width) => [width ?? 0, BAR_HEIGHT],
                    computeLayoutSize: () => ({ minHeight: BAR_HEIGHT, maxHeight: BAR_HEIGHT, minWidth: 0 }),
                    draw(ctx, node, width, posY) {
                        const state = node._fvmBatch;
                        const margin = 15;
                        const barWidth = width - margin * 2;
                        const y = posY + 2;
                        const height = BAR_HEIGHT - 4;
                        const finished = state && !state.error && state.total > 0 && state.remaining === 0;

                        ctx.save();
                        ctx.fillStyle = "#222";
                        ctx.beginPath();
                        ctx.roundRect(margin, y, barWidth, height, 4);
                        ctx.fill();

                        if (state && !state.error && state.total > 0) {
                            const fraction = Math.max(0, Math.min(1, state.fraction || 0));
                            ctx.fillStyle = finished ? "#2f6b4f" : "#2f4f6b";
                            ctx.beginPath();
                            ctx.roundRect(margin, y, Math.max(3, barWidth * fraction), height, 4);
                            ctx.fill();
                        }
                        ctx.strokeStyle = "#555";
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.roundRect(margin, y, barWidth, height, 4);
                        ctx.stroke();

                        let caption, color;
                        if (!state) {
                            caption = "no directory"; color = "#888";
                        } else if (state.error) {
                            caption = String(state.error).slice(0, 60); color = "#f88";
                        } else {
                            const percent = state.total ? Math.round((state.fraction || 0) * 100) : 0;
                            caption = `${state.position} / ${state.total}  (${percent}%)` +
                                (finished ? "  finished" : `  · ${state.remaining} left`);
                            color = finished ? "#bfb" : "#ddd";
                        }
                        const textY = y + height / 2;
                        ctx.font = "11px Arial";
                        ctx.textBaseline = "middle";
                        ctx.textAlign = "left";
                        ctx.fillStyle = color;
                        ctx.fillText(caption, margin + 8, textY);
                        if (state?.filename) {
                            const used = ctx.measureText(caption).width + 24;
                            let name = String(state.filename);
                            const room = barWidth - used - 8;
                            while (name.length > 4 && ctx.measureText(name).width > room) {
                                name = "…" + name.slice(2);
                            }
                            if (room > 30) {
                                ctx.textAlign = "right";
                                ctx.fillStyle = "#aaa";
                                ctx.fillText(name, width - margin - 8, textY);
                            }
                        }
                        ctx.restore();
                    },
                });

                this.addWidget("button", "Reset progress", "reset", async () => {
                    const directory = String(widgetValue(this, "directory") || "").trim();
                    if (!directory) return;
                    await api.fetchApi("/fvmtools/batch/reset", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            directory,
                            tracker: String(widgetValue(this, "tracker") || "default"),
                        }),
                    });
                    await refreshStatus(this);
                });

                this.addWidget("button", "Refresh count", "refresh", async () => {
                    await refreshStatus(this);
                });

                // Re-read the folder whenever the inputs that define it change.
                const node = this;
                for (const name of ["directory", "tracker", "sort_by", "include_subdirs", "pass_subdir", "fail_subdir"]) {
                    const widget = this.widgets?.find((w) => w.name === name);
                    if (!widget) continue;
                    const callback = widget.callback;
                    widget.callback = function () {
                        const value = callback?.apply(this, arguments);
                        setTimeout(() => refreshStatus(node), 50);
                        return value;
                    };
                }

                setTimeout(() => refreshStatus(this), 200);
            } catch (error) {
                console.error("[FVMTools] BatchLoadImage widget setup failed:", error);
            }
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            const payload = message?.fvm_batch?.[0];
            if (payload) {
                this._fvmBatch = payload;
                this.setDirtyCanvas(true);
            }
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            setTimeout(() => refreshStatus(this), 300);
            return result;
        };
    },
});
