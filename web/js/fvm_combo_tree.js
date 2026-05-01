// FVMtools combo-tree dropdown enhancer.
// Detects slash-separated entries (e.g. "indoor/business/skyscraper_lobby") in
// COMBO dropdowns of specific FVMtools nodes and renders them as collapsible
// folder trees. Adapted from pysssss/ComfyUI-Custom-Scripts betterCombos.js.

import { app } from "../../scripts/app.js";
import { $el } from "../../scripts/ui.js";

const TREE_ENABLED_NODES = new Set([
    "FVM_SMP_LocationGenerator",
    "FVM_JB_LocationBlock",
    "FVM_SMP_OutfitGenerator",
]);

const TREE_WIDGET_NAMES = new Set([
    "location_set",
    "outfit_set",
]);

app.registerExtension({
    name: "fvmtools.ComboTree",
    init() {
        $el("style", {
            textContent: `
                .fvm-combo-folder { opacity: 0.85; cursor: pointer; }
                .fvm-combo-folder:hover { background-color: rgba(255, 255, 255, 0.08); }
                .fvm-combo-folder-arrow { display: inline-block; width: 14px; }
                .fvm-combo-prefix { display: none; opacity: 0.5; font-size: 0.85em; }

                /* Filter active: revert to flat list with prefix shown */
                .litecontextmenu:has(input:not(:placeholder-shown)) .fvm-combo-folder-contents {
                    display: block !important;
                }
                .litecontextmenu:has(input:not(:placeholder-shown)) .fvm-combo-folder {
                    display: none;
                }
                .litecontextmenu:has(input:not(:placeholder-shown)) .fvm-combo-prefix {
                    display: inline;
                }
                .litecontextmenu:has(input:not(:placeholder-shown)) .litemenu-entry {
                    padding-left: 2px !important;
                }
            `,
            parent: document.body,
        });

        const buildTree = (menu) => {
            const items = Array.from(menu.querySelectorAll(".litemenu-entry"));
            if (!items.length) return;

            // Skip if no entry contains a slash — nothing to nest.
            const splitBy = /\/|\\/;
            const hasSlash = items.some(it => {
                const v = it.getAttribute("data-value") || "";
                return splitBy.test(v);
            });
            if (!hasSlash) return;

            const folderMap = new Map();
            const itemsSymbol = Symbol("items");

            for (const item of items) {
                const value = item.getAttribute("data-value") || "";
                const path = value.split(splitBy);

                item.textContent = path[path.length - 1];
                if (path.length > 1) {
                    const prefix = $el("span.fvm-combo-prefix", {
                        textContent: path.slice(0, -1).join("/") + "/",
                    });
                    item.prepend(prefix);
                }

                if (path.length === 1) continue;

                item.remove();

                let currentLevel = folderMap;
                for (let i = 0; i < path.length - 1; i++) {
                    const folder = path[i];
                    if (!currentLevel.has(folder)) {
                        currentLevel.set(folder, new Map());
                    }
                    currentLevel = currentLevel.get(folder);
                }
                if (!currentLevel.has(itemsSymbol)) {
                    currentLevel.set(itemsSymbol, []);
                }
                currentLevel.get(itemsSymbol).push(item);
            }

            const createFolderElement = (name) => {
                return $el("div.litemenu-entry.fvm-combo-folder", {
                    innerHTML: `<span class="fvm-combo-folder-arrow">▶</span> ${name}`,
                });
            };

            const insertStructure = (parent, map, level = 0) => {
                for (const [folderName, content] of map.entries()) {
                    if (folderName === itemsSymbol) continue;

                    const folderEl = createFolderElement(folderName);
                    folderEl.style.paddingLeft = `${level * 10 + 5}px`;
                    parent.appendChild(folderEl);

                    const child = $el("div.fvm-combo-folder-contents", {
                        style: { display: "none" },
                    });
                    const innerItems = content.get(itemsSymbol) || [];
                    for (const it of innerItems) {
                        it.style.paddingLeft = `${(level + 1) * 10 + 14}px`;
                        child.appendChild(it);
                    }
                    insertStructure(child, content, level + 1);
                    parent.appendChild(child);

                    folderEl.addEventListener("click", (e) => {
                        e.stopPropagation();
                        const arrow = folderEl.querySelector(".fvm-combo-folder-arrow");
                        const collapsed = child.style.display === "none";
                        child.style.display = collapsed ? "block" : "none";
                        arrow.textContent = collapsed ? "▼" : "▶";
                    });
                }
            };

            const anchor = items[0]?.parentElement || menu;
            insertStructure(anchor, folderMap);
        };

        const observer = new MutationObserver((mutations) => {
            const node = app.canvas?.current_node;
            if (!node || !TREE_ENABLED_NODES.has(node.comfyClass)) return;

            for (const mutation of mutations) {
                for (const added of mutation.addedNodes) {
                    if (!added.classList?.contains("litecontextmenu")) continue;
                    const overWidget = app.canvas.getWidgetAtCursor?.();
                    if (!overWidget || !TREE_WIDGET_NAMES.has(overWidget.name)) continue;
                    requestAnimationFrame(() => {
                        // Only apply to the searchable filter dropdown, not right-click menu.
                        if (!added.querySelector(".comfy-context-menu-filter")) return;
                        try {
                            buildTree(added);
                        } catch (err) {
                            console.error("[fvm_combo_tree] buildTree failed:", err);
                        }
                    });
                }
            }
        });

        observer.observe(document.body, { childList: true, subtree: false });
    },
});
