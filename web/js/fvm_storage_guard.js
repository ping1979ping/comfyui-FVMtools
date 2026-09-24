import { app } from "../../../scripts/app.js";

// Storage guard: ComfyUI keeps a draft of every open workflow in localStorage
// (Comfy.Workflow.Draft.v2:*, ~1 MB each for big graphs). The browser gives an
// origin roughly 5 M characters, and the frontend swallows the QuotaExceeded
// error — once the store is full, drafts silently stop updating and a reload
// brings back an old state. This guard
//   1. removes legacy draft blobs the current frontend no longer reads, and
//   2. warns loudly while there is still time to save.

const LEGACY_KEYS = ["Comfy.Workflow.Drafts", "Comfy.Workflow.DraftOrder"];
const LIMIT_CHARS = 5 * 1024 * 1024;      // Chromium: ~5 M UTF-16 chars per origin
const WARN_AT = 0.8;
const CHECK_EVERY_MS = 60 * 1000;

function usedChars() {
    let total = 0;
    for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i) ?? "";
        total += key.length + (localStorage.getItem(key)?.length ?? 0);
    }
    return total;
}

function removeLegacy() {
    let freed = 0;
    for (const key of LEGACY_KEYS) {
        const value = localStorage.getItem(key);
        if (value === null) continue;
        freed += key.length + value.length;
        localStorage.removeItem(key);
    }
    return freed;
}

function toast(severity, summary, detail) {
    const api = app.extensionManager?.toast;
    if (api?.add) api.add({ severity, summary, detail, life: severity === "warn" ? 0 : 6000 });
    else console.warn(`[FVMTools] ${summary}: ${detail}`);
}

let warned = false;

function check() {
    try {
        const fraction = usedChars() / LIMIT_CHARS;
        if (fraction >= WARN_AT && !warned) {
            warned = true;
            toast(
                "warn",
                `Browser-Speicher ${Math.round(fraction * 100)} % voll`,
                "ComfyUI kann Entwürfe bald nicht mehr sichern — ein Neuladen würde " +
                    "dann einen alten Stand zeigen. Jetzt speichern (Strg+S) und " +
                    "ungenutzte Workflow-Tabs schließen."
            );
        } else if (fraction < WARN_AT - 0.1) {
            warned = false;
        }
    } catch (error) {
        console.error("[FVMTools] storage check failed:", error);
    }
}

app.registerExtension({
    name: "FVMTools.StorageGuard",
    async setup() {
        try {
            const freed = removeLegacy();
            if (freed > 0) {
                const mb = ((freed * 2) / 1024 / 1024).toFixed(1);
                console.log(`[FVMTools] removed legacy workflow drafts, freed ~${mb} MB`);
                toast("info", "Alte Workflow-Entwürfe entfernt", `~${mb} MB Browser-Speicher frei`);
            }
        } catch (error) {
            console.error("[FVMTools] legacy draft cleanup failed:", error);
        }
        check();
        setInterval(check, CHECK_EVERY_MS);
    },
});
