import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// SenseNova Interleave Preview renders text and images in their original
// interleaved order on the node itself, instead of ComfyUI's default
// "all images stacked at the top" layout. The backend pushes a `parts`
// array via `ui.parts`; this extension turns it into a single scrollable
// markdown-like flow (think -> text -> image -> text -> image ...).

const CONTAINER_STYLES = {
    padding: "8px",
    boxSizing: "border-box",
    overflow: "auto",
    fontFamily: "ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif",
    fontSize: "13px",
    lineHeight: "1.5",
    color: "var(--input-text, #ddd)",
    background: "var(--comfy-input-bg, #1e1e1e)",
    border: "1px solid var(--border-color, #333)",
    borderRadius: "6px",
    whiteSpace: "normal",
    wordBreak: "break-word",
};

const TEXT_STYLES = {
    margin: "0 0 8px 0",
    whiteSpace: "pre-wrap",
};

const THINK_STYLES = {
    margin: "0 0 8px 0",
    padding: "6px 8px",
    borderLeft: "3px solid var(--node-selected-color, #6c757d)",
    background: "var(--comfy-menu-bg, #2a2a2a)",
    color: "var(--descrip-text, #aaa)",
    fontStyle: "italic",
    whiteSpace: "pre-wrap",
};

const IMG_WRAP_STYLES = {
    margin: "0 0 8px 0",
    textAlign: "center",
};

const IMG_STYLES = {
    maxWidth: "100%",
    maxHeight: "480px",
    borderRadius: "4px",
    border: "1px solid var(--border-color, #333)",
};

const PLACEHOLDER_STYLES = {
    color: "var(--descrip-text, #888)",
    fontStyle: "italic",
};

function applyStyles(el, styles) {
    for (const [key, value] of Object.entries(styles)) {
        el.style[key] = value;
    }
}

function buildImageUrl(part) {
    const params = new URLSearchParams({
        filename: part.filename || "",
        type: part.image_type || "temp",
        subfolder: part.subfolder || "",
        // Bust the browser cache because temp filenames may be reused across runs.
        rand: Math.random().toString(36).slice(2),
    });
    // `api.apiURL` handles ComfyUI's optional /api prefix consistently; fall
    // back to the bare `/view` endpoint registered by the server.
    return api?.apiURL ? api.apiURL(`/view?${params}`) : `/view?${params}`;
}

function renderParts(container, parts) {
    container.innerHTML = "";
    if (!parts || parts.length === 0) {
        const empty = document.createElement("div");
        applyStyles(empty, PLACEHOLDER_STYLES);
        empty.textContent = "(no interleaved output)";
        container.appendChild(empty);
        return;
    }

    for (const part of parts) {
        if (part.type === "text") {
            const div = document.createElement("div");
            applyStyles(div, TEXT_STYLES);
            div.textContent = part.text || "";
            container.appendChild(div);
        } else if (part.type === "think") {
            const details = document.createElement("details");
            applyStyles(details, THINK_STYLES);
            const summary = document.createElement("summary");
            summary.textContent = "think";
            summary.style.cursor = "pointer";
            summary.style.fontStyle = "normal";
            summary.style.fontWeight = "600";
            details.appendChild(summary);
            const body = document.createElement("div");
            body.style.marginTop = "4px";
            body.textContent = part.text || "";
            details.appendChild(body);
            container.appendChild(details);
        } else if (part.type === "image") {
            const wrap = document.createElement("div");
            applyStyles(wrap, IMG_WRAP_STYLES);
            if (part.missing || !part.filename) {
                const span = document.createElement("span");
                applyStyles(span, PLACEHOLDER_STYLES);
                span.textContent = `[image:${part.index} missing]`;
                wrap.appendChild(span);
            } else {
                const img = document.createElement("img");
                applyStyles(img, IMG_STYLES);
                img.alt = `image ${part.index}`;
                img.src = buildImageUrl(part);
                wrap.appendChild(img);
            }
            container.appendChild(wrap);
        }
    }
}

app.registerExtension({
    name: "sensenova.interleave_preview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "SenseNovaInterleavePreview") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const container = document.createElement("div");
            applyStyles(container, CONTAINER_STYLES);

            const empty = document.createElement("div");
            applyStyles(empty, PLACEHOLDER_STYLES);
            empty.textContent = "Interleave preview output will appear here after the workflow runs.";
            container.appendChild(empty);

            // ComfyUI's addDOMWidget returns a widget that auto-resizes with the node.
            const widget = this.addDOMWidget?.("preview", "interleave_preview", container, {
                serialize: false,
                hideOnZoom: false,
            });

            if (widget) {
                this._snInterleavePreviewContainer = container;
                this._snInterleavePreviewWidget = widget;
            } else {
                // Fallback for older ComfyUI builds without addDOMWidget.
                this._snInterleavePreviewContainer = container;
                container.style.minHeight = "200px";
                if (this.widgets_up && this.contentEl) {
                    this.contentEl.appendChild(container);
                }
            }

            // Suppress the default `node.imgs` rendering even if a downstream
            // path tries to attach images here, since we render them inline.
            this.imgs = [];
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const container = this._snInterleavePreviewContainer;
            if (!container) {
                return;
            }
            const parts = message?.parts;
            renderParts(container, Array.isArray(parts) ? parts : []);

            // Default ComfyUI behavior would also paint message.images on top
            // of the node; clear that so our inline rendering is the single
            // source of truth.
            this.imgs = [];
            this.setDirtyCanvas?.(true, true);
        };
    },
});
