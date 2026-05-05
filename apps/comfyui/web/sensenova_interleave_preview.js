import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// SenseNova Interleave Preview renders text and images in their original
// interleaved order on the node. The backend pushes a structured
// `ui.parts` array; we map each entry to a DOM node here.

const STYLE_ID = "sensenova-interleave-preview-styles";
const STYLE_CSS = `
.sn-interleave {
    padding: 8px;
    box-sizing: border-box;
    overflow: auto;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 13px;
    line-height: 1.5;
    color: var(--input-text, #ddd);
    background: var(--comfy-input-bg, #1e1e1e);
    border: 1px solid var(--border-color, #333);
    border-radius: 6px;
    word-break: break-word;
}
.sn-interleave > * { margin: 0 0 8px 0; }
.sn-interleave-text { white-space: pre-wrap; }
.sn-interleave-think {
    padding: 6px 8px;
    border-left: 3px solid var(--node-selected-color, #6c757d);
    background: var(--comfy-menu-bg, #2a2a2a);
    color: var(--descrip-text, #aaa);
    font-style: italic;
    white-space: pre-wrap;
}
.sn-interleave-think summary {
    cursor: pointer;
    font-style: normal;
    font-weight: 600;
}
.sn-interleave-think > div { margin-top: 4px; }
.sn-interleave-image { text-align: center; }
.sn-interleave-image img {
    max-width: 100%;
    max-height: 480px;
    border-radius: 4px;
    border: 1px solid var(--border-color, #333);
}
.sn-interleave-placeholder {
    color: var(--descrip-text, #888);
    font-style: italic;
}
`;

function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = STYLE_CSS;
    document.head.appendChild(style);
}

function buildImageUrl(part) {
    const params = new URLSearchParams({
        filename: part.filename || "",
        type: part.image_type || "temp",
        subfolder: part.subfolder || "",
        // Cache-bust because temp filenames may collide across runs.
        rand: Math.random().toString(36).slice(2),
    });
    return api?.apiURL ? api.apiURL(`/view?${params}`) : `/view?${params}`;
}

const RENDERERS = {
    text(part) {
        const div = document.createElement("div");
        div.className = "sn-interleave-text";
        div.textContent = part.text || "";
        return div;
    },
    think(part) {
        const details = document.createElement("details");
        details.className = "sn-interleave-think";
        const summary = document.createElement("summary");
        summary.textContent = "think";
        details.appendChild(summary);
        const body = document.createElement("div");
        body.textContent = part.text || "";
        details.appendChild(body);
        return details;
    },
    image(part) {
        const wrap = document.createElement("div");
        wrap.className = "sn-interleave-image";
        if (part.missing || !part.filename) {
            const span = document.createElement("span");
            span.className = "sn-interleave-placeholder";
            span.textContent = `[image:${part.index} missing]`;
            wrap.appendChild(span);
        } else {
            const img = document.createElement("img");
            img.alt = `image ${part.index}`;
            img.src = buildImageUrl(part);
            wrap.appendChild(img);
        }
        return wrap;
    },
};

function renderParts(container, parts) {
    container.innerHTML = "";
    if (!parts?.length) {
        const empty = document.createElement("div");
        empty.className = "sn-interleave-placeholder";
        empty.textContent = "(no interleaved output)";
        container.appendChild(empty);
        return;
    }
    for (const part of parts) {
        const renderer = RENDERERS[part.type];
        if (renderer) container.appendChild(renderer(part));
    }
}

app.registerExtension({
    name: "sensenova.interleave_preview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "SenseNovaInterleavePreview") return;
        ensureStyles();

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const container = document.createElement("div");
            container.className = "sn-interleave";
            const hint = document.createElement("div");
            hint.className = "sn-interleave-placeholder";
            hint.textContent = "Interleave preview output will appear here after the workflow runs.";
            container.appendChild(hint);

            this.addDOMWidget?.("preview", "interleave_preview", container, {
                serialize: false,
                hideOnZoom: false,
            });
            this._snContainer = container;
            // Suppress ComfyUI's default node-header image strip; we render images inline.
            this.imgs = [];
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            if (!this._snContainer) return;
            renderParts(this._snContainer, Array.isArray(message?.parts) ? message.parts : []);
            this.imgs = [];
            this.setDirtyCanvas?.(true, true);
        };
    },
});
