import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// SenseNova Interleave Preview displays the interleaved text/image output
// from `SenseNovaU1LocalInterleave`. ComfyUI core renders `ui.images` on
// nodes but ignores `ui.text`, so we attach a read-only multiline widget
// and write the markdown produced by the backend into it on every run.
app.registerExtension({
    name: "sensenova.interleave_preview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "SenseNovaInterleavePreview") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const widget = ComfyWidgets.STRING(
                this,
                "preview",
                ["STRING", { multiline: true, default: "" }],
                app,
            ).widget;

            const inputEl = widget.inputEl;
            if (inputEl) {
                inputEl.readOnly = true;
                inputEl.placeholder = "Interleave preview output will appear here after the workflow runs.";
                inputEl.style.opacity = "0.9";
                inputEl.style.fontFamily = "ui-monospace, SFMono-Regular, Menlo, monospace";
                inputEl.style.fontSize = "12px";
            }
            // Frontend-only widget: never serialize into the saved workflow,
            // otherwise it would shift the widgets_values index for existing JSONs.
            widget.serialize = false;
            widget.serializeValue = () => undefined;

            this._snInterleavePreviewWidget = widget;
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const widget = this._snInterleavePreviewWidget;
            if (!widget) {
                return;
            }
            const text = message?.text;
            if (typeof text === "string") {
                widget.value = text;
            } else if (Array.isArray(text)) {
                widget.value = text.join("\n\n");
            } else {
                widget.value = "";
            }
            this.setDirtyCanvas?.(true, true);
        };
    },
});
