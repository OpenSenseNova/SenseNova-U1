try:
    from .nodes import comfy_entrypoint
except ImportError:  # pragma: no cover - supports direct pytest collection
    from nodes import comfy_entrypoint

# ComfyUI auto-loads every JS file under this directory as a frontend extension.
# Used to render `ui.text` produced by SenseNovaInterleavePreview, which the
# stock frontend does not display on the node itself.
WEB_DIRECTORY = "./web"

__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
