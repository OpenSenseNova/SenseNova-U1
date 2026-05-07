# Register `<comfyui>/models/gguf` as a model folder before nodes.py is imported,
# so SenseNovaU1LocalLoader's `gguf_checkpoint` dropdown can be populated via
# folder_paths.get_filename_list("gguf"). Tolerant when folder_paths isn't
# importable (e.g. running tests outside ComfyUI).
try:
    import os as _os

    import folder_paths as _folder_paths

    _gguf_dir = _os.path.join(_folder_paths.models_dir, "gguf")
    _existing = _folder_paths.folder_names_and_paths.get("gguf")
    if _existing is None:
        _folder_paths.folder_names_and_paths["gguf"] = ([_gguf_dir], {".gguf"})
    else:
        _paths, _exts = _existing
        if _gguf_dir not in _paths:
            _paths.append(_gguf_dir)
        _exts.add(".gguf")
except Exception:  # pragma: no cover - non-ComfyUI env or registration race
    pass

try:
    from .nodes import comfy_entrypoint
except ImportError:  # pragma: no cover - supports direct pytest collection
    from nodes import comfy_entrypoint

# ComfyUI auto-loads every JS file under this directory as a frontend extension.
# Used to render `ui.text` produced by SenseNovaInterleavePreview, which the
# stock frontend does not display on the node itself.
WEB_DIRECTORY = "./web"

__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
