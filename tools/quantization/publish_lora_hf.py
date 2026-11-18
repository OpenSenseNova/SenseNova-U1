"""Publish a validated SenseNova LoRA adapter to a Hugging Face model repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

LORA_MANIFEST_SCHEMA = "sensenova-lora-manifest-v1"
DOWN_SUFFIX = ".lora_down.weight"
UP_SUFFIX = ".lora_up.weight"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_lora_file(path: str | Path) -> dict[str, Any]:
    """Validate a LoRA Safetensors file and return upload metadata."""
    path = Path(path).resolve()
    if path.suffix.lower() != ".safetensors":
        raise ValueError(f"LoRA adapter must be a .safetensors file: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)

    down = {}
    up = {}
    alpha = set()
    dtypes = set()
    ranks = set()
    with safe_open(path, framework="pt", device="cpu") as reader:
        keys = set(reader.keys())
        for key in sorted(keys):
            tensor = reader.get_tensor(key)
            if not torch.is_floating_point(tensor) or not torch.isfinite(tensor).all():
                raise ValueError(f"LoRA tensor must be finite floating point: {key}")
            dtypes.add(str(tensor.dtype).removeprefix("torch."))
            if key.endswith(DOWN_SUFFIX):
                base = key[: -len(DOWN_SUFFIX)]
                down[base] = tuple(tensor.shape)
                ranks.add(int(tensor.shape[0]))
            elif key.endswith(UP_SUFFIX):
                base = key[: -len(UP_SUFFIX)]
                up[base] = tuple(tensor.shape)
            elif key.endswith(".alpha"):
                alpha.add(key[: -len(".alpha")])

    if not down:
        raise ValueError("LoRA file has no .lora_down.weight tensors")
    if missing := sorted(set(down) - set(up)):
        raise ValueError(f"LoRA file is missing paired up tensors: {missing[:5]}")
    if extra := sorted(set(up) - set(down)):
        raise ValueError(f"LoRA file has unpaired up tensors: {extra[:5]}")
    for base in down:
        if down[base][0] != up[base][1]:
            raise ValueError(f"LoRA rank mismatch for {base}: down={down[base]}, up={up[base]}")

    return {
        "adapter_file": path.name,
        "sha256": _sha256(path),
        "tensor_count": len(keys),
        "layer_count": len(down),
        "ranks": sorted(ranks),
        "dtypes": sorted(dtypes),
        "alpha_layers": len(alpha),
        "loader": "sensenova_u1.utils.lora.load_and_merge_lora_weight_from_safetensors",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-path", required=True, help="Local LoRA .safetensors file.")
    parser.add_argument(
        "--adapter-name",
        default=None,
        help="Filename to use in the HF repository. Defaults to the local filename.",
    )
    parser.add_argument("--repo-id", required=True, help="Hugging Face adapter repository.")
    parser.add_argument("--base-model", required=True, help="Compatible base model repository ID.")
    parser.add_argument("--purpose", required=True, help="Release purpose recorded in the upload commit description.")
    parser.add_argument("--source-revision", default=None, help="Base model revision used with this adapter.")
    parser.add_argument("--readme", default=None, help="Optional prepared README.md to upload with the adapter.")
    parser.add_argument(
        "--path-in-repo", default=None, help="Optional destination directory inside the adapter repository."
    )
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable containing the HF token.")
    parser.add_argument("--revision", default="main", help="Branch receiving the upload (default: main).")
    parser.add_argument("--tag", default=None, help="Optional tag to create at the upload commit.")
    parser.add_argument("--tag-message", default=None, help="Optional tag message.")
    parser.add_argument("--replace-tag", action="store_true", help="Delete an existing tag before recreating it.")
    parser.add_argument("--create-repo", action="store_true", help="Create the adapter repository when missing.")
    parser.add_argument(
        "--private", action=argparse.BooleanOptionalAction, default=True, help="Visibility for --create-repo."
    )
    parser.add_argument("--commit-message", default=None, help="Optional upload commit title.")
    parser.add_argument("--commit-description", default=None, help="Optional upload commit description.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without contacting HF.")
    return parser.parse_args()


def _write_generated_readme(path: Path, *, adapter_name: str, base_model: str, source_revision: str | None) -> None:
    revision = f" at revision `{source_revision}`" if source_revision else ""
    path.write_text(
        f"""# {adapter_name}

This repository contains a SenseNova LoRA adapter for `{base_model}`{revision}.

The adapter is not a standalone model. Load it with the matching SenseNova-U1
runtime and pass its local path through `--lora_path`.

```bash
python examples/t2i/inference.py \\
  --model_path {base_model} \\
  --lora_path ./{adapter_name}.safetensors \\
  --prompt \"A cinematic mountain lake at sunrise\"
```

See the [SenseNova-U1 repository](https://github.com/OpenSenseNova/SenseNova-U1)
for runtime requirements and task-specific commands.
"""
    )


def main() -> None:
    args = _parse_args()
    adapter = validate_lora_file(args.lora_path)
    adapter_name = args.adapter_name or adapter["adapter_file"]
    if Path(adapter_name).name != adapter_name or Path(adapter_name).suffix.lower() != ".safetensors":
        raise ValueError("--adapter-name must be a filename ending in .safetensors")
    token = os.environ.get(args.token_env)
    if not token and not args.dry_run:
        raise RuntimeError(
            f"Set the Hugging Face token in ${args.token_env}; tokens are not accepted as CLI arguments."
        )

    plan = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "path_in_repo": args.path_in_repo or ".",
        "purpose": args.purpose,
        "base_model": args.base_model,
        "adapter_file": adapter_name,
        "source_file": adapter["adapter_file"],
        "sha256": adapter["sha256"],
        "tag": args.tag,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory(prefix="sensenova-lora-publish-") as temporary_dir:
        staging = Path(temporary_dir)
        shutil.copy2(args.lora_path, staging / adapter_name)
        manifest_adapter = {**adapter, "adapter_file": adapter_name, "source_file": adapter["adapter_file"]}
        manifest = {
            "schema": LORA_MANIFEST_SCHEMA,
            "adapter": manifest_adapter,
            "base_model": args.base_model,
            "base_revision": args.source_revision,
            "purpose": args.purpose,
        }
        (staging / "lora_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        if args.readme:
            shutil.copy2(args.readme, staging / "README.md")
        else:
            _write_generated_readme(
                staging / "README.md",
                adapter_name=Path(adapter_name).stem,
                base_model=args.base_model,
                source_revision=args.source_revision,
            )

        api = HfApi(token=token)
        if args.create_repo:
            api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True, token=token)
        commit = api.upload_folder(
            repo_id=args.repo_id,
            folder_path=staging,
            path_in_repo=args.path_in_repo,
            revision=args.revision,
            repo_type="model",
            commit_message=args.commit_message or f"release: publish {args.purpose}",
            commit_description=args.commit_description or args.purpose,
            token=token,
        )
        commit_hash = getattr(commit, "commit_hash", None) or args.revision
        if args.tag:
            if args.replace_tag:
                api.delete_tag(args.repo_id, tag=args.tag, repo_type="model", token=token)
            api.create_tag(
                args.repo_id,
                tag=args.tag,
                revision=commit_hash,
                tag_message=args.tag_message or args.purpose,
                repo_type="model",
                token=token,
            )
    print(json.dumps({**plan, "commit": commit_hash}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
