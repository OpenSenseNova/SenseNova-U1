"""Publish validated Native/GGUF artifacts to a Hugging Face model repository."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

from conversion_utils import CONFIG_FILES

MANIFEST_SCHEMA = "sensenova-artifact-manifest-v1"


def _safe_relative_path(root: Path, value: str) -> Path:
    """Resolve a manifest path while keeping it inside the release directory."""
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Manifest path escapes the release directory: {value!r}")
    path = root.joinpath(*relative.parts)
    if root not in path.resolve().parents and path.resolve() != root.resolve():
        raise ValueError(f"Manifest path escapes the release directory: {value!r}")
    return path


def _load_manifest(source_dir: Path) -> dict[str, Any]:
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing release manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported manifest schema in {manifest_path}: {manifest.get('schema')!r}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"Manifest has no artifacts: {manifest_path}")
    return manifest


def validate_release_tree(source_dir: str | Path) -> dict[str, Any]:
    """Validate the artifacts, reports, resources, and manifest before upload."""
    source_dir = Path(source_dir).resolve()
    manifest = _load_manifest(source_dir)
    upload_paths = {"manifest.json"}
    for item in manifest["artifacts"]:
        if not isinstance(item, dict):
            raise ValueError("Manifest artifacts must be objects")
        artifact = _safe_relative_path(source_dir, item["path"])
        report = _safe_relative_path(source_dir, item["conversion_report"])
        if not artifact.is_file():
            raise FileNotFoundError(f"Manifest artifact does not exist: {artifact}")
        if not report.is_file():
            raise FileNotFoundError(f"Manifest conversion report does not exist: {report}")
        artifact_dir = artifact.parent
        for resource in CONFIG_FILES:
            resource_path = artifact_dir / resource
            if not resource_path.is_file():
                raise FileNotFoundError(f"Missing resource for {artifact.name}: {resource_path}")
            upload_paths.add(resource_path.relative_to(source_dir).as_posix())
        upload_paths.add(artifact.relative_to(source_dir).as_posix())
        upload_paths.add(report.relative_to(source_dir).as_posix())
        quantization_config = item.get("quantization_config")
        if quantization_config is not None:
            quantization_path = _safe_relative_path(source_dir, quantization_config)
            if not quantization_path.is_file():
                raise FileNotFoundError(f"Manifest quantization config does not exist: {quantization_path}")
            upload_paths.add(quantization_path.relative_to(source_dir).as_posix())
    for readme in source_dir.glob("README*.md"):
        upload_paths.add(readme.name)
    return {
        "source_dir": str(source_dir),
        "manifest": manifest,
        "artifact_count": len(manifest["artifacts"]),
        "upload_paths": sorted(upload_paths),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", required=True, help="Validated format directory or combined sensenova output directory."
    )
    parser.add_argument(
        "--repo-id", required=True, help="Hugging Face model repository, for example sensenova/SenseNova-U1.6-8B-MoT."
    )
    parser.add_argument(
        "--path-in-repo", default=None, help="Optional destination directory inside the model repository."
    )
    parser.add_argument("--purpose", required=True, help="Release purpose recorded in the upload commit description.")
    parser.add_argument(
        "--token-env", default="HF_TOKEN", help="Environment variable containing the HF token (default: HF_TOKEN)."
    )
    parser.add_argument("--revision", default="main", help="Branch receiving the upload (default: main).")
    parser.add_argument("--tag", default=None, help="Optional tag to create at the upload commit.")
    parser.add_argument("--tag-message", default=None, help="Optional tag message.")
    parser.add_argument(
        "--replace-tag", action="store_true", help="Delete an existing tag before recreating it at this upload commit."
    )
    parser.add_argument(
        "--create-repo", action="store_true", help="Create the target repository when it does not exist."
    )
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Visibility for --create-repo (default: private).",
    )
    parser.add_argument("--commit-message", default=None, help="Optional upload commit title.")
    parser.add_argument("--commit-description", default=None, help="Optional upload commit description.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print the upload plan without contacting HF."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    validation = validate_release_tree(args.source_dir)
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
        "artifacts": validation["artifact_count"],
        "files": len(validation["upload_paths"]),
        "tag": args.tag,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    if args.create_repo:
        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True, token=token)
    commit = api.upload_folder(
        repo_id=args.repo_id,
        folder_path=validation["source_dir"],
        path_in_repo=args.path_in_repo,
        revision=args.revision,
        repo_type="model",
        allow_patterns=validation["upload_paths"],
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
