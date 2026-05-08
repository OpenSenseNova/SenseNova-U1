from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_NODE_NAME = "ComfyUI-SenseNova-U1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install the SenseNova-U1 ComfyUI app.")
    parser.add_argument(
        "--comfyui",
        required=True,
        help="Path to the ComfyUI checkout that contains the custom_nodes directory.",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_NODE_NAME,
        help=f"Directory name under ComfyUI/custom_nodes (default: {DEFAULT_NODE_NAME}).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating a symlink.",
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Run pip install -r requirements.txt with the current Python.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing symlink or directory at the target path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_dir = Path(__file__).resolve().parent
    repo_dir = app_dir.parents[1]
    comfyui_dir = Path(args.comfyui).expanduser().resolve()
    custom_nodes = comfyui_dir / "custom_nodes"
    target = custom_nodes / args.name

    if not custom_nodes.is_dir():
        raise SystemExit(f"ComfyUI custom_nodes directory not found: {custom_nodes}")

    if target.exists() or target.is_symlink():
        if not args.force:
            raise SystemExit(
                f"Target already exists: {target}\nRe-run with --force to replace it, or choose another --name."
            )
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)

    if args.copy:
        shutil.copytree(app_dir, target, ignore=shutil.ignore_patterns("__pycache__"))
        action = "Copied"
    else:
        os.symlink(app_dir, target, target_is_directory=True)
        action = "Linked"

    print(f"{action} SenseNova-U1 ComfyUI app:")
    print(f"  {target} -> {app_dir}")

    if args.install_deps:
        requirements = app_dir / "requirements.txt"
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements)])
        print("\nFor local inference, also install the SenseNova-U1 runtime in the ComfyUI Python environment:")
        print(f"  {sys.executable} -m pip install -e {repo_dir}")
        print(f"  export SENSENOVA_U1_SRC={repo_dir / 'src'}")
        print("  Restart ComfyUI.")
    else:
        print("\nNext steps:")
        print(f"  {sys.executable} -m pip install -r {app_dir / 'requirements.txt'}")
        print(f"  {sys.executable} -m pip install -e {repo_dir}  # for local inference")
        print(f"  export SENSENOVA_U1_SRC={repo_dir / 'src'}")
        print("  Restart ComfyUI.")


if __name__ == "__main__":
    main()
