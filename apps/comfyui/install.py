from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_NODE_NAME = "ComfyUI-SenseNova-U1"


def _link_or_junction(source: Path, target: Path) -> str:
    """Create a directory symlink, falling back to an NTFS junction on Windows.

    Windows blocks `os.symlink` unless the user is an administrator or
    Developer Mode is enabled (WinError 1314). A directory junction
    (`mklink /J`) provides the same `Path.resolve()`-followable semantics
    without any privilege; ComfyUI loads through it and the loader's
    auto-discovery still finds the monorepo source.
    """
    try:
        os.symlink(source, target, target_is_directory=True)
        return "Linked"
    except OSError as exc:
        if sys.platform != "win32":
            raise
        try:
            subprocess.check_call(
                ["cmd", "/c", "mklink", "/J", str(target), str(source)],
                stdout=subprocess.DEVNULL,
            )
            return "Junctioned"
        except (subprocess.CalledProcessError, FileNotFoundError) as junc_exc:
            raise SystemExit(
                f"Could not create a symlink at {target}: {exc}\n"
                f"Falling back to `mklink /J` also failed: {junc_exc}\n"
                "Re-run with --copy, enable Windows Developer Mode "
                "(Settings → For Developers), or run this script as Administrator."
            ) from exc


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
        action = _link_or_junction(app_dir, target)

    print(f"{action} SenseNova-U1 ComfyUI app:")
    print(f"  {target} -> {app_dir}")

    if not args.copy:
        # Default symlink (or Windows junction) mode: local_pipeline.py's
        # default_source_path() resolves __file__ through the link back to
        # this monorepo and discovers <repo>/src automatically. No env var
        # needed for local inference.
        print(f"\n{action} mode: SENSENOVA_U1_SRC auto-resolves to")
        print(f"  {repo_dir / 'src'}")
        print("via local_pipeline.default_source_path(), because the loader")
        print("file is linked back into this checkout. Moving or renaming")
        print("the monorepo breaks that link — re-run install.py afterwards.")
    else:
        # --copy mode: files live under <ComfyUI>/custom_nodes/, no symlink
        # to follow, so the user must point SENSENOVA_U1_SRC explicitly.
        print("\nCopy mode: auto-discovery is disabled (no symlink to follow).")
        print(f"Set SENSENOVA_U1_SRC={repo_dir / 'src'}")
        print("in the ComfyUI launch environment, or fill the loader node's")
        print("`sensenova_u1_src` input.")

    if args.install_deps:
        requirements = app_dir / "requirements.txt"
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements)])
        print("\nFor local inference, also install the SenseNova-U1 runtime in the ComfyUI Python environment:")
        print(f"  {sys.executable} -m pip install -e {repo_dir}")
        print("  Restart ComfyUI.")
    else:
        print("\nNext steps:")
        print(f"  {sys.executable} -m pip install -r {app_dir / 'requirements.txt'}")
        print(f"  {sys.executable} -m pip install -e {repo_dir}  # for local inference")
        print("  Restart ComfyUI.")


if __name__ == "__main__":
    main()
