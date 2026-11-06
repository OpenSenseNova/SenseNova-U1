from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: hf_download_worker.py REPO_ID RESULT_JSON")

    repo_id, result_json = sys.argv[1:]
    try:
        from huggingface_hub import snapshot_download

        snapshot_path = snapshot_download(repo_id)
        payload = {"ok": True, "snapshot_path": str(snapshot_path)}
        return_code = 0
    except BaseException as exc:
        payload = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        return_code = 1

    Path(result_json).write_text(json.dumps(payload), encoding="utf-8")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
