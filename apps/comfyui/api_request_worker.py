from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx


def _response_result(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except json.JSONDecodeError:
        return {
            "kind": "response",
            "status_code": response.status_code,
            "json_valid": False,
            "body_text": response.text[:500],
        }
    return {
        "kind": "response",
        "status_code": response.status_code,
        "json_valid": True,
        "body": body,
    }


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: api_request_worker.py REQUEST_JSON RESULT_JSON")

    request_path, result_path = (Path(value) for value in sys.argv[1:])
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        timeout = httpx.Timeout(**request["timeout"])
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                request["url"],
                headers=request["headers"],
                json=request["payload"],
            )
        result = _response_result(response)
        return_code = 0
    except httpx.HTTPError as exc:
        result = {"kind": "transport_error", "error_type": type(exc).__name__}
        return_code = 1
    except BaseException as exc:
        result = {"kind": "worker_error", "error_type": type(exc).__name__}
        return_code = 1

    result_path.write_text(json.dumps(result), encoding="utf-8")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
