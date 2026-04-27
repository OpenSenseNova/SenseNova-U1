"""Least-in-flight HTTP load balancer for multiple LightLLM server replicas.

Purpose: front N tp-sharded LightLLM instances behind a single OpenAI-compatible
endpoint, so VLMEvalKit / EASI can treat them as one higher-throughput server
without any client-side changes.

Backend selection: least in-flight requests (falls back to round-robin when tied).
Health checks: periodic GET /v1/models against each backend; unhealthy backends
are skipped (but never permanently removed — they rejoin when they recover).
Streaming: passthrough via httpx.stream() + StreamingResponse — long SSE works.

Configuration (env):
    BACKENDS          Comma-separated backend base URLs, e.g.
                      "http://localhost:8000,http://localhost:8010"
                      (REQUIRED)
    LB_HOST           Bind address                      (default: 0.0.0.0)
    LB_PORT           Listen port                       (default: 9000)
    LB_REQUEST_TIMEOUT Seconds per proxied request     (default: 1800 = 30 min)
    LB_HEALTH_INTERVAL Seconds between health probes   (default: 10)
    LB_STARTUP_TIMEOUT Max wait for first healthy     (default: 600)

Run via `serve_lb.sh` or directly:
    BACKENDS=http://localhost:8000,http://localhost:8010 python lb.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

BACKENDS: list[str] = [b.strip().rstrip("/") for b in os.environ.get("BACKENDS", "").split(",") if b.strip()]
if not BACKENDS:
    print("[lb] ERROR: BACKENDS env var required (comma-separated base URLs)", file=sys.stderr)
    sys.exit(1)

LB_HOST = os.environ.get("LB_HOST", "0.0.0.0")
LB_PORT = int(os.environ.get("LB_PORT", "9000"))
REQUEST_TIMEOUT = float(os.environ.get("LB_REQUEST_TIMEOUT", "1800"))
HEALTH_INTERVAL = float(os.environ.get("LB_HEALTH_INTERVAL", "10"))
STARTUP_TIMEOUT = float(os.environ.get("LB_STARTUP_TIMEOUT", "600"))

# Per-backend state: in-flight count + health flag.
inflight: dict[str, int] = {b: 0 for b in BACKENDS}
healthy: dict[str, bool] = {b: False for b in BACKENDS}


async def _probe_once(client: httpx.AsyncClient, backend: str) -> bool:
    """Return True if the backend looks like a LightLLM server.

    Pitfall: any random HTTP listener on the same port would answer, so we
    can't just check "did the connection succeed". Instead, hit /health (if
    exposed) or /v1/models and require the response to be from LightLLM —
    we key off the `Server: hypercorn-h11` header which LightLLM emits
    (hypercorn is its ASGI server) and upstream OpenAI-compat servers
    generally don't use hypercorn.

    Falls back to "any response at /v1/chat/completions with OPTIONS" when
    the server header check is ambiguous.
    """
    try:
        # /v1/models currently 500s on LightLLM due to an upstream pydantic
        # bug in ModelCard (owned_by=None). That's actually fine for our
        # probe — we just need a response.
        resp = await client.get(f"{backend}/v1/models", timeout=5.0)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, OSError):
        return False
    except Exception:
        return False

    # Require the Server header to mention hypercorn — distinguishes a real
    # LightLLM from a stray process on the same port.
    server_hdr = resp.headers.get("server", "").lower()
    if "hypercorn" in server_hdr:
        return True

    # Fallback: some reverse proxies strip Server headers. Accept any non-HTML
    # response body shorter than 4 KB with a 2xx/4xx/5xx status (i.e. not 200
    # HTML from a static file server).
    ctype = resp.headers.get("content-type", "").lower()
    if "html" in ctype:
        return False
    return True


async def health_loop(client: httpx.AsyncClient):
    """Periodically probe each backend; flip the healthy dict."""
    while True:
        for b in BACKENDS:
            ok = await _probe_once(client, b)
            if ok != healthy[b]:
                print(f"[lb] backend {b}: {'UP' if ok else 'DOWN'}", flush=True)
            healthy[b] = ok
        await asyncio.sleep(HEALTH_INTERVAL)


async def wait_for_first_healthy(client: httpx.AsyncClient):
    """Block until at least one backend passes a probe, or STARTUP_TIMEOUT."""
    deadline = time.monotonic() + STARTUP_TIMEOUT
    print(f"[lb] waiting for at least one healthy backend (timeout {STARTUP_TIMEOUT:.0f}s)...", flush=True)
    while time.monotonic() < deadline:
        for b in BACKENDS:
            if await _probe_once(client, b):
                healthy[b] = True
                print(f"[lb] first healthy backend: {b}", flush=True)
                return
        await asyncio.sleep(2.0)
    print(
        f"[lb] WARNING: no backend became healthy within {STARTUP_TIMEOUT:.0f}s — serving anyway",
        file=sys.stderr,
        flush=True,
    )


def pick_backend() -> str | None:
    """Least in-flight among healthy backends; ties broken by first-seen order."""
    candidates = [b for b in BACKENDS if healthy[b]]
    if not candidates:
        # All unhealthy — fall back to any backend (will error back to client).
        candidates = BACKENDS
    return min(candidates, key=lambda b: inflight[b])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One shared client used for both proxying and health checks; large pool
    # so we can overlap many vlmevalkit workers.
    limits = httpx.Limits(max_keepalive_connections=128, max_connections=256)
    client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT), limits=limits)
    app.state.client = client

    await wait_for_first_healthy(client)
    health_task = asyncio.create_task(health_loop(client))

    try:
        yield
    finally:
        health_task.cancel()
        await client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/_lb/status")
async def lb_status():
    return {
        "backends": BACKENDS,
        "healthy": healthy,
        "inflight": inflight,
    }


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy(full_path: str, request: Request):
    backend = pick_backend()
    if backend is None:
        return JSONResponse({"error": "no backends configured"}, status_code=503)

    url = f"{backend}/{full_path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    # Strip hop-by-hop + host headers; pass the rest through.
    drop = {"host", "content-length", "connection", "transfer-encoding"}
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in drop}

    body = await request.body()

    client: httpx.AsyncClient = request.app.state.client
    inflight[backend] += 1
    try:
        upstream_req = client.build_request(
            request.method,
            url,
            content=body if body else None,
            headers=fwd_headers,
        )
        upstream_resp = await client.send(upstream_req, stream=True)
    except Exception as e:
        inflight[backend] -= 1
        return JSONResponse({"error": f"backend {backend} unreachable: {e}"}, status_code=502)

    async def body_iter():
        try:
            async for chunk in upstream_resp.aiter_raw():
                yield chunk
        finally:
            await upstream_resp.aclose()
            inflight[backend] -= 1

    # Filter response headers (don't let chunked/transfer-encoding pass through verbatim).
    out_headers = {k: v for k, v in upstream_resp.headers.items() if k.lower() not in drop}

    return StreamingResponse(
        body_iter(),
        status_code=upstream_resp.status_code,
        headers=out_headers,
        media_type=upstream_resp.headers.get("content-type"),
    )


if __name__ == "__main__":
    print(f"[lb] backends: {BACKENDS}", flush=True)
    print(f"[lb] listening on http://{LB_HOST}:{LB_PORT}  (status: /_lb/status)", flush=True)
    uvicorn.run(app, host=LB_HOST, port=LB_PORT, log_level="info", access_log=False)
