import sys
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from test_comfyui_api_models import API_CLIENT, _FakeConfig


class _HangingRequestHandler(BaseHTTPRequestHandler):
    request_started = threading.Event()
    release_request = threading.Event()

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(content_length)
        self.request_started.set()
        self.release_request.wait(timeout=5)
        body = b'{"choices":[{"message":{"content":"done"}}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def log_message(self, _format: str, *args) -> None:
        pass


class _SuccessfulRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(content_length)
        body = b'{"result":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args) -> None:
        pass


class _FakeInterrupt(BaseException):
    pass


class SenseNovaApiInterruptTest(unittest.TestCase):
    def test_post_worker_returns_a_successful_json_response(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _SuccessfulRequestHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        client = API_CLIENT.SenseNovaClient(
            _FakeConfig(api_key="shared-key", base_url=f"http://127.0.0.1:{server.server_port}")
        )

        try:
            result = client._post_json("/success", {"payload": "request"}, timeout=10)
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

        self.assertEqual(result, {"result": "ok"})

    def test_write_timeout_is_not_retried(self) -> None:
        client = API_CLIENT.SenseNovaClient(_FakeConfig(api_key="shared-key", base_url="https://token.sensenova.cn/v1"))
        transport_result = {"kind": "transport_error", "error_type": "WriteTimeout"}

        with mock.patch.object(API_CLIENT, "_post_json_interruptibly", return_value=transport_result) as post:
            with self.assertRaisesRegex(RuntimeError, "WriteTimeout"):
                client._post_json("/chat/completions", {"payload": "request"}, timeout=120)

        post.assert_called_once()

    def test_comfyui_interrupt_terminates_an_in_flight_post(self) -> None:
        interrupted = threading.Event()
        completed = threading.Event()
        errors: list[BaseException] = []
        _HangingRequestHandler.request_started.clear()
        _HangingRequestHandler.release_request.clear()

        server = ThreadingHTTPServer(("127.0.0.1", 0), _HangingRequestHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        model_management = types.ModuleType("comfy.model_management")

        def throw_if_interrupted() -> None:
            if interrupted.is_set():
                raise _FakeInterrupt

        model_management.throw_exception_if_processing_interrupted = throw_if_interrupted
        comfy = types.ModuleType("comfy")
        comfy.model_management = model_management

        client = API_CLIENT.SenseNovaClient(
            _FakeConfig(api_key="shared-key", base_url=f"http://127.0.0.1:{server.server_port}")
        )

        def run_request() -> None:
            try:
                client._post_json("/wait", {"payload": "request"}, timeout=10)
            except BaseException as exc:
                errors.append(exc)
            finally:
                completed.set()

        request_thread = threading.Thread(target=run_request, daemon=True)
        try:
            with mock.patch.dict(
                sys.modules,
                {"comfy": comfy, "comfy.model_management": model_management},
            ):
                request_thread.start()
                self.assertTrue(_HangingRequestHandler.request_started.wait(timeout=2))
                interrupted.set()
                stopped_after_cancel = completed.wait(timeout=1)
        finally:
            _HangingRequestHandler.release_request.set()
            request_thread.join(timeout=2)
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=2)

        self.assertTrue(stopped_after_cancel, "in-flight POST ignored the ComfyUI interrupt")
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], _FakeInterrupt)


if __name__ == "__main__":
    unittest.main()
