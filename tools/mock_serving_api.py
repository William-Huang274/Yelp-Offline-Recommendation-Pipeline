#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from batch_infer_demo import (
    DEFAULT_INPUT_PATH,
    load_release_reference,
    load_serving_config,
    rank_payload,
    read_json,
    warm_demo_assets,
)
from replay_store import load_replay_store


def health_payload() -> dict[str, Any]:
    release_ref = load_release_reference()
    serving_config = load_serving_config()
    return {
        "status": "ok",
        "service": "mock_serving_api",
        "mode": "mock_http_service",
        "release_id": release_ref["release_id"],
        "current_output_surface": release_ref["current_output_surface"],
        "model_version": serving_config["model_version"],
        "default_strategy": serving_config["default_strategy"],
        "allowed_strategies": serving_config["allowed_strategies"],
        "latency_budget_ms": serving_config["latency_budget_ms"],
        "stage11_live_default_mode": serving_config.get("stage11_live_default_mode", "off"),
    }


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "MockServingAPI/1.0"

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._send_json(200, health_payload())
            return
        self._send_json(404, {"error": "not_found", "path": self.path})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/rank":
            self._send_json(404, {"error": "not_found", "path": self.path})
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self._send_json(400, {"error": "empty_body"})
            return
        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            result = rank_payload(payload)
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": "invalid_json", "detail": str(exc)})
            return
        except Exception as exc:  # pragma: no cover - defensive service path
            self._send_json(400, {"error": "rank_failed", "detail": str(exc)})
            return
        self._send_json(200, result)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[HTTP] {self.address_string()} - {fmt % args}")


def run_self_test() -> int:
    sample = read_json(Path(DEFAULT_INPUT_PATH))
    replay_store = load_replay_store()
    payload = {
        "health": health_payload(),
        "legacy_rank_result": rank_payload(sample),
        "replay_rank_result": rank_payload(
            {
                "request_id": replay_store.sample_request_id,
                "strategy": "reward_rerank",
                "debug": True,
                "include_fallback_demo": True,
                "stage11_live_mode": "remote_dry_run",
            }
        ),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mock serving API for the current Yelp ranking stack."
    )
    parser.add_argument("--host", default="127.0.0.1", help="listen host")
    parser.add_argument("--port", type=int, default=8000, help="listen port")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="print /health and /rank sample payloads without starting the server",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="do not preload replay assets before serving",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        return run_self_test()

    if not args.skip_warmup:
        started = time.perf_counter()
        warm_demo_assets(include_replay=True)
        warm_ms = round((time.perf_counter() - started) * 1000.0, 3)
        print(f"[INFO] warmed demo assets in {warm_ms} ms")

    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"[INFO] mock_serving_api listening on http://{args.host}:{args.port}")
    print("[INFO] GET  /health")
    print("[INFO] POST /rank")
    print(f"[INFO] sample request file: {DEFAULT_INPUT_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
