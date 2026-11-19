"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._inputs: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None) -> str:
        """Store input + start timestamp keyed by request_id/user_id."""
        req_id = request_id or f"req-{len(self.logs) + len(self._open) + 1}"
        start_time = time.time()
        self._open[req_id] = start_time
        self._inputs[req_id] = {
            "user_id": user_id,
            "text": text,
            "timestamp": utc_now_iso(),
            "start_time": start_time,
        }
        return req_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ) -> dict:
        """Store output, layer decision, latency; append to self.logs."""
        req_id = request_id
        start_time = self._open.pop(req_id, None) if req_id else None
        latency = (time.time() - start_time) if start_time else 0.0
        input_data = self._inputs.pop(req_id, {}) if req_id else {}

        log_entry = {
            "request_id": req_id,
            "user_id": user_id,
            "input": input_data.get("text", ""),
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "latency": round(latency, 4),
            "timestamp": utc_now_iso(),
        }
        self.logs.append(log_entry)
        return log_entry

    def export_json(self, filepath: str | None = None) -> str:
        """Write logs to disk (JSON array) under repo-root outputs/ by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)
