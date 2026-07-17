"""Structured logging + an in-process metrics registry (stdlib only).

- ``configure_logging`` installs a JSON (or plain) formatter on the root logger.
- ``Metrics`` counts requests, latencies, grade outcomes, and renders a
  Prometheus text exposition for ``/metrics`` — no prometheus_client dependency.
  For a real fleet you'd export to a proper backend; the interface is the same.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import Counter, defaultdict
from typing import Dict, Optional

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message", "asctime", "taskName"
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge any structured extras attached via logger.info(..., extra={...}).
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if json_logs
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root.addHandler(handler)
    # Silence chatty third-party request loggers; we emit our own access log.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _esc(v: str) -> str:
    # Prometheus label-value escaping: backslash, double-quote, and newline.
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# Cap distinct endpoint labels so a caller that leaks client-controlled paths
# (e.g. raw 404 URLs) can't blow up metric cardinality / memory. Overflow folds
# into a single "other" bucket. The middleware passes matched route templates,
# so this is defence-in-depth.
_MAX_ENDPOINTS = 64
_OTHER = "other"


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: Counter = Counter()          # (endpoint, status) -> n
        self.latency_sum: Dict[str, float] = defaultdict(float)
        self.latency_count: Counter = Counter()     # endpoint -> n
        self.grades: Counter = Counter()            # pass|fail|error|abstain -> n

    def _bounded(self, endpoint: str) -> str:
        if endpoint in self.latency_count or len(self.latency_count) < _MAX_ENDPOINTS:
            return endpoint
        return _OTHER

    def record_request(self, endpoint: str, status: int, latency_ms: float) -> None:
        with self._lock:
            endpoint = self._bounded(endpoint)
            self.requests[(endpoint, status)] += 1
            self.latency_sum[endpoint] += latency_ms
            self.latency_count[endpoint] += 1

    def record_grade(self, result: str) -> None:
        with self._lock:
            self.grades[result] += 1

    def render_prometheus(self, cache_stats: Optional[dict] = None) -> str:
        lines = []
        with self._lock:
            lines.append("# TYPE docsthatrun_requests_total counter")
            for (endpoint, status), n in sorted(self.requests.items()):
                lines.append(
                    f'docsthatrun_requests_total{{endpoint="{_esc(endpoint)}",'
                    f'status="{status}"}} {n}'
                )
            lines.append("# TYPE docsthatrun_request_latency_ms_sum counter")
            for endpoint, s in sorted(self.latency_sum.items()):
                lines.append(
                    f'docsthatrun_request_latency_ms_sum{{endpoint="{_esc(endpoint)}"}} '
                    f'{round(s, 2)}'
                )
                lines.append(
                    f'docsthatrun_request_latency_ms_count{{endpoint="{_esc(endpoint)}"}} '
                    f'{self.latency_count[endpoint]}'
                )
            lines.append("# TYPE docsthatrun_grades_total counter")
            for result, n in sorted(self.grades.items()):
                lines.append(f'docsthatrun_grades_total{{result="{_esc(result)}"}} {n}')
        if cache_stats:
            lines.append("# TYPE docsthatrun_cache_hits_total counter")
            lines.append(f'docsthatrun_cache_hits_total {cache_stats.get("hits", 0)}')
            lines.append(f'docsthatrun_cache_misses_total {cache_stats.get("misses", 0)}')
            lines.append("# TYPE docsthatrun_cache_size gauge")
            lines.append(f'docsthatrun_cache_size {cache_stats.get("size", 0)}')
        return "\n".join(lines) + "\n"

    def snapshot(self, cache_stats: Optional[dict] = None) -> dict:
        with self._lock:
            reqs = {f"{e} {s}": n for (e, s), n in self.requests.items()}
            avg = {
                e: round(self.latency_sum[e] / self.latency_count[e], 1)
                for e in self.latency_count
            }
            out = {
                "requests": reqs,
                "avg_latency_ms": avg,
                "grades": dict(self.grades),
            }
        if cache_stats:
            out["cache"] = cache_stats
        return out
