"""Central runtime configuration, loaded once from the environment.

Every tunable the server / sandbox / LLM path uses lives here so behaviour is
env-driven (12-factor) rather than scattered ``os.environ`` reads. Import the
module-level ``settings`` singleton; call ``Settings.from_env()`` in a test to
build an isolated copy. Stdlib-only, so the zero-dependency core still imports it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _csv(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return tuple(p.strip() for p in raw.split(",") if p.strip())


@dataclass(frozen=True)
class Settings:
    # ---- LLM ----------------------------------------------------------------
    model: str = "claude-opus-4-8"
    effort: str = "medium"
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2

    # ---- retrieval / answer -------------------------------------------------
    default_version: str = "v2"
    top_k_default: int = 5
    top_k_max: int = 50
    max_question_chars: int = 2000

    # ---- sandbox execution limits -------------------------------------------
    sandbox_timeout_s: int = 20        # wall-clock
    sandbox_cpu_seconds: int = 10      # RLIMIT_CPU
    sandbox_memory_mb: int = 1024      # RLIMIT_AS (address space); generous so
                                       # a legit pydantic import never false-fails
    sandbox_file_mb: int = 10          # RLIMIT_FSIZE (max file write)

    # ---- answer cache -------------------------------------------------------
    cache_max: int = 256
    cache_ttl_s: float = 900.0

    # ---- rate limiting (per client IP, token bucket) ------------------------
    rate_limit_rpm: int = 60           # sustained requests/min; 0 disables
    rate_limit_burst: int = 20         # bucket capacity

    # ---- server -------------------------------------------------------------
    cors_origins: Tuple[str, ...] = ()
    log_level: str = "INFO"
    log_json: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model=os.environ.get("DOCSTHATRUN_MODEL", cls.model),
            effort=os.environ.get("DOCSTHATRUN_EFFORT", cls.effort),
            llm_timeout_s=_float("DOCSTHATRUN_LLM_TIMEOUT", cls.llm_timeout_s),
            llm_max_retries=_int("DOCSTHATRUN_LLM_RETRIES", cls.llm_max_retries),
            default_version=os.environ.get("DOCSTHATRUN_DEFAULT_VERSION", cls.default_version),
            top_k_default=_int("DOCSTHATRUN_TOP_K", cls.top_k_default),
            top_k_max=_int("DOCSTHATRUN_TOP_K_MAX", cls.top_k_max),
            max_question_chars=_int("DOCSTHATRUN_MAX_QUESTION_CHARS", cls.max_question_chars),
            sandbox_timeout_s=_int("DOCSTHATRUN_SANDBOX_TIMEOUT", cls.sandbox_timeout_s),
            sandbox_cpu_seconds=_int("DOCSTHATRUN_SANDBOX_CPU", cls.sandbox_cpu_seconds),
            sandbox_memory_mb=_int("DOCSTHATRUN_SANDBOX_MEM_MB", cls.sandbox_memory_mb),
            sandbox_file_mb=_int("DOCSTHATRUN_SANDBOX_FILE_MB", cls.sandbox_file_mb),
            cache_max=_int("DOCSTHATRUN_CACHE_MAX", cls.cache_max),
            cache_ttl_s=_float("DOCSTHATRUN_CACHE_TTL", cls.cache_ttl_s),
            rate_limit_rpm=_int("DOCSTHATRUN_RATE_RPM", cls.rate_limit_rpm),
            rate_limit_burst=_int("DOCSTHATRUN_RATE_BURST", cls.rate_limit_burst),
            cors_origins=_csv("DOCSTHATRUN_CORS_ORIGINS", cls.cors_origins),
            log_level=os.environ.get("DOCSTHATRUN_LOG_LEVEL", cls.log_level).upper(),
            log_json=_bool("DOCSTHATRUN_LOG_JSON", cls.log_json),
        )


settings = Settings.from_env()
