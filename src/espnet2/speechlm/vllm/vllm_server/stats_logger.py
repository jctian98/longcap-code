"""JSONL engine-stats logger for the captioner service.

vLLM v1's default ``LoggingStatLogger`` prints through the vllm logger, which
is unreliable once uvicorn reconfigures logging (measured 2026-08-27: zero
periodic stat lines in a service log despite ``log_stats=True``). For serving
benchmarks we want machine-readable scheduler/cache counters anyway, so this
custom ``StatLoggerBase`` appends one JSON line every ``interval`` seconds:

    {"ts": ..., "running": R, "waiting": W, "kv_usage": 0.42,
     "gen_tok": ΔG, "prompt_computed_tok": ΔP, "prompt_cached_tok": ΔC,
     "prefix_queries": Δq, "prefix_hits": Δh, "mm_queries": Δq, "mm_hits": Δh,
     "preempted": Δn}

Token counters are deltas since the previous line; ``prompt_cached_tok`` /
``prompt_computed_tok`` decompose each scheduled prompt into prefix-cache-hit
vs actually-computed tokens (the server-side truth for APC effectiveness).
Enabled via the service's ``--stats-jsonl PATH``.
"""

from __future__ import annotations

import json
import time


def make_jsonl_stat_logger_factory(path: str, interval: float = 5.0):
    """Returns a StatLoggerFactory: ``(vllm_config, engine_idx) -> logger``."""

    def factory(vllm_config, engine_index: int = 0):
        return _JsonlStatLogger(path, interval, engine_index)

    return factory


class _JsonlStatLogger:
    def __init__(self, path: str, interval: float, engine_index: int = 0):
        self._f = open(path, "a", buffering=1)
        self._interval = interval
        self._last_flush = time.time()
        self._acc = self._zero()

    @staticmethod
    def _zero():
        return {"gen_tok": 0, "prompt_computed_tok": 0, "prompt_cached_tok": 0,
                "prefix_queries": 0, "prefix_hits": 0,
                "mm_queries": 0, "mm_hits": 0, "preempted": 0}

    def record(self, scheduler_stats, iteration_stats,
               mm_cache_stats=None, engine_idx: int = 0):
        a = self._acc
        if iteration_stats is not None:
            a["gen_tok"] += getattr(iteration_stats, "num_generation_tokens", 0)
            pts = getattr(iteration_stats, "prompt_token_stats", None)
            if pts is not None:
                a["prompt_computed_tok"] += getattr(pts, "computed", 0)
                a["prompt_cached_tok"] += getattr(pts, "cached_tokens", 0)
            a["preempted"] += getattr(iteration_stats, "num_preempted_reqs", 0)
        if mm_cache_stats is not None:
            a["mm_queries"] += getattr(mm_cache_stats, "queries", 0)
            a["mm_hits"] += getattr(mm_cache_stats, "hits", 0)
        pcs = getattr(scheduler_stats, "prefix_cache_stats", None) \
            if scheduler_stats is not None else None
        if pcs is not None:
            a["prefix_queries"] += getattr(pcs, "queries", 0)
            a["prefix_hits"] += getattr(pcs, "hits", 0)

        now = time.time()
        if now - self._last_flush >= self._interval:
            line = {"ts": round(now, 1), **{k: a[k] for k in a}}
            if scheduler_stats is not None:
                line["running"] = getattr(scheduler_stats, "num_running_reqs", None)
                line["waiting"] = getattr(scheduler_stats, "num_waiting_reqs", None)
                kv = getattr(scheduler_stats, "kv_cache_usage", None)
                line["kv_usage"] = round(kv, 4) if kv is not None else None
            line["dt"] = round(now - self._last_flush, 2)
            self._f.write(json.dumps(line) + "\n")
            self._last_flush = now
            self._acc = self._zero()

    def log_engine_initialized(self):
        self._f.write(json.dumps({"ts": round(time.time(), 1),
                                  "event": "engine_initialized"}) + "\n")

    def log(self):
        pass

    def record_sleep_state(self, is_awake: int, level: int):
        pass
