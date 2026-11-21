"""Structured logger kernel bindings — logger.c.

Wraps C implementation in ``kernel/logger.c`` with pure-Python fallback.
Supports JSON output, severity levels, and per-rank logging.
"""

import json
import sys
import time
import threading
from typing import Optional, Dict, Any

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_core_kernel")


class Logger:
    """Structured logger with severity levels and JSON output.

    Levels: DEBUG, INFO, WARN, ERROR, FATAL.
    """

    LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3, "FATAL": 4}

    def __init__(self, name: str = "sneppx", level: str = "INFO",
                 json_output: bool = False, rank: int = 0):
        self.name = name
        self.level = level
        self.json_output = json_output
        self.rank = rank
        self._lock = threading.Lock()
        self._has_c = _HAS_C

    def _should_log(self, level: str) -> bool:
        return self.LEVELS.get(level, 0) >= self.LEVELS.get(self.level, 0)

    def _format(self, level: str, message: str, extra: Optional[dict] = None) -> str:
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "level": level,
            "name": self.name,
            "rank": self.rank,
            "message": message,
        }
        if extra:
            record.update(extra)
        if self.json_output:
            return json.dumps(record)
        return f"[{record['timestamp']}] [{level}] [{self.name}] {message}"

    def debug(self, message: str, **kwargs) -> None:
        if self._should_log("DEBUG"):
            self._emit("DEBUG", message, kwargs)

    def info(self, message: str, **kwargs) -> None:
        if self._should_log("INFO"):
            self._emit("INFO", message, kwargs)

    def warn(self, message: str, **kwargs) -> None:
        if self._should_log("WARN"):
            self._emit("WARN", message, kwargs)

    def error(self, message: str, **kwargs) -> None:
        if self._should_log("ERROR"):
            self._emit("ERROR", message, kwargs)

    def fatal(self, message: str, **kwargs) -> None:
        if self._should_log("FATAL"):
            self._emit("FATAL", message, kwargs)

    def _emit(self, level: str, message: str, extra: dict) -> None:
        with self._lock:
            line = self._format(level, message, extra)
            if level in ("ERROR", "FATAL"):
                print(line, file=sys.stderr)
            else:
                print(line)

    def set_level(self, level: str) -> None:
        if level in self.LEVELS:
            self.level = level

    def named(self, sub_name: str) -> "Logger":
        return Logger(f"{self.name}.{sub_name}", self.level, self.json_output, self.rank)
