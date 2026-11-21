"""Structured (JSON-lines) logging (pure Python)."""

import json
import sys
import time
from typing import Any, Dict, List, Optional


class StructuredLogger:
    """Emits ``{level, logger, message, timestamp, **fields}`` records.

    Records are appended to ``self.records`` and optionally written as JSON
    lines to ``stream``.
    """

    def __init__(self, name: str = "root", stream=None, level: str = "INFO"):
        self.name = name
        self.stream = stream
        self.level = level.upper()
        self.records: List[Dict[str, Any]] = []

    _ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}

    def _log(self, level: str, message: str, **fields) -> Dict[str, Any]:
        record = {
            "level": level,
            "logger": self.name,
            "message": message,
            "timestamp": time.time(),
            **fields,
        }
        self.records.append(record)
        if self.stream is not None:
            if self._ORDER[level] >= self._ORDER.get(self.level, 20):
                self.stream.write(json.dumps(record) + "\n")
        return record

    def debug(self, message: str, **fields):
        return self._log("DEBUG", message, **fields)

    def info(self, message: str, **fields):
        return self._log("INFO", message, **fields)

    def warning(self, message: str, **fields):
        return self._log("WARNING", message, **fields)

    def error(self, message: str, **fields):
        return self._log("ERROR", message, **fields)

    def get_records(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        if level is None:
            return list(self.records)
        return [r for r in self.records if r["level"] == level]

    def clear(self) -> None:
        self.records.clear()


def get_logger(
    name: str = "root", stream=None, level: str = "INFO"
) -> StructuredLogger:
    return StructuredLogger(name, stream=stream, level=level)
