from __future__ import annotations

import json
import logging
from pathlib import Path

class JsonFormatter(logging.Formatter):
    """
    Custom JSON formatter for logging, enables to write warning and not loose them if things go wrong further away.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "logger": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
            "file": record.filename,
            "line": record.lineno,
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)