"""Journalisation : lisible en développement, JSON une ligne par événement en exploitation.

``configure_logging()`` lit :class:`~finlam_prod.config.Settings` ; avec
``FINLAM_LOG_JSON=1`` chaque enregistrement devient un objet JSON
(``ts``, ``level``, ``logger``, ``msg`` + champs ``extra``), ce qu'un
collecteur (Loki, ELK, journald) indexe sans analyse de texte.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

from finlam_prod.config import Settings, get_settings

_STANDARD = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Un enregistrement → une ligne JSON, champs ``extra`` inclus."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(settings: Settings | None = None, level: str | None = None) -> None:
    """Configure la racine une seule fois (idempotent)."""
    settings = settings or get_settings()
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_finlam", False):
            root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler._finlam = True  # type: ignore[attr-defined]
    if settings.log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(level or settings.log_level)
    for noisy in ("httpx", "urllib3", "mlflow", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
