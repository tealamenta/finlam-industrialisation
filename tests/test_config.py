"""Configuration et journalisation."""

import json
import logging

import pytest

from finlam_prod.config import Settings
from finlam_prod.logging_utils import JsonFormatter, configure_logging


def test_settings_defaults_and_env_override(tmp_path):
    base = Settings.from_env({}, dotenv=None)
    assert base.max_pages == 200 and base.api_key == "" and base.log_json is False
    custom = Settings.from_env(
        {"FINLAM_MAX_PAGES": "5", "FINLAM_LOG_JSON": "1", "FINLAM_MODELS_DIR": "m"}, dotenv=None
    )
    assert custom.max_pages == 5 and custom.log_json is True and str(custom.models_dir) == "m"
    assert custom.describe()["api_key"] == ""


def test_settings_dotenv_then_env_priority(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("FINLAM_MAX_PAGES=7\nFINLAM_API_KEY='k'\n# commentaire\n", encoding="utf-8")
    s = Settings.from_env({"FINLAM_MAX_PAGES": "9"}, dotenv=dotenv)
    assert s.max_pages == 9 and s.api_key == "k"
    assert s.describe()["api_key"] == "***"
    assert "FINLAM_API_KEY" not in s.source


def test_settings_rejects_bad_values():
    with pytest.raises(ValueError, match="entier"):
        Settings.from_env({"FINLAM_MAX_PAGES": "beaucoup"}, dotenv=None)
    with pytest.raises(ValueError, match="LOG_LEVEL"):
        Settings.from_env({"FINLAM_LOG_LEVEL": "bavard"}, dotenv=None)


def test_json_formatter_emits_one_json_line_with_extra():
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "bonjour %s", ("monde",), None)
    record.route = "/issues"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "bonjour monde" and payload["route"] == "/issues"
    assert payload["level"] == "INFO" and payload["ts"].endswith("Z")


def test_configure_logging_is_idempotent():
    settings = Settings.from_env({"FINLAM_LOG_JSON": "1"}, dotenv=None)
    configure_logging(settings)
    configure_logging(settings)
    ours = [h for h in logging.getLogger().handlers if getattr(h, "_finlam", False)]
    assert len(ours) == 1 and isinstance(ours[0].formatter, JsonFormatter)
