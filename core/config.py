"""Shared configuration loader for outfit_config.ini."""

import configparser
import os

_config_cache = None
_config_mtime = 0.0


def _get_project_root():
    """Get the project root directory (parent of core/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_config_path():
    return os.path.join(_get_project_root(), "outfit_config.ini")


def get_config():
    """Read and return ConfigParser. Re-reads automatically if file changed on disk."""
    global _config_cache, _config_mtime
    config_path = _get_config_path()

    try:
        mtime = os.path.getmtime(config_path)
    except OSError:
        return configparser.ConfigParser()

    if _config_cache is not None and mtime == _config_mtime:
        return _config_cache

    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    _config_cache = config
    _config_mtime = mtime
    return config


def get_model_path(key, fallback=""):
    """Get a path from [models] section. Returns empty string if not set."""
    config = get_config()
    return config.get("models", key, fallback=fallback).strip()
