# Databricks notebook source
"""Rename the online-feature Lakeflow pipeline only. Does not rename any UC table."""

from src.n00_shared.feature_store import rename_online_sync_pipeline
from src.n00_shared.runtime import load_settings

settings = load_settings()
pid = rename_online_sync_pipeline(settings)
msg = f"pipeline={pid or 'not-found'} name={settings.online_sync_pipeline_name}"
print(msg)
try:
    dbutils.notebook.exit(msg)  # noqa: F821
except Exception:
    pass
