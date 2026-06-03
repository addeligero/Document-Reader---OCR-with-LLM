from __future__ import annotations

from .models import AppSetting


def setting_bool(key: str) -> bool:
    setting = AppSetting.query.get(key)
    return (setting.value if setting else "false").lower() == "true"
