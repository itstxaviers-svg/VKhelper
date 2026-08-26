from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path) -> None:
    """Small .env reader, deliberately avoiding another runtime dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    ai_provider: str
    kie_api_key: str
    kie_model: str
    vk_group_token: str
    vk_group_id: str
    database_path: Path
    manager_vk_id: str
    manager_vk_url: str
    timezone: str
    daily_summary_hour: int
    business: dict
    knowledge: str

    @classmethod
    def from_project_root(cls, root: Path) -> "Settings":
        load_dotenv(root / ".env")
        # BotHost mounts its writable database volume at /app/data.  Keep the
        # bundled community facts beside the application code so the mount
        # cannot hide them at runtime.
        configured_data = root / "data"
        bundled_data = Path(__file__).resolve().parent
        data_source = bundled_data
        business: dict | None = None
        for candidate in (configured_data, bundled_data):
            try:
                business = json.loads((candidate / "business.json").read_text(encoding="utf-8"))
                data_source = candidate
                break
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        if business is None:
            raise RuntimeError("business.json is unavailable")
        return cls(
            ai_provider=os.getenv("AI_PROVIDER", "kie").lower(),
            kie_api_key=os.getenv("KIE_API_KEY", ""),
            kie_model=os.getenv("KIE_MODEL", "gpt-5-2"),
            vk_group_token=os.getenv("VK_GROUP_TOKEN", ""),
            vk_group_id=os.getenv("VK_GROUP_ID", ""),
            database_path=Path(os.getenv("DATABASE_PATH", str(root / "data" / "vkhelper.sqlite3"))),
            manager_vk_id=os.getenv("MANAGER_VK_ID", business.get("manager_vk_id", "")),
            manager_vk_url=os.getenv("MANAGER_VK_URL", business.get("manager_vk_url", "")),
            timezone=os.getenv("TIMEZONE", "Asia/Yekaterinburg"),
            daily_summary_hour=int(os.getenv("DAILY_SUMMARY_HOUR", "20")),
            business=business,
            knowledge=(data_source / "knowledge.md").read_text(encoding="utf-8"),
        )
