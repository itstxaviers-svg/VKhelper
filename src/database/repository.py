from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path


LEAD_FIELDS = ("child_name", "child_grade", "parent_name", "parent_phone")


class Repository:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def _migrate(self) -> None:
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                vk_user_id TEXT PRIMARY KEY, first_seen TEXT NOT NULL, last_seen TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, vk_user_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')), content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_user_created ON messages(vk_user_id, id DESC);
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT, vk_user_id TEXT NOT NULL UNIQUE,
                child_name TEXT, child_grade TEXT, parent_name TEXT, parent_phone TEXT,
                status TEXT NOT NULL, contact_consent INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS processed_events (
                event_id TEXT PRIMARY KEY, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS daily_summaries (
                summary_date TEXT PRIMARY KEY, sent_at TEXT NOT NULL
            );
        """)
        self.connection.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def touch_user(self, user_id: str) -> bool:
        now = self._now()
        existed = self.connection.execute("SELECT 1 FROM users WHERE vk_user_id = ?", (user_id,)).fetchone() is not None
        if existed:
            self.connection.execute("UPDATE users SET last_seen = ? WHERE vk_user_id = ?", (now, user_id))
        else:
            self.connection.execute("INSERT INTO users(vk_user_id, first_seen, last_seen) VALUES (?, ?, ?)", (user_id, now, now))
        self.connection.commit()
        return not existed

    def save_message(self, user_id: str, role: str, content: str) -> None:
        self.connection.execute("INSERT INTO messages(vk_user_id, role, content, created_at) VALUES (?, ?, ?, ?)", (user_id, role, content, self._now()))
        self.connection.commit()

    def history(self, user_id: str, limit: int = 20) -> list[dict]:
        rows = self.connection.execute("SELECT role, content FROM messages WHERE vk_user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_lead(self, user_id: str) -> dict | None:
        row = self.connection.execute("SELECT * FROM leads WHERE vk_user_id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    def ensure_lead(self, user_id: str) -> dict:
        lead = self.get_lead(user_id)
        if lead:
            return lead
        now = self._now()
        self.connection.execute("INSERT INTO leads(vk_user_id, status, created_at, updated_at) VALUES (?, 'COLLECTING_CONTACTS', ?, ?)", (user_id, now, now))
        self.connection.commit()
        return self.get_lead(user_id)  # type: ignore[return-value]

    def update_lead(self, user_id: str, values: dict[str, str | None], status: str | None = None, consent: bool | None = None) -> dict:
        lead = self.ensure_lead(user_id)
        updates: dict[str, object] = {key: value.strip() for key, value in values.items() if key in LEAD_FIELDS and isinstance(value, str) and value.strip()}
        if status:
            updates["status"] = status
        if consent is not None:
            updates["contact_consent"] = int(consent)
        if updates:
            updates["updated_at"] = self._now()
            columns = ", ".join(f"{column} = ?" for column in updates)
            self.connection.execute(f"UPDATE leads SET {columns} WHERE vk_user_id = ?", (*updates.values(), user_id))
            self.connection.commit()
        return self.get_lead(user_id) or lead

    def claim_event(self, event_id: str) -> bool:
        try:
            self.connection.execute("INSERT INTO processed_events(event_id, created_at) VALUES (?, ?)", (event_id, self._now()))
            self.connection.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def daily_incoming_stats(self, start_at: str, end_at: str) -> tuple[int, int]:
        row = self.connection.execute(
            "SELECT COUNT(*) AS messages, COUNT(DISTINCT vk_user_id) AS users FROM messages WHERE role = 'user' AND created_at >= ? AND created_at < ?",
            (start_at, end_at),
        ).fetchone()
        return int(row["messages"]), int(row["users"])

    def daily_summary_sent(self, summary_date: str) -> bool:
        return self.connection.execute("SELECT 1 FROM daily_summaries WHERE summary_date = ?", (summary_date,)).fetchone() is not None

    def mark_daily_summary_sent(self, summary_date: str) -> None:
        self.connection.execute("INSERT OR IGNORE INTO daily_summaries(summary_date, sent_at) VALUES (?, ?)", (summary_date, self._now()))
        self.connection.commit()

    def consented_leads_updated_between(self, start_at: str, end_at: str) -> list[dict]:
        rows = self.connection.execute(
            "SELECT vk_user_id, child_name, child_grade, parent_name, parent_phone FROM leads WHERE contact_consent = 1 AND updated_at >= ? AND updated_at < ? ORDER BY updated_at",
            (start_at, end_at),
        ).fetchall()
        return [dict(row) for row in rows]
