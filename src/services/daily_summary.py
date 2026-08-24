from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

from src.database.repository import Repository


class DailySummaryService:
    def __init__(self, repository: Repository, manager_vk_id: str, timezone: str, summary_hour: int = 20) -> None:
        self.repository = repository
        self.manager_vk_id = manager_vk_id
        self.zone = ZoneInfo(timezone)
        self.summary_hour = summary_hour

    def run_if_due(self, send_message, now: datetime | None = None) -> bool:
        if not self.manager_vk_id:
            return False
        local_now = (now or datetime.now(self.zone)).astimezone(self.zone)
        if local_now.hour < self.summary_hour:
            return False
        summary_date = local_now.date().isoformat()
        if self.repository.daily_summary_sent(summary_date):
            return False
        start = datetime.combine(local_now.date(), time.min, tzinfo=self.zone)
        end = start + timedelta(days=1)
        start_at, end_at = start.astimezone(UTC).isoformat(), end.astimezone(UTC).isoformat()
        messages, users = self.repository.daily_incoming_stats(start_at, end_at)
        if not messages:
            return False
        lines = [f"Итоги сообщений за {summary_date}", "", f"Новых сообщений: {messages}", f"Написали пользователей: {users}"]
        leads = self.repository.consented_leads_updated_between(start_at, end_at)
        if leads:
            lines.extend(["", "Заявки с согласием на передачу контактов:"])
            for lead in leads:
                lines.extend([
                    "",
                    f"Родитель: {lead['parent_name'] or 'не указано'}",
                    f"Ребёнок: {lead['child_name'] or 'не указано'}",
                    f"Класс: {lead['child_grade'] or 'не указан'}",
                    f"Телефон: {lead['parent_phone'] or 'не указан'}",
                    f"VK user_id: {lead['vk_user_id']}",
                ])
        send_message(self.manager_vk_id, "\n".join(lines))
        self.repository.mark_daily_summary_sent(summary_date)
        return True
