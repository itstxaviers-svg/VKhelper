import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from src.database.repository import Repository
from src.services.daily_summary import DailySummaryService


class DailySummaryTests(unittest.TestCase):
    def test_sends_once_after_20_if_messages_exist(self):
        repo = Repository(":memory:")
        repo.touch_user("1")
        repo.save_message("1", "user", "Здравствуйте")
        sent = []
        service = DailySummaryService(repo, "42", "Asia/Yekaterinburg")
        now = datetime(2026, 8, 23, 20, 5, tzinfo=ZoneInfo("Asia/Yekaterinburg"))
        self.assertTrue(service.run_if_due(lambda user, text: sent.append((user, text)), now))
        self.assertIn("Новых сообщений: 1", sent[0][1])
        self.assertFalse(service.run_if_due(lambda user, text: sent.append((user, text)), now))

    def test_summary_includes_only_consented_lead_contacts(self):
        repo = Repository(":memory:")
        repo.touch_user("1")
        repo.save_message("1", "user", "Здравствуйте")
        repo.update_lead("1", {"parent_name": "Елена", "child_name": "Маша", "child_grade": "5", "parent_phone": "+7 999 123-45-67"}, consent=True)
        sent = []
        service = DailySummaryService(repo, "42", "Asia/Yekaterinburg")
        now = datetime.now(ZoneInfo("Asia/Yekaterinburg")).replace(hour=20, minute=5)
        service.run_if_due(lambda user, text: sent.append(text), now)
        self.assertIn("Телефон: +7 999 123-45-67", sent[0])
