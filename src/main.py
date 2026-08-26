from __future__ import annotations

import logging
from pathlib import Path

from src.ai.kie import KieProvider
from src.config.settings import Settings
from src.database.repository import Repository
from src.services.conversation import ConversationService
from src.services.daily_summary import DailySummaryService
from src.vk.client import VKClient


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(__file__).resolve().parents[1]
    settings = Settings.from_project_root(root)
    if settings.ai_provider != "kie":
        raise RuntimeError(f"Unsupported AI_PROVIDER: {settings.ai_provider}")
    repository = Repository(settings.database_path)
    service = ConversationService(repository, KieProvider(settings.kie_api_key, settings.kie_model), settings.business, settings.knowledge, settings.manager_vk_id, settings.manager_vk_url)
    vk = VKClient(settings.vk_group_token, settings.vk_group_id)
    summaries = DailySummaryService(repository, settings.manager_vk_id, settings.timezone, settings.daily_summary_hour)

    def on_message(message: dict) -> None:
        user_id = str(message.get("from_id", ""))
        text = str(message.get("text", ""))
        message_id = message.get("conversation_message_id") or message.get("id")
        event_id = f"{message.get('peer_id')}:{message_id}"
        if not user_id or not text or not repository.claim_event(event_id):
            return
        logging.info("vk_message_received", extra={"vk_user_id": user_id})
        reply = service.handle(user_id, text)
        vk.send_message(user_id, reply.text)
        logging.info("reply_sent", extra={"vk_user_id": user_id})
        if reply.notify_manager and reply.lead:
            vk.send_message(settings.manager_vk_id, service.manager_notification(reply.lead, user_id))
            repository.update_lead(user_id, {}, status="HANDED_TO_MANAGER")
            logging.info("lead_handed_to_manager", extra={"vk_user_id": user_id})

    def on_tick() -> None:
        try:
            if summaries.run_if_due(vk.send_message):
                logging.info("daily_summary_sent")
        except Exception:
            logging.warning("daily_summary_failed")

    logging.info("bot_started")
    vk.poll_forever(on_message, on_tick)


if __name__ == "__main__":
    main()
