from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


INTENTS = {
    "GREETING", "ADDRESS", "PRICE", "SCHEDULE", "ACTIVITY_STATUS",
    "MONTHLY_FREQUENCY", "ABOUT", "AVAILABILITY", "ENROLLMENT", "CONTACT_MANAGER",
    "CONSENT_TO_CONTACT", "ADVERTISEMENT", "GENERAL_QUESTION", "UNKNOWN",
}


@dataclass
class AIResult:
    intent: str = "UNKNOWN"
    reply: str = ""
    lead_detected: bool = False
    contact_consent: bool = False
    extracted_data: dict[str, str | None] | None = None

    @classmethod
    def from_dict(cls, raw: dict) -> "AIResult":
        extracted = raw.get("extracted_data") or {}
        if not isinstance(extracted, dict):
            extracted = {}
        intent = str(raw.get("intent", "UNKNOWN")).upper()
        return cls(
            intent=intent if intent in INTENTS else "UNKNOWN",
            reply=str(raw.get("reply", ""))[:1200],
            lead_detected=bool(raw.get("lead_detected", False)),
            contact_consent=bool(raw.get("contact_consent", False)),
            extracted_data={key: extracted.get(key) for key in ("child_name", "child_grade", "parent_name", "parent_phone")},
        )


class AIProvider(Protocol):
    def analyze(self, message: str, history: list[dict], business: dict, knowledge: str, lead: dict | None) -> AIResult: ...
