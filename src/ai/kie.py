from __future__ import annotations

import json
import urllib.error
import urllib.request

from src.http_client import tls_context

from .gemini import SYSTEM_INSTRUCTION
from .provider import AIProvider, AIResult


class KieProviderError(RuntimeError):
    pass


class KieProvider(AIProvider):
    """Kie.ai chat-completions adapter.

    Kie exposes model-specific, OpenAI-compatible chat endpoints.  The model
    name is deliberately an environment setting so it can be changed without a
    code deployment.
    """

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise KieProviderError("KIE_API_KEY is not configured")
        self.api_key = api_key
        self.model = model

    def analyze(self, message: str, history: list[dict], business: dict, knowledge: str, lead: dict | None) -> AIResult:
        context = {
            "business": business,
            "knowledge": knowledge[:12000],
            "lead": lead or {},
            "history": history,
            "message": message,
        }
        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
            "stream": False,
        }
        endpoint = f"https://api.kie.ai/{self.model}/v1/chat/completions"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60, context=tls_context()) as response:
                body = json.loads(response.read().decode())
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise KieProviderError("Kie returned a non-text response")
            return AIResult.from_dict(json.loads(content))
        except KieProviderError:
            raise
        except urllib.error.HTTPError as exc:
            raise KieProviderError(f"Kie HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise KieProviderError(f"Kie network error: {type(exc.reason).__name__}") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise KieProviderError("Kie returned an invalid response") from exc
