import json
import unittest
from unittest.mock import patch

from src.ai.gemini import AIProviderError, GeminiProvider


class FakeHTTPResponse:
    def __init__(self, body: dict):
        self.body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.body


class GeminiProviderTests(unittest.TestCase):
    @patch("src.ai.gemini.tls_context", return_value=None)
    @patch("src.ai.gemini.urllib.request.urlopen")
    def test_structured_response_and_production_generation_settings(self, urlopen, _tls_context):
        result_json = {
            "intent": "GENERAL_QUESTION",
            "reply": "Понимаю вас. Давайте спокойно разберёмся.",
            "lead_detected": False,
            "contact_consent": False,
            "extracted_data": {
                "child_name": None,
                "child_grade": None,
                "parent_name": None,
                "parent_phone": None,
            },
        }
        urlopen.return_value = FakeHTTPResponse({
            "candidates": [{
                "finishReason": "STOP",
                "content": {"parts": [
                    {"thought": True, "text": "internal reasoning"},
                    {"text": json.dumps(result_json, ensure_ascii=False)},
                ]},
            }],
        })

        result = GeminiProvider("secret", "gemini-3.5-flash").analyze(
            "Я растерялась", [], {}, "", None,
        )

        self.assertEqual(result.intent, "GENERAL_QUESTION")
        self.assertIn("разберёмся", result.reply)
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode())
        config = payload["generationConfig"]
        self.assertEqual(config["thinkingConfig"], {"thinkingLevel": "minimal"})
        self.assertEqual(config["maxOutputTokens"], 4096)
        self.assertNotIn("temperature", config)
        self.assertIn("AVAILABILITY", config["responseJsonSchema"]["properties"]["intent"]["enum"])

    @patch("src.ai.gemini.tls_context", return_value=None)
    @patch("src.ai.gemini.urllib.request.urlopen")
    def test_truncated_json_becomes_safe_provider_error(self, urlopen, _tls_context):
        urlopen.return_value = FakeHTTPResponse({
            "candidates": [{
                "finishReason": "MAX_TOKENS",
                "content": {"parts": [{"text": '{"intent":"GENERAL'}]},
            }],
        })

        with self.assertRaisesRegex(AIProviderError, "truncated"):
            GeminiProvider("secret", "gemini-3.5-flash").analyze("Привет", [], {}, "", None)


if __name__ == "__main__":
    unittest.main()
