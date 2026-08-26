import json
import unittest
from unittest.mock import patch

from src.ai.kie import KieProvider


class FakeHTTPResponse:
    def __init__(self, body: dict):
        self.body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.body


class KieProviderTests(unittest.TestCase):
    @patch("src.ai.kie.tls_context", return_value=None)
    @patch("src.ai.kie.urllib.request.urlopen")
    def test_uses_kie_chat_endpoint_and_parses_structured_reply(self, urlopen, _tls_context):
        result_json = {
            "intent": "GENERAL_QUESTION",
            "reply": "С удовольствием помогу.",
            "lead_detected": False,
            "contact_consent": False,
            "extracted_data": {},
        }
        urlopen.return_value = FakeHTTPResponse({
            "choices": [{"message": {"content": json.dumps(result_json, ensure_ascii=False)}}]
        })

        result = KieProvider("secret", "gpt-5-2").analyze("Привет", [], {}, "", None)

        self.assertEqual(result.intent, "GENERAL_QUESTION")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.kie.ai/gpt-5-2/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        self.assertFalse(json.loads(request.data.decode())["stream"])


if __name__ == "__main__":
    unittest.main()
