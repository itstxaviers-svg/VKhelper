from __future__ import annotations

import json
import logging
import random
import time
import urllib.parse
import urllib.request

from src.http_client import tls_context


class VKClientError(RuntimeError):
    pass


class VKClient:
    API_VERSION = "5.199"

    def __init__(self, token: str, group_id: str) -> None:
        if not token or not group_id:
            raise VKClientError("VK_GROUP_TOKEN and VK_GROUP_ID must be configured")
        self.token, self.group_id = token, group_id

    def _call(self, method: str, **params):
        params.update(access_token=self.token, v=self.API_VERSION)
        url = f"https://api.vk.com/method/{method}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=25, context=tls_context()) as response:
                data = json.loads(response.read().decode())
        except Exception as exc:
            raise VKClientError(f"VK request failed: {method}") from exc
        if "error" in data:
            raise VKClientError(f"VK error {data['error'].get('error_code')}: {method}")
        return data["response"]

    def long_poll_server(self) -> dict:
        return self._call("groups.getLongPollServer", group_id=self.group_id)

    def send_message(self, user_id: str, text: str) -> None:
        self._call("messages.send", user_id=user_id, random_id=random.randint(1, 2**31 - 1), message=text)

    def poll_forever(self, on_message, on_tick=None) -> None:
        server = None
        while True:
            try:
                if server is None:
                    server = self.long_poll_server()
                if on_tick:
                    on_tick()
                query = urllib.parse.urlencode({"act": "a_check", "key": server["key"], "wait": 25, "ts": server["ts"]})
                with urllib.request.urlopen(f"{server['server']}?{query}", timeout=35, context=tls_context()) as response:
                    data = json.loads(response.read().decode())
                if data.get("failed"):
                    server = self.long_poll_server()
                    continue
                server["ts"] = data["ts"]
                for update in data.get("updates", []):
                    if update.get("type") == "message_new":
                        on_message(update["object"]["message"])
            except Exception:
                logging.exception("vk_long_poll_error")
                time.sleep(2)
                server = None
