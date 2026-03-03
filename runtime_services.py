import math
import random
import threading
import time
from collections import deque

import requests


class TelemetryBuffer:
    def __init__(self, supabase_url, supabase_key, batch_size, metrics, logger):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.batch_size = batch_size
        self.metrics = metrics
        self.logger = logger
        self.queue = deque()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.last_payload_by_event = {}
        self.worker = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        self.worker.join(timeout=2)

    def enqueue(self, event, payload):
        safe_payload = self._sanitize_for_json(payload)
        if self.last_payload_by_event.get(event) == safe_payload:
            return
        self.last_payload_by_event[event] = safe_payload
        with self.lock:
            self.queue.append((event, safe_payload))

    def _sanitize_for_json(self, value):
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {k: self._sanitize_for_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize_for_json(v) for v in value]
        return value

    def _run(self):
        while not self.stop_event.is_set():
            batch = []
            with self.lock:
                while self.queue and len(batch) < self.batch_size:
                    batch.append(self.queue.popleft())
            if not batch:
                time.sleep(0.2)
                continue
            for event, payload in batch:
                if not self._send_with_retry(event, payload):
                    self.metrics.db_send_errors += 1
                else:
                    self.metrics.db_sent_events += 1

    def _send_with_retry(self, event, safe_payload, retries=3):
        for attempt in range(retries + 1):
            try:
                requests.post(
                    f"{self.supabase_url}/rest/v1/logs",
                    headers={
                        "apikey": self.supabase_key,
                        "Authorization": f"Bearer {self.supabase_key}",
                        "Content-Type": "application/json",
                        "Prefer": "return=minimal",
                    },
                    json={
                        "ts": int(time.time() * 1000),
                        "event": event,
                        "symbol": safe_payload.get("symbol"),
                        "data": safe_payload,
                    },
                    timeout=5,
                )
                return True
            except Exception as e:
                if attempt >= retries:
                    self.logger.error(
                        "SUPABASE WRITE FAILED | event=%s | payload=%s | error=%s",
                        event,
                        safe_payload,
                        e,
                    )
                    return False
                base = 0.3 * (2 ** attempt)
                time.sleep(base + random.uniform(0, 0.25))
        return False


def send_telegram_alert(bot_token, chat_id, logger, metrics, text):
    if not bot_token or not chat_id:
        logger.debug("telegram alert skipped: token/chat_id not configured")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=5,
        )
        metrics.alerts_sent += 1
    except Exception as e:
        logger.error("TELEGRAM ALERT FAILED | text=%s | error=%s", text, e)
