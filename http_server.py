import os
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.getenv("PORT", "10000"))
METRICS_GETTER = None


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            body = METRICS_GETTER() if METRICS_GETTER else "metrics_unavailable 1\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()

    def log_message(self, format, *args):
        return


def run_http_server(stop_event, port=PORT, metrics_getter=None):
    global METRICS_GETTER
    METRICS_GETTER = metrics_getter
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    server.timeout = 1
    while not stop_event.is_set():
        server.handle_request()
