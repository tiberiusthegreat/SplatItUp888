#!/usr/bin/env python3
"""Serve a local SuperSplat build and one PLY from the same origin."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import sys


dist = Path(sys.argv[1]).resolve()
splat = Path(sys.argv[2]).resolve()
port = int(sys.argv[3]) if len(sys.argv) > 3 else 3010
host = sys.argv[4] if len(sys.argv) > 4 else "127.0.0.1"


class SuperSplatHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def translate_path(self, path):
        if urlsplit(path).path == f"/{splat.name}":
            return str(splat)
        return super().translate_path(path)


handler = partial(SuperSplatHandler, directory=str(dist))
server = ThreadingHTTPServer((host, port), handler)
server.serve_forever()
