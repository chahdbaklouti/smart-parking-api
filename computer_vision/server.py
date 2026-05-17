"""
server.py — assembles the server from its parts.

"""

from api import app


def start_server():
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )