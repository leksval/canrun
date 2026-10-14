#!/usr/bin/env python3
"""
Protocol V2 subprocess test.

Speaks length-prefixed JSON-RPC to plugin.py: initialize, ping, then execute.
The compatibility command uses the protocol-probe shortcut so the test does not call Steam.
A second ping is sent while that command sleeps, and must be answered before completion.
"""

import json
import os
import queue
import struct
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

PLUGIN = Path(__file__).resolve().parent.parent / "plugin.py"


def _read_exact(stream, count):
    data = b""
    while len(data) < count:
        chunk = stream.read(count - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def _reader(stdout, messages):
    try:
        while True:
            header = _read_exact(stdout, 4)
            if not header:
                break
            length = struct.unpack(">I", header)[0]
            payload = _read_exact(stdout, length)
            if payload is None:
                messages.put(RuntimeError("truncated Protocol V2 payload"))
                break
            messages.put(json.loads(payload.decode("utf-8")))
    except Exception as exc:
        messages.put(exc)
    finally:
        messages.put(None)


def _send(stdin, message):
    payload = json.dumps(message).encode("utf-8")
    stdin.write(struct.pack(">I", len(payload)) + payload)
    stdin.flush()


def test_protocol_v2_initialize_ping_and_execute():
    """initialize, ping, and check_compatibility use JSON-RPC 2.0 framing."""
    env = os.environ.copy()
    env["CANRUN_PROTOCOL_PROBE"] = "1"
    messages = queue.Queue()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        proc = subprocess.Popen(
            [sys.executable, str(PLUGIN)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=tmp,
            env=env,
        )
        threading.Thread(target=_reader, args=(proc.stdout, messages), daemon=True).start()
        stderr = []

        def _drain():
            stderr.append(proc.stderr.read())

        threading.Thread(target=_drain, daemon=True).start()

        def next_message(timeout):
            item = messages.get(timeout=timeout)
            if isinstance(item, Exception):
                raise item
            if item is None:
                raise AssertionError(f"plugin closed stdout; stderr={b''.join(stderr)!r}")
            assert item.get("jsonrpc") == "2.0", item
            return item

        try:
            _send(proc.stdin, {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocol_version": "2.0", "engine_version": "1.0.0"},
            })
            initialized = next_message(40)
            assert initialized["id"] == 1
            assert initialized["result"]["protocol_version"] == "2.0"
            assert initialized["result"]["name"] == "canrun"
            command_names = {command["name"] for command in initialized["result"]["commands"]}
            assert {"check_compatibility", "detect_hardware", "auto_detect"} <= command_names

            _send(proc.stdin, {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "ping",
                "params": {"timestamp": 111},
            })
            ping = next_message(5)
            assert ping["id"] == 2
            assert ping["result"]["timestamp"] == 111

            _send(proc.stdin, {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "execute",
                "params": {
                    "function": "check_compatibility",
                    "arguments": {"game_name": "protocol-probe"},
                    "context": [],
                    "system_info": "",
                },
            })
            _send(proc.stdin, {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "ping",
                "params": {"timestamp": 222},
            })

            saw_pong = False
            saw_complete = False
            for _ in range(6):
                message = next_message(10)
                if message.get("id") == 4:
                    assert message["result"]["timestamp"] == 222
                    assert saw_complete is False
                    saw_pong = True
                elif message.get("method") == "stream":
                    assert message["params"]["request_id"] == 3
                elif message.get("method") == "complete":
                    assert saw_pong, "ping was not answered before the command finished"
                    assert message["params"]["request_id"] == 3
                    assert message["params"]["success"] is True
                    assert message["params"]["data"] == "protocol-probe-ok"
                    saw_complete = True
                    break
                elif message.get("method") == "error":
                    raise AssertionError(message)
            assert saw_pong and saw_complete
        finally:
            proc.kill()
            proc.wait(timeout=5)
