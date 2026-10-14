"""Run real titles through plugin.py the way G-Assist does.

Starts one persistent Protocol V2 process, initializes it, then executes
check_compatibility and auto_detect. This calls Steam. It is not part of pytest.
"""

import json
import queue
import struct
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, ".")

from canrun_game_fetcher import steam_title_matches

PLUGIN = Path(__file__).resolve().parent.parent / "plugin.py"

# (command, argument, words the answer title must match)
CASES = [
    ("auto_detect", "can i run dawnwalker", "dawnwalker"),
    ("auto_detect", "can i run Dawn of the Walkers", "Dawn of the Walkers"),
    ("auto_detect", "can i run dawnwlaker", "dawnwlaker"),
    ("auto_detect", "Can I run the blood of dawnwalker?", "the blood of dawnwalker"),
    ("check_compatibility", "Elden Ring", "Elden Ring"),
    ("check_compatibility", "Cyberpunk 2077", "Cyberpunk 2077"),
    ("check_compatibility", "Hades", "Hades"),
    ("check_compatibility", "Hades II", "Hades II"),
    ("check_compatibility", "Baldur's Gate 3", "Baldur's Gate 3"),
    ("check_compatibility", "Counter-Strike 2", "Counter-Strike 2"),
    ("check_compatibility", "Stardew Valley", "Stardew Valley"),
    ("check_compatibility", "Hollow Knight", "Hollow Knight"),
    ("check_compatibility", "Celeste", "Celeste"),
    ("check_compatibility", "Portal 2", "Portal 2"),
    ("check_compatibility", "The Witcher 3", "The Witcher 3"),
    ("check_compatibility", "Red Dead Redemption 2", "Red Dead Redemption 2"),
    ("check_compatibility", "Diablo IV", "Diablo IV"),
    ("check_compatibility", "Doom Eternal", "Doom Eternal"),
    ("check_compatibility", "Sekiro", "Sekiro"),
    ("check_compatibility", "Valheim", "Valheim"),
    ("check_compatibility", "Terraria", "Terraria"),
    ("check_compatibility", "Deep Rock Galactic", "Deep Rock Galactic"),
    ("check_compatibility", "Lethal Company", "Lethal Company"),
    ("check_compatibility", "Balatro", "Balatro"),
    ("check_compatibility", "Black Myth: Wukong", "Black Myth: Wukong"),
    ("check_compatibility", "Dave the Diver", "Dave the Diver"),
    ("check_compatibility", "Vampire Survivors", "Vampire Survivors"),
    ("check_compatibility", "Slay the Spire", "Slay the Spire"),
]


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


def _answer_title(text: str) -> str:
    first = (text or "").split("\n", 1)[0].strip()
    for ending in (
        " can run on this PC. Performance is excellent.",
        " can run on this PC.",
        " does not meet the minimum requirements on this PC.",
    ):
        if first.endswith(ending):
            return first[: -len(ending)].strip()
    return ""


def main() -> None:
    messages = queue.Queue()
    proc = subprocess.Popen(
        [sys.executable, str(PLUGIN)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=str(PLUGIN.parent),
    )
    threading.Thread(target=_reader, args=(proc.stdout, messages), daemon=True).start()

    def next_message(timeout):
        item = messages.get(timeout=timeout)
        if isinstance(item, Exception):
            raise item
        if item is None:
            raise RuntimeError("plugin closed stdout")
        return item

    passed = 0
    failed = []
    try:
        _send(proc.stdin, {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocol_version": "2.0", "engine_version": "1.0.0"},
        })
        initialized = next_message(40)
        names = {command["name"] for command in initialized["result"]["commands"]}
        print(
            f"initialized {initialized['result']['name']} "
            f"protocol {initialized['result']['protocol_version']} commands {sorted(names)}",
            flush=True,
        )
        if initialized.get("id") != 1 or "check_compatibility" not in names or "auto_detect" not in names:
            raise RuntimeError(initialized)

        for index, (command, argument, query) in enumerate(CASES, start=2):
            key = "user_input" if command == "auto_detect" else "game_name"
            _send(proc.stdin, {
                "jsonrpc": "2.0",
                "id": index,
                "method": "execute",
                "params": {
                    "function": command,
                    "arguments": {key: argument},
                    "context": [],
                    "system_info": "",
                },
            })
            streamed = False
            answer = ""
            error = ""
            for _ in range(8):
                message = next_message(90)
                if message.get("method") == "stream" and message.get("params", {}).get("request_id") == index:
                    streamed = True
                    continue
                if message.get("method") == "complete" and message.get("params", {}).get("request_id") == index:
                    answer = str(message["params"].get("data") or "")
                    break
                if message.get("method") == "error":
                    error = str(message)
                    break
            title = _answer_title(answer)
            ok = bool(title) and steam_title_matches(query, title) and streamed and not error
            passed += int(ok)
            status = "ok" if ok else "MISS"
            print(f"{status} {command} {argument!r} -> {title or answer.split(chr(10), 1)[0][:120]}", flush=True)
            if not ok:
                failed.append((argument, title or error or answer[:160]))
    finally:
        proc.kill()
        proc.wait(timeout=5)

    print(f"\nPASSED {passed}/{len(CASES)}", flush=True)
    for argument, detail in failed:
        print(f"  FAIL {argument}: {detail}", flush=True)
    if passed < 20 or failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
