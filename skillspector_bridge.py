import os
import pathlib
import sys
import time
import uuid

MAILBOX = pathlib.Path(os.environ.get("SKILLSPECTOR_MAILBOX", r"C:\temp\skillspector-mailbox"))
TIMEOUT = int(os.environ.get("SKILLSPECTOR_BRIDGE_TIMEOUT", "90"))

MAILBOX.mkdir(parents=True, exist_ok=True)
uid = str(uuid.uuid4())
req_file = MAILBOX / f"{uid}.req"
resp_file = MAILBOX / f"{uid}.resp"

prompt = sys.stdin.read()
req_file.write_text(prompt, encoding="utf-8")

for _ in range(TIMEOUT * 2):  # poll every 0.5 s
    time.sleep(0.5)
    if resp_file.exists():
        try:
            print(resp_file.read_text(encoding="utf-8"))
        finally:
            req_file.unlink(missing_ok=True)
            resp_file.unlink(missing_ok=True)
        sys.exit(0)

req_file.unlink(missing_ok=True)
sys.stderr.write(f"skillspector_bridge: timed out after {TIMEOUT}s\n")
sys.exit(1)
