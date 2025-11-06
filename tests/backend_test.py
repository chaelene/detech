"""Manual curl smoke tests for the DETECH backend service.

Run this module directly to issue representative curl requests against a
locally running backend instance. The commands mirror the automated tests and
are useful when validating deployments from the shell.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class CurlCommand:
    name: str
    command: List[str]


SESSION_EXAMPLE = {
    "session_id": "demo-session",
    "wallet_pubkey": "ExampleWalletPubKey",
    "sdp": "ZGVtbw==",
}


CURL_COMMANDS: List[CurlCommand] = [
    CurlCommand(
        name="health",
        command=["curl", "-s", "http://localhost:8000/health"],
    ),
    CurlCommand(
        name="stream",
        command=[
            "curl",
            "-s",
            "-X",
            "POST",
            "http://localhost:8000/stream",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(SESSION_EXAMPLE),
        ],
    ),
]


def run_all() -> None:
    for cmd in CURL_COMMANDS:
        print(f"\n$ {' '.join(cmd.command)}")
        subprocess.run(cmd.command, check=False)


if __name__ == "__main__":
    run_all()

