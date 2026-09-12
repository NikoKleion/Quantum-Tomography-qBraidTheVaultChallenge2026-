"""Connection check for the qBraid Vault Challenge: prints state() and the top 5
leaderboard rows. Neither call enrolls you or spends budget.

The API key comes from QBRAID_API_KEY or from a file named qbraid_key.txt or
.qbraid_key that holds only the key, in engine/, Downloads, Desktop, Documents,
or the home folder. The key is never printed.

Usage: python engine/connect.py
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_key() -> str:
    env = os.environ.get("QBRAID_API_KEY")
    if env and env.strip():
        return env.strip()
    here = Path(__file__).parent
    home = Path.home()
    for d in (here, home / "Downloads", home / "Desktop",
              home / "Documents", home):
        for name in ("qbraid_key.txt", ".qbraid_key"):
            p = d / name
            if p.exists():
                key = p.read_text(encoding="utf-8-sig").strip()
                if key:
                    print(f"[key found in {p.name}]")   # never prints the value
                    return key
    raise SystemExit(
        "No API key found. Put it in env var QBRAID_API_KEY, or save a file "
        "'qbraid_key.txt' containing only your key in engine/, Downloads, Desktop, "
        "Documents, or your home folder."
    )


def main():
    from vault_client import VaultClient
    try:
        from qbraid import QbraidSessionV1
    except ImportError:
        from qbraid_core import QbraidSessionV1

    key = _find_key()
    client = VaultClient(QbraidSessionV1(api_key=key))

    print("\n=== state() (does not enroll) ===")
    try:
        st = client.state()
        print(st)
    except Exception as e:                               # noqa: BLE001
        print("state() failed:", type(e).__name__, str(e)[:300])
        return

    print("\n=== leaderboard top 5 (does not enroll) ===")
    try:
        for row in client.leaderboard(page=1, limit=5):
            print("  ", row)
    except Exception as e:                               # noqa: BLE001
        print("leaderboard() failed:", type(e).__name__, str(e)[:300])


if __name__ == "__main__":
    main()
