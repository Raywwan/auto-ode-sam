"""Idempotently fetch the MONAI SwinViT SSL pretrained checkpoint used to
warm-start the V9 Stage-1 proposer.

Run on a fresh machine: `python scripts/fetch_pretrained.py`.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

SSL_URL = (
    "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/"
    "download/0.8.1/model_swinvit.pt"
)
DEST = Path(__file__).resolve().parent.parent / "checkpoints" / "pretrained" / "model_swinvit.pt"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size > 1 << 27:   # > 128 MB
        print(f"[fetch_pretrained] already present: {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
        return 0
    print(f"[fetch_pretrained] downloading {SSL_URL} -> {DEST}")

    def _hook(n, bs, total):
        pct = (n * bs) / max(1, total) * 100.0
        sys.stdout.write(f"\r  {pct:5.1f}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(SSL_URL, str(DEST), reporthook=_hook)
    sys.stdout.write("\n")
    print(f"[fetch_pretrained] done: {DEST.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
