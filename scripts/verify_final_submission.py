"""Verify the immutable snapshot extracted from the submitted v45 archive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "final_submission"
EXPECTED = {
    "artifacts/official_multimodal.joblib": (
        "e7604e6868e46f673428aebf8439eb87e3875fdfbd8c3c2e4cfd8c7ac7d9d53a"
    ),
    "run.py": "fccacca5a818dfa4c8b3a7c3bf7241c681f62289d93b86958716bd3f63d2152f",
    "src/ozon_quality/official_explanations.py": (
        "aeae933175685b873c2302ee48cb5b0637b10b281b517462e1de71ee65057c35"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for relative, expected in EXPECTED.items():
        path = SNAPSHOT / relative
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"hash mismatch: {relative}: {actual} != {expected}")

    metadata = json.loads((SNAPSHOT / "metadata.json").read_text(encoding="utf-8"))
    if metadata != {
        "image": "odsai/ecup26-quality-baseline:1.0",
        "entry_point": "python -u run.py",
    }:
        raise SystemExit(f"unexpected metadata: {metadata!r}")
    print("final v45 snapshot: OK")


if __name__ == "__main__":
    main()

