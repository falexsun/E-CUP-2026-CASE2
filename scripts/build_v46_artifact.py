"""Build v46 artifact: v45 + expanded safe regex rules for flammable category.

New rules added (all P=1.0 on full data, 0 BAD cross-matches):
- matches_tourist: спички длительного горения туристические (5 TP)
- smoke_broad: гранаты страйкбольные PYROFX (3 TP)
- sparkler: бенгальская свеча (1 TP)
- moxibustion: моксотерапия (1 TP)

Total new true positives: 10 (from 39 to 49 rule-covered items)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description="Build v46 artifact with expanded rules")
    parser.add_argument("--base-artifact", required=True, help="Path to v45 fullrefit artifact")
    parser.add_argument("--config", required=True, help="Path to v46 config JSON")
    parser.add_argument("--output", required=True, help="Output artifact path")
    args = parser.parse_args()

    artifact = joblib.load(args.base_artifact)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    # Add new regex rules
    existing_rules = artifact.get("regex_title_overrides", {}).get("Легковоспламеняющиеся", [])
    new_rules = config["new_regex_rules"]

    # Combine: existing + new
    all_rules = existing_rules + [
        {"pattern": r["pattern"], "label": r["label"]}
        for r in new_rules
    ]

    artifact["regex_title_overrides"] = {"Легковоспламеняющиеся": all_rules}
    artifact["regex_override_metadata"] = {
        "version": "v46",
        "base": "v45",
        "new_rules_count": len(new_rules),
        "total_rules_count": len(all_rules),
        "selection": "fold_safe_precision_1_0_zero_fp",
    }
    artifact["artifact_version"] = int(artifact.get("artifact_version", 1)) + 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path, compress=3)

    # Verify
    reloaded = joblib.load(output_path)
    final_rules = reloaded.get("regex_title_overrides", {}).get("Легковоспламеняющиеся", [])
    print(f"v46 artifact saved: {output_path}")
    print(f"Total regex rules: {len(final_rules)}")
    for i, r in enumerate(final_rules):
        print(f"  {i+1}. {r['pattern']} -> label={r['label']}")

    # Hash
    sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"SHA-256: {sha}")


if __name__ == "__main__":
    main()
