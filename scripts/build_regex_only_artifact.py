#!/usr/bin/env python3
"""Build v45 with only high-precision, rule-derived title regexes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from build_title_override_artifact import FLAMMABLE_TITLE_RULES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()

    artifact = joblib.load(args.base_artifact)
    # The sixth exploratory rule had one false positive in released labels. Keep only
    # rule families that are both entailed by the task definition and perfectly precise.
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        accepted_names = set(config["regex_rules"])
    else:
        accepted_names = {
            rule["name"]
            for rule in FLAMMABLE_TITLE_RULES
            if rule["name"] != "dry_fuel_with_ignition_source"
        }
    accepted = [rule for rule in FLAMMABLE_TITLE_RULES if rule["name"] in accepted_names]
    if {rule["name"] for rule in accepted} != accepted_names:
        raise ValueError("Configured regex rule is not defined")
    artifact.pop("exact_title_overrides", None)
    artifact.pop("regex_title_overrides", None)
    artifact.pop("title_override_metadata", None)
    flammable_knn = artifact["knn_overrides"]["Легковоспламеняющиеся"]
    flammable_knn.pop("nearest_override_threshold", None)
    artifact["regex_title_overrides"] = {"Легковоспламеняющиеся": accepted}
    artifact["regex_override_metadata"] = {
        "selection": "task_definition_and_zero_false_positives_in_released_data",
        "rule_names": [rule["name"] for rule in accepted],
        "excluded": ["exact_title_lookup", "nearest_hard_override"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=3)
    print(artifact["regex_override_metadata"])


if __name__ == "__main__":
    main()
