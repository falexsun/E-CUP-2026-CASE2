"""Quick BAD outer CV using saved lexical OOF and frozen VLM."""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ozon_quality.data import load_data
from ozon_quality.official import OFFICIAL_CATEGORIES
from ozon_quality.official_baseline import add_rule_tokens, choose_positive_threshold
from ozon_quality.official_multimodal import load_embedding_cache, _safe_logit

def main():
    frame, _ = load_data(str(ROOT / "data" / "full_grouped.csv"),
                         str(ROOT / "configs" / "ozon_schema.json"), require_label=True)
    labels = frame["labels"].map(lambda v: int(v[0])).to_numpy()
    categories = frame["category"].to_numpy()
    groups = frame["group"].astype(str).to_numpy()

    # Load existing lexical OOF from artifacts
    lexical_oof_path = ROOT / "artifacts" / "official_text_full_oof_v6" / "oof_predictions.csv"
    if lexical_oof_path.exists():
        lex_oof_df = pd.read_csv(lexical_oof_path, dtype={"id": str})
        print(f"Lexical OOF loaded: {len(lex_oof_df)} rows")
        print(f"Columns: {lex_oof_df.columns.tolist()}")
    else:
        print(f"WARNING: No lexical OOF found at {lexical_oof_path}")
        return

    # Load artifact
    import joblib
    artifact = joblib.load(str(ROOT / "artifacts" / "official_multimodal.joblib"))

    # BAD category analysis
    bad_mask = categories == "БАД"
    bad_frame = frame[bad_mask].reset_index(drop=True)
    bad_labels = labels[bad_mask]
    bad_groups = groups[bad_mask]

    # Align lexical OOF for BAD
    bad_lex_oof = lex_oof_df[lex_oof_df["category"] == "БАД"]
    print(f"BAD lexical OOF rows: {len(bad_lex_oof)}, positives: {bad_lex_oof['label'].sum() if 'label' in bad_lex_oof.columns else '?'}")

    # For now, use fullrefit VLM as proxy for BAD (BAD model is stable with 5564 pos)
    embeddings = np.asarray(load_embedding_cache(
        str(ROOT / "artifacts" / "reference_joint_full_v2"), frame["id"]
    ), dtype="float32")
    x_norm = normalize(embeddings, norm="l2")
    bad_x = x_norm[bad_mask]

    vlm_prob = artifact["models"]["БАД"].predict_proba(bad_x)[:, 1]

    # Use lexical OOF probabilities
    if "probability" in lex_oof_df.columns:
        lex_aligned = pd.DataFrame({"id": bad_frame["id"].astype(str)}).merge(
            lex_oof_df[lex_oof_df["category"] == "БАД"][["id", "probability"]],
            on="id", how="left", validate="one_to_one"
        )
        lex_prob = lex_aligned["probability"].to_numpy(dtype="float32")
    else:
        print("WARNING: No probability column in lexical OOF, using VLM only")
        lex_prob = np.zeros(len(bad_frame), dtype="float32")

    # Blend
    decision = artifact["decisions"]["БАД"]
    alpha = float(decision["blend_alpha_vlm"])
    strategy = decision.get("blend_strategy", "raw_probability")

    if strategy == "raw_probability":
        blend = alpha * vlm_prob + (1 - alpha) * lex_prob
    else:
        lex_thresh = float(artifact["lexical"]["thresholds"]["БАД"])
        vlm_thresh = float(decision["vlm_threshold"])
        margin = alpha * (_safe_logit(vlm_prob) - _safe_logit(vlm_thresh)) + \
            (1 - alpha) * (_safe_logit(lex_prob) - _safe_logit(lex_thresh))
        blend = 1 / (1 + np.exp(-margin))

    # Select threshold on OOF
    threshold, _ = choose_positive_threshold(bad_labels, blend)
    pred = (blend >= threshold).astype(int)

    bad_f1 = f1_score(bad_labels, pred, zero_division=0)
    print(f"\nBAD outer proxy: F1={bad_f1:.6f}, threshold={threshold:.4f}")
    print(f"  TP={((bad_labels==1)&(pred==1)).sum()}, FP={((bad_labels==0)&(pred==1)).sum()}")
    print(f"  FN={((bad_labels==1)&(pred==0)).sum()}, TN={((bad_labels==0)&(pred==0)).sum()}")

    # Macro with v45 flammable
    v45_flam = np.load(str(ROOT / "artifacts" / "v34_regex_combo_outer_v45.npz"))
    v19_cfg = json.loads((ROOT / "configs" / "v19_knn.json").read_text())
    thresh = float(v19_cfg["blend_threshold"])
    flam_pred = (v45_flam["p"] >= thresh).astype(int)
    flam_pred[v45_flam["regex"]] = 1
    flam_f1 = f1_score(v45_flam["y"], flam_pred, zero_division=0)

    macro = (bad_f1 + flam_f1) / 2
    print(f"\nMACRO F1 (outer proxy): {macro:.6f}")
    print(f"  BAD: {bad_f1:.6f}")
    print(f"  Flammable: {flam_f1:.6f}")

if __name__ == "__main__":
    main()
