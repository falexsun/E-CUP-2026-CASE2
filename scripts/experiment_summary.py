"""Comprehensive experiment summary for Ozon Quality Control research."""

# Summary of all experiments conducted in this session

EXPERIMENTS = """
## Experiment Summary — Autonomous Research Session

### Honest Baseline (outer CV, leakage-safe)
- v19 flammable F1: 0.6742
- v45+regex flammable F1: 0.7757 (P=0.8122, R=0.7424, AP=0.6762)
- BAD outer F1: 0.9359
- **Macro F1 proxy: 0.8558**
- **Target: 0.90 (gap: 0.0442)**

### Per-fold Flammable F1 (v45+regex)
| Fold | Pos | v19 F1 | v45 F1 | 
|------|-----|--------|--------|
| 0    | 40  | 0.8000 | 0.8861 |
| 1    | 39  | 0.8861 | 0.8537 |
| 2    | 40  | 0.7595 | 0.8675 |
| 3    | 40  | 0.3607 | 0.3509 |
| 4    | 39  | 0.4688 | 0.7949 |

### Error Analysis (51 FN)
| Family | Count | % | 
|--------|-------|---|
| party_cracker (entity-426) | 27 | 52.9% |
| burner_gas | 7 | 13.7% |
| gift_set | 6 | 11.8% |
| candle_sparkler | 4 | 7.8% |
| other | 4 | 7.8% |
| ignition_fuel | 3 | 5.9% |

**Key finding**: 27/51 FNs are from entity-426 (single seller of pneumatic crackers labeled as flammable). This is label noise, not a model error.

### Experiments Conducted

| Experiment | Result | Delta vs v45 | Status |
|------------|--------|---------------|--------|
| Multi-image embeddings (mean/max) | F1 drops in all configs | -0.09 to -0.12 | REJECTED |
| VLM binary classifier (Qwen3-VL-2B-Instruct) | Flammable F1=0.63, AP=0.49 | -0.054 stacked | REJECTED |
| OCR features (498 sample, optimistic) | +0.046 F1 on subset | +0.046 | OPTIMISTIC |
| OCR features (12971 full, leakage-safe) | Strategy 1: -0.009, Strategy 2: -0.210 | negative | REJECTED |
| OCR targeted boost (predeclared thresholds) | 0 TP, 12-16 FP in all configs | negative | REJECTED |
| v47 party cracker rules | Post-hoc fold overfit | N/A | INVALID |
| v45_v46_fast_audit | Fullrefit data leakage | N/A | INVALID |
| InternVL3.5-2B | Incompatible with transformers 5.15.0 | N/A | BLOCKED |

### Why 0.90 is not achievable with current approach

1. **No new independent signal found**: OCR, multi-image, VLM all failed to add value
2. **Label noise dominates errors**: 53% of FNs are from one mislabeled seller
3. **Remaining 24 FNs require fundamentally new features**: burners, gift sets, candles need visual understanding beyond current VLM capability
4. **v45 is near-optimal for available signal**: All augmentation attempts hurt performance

### What would be needed for 0.90

1. **Label correction**: Remove or relabel entity-426 party crackers (would save ~0.02 macro F1)
2. **Stronger VLM**: InternVL3.5/MiniCPM-V with proper integration (needs transformers upgrade)
3. **Contrastive learning**: Supervised contrastive loss with hard negatives (needs careful implementation)
4. **OCR on all images**: Full OCR with batch processing (already extracted, but doesn't help)

### Recommendation

v45 (public 0.8142) is the best achievable with current signal and models. The gap to 0.90 requires either:
- Label noise correction (entity-426)
- A fundamentally stronger vision model
- New feature engineering beyond text+image embeddings

No stable candidate improving v45 was found in this session.
"""

print(EXPERIMENTS)
