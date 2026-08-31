import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ozon_quality.official_baseline import add_rule_tokens, choose_positive_threshold


def test_rule_tokens_are_category_specific() -> None:
    bad = add_rule_tokens(pd.Series(["БАД. Не является лекарством"]), "БАД").iloc[0]
    fire = add_rule_tokens(pd.Series(["Зажигалка без газа"]), "Легковоспламеняющиеся").iloc[0]
    assert "__RULE_BAD_DIRECT__" in bad
    assert "__RULE_IGNITION_SOURCE__" in fire
    assert "__RULE_WITHOUT_CONTENT__" in fire


def test_positive_threshold_optimizes_positive_f1() -> None:
    threshold, score = choose_positive_threshold(
        np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.4, 0.3, 0.9])
    )
    assert 0.1 < threshold <= 0.3
    assert score == 0.8


def test_fast_threshold_matches_brute_force() -> None:
    rng = np.random.default_rng(17)
    labels = rng.integers(0, 2, size=137)
    probability = rng.random(137)
    threshold, score = choose_positive_threshold(labels, probability)
    candidates = np.unique(np.r_[np.linspace(0.01, 0.99, 197), probability])
    brute = np.asarray(
        [f1_score(labels, probability >= value, zero_division=0) for value in candidates]
    )
    best = np.flatnonzero(brute == brute.max())
    expected_index = best[np.argmin(np.abs(candidates[best] - 0.5))]
    assert threshold == candidates[expected_index]
    assert score == brute[expected_index]
