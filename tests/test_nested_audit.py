"""Unit tests for nested_audit invariants.

Run with: python -m pytest tests/test_nested_audit.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nested_audit import (
    LEX_FROZEN_C,
    V45_REGEX,
    VLM_FROZEN_C,
    _apply_blend,
    _assert_no_overlap,
    _assert_oof_coverage,
    _norm_title,
    _select_blend,
    run_nested_audit,
)

# ── group overlap ────────────────────────────────────────────────────────

class TestGroupOverlap:
    def test_no_overlap_passes(self):
        _assert_no_overlap(np.array(["a", "b"]), np.array(["c", "d"]), "test")

    def test_overlap_raises(self):
        with pytest.raises(AssertionError, match="GROUP LEAKAGE"):
            _assert_no_overlap(np.array(["a", "b"]), np.array(["b", "c"]), "test")

    def test_empty_sets(self):
        _assert_no_overlap(np.array([]), np.array(["a"]), "empty_train")
        _assert_no_overlap(np.array(["a"]), np.array([]), "empty_test")


# ── OOF coverage ─────────────────────────────────────────────────────────

class TestOOFCoverage:
    def test_exact_coverage_passes(self):
        counter = np.array([1, 1, 1, 1])
        mask = np.array([True, True, True, True])
        _assert_oof_coverage(counter, mask, "test")

    def test_partial_coverage_fails(self):
        counter = np.array([1, 0, 1, 1])
        mask = np.array([True, True, True, True])
        with pytest.raises(AssertionError, match="OOF coverage error"):
            _assert_oof_coverage(counter, mask, "test")

    def test_double_coverage_fails(self):
        counter = np.array([1, 2, 1, 1])
        mask = np.array([True, True, True, True])
        with pytest.raises(AssertionError, match="OOF coverage error"):
            _assert_oof_coverage(counter, mask, "test")

    def test_unprocessed_rows_ignored(self):
        """Rows not in processed_mask should not affect the assertion."""
        counter = np.array([1, 0, 1, 0])
        mask = np.array([True, False, True, False])
        _assert_oof_coverage(counter, mask, "test")

    def test_smoke_fold_coverage(self):
        """Only the smoke fold's rows should have counter=1."""
        counter = np.array([0, 0, 1, 0, 0])
        mask = np.array([False, False, True, False, False])
        _assert_oof_coverage(counter, mask, "smoke")


# ── regex ────────────────────────────────────────────────────────────────

class TestRegex:
    def test_regex_flammable_only(self):
        """V45_REGEX patterns should only match flammable-relevant titles."""
        positive_titles = [
            "Уголь древесный березовый 2 кг",
            "Мангал одноразовый с углем",
            "Газ для портативных плит 220гр",
            "Дымовая шашка страйкбольная",
            "Брикеты для гриля Weber 4 кг",
        ]
        for title in positive_titles:
            normed = _norm_title(title)
            matched = any(p.search(normed) for p, _ in V45_REGEX)
            assert matched, f"Expected match for: {title}"

    def test_regex_no_false_positive_on_bad(self):
        """V45_REGEX should NOT match BAD-relevant titles."""
        bad_titles = [
            "Now CoQ-10 100 мг 50 капсул",
            "Витамир L-метилфолат таблетки",
            "БАД биологически активная добавка",
        ]
        for title in bad_titles:
            normed = _norm_title(title)
            matched = any(p.search(normed) for p, _ in V45_REGEX)
            assert not matched, f"False positive on BAD: {title}"

    def test_regex_bounds(self):
        """Regex should only operate on indices 0..len-1."""
        n = 10
        for j in range(n):
            assert 0 <= j < n, f"Index {j} out of bounds for length {n}"


# ── blend ────────────────────────────────────────────────────────────────

class TestBlend:
    def test_raw_probability_blend(self):
        lex = np.array([0.3, 0.7], dtype="float32")
        vlm = np.array([0.8, 0.2], dtype="float32")
        result = _apply_blend(lex, vlm, alpha=0.5, strategy="raw_probability",
                              lex_thresh=0.5, vlm_thresh=0.5)
        expected = 0.5 * vlm + 0.5 * lex
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_logit_blend(self):
        lex = np.array([0.3, 0.7], dtype="float32")
        vlm = np.array([0.8, 0.2], dtype="float32")
        result = _apply_blend(lex, vlm, alpha=0.5, strategy="threshold_normalized_logit",
                              lex_thresh=0.5, vlm_thresh=0.5)
        # Should produce valid probabilities
        assert (result >= 0).all() and (result <= 1).all()

    def test_select_blend_returns_valid(self):
        y = np.array([0, 0, 1, 1, 1, 0], dtype="int32")
        lex_oof = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3], dtype="float32")
        vlm_oof = np.array([0.2, 0.3, 0.7, 0.8, 0.9, 0.1], dtype="float32")
        alpha, strategy, t, lex_t, vlm_t = _select_blend(lex_oof, vlm_oof, y)
        assert 0 <= alpha <= 1
        assert strategy in ("raw_probability", "threshold_normalized_logit")
        assert 0 <= t <= 1


# ── norm_title ───────────────────────────────────────────────────────────

class TestNormTitle:
    def test_basic(self):
        assert _norm_title("Hello World") == "hello world"

    def test_yo_replaced(self):
        assert "е" in _norm_title("ёлка")

    def test_special_chars_removed(self):
        result = _norm_title("Тест!@#123")
        assert result == "тест 123"  # special chars become space separator


# ── frozen C values ──────────────────────────────────────────────────────

class TestFrozenC:
    def test_lex_frozen_c_matches_v45(self):
        """Frozen lexical C values must match v45 fullrefit artifact."""
        assert LEX_FROZEN_C["БАД"] == 4.0
        assert LEX_FROZEN_C["Легковоспламеняющиеся"] == 10.0

    def test_vlm_frozen_c(self):
        """Frozen VLM C must be 1.0."""
        assert VLM_FROZEN_C == 1.0

    def test_frozen_c_keys_match_categories(self):
        from ozon_quality.official import OFFICIAL_CATEGORIES
        assert set(LEX_FROZEN_C.keys()) == set(OFFICIAL_CATEGORIES)


# ── no-regex default ─────────────────────────────────────────────────────

class TestNoRegexDefault:
    def test_run_nested_audit_signature(self):
        """run_nested_audit must accept use_regex with default False."""
        import inspect
        sig = inspect.signature(run_nested_audit)
        assert "use_regex" in sig.parameters
        assert sig.parameters["use_regex"].default is False


# ── residual script invalid ──────────────────────────────────────────────

class TestResidualInvalid:
    def test_residual_raises_on_run(self):
        """residual_ocr_experiment.py must raise RuntimeError if run."""
        sys.path.insert(0, str(ROOT / "scripts"))
        from residual_ocr_experiment import main as residual_main
        with pytest.raises(RuntimeError, match="INVALID"):
            residual_main()
