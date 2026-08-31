import pandas as pd

from ozon_quality.official import RESULT_PATTERN, format_result
from ozon_quality.official_explanations import _contradicts_decision, deterministic_comment


def test_bad_comment_uses_direct_marking() -> None:
    row = pd.Series(
        {
            "category": "БАД",
            "title": "Добавка",
            "description": "Биологически активная добавка к пище",
        }
    )
    comment = deterministic_comment(row, 1)
    assert "маркировка" in comment
    assert RESULT_PATTERN.fullmatch(format_result(comment, 1))


def test_flammable_negative_comment_is_contract_safe() -> None:
    row = pd.Series(
        {
            "category": "Легковоспламеняющиеся",
            "title": "Мангал",
            "description": "Поставляется без угля",
        }
    )
    assert RESULT_PATTERN.fullmatch(format_result(deterministic_comment(row, 0), 0))


def test_semantic_contradiction_is_detected() -> None:
    assert _contradicts_decision("Товар соответствует положительному классу.", 0)
    assert _contradicts_decision("Товар не соответствует категории БАД.", 1)
    assert not _contradicts_decision("Товар не соответствует категории БАД.", 0)
