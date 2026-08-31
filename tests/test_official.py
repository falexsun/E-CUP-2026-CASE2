import pandas as pd
import pytest

from ozon_quality.official import format_result, official_category_f1, validate_submission


def test_official_metric_averages_positive_f1_by_category() -> None:
    result = official_category_f1(
        [1, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 1, 0],
        ["БАД", "БАД", "БАД", "Легковоспламеняющиеся", "Легковоспламеняющиеся", "Легковоспламеняющиеся"],
    )
    assert result["f1_БАД"] == pytest.approx(2 / 3)
    assert result["f1_Легковоспламеняющиеся"] == pytest.approx(2 / 3)
    assert result["official_score"] == pytest.approx(2 / 3)


def test_official_result_contract() -> None:
    value = format_result("Прямое указание на маркировку товара найдено в описании.", 1)
    frame = pd.DataFrame({"id": [7], "result": [value]})
    validate_submission(frame, pd.Series([7]))
    assert value.endswith("<вердикт>не бан")
