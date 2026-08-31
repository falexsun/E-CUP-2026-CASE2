"""Training and inference orchestration for single- and multi-label targets."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler

from ozon_quality.data import load_data
from ozon_quality.features import build_features
from ozon_quality.lexical import build_lexical_classifier
from ozon_quality.metrics import (
    choose_binary_threshold,
    choose_multilabel_thresholds,
    classification_metrics,
    multilabel_metrics,
)


def _base_estimator(seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        ),
    )


def _classifier(seed: int, task_type: str) -> Any:
    base = _base_estimator(seed)
    return OneVsRestClassifier(base) if task_type == "multi_label" else base


def _feature_slices(metadata: dict[str, object]) -> dict[str, slice]:
    blocks = metadata["blocks"]
    assert isinstance(blocks, dict)
    llm_text = blocks["llm_text"]
    vlm_text = blocks["vlm_text"]
    image = blocks["vlm_image"]
    meta = blocks["metadata"]
    total = int(metadata["feature_dimensions"])
    return {
        "text": slice(int(llm_text[0]), int(vlm_text[1])),
        "image": slice(int(image[0]), int(image[1])),
        "metadata": slice(int(meta[0]), int(meta[1])),
        "fusion": slice(0, total),
    }


def _encode_targets(
    train_labels: pd.Series,
    valid_labels: pd.Series,
    configured_type: str,
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    if train_labels.empty or valid_labels.empty:
        raise ValueError("Train and validation must be non-empty")
    lengths = pd.concat([train_labels.map(len), valid_labels.map(len)], ignore_index=True)
    inferred_multi = lengths.max() > 1 or lengths.min() == 0
    task_type = "multi_label" if configured_type == "multi_label" or inferred_multi else "single_label"
    if configured_type == "single_label" and inferred_multi:
        raise ValueError("task_type=single_label conflicts with empty or multiple labels")
    if task_type == "multi_label":
        encoder = MultiLabelBinarizer()
        y_train = encoder.fit_transform(train_labels)
        unknown = sorted(set().union(*valid_labels.tolist()) - set(encoder.classes_))
        if unknown:
            raise ValueError(f"Validation contains labels absent from train: {unknown}")
        y_valid = encoder.transform(valid_labels)
        classes = list(map(str, encoder.classes_))
    else:
        encoder = LabelEncoder()
        train_scalar = train_labels.map(lambda values: values[0])
        valid_scalar = valid_labels.map(lambda values: values[0])
        y_train = encoder.fit_transform(train_scalar)
        unknown = sorted(set(valid_scalar) - set(encoder.classes_))
        if unknown:
            raise ValueError(f"Validation contains labels absent from train: {unknown}")
        y_valid = encoder.transform(valid_scalar)
        classes = list(map(str, encoder.classes_))
    if len(classes) < 2:
        raise ValueError("Training requires at least two target classes")
    return y_train, y_valid, classes, task_type


def _evaluate(
    model: Any,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    task_type: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, float | list[float] | None]:
    probabilities = np.asarray(model.predict_proba(x_valid))
    if task_type == "multi_label":
        thresholds, class_scores = choose_multilabel_thresholds(y_valid, probabilities)
        prediction = (probabilities >= thresholds).astype("int8")
        metrics = multilabel_metrics(y_valid, prediction)
        metrics["per_class_threshold_f1"] = class_scores.tolist()
        return metrics, prediction, probabilities, thresholds.tolist()
    threshold = None
    if probabilities.shape[1] == 2:
        threshold, _ = choose_binary_threshold(y_valid, probabilities[:, 1])
        prediction = (probabilities[:, 1] >= threshold).astype("int64")
    else:
        prediction = probabilities.argmax(axis=1)
    return classification_metrics(y_valid, prediction), prediction, probabilities, threshold


def _evaluate_probabilities(
    probabilities: np.ndarray,
    y_valid: np.ndarray,
    task_type: str,
) -> tuple[dict[str, Any], np.ndarray, float | list[float] | None]:
    if task_type == "multi_label":
        thresholds, class_scores = choose_multilabel_thresholds(y_valid, probabilities)
        prediction = (probabilities >= thresholds).astype("int8")
        metrics = multilabel_metrics(y_valid, prediction)
        metrics["per_class_threshold_f1"] = class_scores.tolist()
        return metrics, prediction, thresholds.tolist()
    threshold = None
    if probabilities.shape[1] == 2:
        threshold, _ = choose_binary_threshold(y_valid, probabilities[:, 1])
        prediction = (probabilities[:, 1] >= threshold).astype("int64")
    else:
        prediction = probabilities.argmax(axis=1)
    return classification_metrics(y_valid, prediction), prediction, threshold


def _search_blend(
    dense_probability: np.ndarray,
    lexical_probability: np.ndarray,
    y_valid: np.ndarray,
    task_type: str,
) -> tuple[float, dict[str, Any], np.ndarray, np.ndarray, float | list[float] | None]:
    best: tuple[float, float, dict[str, Any], np.ndarray, np.ndarray, float | list[float] | None] | None = None
    for dense_weight in np.linspace(0, 1, 41):
        probability = dense_weight * dense_probability + (1 - dense_weight) * lexical_probability
        metrics, prediction, threshold = _evaluate_probabilities(probability, y_valid, task_type)
        score = float(metrics["macro_f1"])
        tie_break = -abs(float(dense_weight) - 0.5)
        candidate = (score, tie_break, metrics, prediction, probability, threshold)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
            best_weight = float(dense_weight)
    assert best is not None
    return best_weight, best[2], best[3], best[4], best[5]


def _apply_fixed_decision(
    probability: np.ndarray,
    y_valid: np.ndarray,
    task_type: str,
    threshold: float | list[float] | None,
) -> tuple[dict[str, float], np.ndarray]:
    if task_type == "multi_label":
        values = np.asarray(threshold, dtype="float32")
        if values.shape != (probability.shape[1],):
            raise ValueError("OOF threshold count does not match target classes")
        prediction = (probability >= values[None, :]).astype("int8")
        return multilabel_metrics(y_valid, prediction), prediction
    if probability.shape[1] == 2 and threshold is not None:
        prediction = (probability[:, 1] >= float(threshold)).astype("int64")
    else:
        prediction = probability.argmax(axis=1)
    return classification_metrics(y_valid, prediction), prediction


def _write_diagnostics(
    output_dir: Path,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probability: np.ndarray,
    classes: list[str],
    task_type: str,
) -> None:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=None if task_type == "multi_label" else np.arange(len(classes)),
        average=None,
        zero_division=0,
    )
    pd.DataFrame(
        {
            "class": classes,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    ).to_csv(output_dir / "per_class_metrics.csv", index=False)
    errors: list[dict[str, Any]] = []
    if task_type == "single_label":
        matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
        pd.DataFrame(matrix, index=classes, columns=classes).to_csv(output_dir / "confusion_matrix.csv")
        for row_index, (truth, pred) in enumerate(zip(y_true, y_pred, strict=True)):
            if truth == pred:
                continue
            common = {
                "id": frame.iloc[row_index]["id"],
                "true": classes[int(truth)],
                "predicted": classes[int(pred)],
                "text": frame.iloc[row_index]["text"],
                "images": json.dumps(frame.iloc[row_index]["images"], ensure_ascii=False),
            }
            errors.extend(
                [
                    common
                    | {
                        "error": "FP",
                        "class": classes[int(pred)],
                        "confidence": float(probability[row_index, pred]),
                    },
                    common
                    | {
                        "error": "FN",
                        "class": classes[int(truth)],
                        "confidence": float(1 - probability[row_index, truth]),
                    },
                ]
            )
    else:
        for row_index in range(len(frame)):
            for class_index, label in enumerate(classes):
                if y_true[row_index, class_index] == y_pred[row_index, class_index]:
                    continue
                errors.append(
                    {
                        "id": frame.iloc[row_index]["id"],
                        "error": "FN" if y_true[row_index, class_index] else "FP",
                        "class": label,
                        "true": int(y_true[row_index, class_index]),
                        "predicted": int(y_pred[row_index, class_index]),
                        "confidence": float(
                            probability[row_index, class_index]
                            if not y_true[row_index, class_index]
                            else 1 - probability[row_index, class_index]
                        ),
                        "text": frame.iloc[row_index]["text"],
                        "images": json.dumps(frame.iloc[row_index]["images"], ensure_ascii=False),
                    }
                )
    error_frame = pd.DataFrame(errors)
    if not error_frame.empty:
        error_frame.sort_values("confidence", ascending=False).head(100).to_csv(
            output_dir / "top_100_errors.csv", index=False
        )
        for error_type in ("FP", "FN"):
            error_frame[error_frame["error"].eq(error_type)].sort_values(
                "confidence", ascending=False
            ).head(100).to_csv(output_dir / f"top_100_{error_type.casefold()}.csv", index=False)
    else:
        empty_errors = pd.DataFrame(columns=["id", "error", "class", "confidence"])
        empty_errors.to_csv(output_dir / "top_100_errors.csv", index=False)
        empty_errors.to_csv(output_dir / "top_100_fp.csv", index=False)
        empty_errors.to_csv(output_dir / "top_100_fn.csv", index=False)


def _write_difficulty_slices(
    output_dir: Path,
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    probability: np.ndarray,
    validation_outputs: dict[str, tuple[np.ndarray, np.ndarray]],
    task_type: str,
) -> None:
    if task_type == "single_label":
        sorted_probability = np.sort(probability, axis=1)
        uncertainty = sorted_probability[:, -1] - sorted_probability[:, -2]
        conflict = validation_outputs["text"][0] != validation_outputs["image"][0]
    else:
        uncertainty = np.min(np.abs(probability - 0.5), axis=1)
        conflict = np.any(
            validation_outputs["text"][0] != validation_outputs["image"][0], axis=1
        )
    audit = predictions.copy()
    audit["uncertainty_margin"] = uncertainty
    audit["text"] = frame["text"].to_numpy()
    audit["images"] = frame["images"].map(
        lambda value: json.dumps(value, ensure_ascii=False)
    )
    audit.sort_values("uncertainty_margin", ascending=True).head(100).to_csv(
        output_dir / "uncertain_100.csv", index=False
    )
    audit.loc[conflict].to_csv(output_dir / "multimodal_conflicts.csv", index=False)


def train(
    train_path: str,
    valid_path: str,
    output: str,
    *,
    schema: str | None,
    text_model: str,
    vision_model: str,
    device: str,
    batch_size: int,
    seed: int,
    text_revision: str | None = None,
    vision_revision: str | None = None,
    blend_config: str | None = None,
    refit_all: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame, config = load_data(train_path, schema, require_label=True)
    valid_frame, valid_config = load_data(valid_path, schema, require_label=True)
    if config["columns"] != valid_config["columns"]:
        raise ValueError("Train and validation resolved to different schemas; provide --schema")
    y_train, y_valid, classes, task_type = _encode_targets(
        train_frame["labels"], valid_frame["labels"], str(config.get("task_type", "auto"))
    )
    cache_dir = output_dir / "cache"
    x_train, feature_meta = build_features(
        train_frame,
        text_model=text_model,
        vision_model=vision_model,
        text_revision=text_revision,
        vision_revision=vision_revision,
        device=device,
        batch_size=batch_size,
        cache_dir=cache_dir,
    )
    x_valid, valid_meta = build_features(
        valid_frame,
        text_model=text_model,
        vision_model=vision_model,
        text_revision=text_revision,
        vision_revision=vision_revision,
        device=device,
        batch_size=batch_size,
        cache_dir=cache_dir,
    )
    if x_train.shape[1] != x_valid.shape[1]:
        raise ValueError("Train/validation feature dimensions differ")
    slices = _feature_slices(feature_meta)
    models: dict[str, Any] = {}
    model_metrics: dict[str, Any] = {}
    fusion_prediction = fusion_probability = None
    validation_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, feature_slice in slices.items():
        model = _classifier(seed, task_type)
        model.fit(x_train[:, feature_slice], y_train)
        metrics, prediction, probability, threshold = _evaluate(
            model, x_valid[:, feature_slice], y_valid, task_type
        )
        model_metrics[name] = metrics | {"threshold": threshold}
        models[name] = model
        validation_outputs[name] = (prediction, probability)
        if name == "fusion":
            fusion_prediction, fusion_probability = prediction, probability
    assert fusion_prediction is not None and fusion_probability is not None
    lexical_model = build_lexical_classifier(seed, task_type)
    lexical_model.fit(train_frame["text"], y_train)
    lexical_probability = np.asarray(lexical_model.predict_proba(valid_frame["text"]))
    lexical_metrics, lexical_prediction, lexical_threshold = _evaluate_probabilities(
        lexical_probability, y_valid, task_type
    )
    model_metrics["lexical"] = lexical_metrics | {"threshold": lexical_threshold}
    validation_outputs["lexical"] = (lexical_prediction, lexical_probability)
    if blend_config:
        frozen_blend = json.loads(Path(blend_config).read_text(encoding="utf-8"))
        if frozen_blend.get("task_type") != task_type or frozen_blend.get("classes") != classes:
            raise ValueError("OOF blend config task/classes do not match this training run")
        dense_weight = float(frozen_blend["dense_weight"])
        ensemble_threshold = frozen_blend.get("threshold")
        ensemble_probability = (
            dense_weight * fusion_probability + (1 - dense_weight) * lexical_probability
        )
        ensemble_metrics, ensemble_prediction = _apply_fixed_decision(
            ensemble_probability, y_valid, task_type, ensemble_threshold
        )
        blend_source = str(Path(blend_config).resolve())
    else:
        (
            dense_weight,
            ensemble_metrics,
            ensemble_prediction,
            ensemble_probability,
            ensemble_threshold,
        ) = _search_blend(fusion_probability, lexical_probability, y_valid, task_type)
        blend_source = "validation"
    model_metrics["ensemble"] = ensemble_metrics | {
        "threshold": ensemble_threshold,
        "dense_weight": dense_weight,
        "lexical_weight": 1 - dense_weight,
    }
    validation_outputs["ensemble"] = (ensemble_prediction, ensemble_probability)
    metrics = dict(model_metrics["ensemble"])
    metrics.update(
        {
            "task_type": task_type,
            "classes": classes,
            "train_rows": len(train_frame),
            "valid_rows": len(valid_frame),
            "valid_images_declared": int(valid_frame["images"].map(bool).sum()),
            "runtime_seconds": time.perf_counter() - started,
            "blend_selected_on": blend_source,
            "refit_all": refit_all,
        }
    )
    artifact_dense_model = models["fusion"]
    artifact_lexical_model = lexical_model
    if refit_all:
        all_features = np.concatenate([x_train, x_valid], axis=0)
        all_targets = np.concatenate([y_train, y_valid], axis=0)
        artifact_dense_model = _classifier(seed, task_type)
        artifact_dense_model.fit(all_features, all_targets)
        artifact_lexical_model = build_lexical_classifier(seed, task_type)
        artifact_lexical_model.fit(
            pd.concat([train_frame["text"], valid_frame["text"]], ignore_index=True),
            all_targets,
        )
    artifact = {
        "model": artifact_dense_model,
        "lexical_model": artifact_lexical_model,
        "dense_weight": dense_weight,
        "classes": classes,
        "task_type": task_type,
        "threshold": ensemble_threshold,
        "text_model": text_model,
        "vision_model": vision_model,
        "text_revision": text_revision,
        "vision_revision": vision_revision,
        "feature_metadata": feature_meta,
        "schema": config,
        "seed": seed,
    }
    joblib.dump(artifact, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(model_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "artifact_version": 3,
        "task_type": task_type,
        "text_model": text_model,
        "vision_model": vision_model,
        "text_revision": text_revision,
        "vision_revision": vision_revision,
        "ensemble": {"dense_weight": dense_weight, "lexical_weight": 1 - dense_weight},
        "blend_selected_on": blend_source,
        "refit_all": refit_all,
        "feature_metadata": feature_meta,
        "validation_feature_metadata": valid_meta,
        "threshold_selected_on": str(Path(valid_path).resolve()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "schema.resolved.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    if schema:
        shutil.copy2(schema, output_dir / "schema.input.json")
    predictions = pd.DataFrame({"id": valid_frame["id"]})
    if task_type == "multi_label":
        predictions["labels"] = [
            json.dumps([label for label, active in zip(classes, row, strict=True) if active], ensure_ascii=False)
            for row in ensemble_prediction
        ]
        predictions["true_labels"] = valid_frame["labels"].map(
            lambda value: json.dumps(value, ensure_ascii=False)
        )
    else:
        predictions["prediction"] = np.asarray(classes)[ensemble_prediction]
        predictions["label"] = valid_frame["labels"].map(lambda value: value[0])
    for index, label in enumerate(classes):
        predictions[f"probability_{label}"] = ensemble_probability[:, index]
    for name, (prediction, _) in validation_outputs.items():
        if task_type == "single_label":
            predictions[f"prediction_{name}"] = np.asarray(classes)[prediction]
        else:
            predictions[f"prediction_{name}"] = [
                json.dumps(
                    [label for label, active in zip(classes, row, strict=True) if active],
                    ensure_ascii=False,
                )
                for row in prediction
            ]
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    _write_difficulty_slices(
        output_dir,
        valid_frame,
        predictions,
        ensemble_probability,
        validation_outputs,
        task_type,
    )
    _write_diagnostics(
        output_dir,
        valid_frame,
        y_valid,
        ensemble_prediction,
        ensemble_probability,
        classes,
        task_type,
    )
    return metrics


def predict(
    input_path: str,
    model_path: str,
    output: str,
    *,
    schema: str | None,
    device: str,
    batch_size: int,
    cache_dir: str | None,
    prediction_column: str,
) -> pd.DataFrame:
    artifact = joblib.load(model_path)
    effective_schema = schema
    if effective_schema is None:
        embedded = Path(model_path).parent / "schema.resolved.json"
        if embedded.exists():
            effective_schema = str(embedded)
    frame, _ = load_data(input_path, effective_schema, require_label=False)
    features, _ = build_features(
        frame,
        text_model=artifact["text_model"],
        vision_model=artifact["vision_model"],
        text_revision=artifact.get("text_revision"),
        vision_revision=artifact.get("vision_revision"),
        device=device,
        batch_size=batch_size,
        cache_dir=cache_dir,
    )
    expected = int(artifact["feature_metadata"]["feature_dimensions"])
    if features.shape[1] != expected:
        raise ValueError(f"Feature dimension mismatch: expected {expected}, got {features.shape[1]}")
    dense_probability = np.asarray(artifact["model"].predict_proba(features))
    lexical_probability = np.asarray(
        artifact["lexical_model"].predict_proba(frame["text"])
    )
    probability = (
        float(artifact["dense_weight"]) * dense_probability
        + (1 - float(artifact["dense_weight"])) * lexical_probability
    )
    classes = np.asarray(artifact["classes"])
    if artifact["task_type"] == "multi_label":
        prediction = probability >= np.asarray(artifact["threshold"])[None, :]
        values = [
            json.dumps(classes[row].tolist(), ensure_ascii=False) for row in prediction
        ]
    elif probability.shape[1] == 2 and artifact["threshold"] is not None:
        values = classes[(probability[:, 1] >= artifact["threshold"]).astype("int64")]
    else:
        values = classes[probability.argmax(axis=1)]
    result = pd.DataFrame({"id": frame["id"], prediction_column: values})
    for index, label in enumerate(classes):
        result[f"probability_{label}"] = probability[:, index]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result
