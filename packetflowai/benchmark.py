"""Dataset adapters, deterministic splits, and security evaluation reports."""

import csv
import json
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .taxonomy import ATTACK_TYPES, normalize_label


@dataclass(frozen=True)
class DatasetRecord:
    features: dict[str, Any]
    native_label: str
    normalized_label: str
    dataset_id: str
    group_id: str
    provenance: dict[str, Any]


class DatasetAdapter:
    dataset_id = "base"
    label_fields: tuple[str, ...] = ("label",)
    group_fields: tuple[str, ...] = ("session_id", "capture_id", "source_ip")

    def native_label(self, row: Mapping[str, Any]) -> Any:
        for field in self.label_fields:
            if field in row and row[field] not in {None, ""}:
                return row[field]
        raise ValueError(f"{self.dataset_id} row has no native label in {self.label_fields}")

    def group_id(self, row: Mapping[str, Any], index: int) -> str:
        for field in self.group_fields:
            if row.get(field) not in {None, ""}:
                return f"{field}:{row[field]}"
        return f"row:{index}"

    def features(self, row: Mapping[str, Any]) -> dict[str, Any]:
        excluded = set(self.label_fields)
        return {str(key): value for key, value in row.items() if key not in excluded}

    def adapt(self, rows: Iterable[Mapping[str, Any]]) -> list[DatasetRecord]:
        records = []
        for index, row in enumerate(rows):
            native = self.native_label(row)
            records.append(DatasetRecord(
                features=self.features(row),
                native_label=str(native),
                normalized_label=normalize_label(native),
                dataset_id=self.dataset_id,
                group_id=self.group_id(row, index),
                provenance={
                    "row_index": index,
                    "native_label_field": next(field for field in self.label_fields if field in row),
                },
            ))
        return records

    def from_csv(self, path: Path) -> list[DatasetRecord]:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            return self.adapt(csv.DictReader(source))


class CICIDS2017Adapter(DatasetAdapter):
    dataset_id = "cicids2017"
    label_fields = ("Label", "label")
    group_fields = ("Capture", "Flow ID", "Source IP", "Src IP")


class CSECICIDS2018Adapter(DatasetAdapter):
    dataset_id = "cse-cic-ids2018"
    label_fields = ("Label", "label")
    group_fields = ("Timestamp", "Src IP", "Source IP")


class UNSWNB15Adapter(DatasetAdapter):
    dataset_id = "unsw-nb15"
    label_fields = ("attack_cat", "label")
    group_fields = ("srcip", "id")


class LegacyPacketAdapter(DatasetAdapter):
    dataset_id = "packet-tag-explanation"
    label_fields = ("label", "Label", "attack_type", "Attack Type", "category", "class")
    group_fields = ("capture_id", "session_id")


ADAPTERS = {
    "cicids2017": CICIDS2017Adapter,
    "cse-cic-ids2018": CSECICIDS2018Adapter,
    "unsw-nb15": UNSWNB15Adapter,
    "packet-tag-explanation": LegacyPacketAdapter,
}


def grouped_split(records: Sequence[DatasetRecord], seed: int = 42,
                  train_fraction: float = 0.7, validation_fraction: float = 0.15
                  ) -> tuple[list[DatasetRecord], list[DatasetRecord], list[DatasetRecord]]:
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must leave non-empty test allocation")
    groups = sorted({record.group_id for record in records})
    random.Random(seed).shuffle(groups)
    train_end = max(1, int(len(groups) * train_fraction))
    validation_end = max(train_end + 1, int(len(groups) * (train_fraction + validation_fraction)))
    train_groups = set(groups[:train_end])
    validation_groups = set(groups[train_end:validation_end])
    return (
        [record for record in records if record.group_id in train_groups],
        [record for record in records if record.group_id in validation_groups],
        [record for record in records if record.group_id not in train_groups | validation_groups],
    )


@dataclass(frozen=True)
class BenchmarkReport:
    dataset_id: str
    model_id: str
    sample_count: int
    labels: tuple[str, ...]
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    false_positive_rate: float
    false_negative_rate: float
    confusion_matrix: list[list[int]]
    pr_auc: float | None
    roc_auc: float | None
    unknown_detection_rate: float
    calibration_error: float | None
    native_label_examples: tuple[str, ...]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")


def evaluate_predictions(dataset_id: str, model_id: str, native_labels: Sequence[str],
                         normalized_targets: Sequence[str], predictions: Sequence[str],
                         malicious_scores: Sequence[float] | None = None,
                         calibration_error: float | None = None) -> BenchmarkReport:
    if not (len(native_labels) == len(normalized_targets) == len(predictions)):
        raise ValueError("benchmark arrays have different lengths")
    labels = tuple(label for label in ATTACK_TYPES if label in set(normalized_targets) | set(predictions))
    precision, recall, f1, _ = precision_recall_fscore_support(
        normalized_targets, predictions, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(normalized_targets, predictions, labels=labels)
    benign_index = labels.index("benign") if "benign" in labels else None
    if benign_index is None:
        false_positive_rate = 0.0
        false_negative_rate = 0.0
    else:
        false_positives = int(matrix[benign_index, :].sum() - matrix[benign_index, benign_index])
        true_negatives = int(matrix[benign_index, benign_index])
        false_negatives = int(matrix[:, benign_index].sum() - matrix[benign_index, benign_index])
        true_positives = int(matrix.sum() - matrix[benign_index, :].sum() - false_negatives)
        false_positive_rate = false_positives / max(1, false_positives + true_negatives)
        false_negative_rate = false_negatives / max(1, false_negatives + true_positives)
    binary_targets = np.asarray([target != "benign" for target in normalized_targets], dtype=int)
    pr_auc = roc_auc = None
    if malicious_scores is not None and len(set(binary_targets.tolist())) > 1:
        pr_auc = float(average_precision_score(binary_targets, malicious_scores))
        roc_auc = float(roc_auc_score(binary_targets, malicious_scores))
    unknown_count = sum(prediction.upper() in {"UNKNOWN", "SUSPICIOUS"} for prediction in predictions)
    return BenchmarkReport(
        dataset_id=dataset_id,
        model_id=model_id,
        sample_count=len(predictions),
        labels=labels,
        macro_f1=float(np.mean(f1)) if len(f1) else 0.0,
        per_class={
            label: {"precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index])}
            for index, label in enumerate(labels)
        },
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        confusion_matrix=matrix.tolist(),
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        unknown_detection_rate=unknown_count / max(1, len(predictions)),
        calibration_error=calibration_error,
        native_label_examples=tuple(sorted(set(native_labels))[:20]),
    )


def benchmark_csv(dataset: str, path: Path, model_id: str,
                  predictor: Callable[[DatasetRecord], tuple[str, float]], output: Path) -> BenchmarkReport:
    adapter_type = ADAPTERS.get(dataset)
    if adapter_type is None:
        raise ValueError(f"unknown dataset adapter: {dataset}")
    records = adapter_type().from_csv(path)
    predictions = [predictor(record) for record in records]
    report = evaluate_predictions(
        dataset,
        model_id,
        [record.native_label for record in records],
        [record.normalized_label for record in records],
        [prediction[0] for prediction in predictions],
        [prediction[1] for prediction in predictions],
    )
    report.write(output)
    return report
