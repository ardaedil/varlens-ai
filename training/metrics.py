from __future__ import annotations

from typing import Sequence


def confusion_matrix(
    references: Sequence[int],
    predictions: Sequence[int],
    *,
    label_ids: Sequence[int],
) -> list[list[int]]:
    index_by_label = {label_id: index for index, label_id in enumerate(label_ids)}
    matrix = [[0 for _ in label_ids] for _ in label_ids]
    for reference, prediction in zip(references, predictions):
        matrix[index_by_label[reference]][index_by_label[prediction]] += 1
    return matrix


def compute_classification_metrics(
    references: Sequence[int],
    predictions: Sequence[int],
    *,
    id2label: dict[int, str] | None = None,
) -> dict[str, object]:
    if len(references) != len(predictions):
        raise ValueError("references and predictions must be the same length.")
    if not references:
        raise ValueError("At least one prediction is required to compute metrics.")

    label_ids = sorted(id2label) if id2label else sorted(set(references) | set(predictions))
    labels = [id2label[label_id] if id2label else str(label_id) for label_id in label_ids]
    matrix = confusion_matrix(references, predictions, label_ids=label_ids)

    per_class: dict[str, dict[str, float | int]] = {}
    accuracy_numerator = 0
    total = len(references)
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for row_index, label in enumerate(labels):
        true_positive = matrix[row_index][row_index]
        support = sum(matrix[row_index])
        predicted = sum(matrix[column_index][row_index] for column_index in range(len(matrix)))
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 0.0
        if precision + recall:
            f1 = 2 * precision * recall / (precision + recall)

        accuracy_numerator += true_positive
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }

    label_count = len(labels)
    return {
        "accuracy": round(accuracy_numerator / total, 6),
        "macro_precision": round(macro_precision / label_count, 6),
        "macro_recall": round(macro_recall / label_count, 6),
        "macro_f1": round(macro_f1 / label_count, 6),
        "balanced_accuracy": round(macro_recall / label_count, 6),
        "support": total,
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }
