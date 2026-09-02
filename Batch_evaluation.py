import csv
import json
import re
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ==================================================
# 路径设置
# ==================================================

OUTPUT_DIR = Path("./output")
OUTPUT_CSV = Path("./evaluation_summary.csv")


# AVeriTeC 标准标签，顺序同时决定 confusion matrix 的行列顺序
LABELS = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]


def safe_name(text):
    """
    将标签转换成适合用作 CSV 列名的形式。

    例如：
    Not Enough Evidence -> not_enough_evidence
    """

    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)

    return text.strip("_")


def parse_directory_metadata(file_path):
    """
    从文件所在目录读取：
    1. reranker_method: Cross Encoder / No reranker
    2. retriever: BM25 / Hybrid / Dense

    预期目录示例：
    output/Cross Encoder/BM25/xxx_result.json
    """

    relative_path = file_path.relative_to(OUTPUT_DIR)
    directory_parts = relative_path.parts[:-1]

    reranker_method = ""
    retriever = ""

    for part in directory_parts:
        normalized_part = part.strip().lower()

        if normalized_part == "cross encoder":
            reranker_method = "Cross Encoder"

        elif normalized_part in {
            "no reranker",
            "no_reranker",
            "no-reranker",
        }:
            reranker_method = "No reranker"

        elif normalized_part == "bm25":
            retriever = "BM25"

        elif normalized_part == "hybrid":
            retriever = "Hybrid"

        elif normalized_part == "dense":
            retriever = "Dense"

    return {
        "reranker_method": reranker_method,
        "retriever": retriever,
        "relative_path": str(relative_path),
    }


def parse_filename_metadata(file_path):
    """
    解析文件名中的实验设置。

    支持的 chunk method：
    - sentence
    - word_100_25
    - word_200_50

    示例文件名：
    sentence_retrieval_top10_rerank_True_Dedup_False_qwen2_5_7b_result.json

    解析结果：
    chunk_method = sentence
    top_k        = 10
    rerank       = True
    dedup        = False
    llm          = qwen2_5_7b
    """

    filename = file_path.name

    pattern = re.compile(
        r"^(?P<chunk_method>sentence|word_\d+_\d+)"
        r"(?:_retrieval)?"
        r"_top(?P<top_k>\d+)"
        r"_rerank_(?P<rerank>True|False)"
        r"_Dedup_(?P<dedup>True|False)"
        r"_(?P<llm>.+?)"
        r"_result\.json$",
        re.IGNORECASE,
    )

    match = pattern.match(filename)

    if match is None:
        raise ValueError(
            f"Cannot parse filename metadata: {filename}"
        )

    metadata = match.groupdict()

    metadata["top_k"] = int(metadata["top_k"])

    # 先存为真正的布尔值，写入 CSV 时显示为 True/False
    metadata["rerank"] = (
        metadata["rerank"].lower() == "true"
    )

    metadata["dedup"] = (
        metadata["dedup"].lower() == "true"
    )

    return metadata


def normalize_label(label):
    """
    清除标签首尾空格。

    如果不同文件存在一些常见的标签写法差异，
    可以继续在这里增加映射。
    """

    label = str(label).strip()

    label_mapping = {
        "Support": "Supported",
        "SUPPORTED": "Supported",
        "Refute": "Refuted",
        "REFUTED": "Refuted",
        "NEE": "Not Enough Evidence",
        "Not enough evidence": "Not Enough Evidence",
        "Conflicting Evidence": (
            "Conflicting Evidence/Cherrypicking"
        ),
        "Conflicting Evidence / Cherrypicking": (
            "Conflicting Evidence/Cherrypicking"
        ),
    }

    return label_mapping.get(label, label)


def load_predictions(file_path):
    """
    只从结果文件中读取：
    - gold_label
    - predicted_label
    """

    with file_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            "The root of the JSON file must be a list."
        )

    gold_labels = []
    predicted_labels = []
    skipped_records = 0

    for record_index, record in enumerate(records):
        gold_label = record.get("gold_label")
        predicted_label = record.get("predicted_label")

        if gold_label is None or predicted_label is None:
            skipped_records += 1

            print(
                f"Warning: skipping record {record_index} "
                f"in {file_path.name}: missing label"
            )

            continue

        gold_label = normalize_label(gold_label)
        predicted_label = normalize_label(predicted_label)

        if gold_label not in LABELS:
            skipped_records += 1

            print(
                f"Warning: skipping record {record_index} "
                f"in {file_path.name}: "
                f"unknown gold label '{gold_label}'"
            )

            continue

        if predicted_label not in LABELS:
            skipped_records += 1

            print(
                f"Warning: skipping record {record_index} "
                f"in {file_path.name}: "
                f"unknown predicted label "
                f"'{predicted_label}'"
            )

            continue

        gold_labels.append(gold_label)
        predicted_labels.append(predicted_label)

    return gold_labels, predicted_labels, skipped_records


def calculate_metrics(gold_labels, predicted_labels):
    """
    计算：
    - Accuracy
    - Macro/Micro/Weighted Precision、Recall、F1
    - 每个类别的 Precision、Recall、F1、Support
    - 4 × 4 confusion matrix
    """

    metrics = {
        "num_evaluated": len(gold_labels),
        "accuracy": accuracy_score(
            gold_labels,
            predicted_labels,
        ),
    }

    # ==================================================
    # 整体平均指标
    # ==================================================

    for average_method in ["macro", "micro", "weighted"]:
        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                gold_labels,
                predicted_labels,
                labels=LABELS,
                average=average_method,
                zero_division=0,
            )
        )

        metrics[f"{average_method}_precision"] = precision
        metrics[f"{average_method}_recall"] = recall
        metrics[f"{average_method}_f1"] = f1

    # ==================================================
    # 每个类别的指标
    # ==================================================

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            gold_labels,
            predicted_labels,
            labels=LABELS,
            average=None,
            zero_division=0,
        )
    )

    for label, p, r, f, s in zip(
        LABELS,
        precision,
        recall,
        f1,
        support,
    ):
        label_key = safe_name(label)

        metrics[f"{label_key}_precision"] = p
        metrics[f"{label_key}_recall"] = r
        metrics[f"{label_key}_f1"] = f
        metrics[f"{label_key}_support"] = int(s)

    # ==================================================
    # Confusion matrix
    #
    # 行：gold label
    # 列：predicted label
    # ==================================================

    matrix = confusion_matrix(
        gold_labels,
        predicted_labels,
        labels=LABELS,
    )

    for gold_index, gold_label in enumerate(LABELS):
        for pred_index, predicted_label in enumerate(LABELS):
            gold_key = safe_name(gold_label)
            predicted_key = safe_name(predicted_label)

            column_name = (
                f"cm_gold_{gold_key}"
                f"_pred_{predicted_key}"
            )

            metrics[column_name] = int(
                matrix[gold_index][pred_index]
            )

    return metrics


def main():
    # 递归搜索 output 目录下所有结果文件
    result_files = sorted(
        OUTPUT_DIR.rglob("*_result.json")
    )

    if not result_files:
        print(
            "No result files found in: "
            f"{OUTPUT_DIR.resolve()}"
        )
        return

    summary_rows = []

    for file_path in result_files:
        try:
            # 目录中的元数据
            directory_metadata = parse_directory_metadata(
                file_path
            )

            # 文件名中的元数据
            filename_metadata = parse_filename_metadata(
                file_path
            )

            # 读取 gold/predicted labels
            (
                gold_labels,
                predicted_labels,
                skipped_records,
            ) = load_predictions(file_path)

            if not gold_labels:
                print(
                    f"Skipped empty result: {file_path.name}"
                )
                continue

            # 计算评估指标
            metrics = calculate_metrics(
                gold_labels,
                predicted_labels,
            )

            row = {
                "filename": file_path.name,
                **directory_metadata,
                **filename_metadata,
                "num_skipped": skipped_records,
                **metrics,
            }

            summary_rows.append(row)

            print(
                f"Processed: "
                f"{directory_metadata['reranker_method']} | "
                f"{directory_metadata['retriever']} | "
                f"{filename_metadata['chunk_method']} | "
                f"{filename_metadata['llm']} | "
                f"accuracy={metrics['accuracy']:.4f} | "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )

        except Exception as error:
            print(
                f"Failed: {file_path} | {error}"
            )

    if not summary_rows:
        print("No valid result files were processed.")
        return

    # 列顺序：先实验元数据，再整体指标，再类别指标和矩阵
    preferred_columns = [
        "filename",
        "relative_path",
        "reranker_method",
        "retriever",
        "chunk_method",
        "top_k",
        "rerank",
        "dedup",
        "llm",
        "num_evaluated",
        "num_skipped",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ]

    # 将剩余的类别指标及 confusion matrix 列添加到后面
    all_columns = set()

    for row in summary_rows:
        all_columns.update(row.keys())

    remaining_columns = sorted(
        all_columns - set(preferred_columns)
    )

    fieldnames = preferred_columns + remaining_columns

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("Evaluation completed.")
    print(f"Files processed: {len(summary_rows)}")
    print(f"CSV saved to: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()