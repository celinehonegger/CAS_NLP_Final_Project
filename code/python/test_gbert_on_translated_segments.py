from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "processed" / "translated_german_segments.csv"
DEFAULT_MODEL_DIR = ROOT / "output" / "models" / "gbert_merged_label_classifier"
DEFAULT_OUTPUT_DIR = DEFAULT_MODEL_DIR / "translated_german_evaluation"


def require_packages():
    try:
        import torch
        from transformers import BertForSequenceClassification, BertTokenizer
    except ImportError as error:
        raise SystemExit(
            "Missing dependency. Install the evaluation stack first:\n"
            "python -m pip install torch transformers\n"
        ) from error

    return {
        "torch": torch,
        "BertForSequenceClassification": BertForSequenceClassification,
        "BertTokenizer": BertTokenizer,
    }


def get_device(torch, requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def model_labels(model):
    labels = {}
    for key, value in model.config.id2label.items():
        labels[int(key)] = value
    return labels


def macro_f1(y_true, y_pred, label_count):
    scores = []
    for label_id in range(label_count):
        true_positive = int(((y_true == label_id) & (y_pred == label_id)).sum())
        false_positive = int(((y_true != label_id) & (y_pred == label_id)).sum())
        false_negative = int(((y_true == label_id) & (y_pred != label_id)).sum())

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)

    return float(np.mean(scores))


def confusion_matrix(y_true, y_pred, label_count):
    matrix = np.zeros((label_count, label_count), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def evaluate(args):
    deps = require_packages()
    torch = deps["torch"]
    BertForSequenceClassification = deps["BertForSequenceClassification"]
    BertTokenizer = deps["BertTokenizer"]

    data = pd.read_csv(args.data, keep_default_na=False)
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    labels_by_id = model_labels(model)
    label_to_id = {label: index for index, label in labels_by_id.items()}

    data = data[data["label"].isin(label_to_id)].copy()
    data["label_id"] = data["label"].map(label_to_id).astype(int)

    device = get_device(torch, args.device)
    model.to(device)
    model.eval()

    predictions = []
    losses = []

    with torch.no_grad():
        for batch in batched(list(data.index), args.batch_size):
            encoded = tokenizer(
                data.loc[batch, "text"].tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            labels = torch.tensor(data.loc[batch, "label_id"].to_numpy(), dtype=torch.long).to(device)
            output = model(**encoded, labels=labels)
            losses.append(float(output.loss.item()))
            predictions.extend(output.logits.argmax(dim=-1).cpu().numpy())

    y_true = data["label_id"].to_numpy()
    y_pred = np.array(predictions)
    data["predicted_label_id"] = y_pred
    data["predicted_label"] = data["predicted_label_id"].map(labels_by_id)
    data["correct"] = data["label"] == data["predicted_label"]

    metrics = {
        "model": str(args.model_dir),
        "data": str(args.data),
        "rows": int(len(data)),
        "labels": label_to_id,
        "device": str(device),
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_f1": macro_f1(y_true, y_pred, len(label_to_id)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, len(label_to_id)).tolist(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.drop(columns=["label_id"]).to_csv(args.output_dir / "predictions.csv", index=False, encoding="utf-8")
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(
        "Translated German evaluation complete. "
        f"Accuracy: {metrics['accuracy']:.3f}; "
        f"macro F1: {metrics['macro_f1']:.3f}. "
        f"Outputs saved to {args.output_dir}."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the fine-tuned merged-label GBERT model on translated German French segments.",
        allow_abbrev=False,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    evaluate(parse_args())
