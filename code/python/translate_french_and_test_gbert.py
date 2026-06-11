from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "processed" / "french_segments.csv"
DEFAULT_TRANSLATED = ROOT / "data" / "processed" / "french_segments_translated_de.csv"
DEFAULT_MODEL_DIR = ROOT / "output" / "models" / "gbert_merged_label_classifier"
DEFAULT_OUTPUT_DIR = DEFAULT_MODEL_DIR / "french_translated_evaluation"

MERGED_LABEL_MAP = {
    "credit claiming": "credit claiming",
    "explaining/sharing responsibility": "implicit blame avoidance",
    "praising": "implicit blame avoidance",
    "shifting responsibility": "explicit blame shifting",
    "blaming": "explicit blame shifting",
}


def require_packages():
    try:
        import torch
        from transformers import BertForSequenceClassification, BertTokenizer
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError as error:
        raise SystemExit(
            "Missing dependency. Install the translation/evaluation stack first:\n"
            "python -m pip install torch transformers sentencepiece protobuf\n"
        ) from error

    return {
        "torch": torch,
        "BertForSequenceClassification": BertForSequenceClassification,
        "BertTokenizer": BertTokenizer,
        "MarianMTModel": MarianMTModel,
        "MarianTokenizer": MarianTokenizer,
    }


def get_device(torch, requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_text(text, max_chars):
    sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())
    chunks = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks or [str(text)]


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def translate_texts(data, args, deps, device):
    torch = deps["torch"]
    MarianMTModel = deps["MarianMTModel"]
    MarianTokenizer = deps["MarianTokenizer"]

    tokenizer = MarianTokenizer.from_pretrained(args.translation_model)
    model = MarianMTModel.from_pretrained(args.translation_model).to(device)
    model.eval()

    chunk_rows = []
    for row_id, text in zip(data["record_id"], data["text"]):
        for chunk in split_text(text, args.translation_chunk_chars):
            chunk_rows.append({"record_id": row_id, "text": chunk})

    translated_chunks = []
    with torch.no_grad():
        for batch in batched(chunk_rows, args.translation_batch_size):
            texts = [row["text"] for row in batch]
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.translation_max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(**encoded, max_new_tokens=args.translation_max_new_tokens)
            translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for row, translation in zip(batch, translations):
                translated_chunks.append({"record_id": row["record_id"], "translated_chunk": translation})

    translated = pd.DataFrame(translated_chunks)
    translated = translated.groupby("record_id")["translated_chunk"].apply(" ".join).reset_index()
    translated = translated.rename(columns={"translated_chunk": "translated_text"})

    return data.merge(translated, on="record_id", how="left")


def load_or_translate(args, deps, device):
    if args.translated_data.exists() and not args.force_translate:
        return pd.read_csv(args.translated_data, keep_default_na=False)

    data = pd.read_csv(args.data, keep_default_na=False)
    translated = translate_texts(data, args, deps, device)
    translated.to_csv(args.translated_data, index=False, encoding="utf-8")
    return translated


def model_labels(model):
    id2label = model.config.id2label
    labels = {}
    for key, value in id2label.items():
        labels[int(key)] = value
    return labels


def align_labels(data, labels_by_id):
    labels = set(labels_by_id.values())
    data = data.copy()
    data["source_label"] = data["label"]

    if set(data["label"]).issubset(labels):
        return data

    merged = data["label"].map(MERGED_LABEL_MAP)
    if merged.isna().any():
        missing = sorted(data.loc[merged.isna(), "label"].unique())
        raise ValueError(f"Cannot map these labels to the model label set: {missing}")

    if set(merged).issubset(labels):
        data["label"] = merged
        return data

    raise ValueError(
        "Dataset labels do not match the model labels. "
        f"Dataset labels: {sorted(data['label'].unique())}; model labels: {sorted(labels)}"
    )


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


def evaluate(translated, args, deps, device):
    torch = deps["torch"]
    BertForSequenceClassification = deps["BertForSequenceClassification"]
    BertTokenizer = deps["BertTokenizer"]

    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    labels_by_id = model_labels(model)
    label_to_id = {label: index for index, label in labels_by_id.items()}
    translated = align_labels(translated, labels_by_id)
    translated = translated[translated["label"].isin(label_to_id)].copy()
    translated["label_id"] = translated["label"].map(label_to_id).astype(int)

    predictions = []
    losses = []

    with torch.no_grad():
        for batch in batched(list(translated.index), args.classification_batch_size):
            texts = translated.loc[batch, "translated_text"].tolist()
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.classification_max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            labels = torch.tensor(translated.loc[batch, "label_id"].to_numpy(), dtype=torch.long).to(device)
            output = model(**encoded, labels=labels)
            losses.append(float(output.loss.item()))
            predictions.extend(output.logits.argmax(dim=-1).cpu().numpy())

    y_true = translated["label_id"].to_numpy()
    y_pred = np.array(predictions)
    translated["predicted_label_id"] = y_pred
    translated["predicted_label"] = translated["predicted_label_id"].map(labels_by_id)
    translated["correct"] = translated["label"] == translated["predicted_label"]

    metrics = {
        "translation_model": args.translation_model,
        "classification_model": str(args.model_dir),
        "data": str(args.data),
        "translated_data": str(args.translated_data),
        "rows": int(len(translated)),
        "labels": label_to_id,
        "device": str(device),
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_f1": macro_f1(y_true, y_pred, len(label_to_id)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, len(label_to_id)).tolist(),
    }

    return translated, metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate French segments to German and evaluate a fine-tuned GBERT classifier.",
        allow_abbrev=False,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--translated-data", type=Path, default=DEFAULT_TRANSLATED)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--translation-model", default="Helsinki-NLP/opus-mt-fr-de")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force-translate", action="store_true")
    parser.add_argument("--translation-batch-size", type=int, default=8)
    parser.add_argument("--translation-max-length", type=int, default=512)
    parser.add_argument("--translation-max-new-tokens", type=int, default=512)
    parser.add_argument("--translation-chunk-chars", type=int, default=700)
    parser.add_argument("--classification-batch-size", type=int, default=8)
    parser.add_argument("--classification-max-length", type=int, default=256)
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    deps = require_packages()
    device = get_device(deps["torch"], args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    translated = load_or_translate(args, deps, device)
    predictions, metrics = evaluate(translated, args, deps, device)

    predictions.to_csv(args.output_dir / "translated_french_predictions.csv", index=False, encoding="utf-8")
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(
        "Translated French evaluation complete. "
        f"Accuracy: {metrics['accuracy']:.3f}; "
        f"macro F1: {metrics['macro_f1']:.3f}. "
        f"Outputs saved to {args.output_dir}."
    )


if __name__ == "__main__":
    main()
