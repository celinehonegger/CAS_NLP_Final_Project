#%%

from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd

#%%
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "processed" / "german_segments_merged.csv"
DEFAULT_OUTPUT = ROOT / "output" / "models" / "gbert_merged_label_classifier"


def require_training_packages():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import BertForSequenceClassification, BertTokenizer
        from transformers import get_linear_schedule_with_warmup
    except ImportError as error:
        raise SystemExit(
            "Missing training dependency. Install the fine-tuning stack first:\n"
            "python3 -m pip install torch transformers\n"
        ) from error

    return {
        "torch": torch,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "BertForSequenceClassification": BertForSequenceClassification,
        "BertTokenizer": BertTokenizer,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    }


def set_seed(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data(path):
    data = pd.read_csv(path, keep_default_na=False)
    needed = {"text", "label", "split"}
    missing = needed - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = data[data["split"].isin(["train", "test"])].copy()
    data = data[(data["text"].str.strip() != "") & (data["label"].str.strip() != "")]
    return data.reset_index(drop=True)


def freeze_bert_layers(model):
    for parameter in model.bert.parameters():
        parameter.requires_grad = False


def build_label_maps(labels):
    label_names = sorted(labels.unique())
    label_to_id = {label: index for index, label in enumerate(label_names)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    return label_to_id, id_to_label


def encode_labels(data, label_to_id):
    data = data.copy()
    data["label_id"] = data["label"].map(label_to_id)
    if data["label_id"].isna().any():
        raise ValueError("Found labels not present in label_to_id.")
    data["label_id"] = data["label_id"].astype(int)
    return data


def make_dataset_class(Dataset):
    class SegmentDataset(Dataset):
        def __init__(self, frame, tokenizer, max_length):
            self.texts = frame["text"].tolist()
            self.labels = frame["label_id"].to_numpy(dtype=np.int64)
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, index):
            encoded = self.tokenizer(
                self.texts[index],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            item = {key: value.squeeze(0) for key, value in encoded.items()}
            item["labels"] = self.labels[index]
            return item

    return SegmentDataset


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


def evaluate(model, loader, device, torch, label_count):
    model.eval()
    predictions = []
    actuals = []
    losses = []

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            losses.append(float(output.loss.item()))
            predictions.extend(output.logits.argmax(dim=-1).cpu().numpy())
            actuals.extend(batch["labels"].cpu().numpy())

    y_true = np.array(actuals)
    y_pred = np.array(predictions)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": accuracy,
        "macro_f1": macro_f1(y_true, y_pred, label_count),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def confusion_matrix(y_true, y_pred, label_count):
    matrix = np.zeros((label_count, label_count), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def train(args):
    deps = require_training_packages()
    torch = deps["torch"]
    DataLoader = deps["DataLoader"]
    BertForSequenceClassification = deps["BertForSequenceClassification"]
    BertTokenizer = deps["BertTokenizer"]
    get_linear_schedule_with_warmup = deps["get_linear_schedule_with_warmup"]
    SegmentDataset = make_dataset_class(deps["Dataset"])

    set_seed(args.seed, torch)
    data = load_data(args.data)

    train_data = data[data["split"] == "train"].copy()
    test_data = data[data["split"] == "test"].copy()
    label_to_id, id_to_label = build_label_maps(train_data["label"])
    data = encode_labels(data, label_to_id)
    train_data = data[data["split"] == "train"].copy()
    test_data = data[data["split"] == "test"].copy()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    if args.freeze_bert:
        freeze_bert_layers(model)

    device = get_device(torch)
    model.to(device)

    train_loader = DataLoader(
        SegmentDataset(train_data, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        SegmentDataset(test_data, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_macro_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(output.loss.item()))

        metrics = evaluate(model, test_loader, device, torch, len(label_to_id))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "test_loss": metrics["loss"],
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
            }
        )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate(model, test_loader, device, torch, len(label_to_id))
    matrix = confusion_matrix(final_metrics["y_true"], final_metrics["y_pred"], len(label_to_id))

    predictions = test_data.copy()
    predictions["predicted_label_id"] = final_metrics["y_pred"]
    predictions["predicted_label"] = predictions["predicted_label_id"].map(id_to_label)
    predictions["correct"] = predictions["label"] == predictions["predicted_label"]

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    predictions.drop(columns=["label_id"]).to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8")

    metrics = {
        "model_name": args.model_name,
        "data": str(Path(args.data)),
        "train_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "labels": label_to_id,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "device": str(device),
        "freeze_bert": args.freeze_bert,
        "history": history,
        "test_loss": final_metrics["loss"],
        "test_accuracy": final_metrics["accuracy"],
        "test_macro_f1": final_metrics["macro_f1"],
        "confusion_matrix": matrix.tolist(),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(
        "Training complete. "
        f"Accuracy: {final_metrics['accuracy']:.3f}; "
        f"macro F1: {final_metrics['macro_f1']:.3f}. "
        f"Device: {device}. "
        f"Outputs saved to {output_dir}."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune deepset/gbert-base on German coded segments with merged labels.",
        allow_abbrev=False,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default="deepset/gbert-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--freeze-bert", action="store_true")
    parser.add_argument("--seed", type=int, default=20260602)
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    train(parse_args())

# %%
