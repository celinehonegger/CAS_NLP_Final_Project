from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "figures" / "academic"

WHITE = "#ffffff"
BLACK = "#111111"
DARK = "#333333"
MID = "#666666"
LIGHT = "#d9d9d9"
PALE = "#f2f2f2"
BLUE = "#2f5d8c"
BLUE_LIGHT = "#b9cbe0"
GREY = "#7a7a7a"
GREY_LIGHT = "#c8c8c8"


DATASETS = [
    {
        "key": "five_label",
        "title": "Five-label GBERT model",
        "data": ROOT / "data" / "processed" / "german_segments.csv",
        "metrics": ROOT / "output" / "models" / "gbert_label_classifier" / "metrics.json",
        "predictions": ROOT / "output" / "models" / "gbert_label_classifier" / "test_predictions.csv",
    },
    {
        "key": "merged_label",
        "title": "Merged-label GBERT model",
        "data": ROOT / "data" / "processed" / "german_segments_merged.csv",
        "metrics": ROOT / "output" / "models" / "gbert_merged_label_classifier" / "metrics.json",
        "predictions": ROOT / "output" / "models" / "gbert_merged_label_classifier" / "test_predictions.csv",
    },
    {
        "key": "translated_french",
        "title": "Translated French evaluation",
        "data": ROOT / "data" / "processed" / "translated_german_segments.csv",
        "metrics": ROOT / "output" / "models" / "gbert_merged_label_classifier" / "translated_german_evaluation" / "metrics.json",
        "predictions": ROOT / "output" / "models" / "gbert_merged_label_classifier" / "translated_german_evaluation" / "predictions.csv",
    },
]


def font(size, bold=False):
    path = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
    if Path(path).exists():
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


TITLE = font(42, True)
SUBTITLE = font(27)
AXIS = font(25)
LABEL = font(28)
SMALL = font(24)
TINY = font(20)
NUMBER = font(32, True)


def canvas(width=1800, height=1200):
    return Image.new("RGB", (width, height), WHITE)


def draw_title(draw, title, subtitle):
    draw.text((95, 70), title, fill=BLACK, font=TITLE)
    draw.text((95, 125), subtitle, fill=MID, font=SUBTITLE)


def text_size(draw, text, image_font):
    box = draw.textbbox((0, 0), str(text), font=image_font)
    return box[2] - box[0], box[3] - box[1]


def wrap(draw, text, image_font, max_width):
    words = str(text).split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if text_size(draw, candidate, image_font)[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def label_order(metrics, labels):
    if metrics and "labels" in metrics:
        lookup = {v: k for k, v in metrics["labels"].items()}
        return [lookup[i] for i in range(len(lookup))]
    return sorted(labels)


def save(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = crop_white_border(image, padding=55)
    image.save(path, dpi=(300, 300))


def crop_white_border(image, padding=55):
    """Remove unused white canvas while preserving a small print margin."""
    array = np.asarray(image)
    non_white = np.any(array < 248, axis=2)
    if not non_white.any():
        return image

    ys, xs = np.where(non_white)
    left = max(int(xs.min()) - padding, 0)
    upper = max(int(ys.min()) - padding, 0)
    right = min(int(xs.max()) + padding, image.width)
    lower = min(int(ys.max()) + padding, image.height)
    return image.crop((left, upper, right, lower))


def label_balance(data, cfg):
    counts = data.pivot_table(index="label", columns="split", values="record_id", aggfunc="count", fill_value=0)
    for col in ["train", "test"]:
        if col not in counts:
            counts[col] = 0
    counts["total"] = counts["train"] + counts["test"]
    counts = counts.sort_values("total")

    image = canvas()
    draw = ImageDraw.Draw(image)
    draw_title(draw, f"{cfg['title']}: label distribution", "Number of cases by label and split")

    left, right = 610, 1580
    top = 235
    row_h = 115 if len(counts) <= 3 else 92
    bar_h = 34
    max_count = int(counts["total"].max())
    scale = (right - left) / max_count

    for tick in range(0, max_count + 1, 50):
        x = left + tick * scale
        draw.line((x, top - 35, x, top + row_h * len(counts) - 25), fill=LIGHT, width=1)
        draw.text((x, top + row_h * len(counts) + 8), str(tick), fill=MID, font=TINY, anchor="ma")

    for i, (label, row) in enumerate(counts.iterrows()):
        y = top + i * row_h
        for j, line in enumerate(wrap(draw, label, SMALL, 445)):
            draw.text((95, y - 2 + j * 29), line, fill=BLACK, font=SMALL)

        train_w = float(row["train"]) * scale
        test_w = float(row["test"]) * scale
        draw.rectangle((left, y, left + train_w, y + bar_h), fill=BLUE)
        draw.rectangle((left + train_w, y, left + train_w + test_w, y + bar_h), fill=GREY_LIGHT)
        draw.text((left + train_w + test_w + 18, y - 1), str(int(row["total"])), fill=BLACK, font=SMALL)

    legend_y = top + row_h * len(counts) + 90
    draw.rectangle((95, legend_y, 128, legend_y + 24), fill=BLUE)
    draw.text((142, legend_y - 2), "Train", fill=BLACK, font=SMALL)
    draw.rectangle((240, legend_y, 273, legend_y + 24), fill=GREY_LIGHT)
    draw.text((287, legend_y - 2), "Test", fill=BLACK, font=SMALL)

    save(image, OUT / cfg["key"] / "label_distribution_academic.png")


def target_distribution(data, cfg):
    frame = data.copy()
    frame["target"] = frame["target"].replace("", "no target")
    counts = frame["target"].value_counts().sort_values(ascending=True)

    image = canvas()
    draw = ImageDraw.Draw(image)
    draw_title(draw, f"{cfg['title']}: target distribution", "Number of cases by coded target")

    left, right = 560, 1580
    top = 245
    row_h = 112
    max_count = int(counts.max())
    scale = (right - left) / max_count

    for tick in range(0, max_count + 1, 50):
        x = left + tick * scale
        draw.line((x, top - 35, x, top + row_h * len(counts) - 28), fill=LIGHT, width=1)
        draw.text((x, top + row_h * len(counts) + 8), str(tick), fill=MID, font=TINY, anchor="ma")

    for i, (target, count) in enumerate(counts.items()):
        y = top + i * row_h
        for j, line in enumerate(wrap(draw, target, SMALL, 400)):
            draw.text((95, y - 2 + j * 29), line, fill=BLACK, font=SMALL)
        draw.rectangle((left, y, left + count * scale, y + 36), fill=BLUE)
        draw.text((left + count * scale + 18, y - 1), str(int(count)), fill=BLACK, font=SMALL)

    save(image, OUT / cfg["key"] / "target_distribution_academic.png")


def metrics_summary(metrics, cfg):
    image = canvas()
    draw = ImageDraw.Draw(image)

    if "test_accuracy" in metrics:
        accuracy = metrics["test_accuracy"]
        macro_f1 = metrics["test_macro_f1"]
        loss = metrics["test_loss"]
        subtitle = "Best checkpoint evaluated on held-out German test set"
    else:
        accuracy = metrics["accuracy"]
        macro_f1 = metrics["macro_f1"]
        loss = metrics["loss"]
        subtitle = "Fine-tuned German model evaluated on translated French segments"

    draw_title(draw, f"{cfg['title']}: model performance", subtitle)

    history = metrics.get("history", [])
    card_y = 300 if history else 235
    cards = [("Accuracy", accuracy), ("Macro F1", macro_f1), ("Loss", loss)]
    for i, (name, value) in enumerate(cards):
        x = 145 + i * 520
        y = card_y
        draw.line((x, y, x + 365, y), fill=BLACK, width=3)
        draw.text((x, y + 50), name, fill=MID, font=AXIS)
        draw.text((x, y + 105), f"{value:.3f}", fill=BLACK, font=font(66, True))

    if history:
        left, top, right, bottom = 175, 690, 1580, 1010
        values = {
            "Accuracy": [row["test_accuracy"] for row in history],
            "Macro F1": [row["test_macro_f1"] for row in history],
        }
        line_plot(draw, (left, top, right, bottom), values, "Epoch", "Score")
        draw.text((left, top - 55), "Performance by epoch", fill=BLACK, font=LABEL)

    save(image, OUT / cfg["key"] / "performance_summary_academic.png")


def line_plot(draw, panel, values_by_name, x_label, y_label):
    left, top, right, bottom = panel
    all_values = [v for values in values_by_name.values() for v in values]
    y_min = 0
    y_max = max(1, max(all_values))
    epochs = len(next(iter(values_by_name.values())))

    draw.line((left, bottom, right, bottom), fill=BLACK, width=2)
    draw.line((left, top, left, bottom), fill=BLACK, width=2)
    for tick in np.linspace(0, 1, 5):
        y = bottom - tick * (bottom - top)
        draw.line((left - 8, y, right, y), fill=LIGHT if tick else BLACK, width=1)
        draw.text((left - 18, y), f"{tick:.2f}", fill=MID, font=TINY, anchor="rm")

    for idx, (name, values) in enumerate(values_by_name.items()):
        color = BLUE if idx == 0 else GREY
        points = []
        for i, value in enumerate(values):
            x = left + (i / max(1, epochs - 1)) * (right - left)
            y = bottom - ((value - y_min) / (y_max - y_min)) * (bottom - top)
            points.append((x, y))
        draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)

        lx = right - 250
        ly = top + idx * 34
        draw.line((lx, ly + 10, lx + 35, ly + 10), fill=color, width=4)
        draw.text((lx + 50, ly), name, fill=BLACK, font=SMALL)

    for i in range(epochs):
        x = left + (i / max(1, epochs - 1)) * (right - left)
        draw.text((x, bottom + 22), str(i + 1), fill=MID, font=TINY, anchor="ma")
    draw.text(((left + right) / 2, bottom + 68), x_label, fill=MID, font=AXIS, anchor="ma")
    draw.text((left - 70, (top + bottom) / 2), y_label, fill=MID, font=AXIS, anchor="mm")


def confusion_matrix(metrics, cfg):
    labels = {value: key for key, value in metrics["labels"].items()}
    label_names = [labels[i] for i in range(len(labels))]
    matrix = np.array(metrics["confusion_matrix"])
    max_value = matrix.max()

    image = canvas(1900, 1300)
    draw = ImageDraw.Draw(image)
    draw_title(draw, f"{cfg['title']}: confusion matrix", "Rows are actual labels; columns are model predictions")

    cell = 180 if len(label_names) <= 3 else 145
    left = 610
    top = 315

    draw.text((left, top - 145), "Predicted label", fill=MID, font=AXIS)
    draw.text((95, top - 55), "Actual label", fill=MID, font=AXIS)

    for j, label in enumerate(label_names):
        x = left + j * cell + cell / 2
        for k, line in enumerate(wrap(draw, label, TINY, cell - 15)):
            draw.text((x, top - 118 + k * 23), line, fill=BLACK, font=TINY, anchor="ma")

    for i, label in enumerate(label_names):
        y = top + i * cell + cell / 2 - 15
        for k, line in enumerate(wrap(draw, label, SMALL, 430)):
            draw.text((95, y + k * 30), line, fill=BLACK, font=SMALL)

    for i in range(len(label_names)):
        for j in range(len(label_names)):
            value = int(matrix[i, j])
            intensity = value / max_value if max_value else 0
            shade = int(250 - intensity * 155)
            fill = (shade, shade, shade)
            x = left + j * cell
            y = top + i * cell
            draw.rectangle((x, y, x + cell, y + cell), fill=fill, outline=WHITE, width=4)
            text_fill = WHITE if intensity > 0.55 else BLACK
            draw.text((x + cell / 2, y + cell / 2), str(value), fill=text_fill, font=NUMBER, anchor="mm")

    save(image, OUT / cfg["key"] / "confusion_matrix_academic.png")


def prediction_errors(predictions, cfg):
    errors = predictions[predictions["correct"] == False]
    counts = errors.groupby(["label", "predicted_label"]).size().sort_values(ascending=True).tail(8)

    image = canvas()
    draw = ImageDraw.Draw(image)
    accuracy = float((predictions["label"] == predictions["predicted_label"]).mean())
    draw_title(draw, f"{cfg['title']}: classification errors", f"Most frequent errors; overall accuracy {accuracy:.3f}")

    if counts.empty:
        draw.text((95, 300), "No classification errors.", fill=BLACK, font=LABEL)
        save(image, OUT / cfg["key"] / "classification_errors_academic.png")
        return

    left, right = 690, 1580
    top = 240
    row_h = 90
    max_count = int(counts.max())
    scale = (right - left) / max_count

    for i, ((actual, predicted), count) in enumerate(counts.items()):
        y = top + i * row_h
        text = f"{actual} → {predicted}"
        for k, line in enumerate(wrap(draw, text, SMALL, 545)):
            draw.text((95, y - 4 + k * 29), line, fill=BLACK, font=SMALL)
        draw.rectangle((left, y, left + count * scale, y + 32), fill=BLUE)
        draw.text((left + count * scale + 18, y - 2), str(int(count)), fill=BLACK, font=SMALL)

    save(image, OUT / cfg["key"] / "classification_errors_academic.png")


def main():
    for cfg in DATASETS:
        data = pd.read_csv(cfg["data"], keep_default_na=False)
        metrics = json.load(open(cfg["metrics"], encoding="utf-8")) if cfg["metrics"].exists() else None
        predictions = pd.read_csv(cfg["predictions"], keep_default_na=False) if cfg["predictions"].exists() else None

        label_balance(data, cfg)
        target_distribution(data, cfg)
        if metrics:
            metrics_summary(metrics, cfg)
            confusion_matrix(metrics, cfg)
        if predictions is not None:
            prediction_errors(predictions, cfg)

    print("Created academic-style figures in output/figures/academic.")


if __name__ == "__main__":
    main()
