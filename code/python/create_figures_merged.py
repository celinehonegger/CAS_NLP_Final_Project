from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "german_segments_merged.csv"
MODEL_DIR = ROOT / "output" / "models" / "gbert_merged_label_classifier"
METRICS_PATH = MODEL_DIR / "metrics.json"
PREDICTIONS_PATH = MODEL_DIR / "test_predictions.csv"
FIGURE_DIR = ROOT / "output" / "figures" / "merged"

BG = "#fbfaf7"
INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#dedbd2"
BLUE = "#3b6ea8"
TEAL = "#2f9c95"
GREEN = "#5a9c5a"
RED = "#b65f5f"
GOLD = "#c99432"
PURPLE = "#7d6fb1"
COLORS = [BLUE, TEAL, GOLD, PURPLE, RED]


def font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


TITLE_FONT = font(34, bold=True)
SUBTITLE_FONT = font(18)
LABEL_FONT = font(18)
SMALL_FONT = font(15)
TINY_FONT = font(12)


def text_size(draw, text, image_font):
    box = draw.textbbox((0, 0), str(text), font=image_font)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw, text, image_font, max_width):
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


def title_block(draw, title, subtitle=None):
    draw.text((60, 42), title, fill=INK, font=TITLE_FONT)
    if subtitle:
        draw.text((60, 86), subtitle, fill=MUTED, font=SUBTITLE_FONT)


def save_canvas(width=1400, height=900):
    return Image.new("RGB", (width, height), BG)


def draw_axis_label(draw, xy, text, image_font=SMALL_FONT, anchor="mm", fill=MUTED):
    draw.text(xy, text, fill=fill, font=image_font, anchor=anchor)


def label_distribution(data):
    counts = data.pivot_table(index="label", columns="split", values="record_id", aggfunc="count", fill_value=0)
    for column in ["train", "test"]:
        if column not in counts:
            counts[column] = 0
    counts["total"] = counts["train"] + counts["test"]
    counts = counts.sort_values("total", ascending=True)

    image = save_canvas()
    draw = ImageDraw.Draw(image)
    title_block(draw, "German Segments: Merged Label Balance", "Training and test rows by merged label")

    left, right, top, bottom = 440, 1240, 190, 760
    max_count = int(counts["total"].max())
    bar_height = 56
    gap = 28
    scale = (right - left) / max_count

    for tick in range(0, max_count + 1, 50):
        x = left + tick * scale
        draw.line((x, top - 18, x, bottom + 10), fill=GRID, width=1)
        draw_axis_label(draw, (x, bottom + 36), str(tick), TINY_FONT)

    for i, (label, row) in enumerate(counts.iterrows()):
        y = top + i * (bar_height + gap)
        train_width = row["train"] * scale
        test_width = row["test"] * scale
        label_lines = wrap_text(draw, label, LABEL_FONT, 360)
        for j, line in enumerate(label_lines):
            draw.text((60, y + 10 + j * 22), line, fill=INK, font=LABEL_FONT)
        draw.rounded_rectangle((left, y, left + train_width, y + bar_height), radius=8, fill=BLUE)
        draw.rounded_rectangle(
            (left + train_width, y, left + train_width + test_width, y + bar_height),
            radius=8,
            fill=TEAL,
        )
        draw.text((left + train_width + test_width + 14, y + 16), str(int(row["total"])), fill=INK, font=LABEL_FONT)

    legend_y = 810
    draw.rounded_rectangle((60, legend_y, 86, legend_y + 26), radius=5, fill=BLUE)
    draw.text((96, legend_y), "Train", fill=INK, font=SMALL_FONT)
    draw.rounded_rectangle((170, legend_y, 196, legend_y + 26), radius=5, fill=TEAL)
    draw.text((206, legend_y), "Test", fill=INK, font=SMALL_FONT)
    image.save(FIGURE_DIR / "label_distribution.png")


def target_distribution(data):
    frame = data.copy()
    frame["target"] = frame["target"].replace("", "no target")
    counts = frame["target"].value_counts().sort_values(ascending=False)

    image = save_canvas()
    draw = ImageDraw.Draw(image)
    title_block(draw, "Target Distribution", "German rows by blame/credit target")

    left, right, top, bottom = 260, 1240, 190, 730
    max_count = int(counts.max())
    bar_width = 160
    slot = (right - left) / len(counts)

    for tick in range(0, max_count + 1, 50):
        y = bottom - (tick / max_count) * (bottom - top)
        draw.line((left - 24, y, right + 10, y), fill=GRID, width=1)
        draw_axis_label(draw, (left - 46, y), str(tick), TINY_FONT, anchor="rm")

    for i, (target, count) in enumerate(counts.items()):
        x = left + i * slot + (slot - bar_width) / 2
        h = count / max_count * (bottom - top)
        color = COLORS[i % len(COLORS)]
        draw.rounded_rectangle((x, bottom - h, x + bar_width, bottom), radius=10, fill=color)
        draw.text((x + bar_width / 2, bottom - h - 34), str(int(count)), fill=INK, font=LABEL_FONT, anchor="mm")
        label_lines = wrap_text(draw, target, SMALL_FONT, 170)
        for j, line in enumerate(label_lines):
            draw.text((x + bar_width / 2, bottom + 24 + j * 19), line, fill=INK, font=SMALL_FONT, anchor="mm")

    image.save(FIGURE_DIR / "target_distribution.png")


def line_chart(draw, panel, values_by_name, y_label):
    left, top, right, bottom = panel
    all_values = [value for values in values_by_name.values() for value in values]
    y_min = min(0, min(all_values))
    y_max = max(all_values)
    if math.isclose(y_min, y_max):
        y_max += 1

    draw.rectangle((left, top, right, bottom), outline=GRID, width=2)
    for i in range(5):
        y = top + i * (bottom - top) / 4
        draw.line((left, y, right, y), fill=GRID, width=1)

    epochs = len(next(iter(values_by_name.values())))
    for name_index, (name, values) in enumerate(values_by_name.items()):
        color = COLORS[name_index % len(COLORS)]
        points = []
        for i, value in enumerate(values):
            x = left + (i / max(1, epochs - 1)) * (right - left)
            y = bottom - ((value - y_min) / (y_max - y_min)) * (bottom - top)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
        for point in points:
            draw.ellipse((point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6), fill=color)

    for i in range(epochs):
        x = left + (i / max(1, epochs - 1)) * (right - left)
        draw_axis_label(draw, (x, bottom + 26), str(i + 1), TINY_FONT)
    draw_axis_label(draw, ((left + right) / 2, bottom + 60), "Epoch", SMALL_FONT)
    draw_axis_label(draw, (left - 50, top), f"{y_max:.2f}", TINY_FONT, anchor="rm")
    draw_axis_label(draw, (left - 50, bottom), f"{y_min:.2f}", TINY_FONT, anchor="rm")
    draw_axis_label(draw, (left - 50, (top + bottom) / 2), y_label, SMALL_FONT, anchor="rm")

    legend_x = left
    legend_y = top - 42
    for name_index, name in enumerate(values_by_name):
        color = COLORS[name_index % len(COLORS)]
        draw.rounded_rectangle((legend_x, legend_y, legend_x + 24, legend_y + 24), radius=5, fill=color)
        draw.text((legend_x + 34, legend_y), name, fill=INK, font=SMALL_FONT)
        legend_x += text_size(draw, name, SMALL_FONT)[0] + 92


def training_history(metrics):
    history = metrics.get("history", [])
    image = save_canvas()
    draw = ImageDraw.Draw(image)
    title_block(
        draw,
        "GBERT Merged-Label Fine-Tuning History",
        f"Final test accuracy {metrics['test_accuracy']:.3f}; macro F1 {metrics['test_macro_f1']:.3f}",
    )

    line_chart(
        draw,
        (150, 210, 1280, 475),
        {
            "Train loss": [row["train_loss"] for row in history],
            "Test loss": [row["test_loss"] for row in history],
        },
        "Loss",
    )
    line_chart(
        draw,
        (150, 610, 1280, 820),
        {
            "Accuracy": [row["test_accuracy"] for row in history],
            "Macro F1": [row["test_macro_f1"] for row in history],
        },
        "Score",
    )

    image.save(FIGURE_DIR / "training_history.png")


def confusion_matrix(metrics):
    labels = {value: key for key, value in metrics["labels"].items()}
    label_names = [labels[index] for index in range(len(labels))]
    matrix = np.array(metrics["confusion_matrix"])
    max_value = matrix.max()

    image = save_canvas(1500, 1050)
    draw = ImageDraw.Draw(image)
    title_block(draw, "Merged-Label Test Confusion Matrix", "Rows are actual labels; columns are model predictions")

    left, top = 420, 245
    cell = 120

    for i, actual in enumerate(label_names):
        lines = wrap_text(draw, actual, SMALL_FONT, 300)
        for j, line in enumerate(lines):
            draw.text((60, top + i * cell + 42 + j * 18), line, fill=INK, font=SMALL_FONT)

    for j, predicted in enumerate(label_names):
        lines = wrap_text(draw, predicted, TINY_FONT, 108)
        for k, line in enumerate(lines):
            draw.text((left + j * cell + cell / 2, 178 + k * 15), line, fill=INK, font=TINY_FONT, anchor="mm")

    for i in range(len(label_names)):
        for j in range(len(label_names)):
            value = int(matrix[i, j])
            intensity = value / max_value if max_value else 0
            base = np.array([238, 244, 242])
            accent = np.array([47, 156, 149])
            rgb = tuple((base * (1 - intensity) + accent * intensity).astype(int))
            x = left + j * cell
            y = top + i * cell
            draw.rectangle((x, y, x + cell, y + cell), fill=rgb, outline=BG, width=3)
            text_color = "white" if intensity > 0.55 else INK
            draw.text((x + cell / 2, y + cell / 2), str(value), fill=text_color, font=LABEL_FONT, anchor="mm")

    draw.text((60, top - 46), "Actual label", fill=MUTED, font=SMALL_FONT)
    draw.text((left, 138), "Predicted label", fill=MUTED, font=SMALL_FONT)
    image.save(FIGURE_DIR / "confusion_matrix.png")


def prediction_summary(predictions):
    summary = predictions.groupby(["label", "predicted_label"]).size().reset_index(name="rows")
    errors = predictions[predictions["correct"] == False]
    top_errors = errors.groupby(["label", "predicted_label"]).size().sort_values(ascending=False).head(8)

    image = save_canvas(1400, 900)
    draw = ImageDraw.Draw(image)
    accuracy = float((predictions["label"] == predictions["predicted_label"]).mean())
    title_block(draw, "Merged-Label Prediction Summary", f"Correct test predictions: {accuracy:.1%}")

    y = 180
    draw.text((70, y), "Most Common Misclassifications", fill=INK, font=font(24, bold=True))
    y += 55
    for (actual, predicted), count in top_errors.items():
        text = f"{actual}  →  {predicted}"
        lines = wrap_text(draw, text, LABEL_FONT, 920)
        draw.rounded_rectangle((70, y - 10, 1290, y + 56 + (len(lines) - 1) * 22), radius=8, fill="#ffffff")
        for j, line in enumerate(lines):
            draw.text((95, y + j * 22), line, fill=INK, font=LABEL_FONT)
        draw.text((1240, y), str(int(count)), fill=RED, font=font(22, bold=True), anchor="rm")
        y += 78 + (len(lines) - 1) * 22

    if summary.empty:
        draw.text((70, y), "No predictions available.", fill=MUTED, font=LABEL_FONT)

    image.save(FIGURE_DIR / "prediction_summary.png")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH, keep_default_na=False)

    label_distribution(data)
    target_distribution(data)

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)
        training_history(metrics)
        confusion_matrix(metrics)

    if PREDICTIONS_PATH.exists():
        predictions = pd.read_csv(PREDICTIONS_PATH, keep_default_na=False)
        prediction_summary(predictions)

    print("Created merged-label figures in output/figures/merged.")


if __name__ == "__main__":
    main()
