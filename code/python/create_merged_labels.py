from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "processed" / "german_segments.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "german_segments_merged.csv"

LABEL_MAP = {
    "credit claiming": "credit claiming",
    "explaining/sharing responsibility": "implicit blame avoidance",
    "praising": "implicit blame avoidance",
    "shifting responsibility": "explicit blame shifting",
    "blaming": "explicit blame shifting",
}


def main():
    data = pd.read_csv(INPUT_PATH, keep_default_na=False)
    original_label = data["label"].copy()
    data["label"] = original_label.map(LABEL_MAP)

    if data["label"].isna().any():
        missing = sorted(original_label[data["label"].isna()].unique())
        raise ValueError(f"Missing merged label mapping for: {missing}")

    columns = [
        "record_id",
        "split",
        "country",
        "conference_date",
        "target",
        "label",
        "text",
        "detected_language",
    ]
    data[columns].to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("Created german_segments_merged.csv.")


if __name__ == "__main__":
    main()
