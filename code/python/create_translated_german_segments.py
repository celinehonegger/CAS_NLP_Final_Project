from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "processed" / "french_segments_translated_de.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "translated_german_segments.csv"
TEST_OUTPUT_PATH = ROOT / "data" / "processed" / "translated_german_segments_test.csv"

LABEL_MAP = {
    "credit claiming": "credit claiming",
    "explaining/sharing responsibility": "implicit blame avoidance",
    "praising": "implicit blame avoidance",
    "shifting responsibility": "explicit blame shifting",
    "blaming": "explicit blame shifting",
}


def main():
    data = pd.read_csv(INPUT_PATH, keep_default_na=False)
    data["label"] = data["label"].map(LABEL_MAP)

    if data["label"].isna().any():
        raise ValueError("Some labels could not be mapped to merged labels.")

    data["text"] = data["translated_text"]
    data["detected_language"] = "de"

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
    data.loc[data["split"] == "test", columns].to_csv(TEST_OUTPUT_PATH, index=False, encoding="utf-8")

    print("Created translated_german_segments.csv and translated_german_segments_test.csv.")


if __name__ == "__main__":
    main()
