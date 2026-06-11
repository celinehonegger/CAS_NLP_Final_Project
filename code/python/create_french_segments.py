from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "processed" / "all_segments.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "french_segments.csv"


def main():
    data = pd.read_csv(INPUT_PATH, keep_default_na=False)
    french = data[data["detected_language"] == "fr"].copy()
    french.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("Created french_segments.csv.")


if __name__ == "__main__":
    main()
