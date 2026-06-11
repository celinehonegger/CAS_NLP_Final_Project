from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

FILES = {
    "CH": "CH_COVID19_media_conferences.xlsx",
    "DE": "DE_COVID19_media_conferences.xlsx",
}

TEST_FRACTION = 0.20
RANDOM_SEED = 20260602

GERMAN_WORDS = {
    "aber", "alle", "als", "auch", "auf", "aus", "bei", "bund", "bundesrat",
    "dass", "dazu", "der", "die", "doch", "ein", "eine", "einem", "einen",
    "einer", "es", "für", "haben", "hat", "heisst", "herr", "hier", "ich",
    "im", "in", "ist", "kann", "kantone", "können", "man", "massnahmen",
    "mit", "nicht", "noch", "schweiz", "sich", "sind", "und", "uns",
    "vielleicht", "von", "was", "wenn", "wir", "zu", "zum",
}

FRENCH_WORDS = {
    "alors", "avec", "avoir", "canton", "cantons", "ce", "cela", "cette",
    "dans", "de", "des", "donc", "du", "elle", "en", "est", "et", "faire",
    "faut", "ici", "il", "je", "la", "le", "les", "leur", "mais", "nous",
    "on", "ou", "pas", "pour", "que", "qui", "sur", "un", "une", "vous",
}


def clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def find_header_row(raw):
    for index, row in raw.iterrows():
        values = [clean_text(value).lower() for value in row.iloc[:3]]
        if values == ["document name", "code", "segment"]:
            return index
    raise ValueError("Could not find header row: Document name, Code, Segment")


def parse_dates(values):
    parsed = []

    for value in values:
        text = clean_text(value)
        date = pd.NaT

        for date_format in ["%d.%m.%y", "%d.%m.%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
            date = pd.to_datetime(text, format=date_format, errors="coerce")
            if not pd.isna(date):
                break

        parsed.append("" if pd.isna(date) else date.strftime("%Y-%m-%d"))

    return pd.Series(parsed, index=values.index)


def tokens(text):
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", str(text).lower())


def detect_language(row):
    if row["country"] == "DE":
        return "de"

    words = tokens(row["text"])
    german_score = sum(word in GERMAN_WORDS for word in words)
    french_score = sum(word in FRENCH_WORDS for word in words)

    if german_score > french_score:
        return "de"
    if french_score > german_score:
        return "fr"
    return "unclear"


def split_code(code):
    parts = [part.strip().lower() for part in code.split(">", 1)]
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", parts[0]


def load_file(country, filename):
    path = RAW_DIR / filename
    raw = pd.read_excel(path, sheet_name="coded segments", header=None)
    header_row = find_header_row(raw)

    data = raw.iloc[header_row + 1 :, :3].copy()
    data.columns = ["document_name", "code", "segment"]

    for column in data.columns:
        data[column] = data[column].map(clean_text)

    data = data[(data["code"] != "") & (data["segment"] != "")].copy()
    data["country"] = country
    data["conference_date"] = parse_dates(data["document_name"])
    parsed = data["code"].map(split_code)
    data["target"] = parsed.map(lambda value: value[0])
    data["label"] = parsed.map(lambda value: value[1])
    data["text"] = data["segment"]

    return data


def assign_split(data):
    rng = np.random.default_rng(RANDOM_SEED)
    data = data.copy()
    data["split"] = "train"

    for _, index in data.groupby("label", sort=True).groups.items():
        index = np.array(list(index))
        if len(index) < 2:
            continue
        shuffled = rng.permutation(index)
        test_count = max(1, round(len(index) * TEST_FRACTION))
        test_count = min(test_count, len(index) - 1)
        data.loc[shuffled[:test_count], "split"] = "test"

    return data


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    frames = [load_file(country, filename) for country, filename in FILES.items()]
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["country", "conference_date", "document_name"])
    data = data.reset_index(drop=True)
    data.insert(0, "record_id", [f"seg_{i:05d}" for i in range(1, len(data) + 1)])
    data = assign_split(data)
    data["detected_language"] = data.apply(detect_language, axis=1)

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

    all_segments = data[data["detected_language"].isin(["de", "fr"])].copy()
    german_segments = data[data["detected_language"] == "de"].copy()

    all_segments[columns].to_csv(PROCESSED_DIR / "all_segments.csv", index=False, encoding="utf-8")
    german_segments[columns].to_csv(PROCESSED_DIR / "german_segments.csv", index=False, encoding="utf-8")

    print("Created all_segments.csv and german_segments.csv.")


if __name__ == "__main__":
    main()
