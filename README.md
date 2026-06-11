# CAS NLP Final Project

This repository contains the code, data, outputs, and final report for a CAS Natural Language Processing project on detecting government credit claiming and blame avoidance strategies in COVID-19 media-conference segments.

The project fine-tunes `deepset/gbert-base` on German-language text segments and evaluates the model on both German data and machine-translated French segments.

## Folder Overview

- `code/python/` contains the preprocessing, training, evaluation, and figure-generation scripts.
- `data/raw/` contains the original Excel files.
- `data/processed/` contains the derived CSV datasets used for training and testing.
- `output/models/` contains the fine-tuned model outputs, metrics, and prediction files.
- `output/figures/` contains figures used to summarize the data and model results.
- `final report/` contains the final project report as a PDF.

## Main Workflow

1. Prepare the raw Excel data with `prepare_data.py`.
2. Create the merged-label dataset with `create_merged_labels.py`.
3. Fine-tune the German GBERT model with `train_gbert_merged.py`.
4. Evaluate the model on translated French segments with `test_gbert_on_translated_segments.py`.
5. Create figures with the figure-generation scripts in `code/python/`.

## Notes

The final model uses three labels: `credit claiming`, `implicit blame avoidance`, and `explicit blame shifting`. French segments are not used for model training; they are translated into German and used as an external evaluation set.
