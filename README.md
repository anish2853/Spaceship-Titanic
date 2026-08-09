# Spaceship Titanic — Submission

Predicts whether a passenger was `Transported` to another dimension, using an
ensemble of XGBoost + LightGBM + CatBoost with group/family-aware feature
engineering and imputation.

## Project structure
```
.
├── train.csv                  # Kaggle training data
├── test.csv                   # Kaggle test data
├── sample_submission.csv      # Kaggle's expected submission format
├── train_improved.py          # Main pipeline: FE -> CV -> model training -> submission.csv
├── eda_plots.py               # Generates all EDA plots
├── eda/                       # EDA plots + summary
│   ├── 01_target_distribution.png
│   ├── 02_missing_values.png
│   ├── 03_age_distribution.png
│   ├── 04_cryosleep_vs_transported.png
│   ├── 05_homeplanet_vs_transported.png
│   ├── 06_spend_distributions.png
│   ├── 07_totalspend_vs_transported.png
│   ├── 08_correlation_heatmap.png
│   ├── 09_vip_vs_transported.png
│   ├── 10_deck_vs_transported.png
│   └── EDA_summary.md
├── submission.csv             # Final Kaggle-format predictions (output of train_improved.py)
├── requirements.txt
├── AI_USAGE.md
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

1. Place `train.csv` and `test.csv` in the project root (same folder as
   `train_improved.py`). These are the files downloaded from the Kaggle
   Spaceship Titanic competition page.
2. Generate the EDA plots (optional, already included in `eda/`):
   ```bash
   python3 eda_plots.py
   ```
3. Run the main training pipeline:
   ```bash
   python3 train_improved.py
   ```
   This will:
   - Engineer features (group/family structure, cabin region, spend
     transforms, age buckets — see `AI_USAGE.md` / code comments for detail)
   - Impute missing values using group-first fallback chains
   - Train XGBoost, LightGBM, and CatBoost with 10-fold Stratified
     Cross-Validation
   - Compare a simple average blend vs. a logistic-regression stack and
     pick the better one on OOF accuracy
   - Print all metrics required for the write-up (accuracy, precision,
     recall, F1, ROC-AUC, log loss, CV mean/std, training time, etc.)
   - Save the final predictions to `submission.csv` in Kaggle's expected
     format (`PassengerId`, `Transported`)

## Output

`submission.csv` has two columns:
```
PassengerId,Transported
0013_01,False
0018_01,True
...
```
Upload this file directly to the Kaggle competition's submission page.

## Model performance (10-fold CV, current run)

| Metric | Value |
|---|---|
| CV Accuracy (mean) | 0.8165 |
| CV Accuracy (std) | 0.0141 |
| Validation Accuracy (OOF) | ~0.816 – 0.819 |
| ROC-AUC | ~0.909 |
| Log loss | ~0.373 – 0.383 |

(Exact numbers vary slightly by strategy chosen — see console output when
you run `train_improved.py`.)
