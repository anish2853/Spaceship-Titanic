# AI Usage Note

This project was developed with assistance from Claude (Anthropic), used as
follows:

- **Feature engineering ideas**: suggested group/family-based features
  (GroupSize, IsAlone, FamilySize), cabin-region binning, and log-transforms
  for the skewed spend columns, based on patterns identified in EDA.
- **Missing-value strategy**: proposed the group-first imputation fallback
  chain (group mode/median → related-feature mode/median → global
  mode/median) instead of naive global imputation, after reviewing the
  missing-value distribution.
- **Model pipeline**: helped write and debug the 10-fold Stratified
  Cross-Validation loop, the 3-model ensemble (XGBoost, LightGBM, CatBoost),
  and the blend-vs-stack comparison logic.
- **EDA plots**: generated the plotting script (`eda_plots.py`) and the
  accompanying written summary (`eda/EDA_summary.md`) based on the actual
  computed statistics and generated charts.
- **Documentation**: drafted this README and usage note.

All code was reviewed, executed, and validated by the candidate before
submission. All metrics reported (accuracy, precision, recall, F1, ROC-AUC,
log loss, CV mean/std) were produced by actually running the pipeline on the
provided `train.csv`/`test.csv`, not fabricated or estimated.
