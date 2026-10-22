# Requirement-to-test traceability

| Requirement | Evidence / verification |
|---|---|
| Required data columns, missing values and labels | `tests/test_preprocessing.py`; `src/preprocessing.py` |
| Exact/near deduplication and cross-label ambiguity review | `tests/test_preprocessing.py`; `reports/cleaning_deduplication_report.json` |
| Campaign-separated holdout | `tests/test_preprocessing.py`; model bundle campaign metadata |
| Three approaches, common test set, and metrics | `tests/test_evaluation.py`; `src/evaluation.py`; `results/model_comparison.csv` |
| Label/feature shortcut audit | `tests/test_evaluation.py`; cleaning report and `reports/model_feature_audit.json` |
| Legitimate urgent false-positive impact | `tests/test_prediction.py`, `tests/test_acceptance.py`; urgent subset columns in result CSV and `reports/acceptance_scenarios.json` |
| Eight harmless perturbations and actual prediction changes | `tests/test_evaluation.py`, `tests/test_acceptance.py`; `prediction_flip_rate` and `reports/eight_perturbations.json` |
| False-positive/false-negative examples | `tests/test_evaluation.py`; `results/error_analysis.csv` and `reports/error_analysis.json` |
| Reusable prediction and abstention | `tests/test_prediction.py`; `src/predict.py` |
| Empty/invalid input and missing model | `tests/test_prediction.py`, `tests/test_acceptance.py` |
| Local-only URL handling and message retention | `src/safety.py`, `app.py`, `docs/threat_model.md` |
| Eight perturbed synthetic inputs | `test_at03_defanged_url_and_punctuation_variations` in `tests/test_acceptance.py` |

Run `python run_tests.py` to write the actual run summary to
`reports/test_results.json`. Generated values are run-specific; do not pre-fill them
with expected or invented scores.
