# Final report draft — PhishShift Message Risk Scanner (ML-07)

> Draft only. Replace bracketed fields after running the project and personal review.
> Do not represent this draft as mentor-approved or independently verified.

## 1. Abstract
PhishShift is an educational web application that classifies simulated SMS and email
text as likely phishing, likely legitimate, or uncertain/manual review. It compares a
rule baseline and two TF-IDF classifiers using campaign-separated evaluation.
**Actual results:** [insert from `results/model_comparison.csv`].

## 2. Introduction and problem statement
Text-based social engineering can imitate ordinary service notices. This project
demonstrates a reproducible local NLP workflow and makes uncertainty visible; it does
not promise to detect every attack.

## 3. Objectives and use case
The objectives are synthetic data generation, leakage-aware evaluation, safe local
prediction, beginner-friendly visualization, and automated tests. Intended use and
user flow are described in [use_case.md](./use_case.md).

## 4. Data and methodology
The locally generated dataset contains 480 fictional records across 20 balanced
synthetic campaign groups. Sources, sanitization, and limitations are documented in
`data/source_manifest.md`. Exact and near duplicates are removed before group splitting.
In the recorded run, 240 intentionally blank SMS subject cells were normalized, no
invalid-label rows were removed, and all 480 rows remained after deduplication. The
audit preserved 80 exact cross-label text twins because their simulated sender
provenance differs while visible message text is identical; one cross-label near-match
was also preserved. There were no exact or near matches crossing from training into
holdout. The campaign holdout contained 120 messages from five campaigns; 360 messages
from 15 campaigns remained for training. Exact campaign names are in
`reports/cleaning_deduplication_report.json`.

## 5. Implementation
Flask serves a responsive interface. Scikit-learn pipelines compare a rule baseline,
TF-IDF + Logistic Regression, and TF-IDF + Linear SVM. Joblib saves fitted artifacts.
Submitted URLs remain plain text and are not requested.

## 6. Results and discussion
The measured heldout results in this run were:

| Model | Accuracy | Precision | Recall | F1 | Urgent legitimate FP rate |
|---|---:|---:|---:|---:|---:|
| Rule-based baseline | 0.517 | 0.528 | 0.317 | 0.396 | 17/42 (0.405) |
| TF-IDF + Logistic Regression | 0.550 | 0.538 | 0.717 | 0.614 | 28/42 (0.667) |
| TF-IDF + Linear SVM | 0.500 | 0.500 | 0.650 | 0.565 | 29/42 (0.690) |

For legitimate/phishing label order, confusion matrices were `[[43, 17], [41, 19]]`
for the rule baseline, `[[23, 37], [17, 43]]` for Logistic Regression, and
`[[21, 39], [21, 39]]` for Linear SVM. On the separate eight-variant perturbation test,
the rule baseline changed one prediction (spaced defanged scheme); both TF-IDF models
changed zero. The fixed-transform pass over all holdout texts changed zero predictions
for these final model artifacts. The unfamiliar-wording AT-02 message and AT-01
legitimate urgent password-reset message are recorded in
`reports/acceptance_scenarios.json`; use those artifacts for the exact current
predictions. The evaluation's elevated urgent-legitimate false-positive rates and
near-chance overall accuracy show the limitations of text-only prediction in this
synthetic test. The previous perfect scores came from highly repeated, class-specific
synthetic wording and a small, easy holdout; the first saved classifier's largest
phishing weights included `hxxps`, `example`, and `invalid`. The current generator
balances link/channel rates, overlaps phrase families, and includes same-text/different-
sender examples to expose missing context. Test outcomes are recorded in
`reports/test_results.json`.

These are results on a small, deliberately synthetic corpus and the high TF-IDF scores
should not be interpreted as real-world performance. Explain weaknesses and any later
failures; do not claim generalization to live messages.

## 7. Conclusion and future work
Summarize what the actual run supports. Possible future study: licensed, privacy-reviewed
public corpora, chronological or source-separated evaluation, calibrated abstention,
and usability testing. Any real data requires additional ethical and legal review.

## 8. References, acknowledgements, and verification
No external data source was used for the synthetic dataset. **Add verified references
only if used.** Mentor details/approval: [not supplied; obtain and verify personally].
Author identity, institution, screenshots, and final result verification:
[complete personally].
