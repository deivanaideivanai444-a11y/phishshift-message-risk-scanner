# PhishShift Message Risk Scanner (ML-07)

An educational, local Flask dashboard for analysing harmless simulated SMS and email
text. It compares a rule baseline with TF-IDF + Logistic Regression and TF-IDF + Linear
SVM. Results are **Likely Phishing**, **Likely Legitimate**, or
**Uncertain — Manual Review Required**.

> **Educational decision-support tool. Results are not a guarantee of safety.**
> Model confidence does not verify a sender or make a link safe. Do not enter real
> passwords, one-time codes, personal messages, or other sensitive information.

## Features
- Reproducible generator for 480 synthetic, balanced messages in 20 campaign groups.
- Schema/label checks, missing-value handling, normalized exact/near-template
  deduplication before campaign-separated holdout splitting.
- Actual accuracy, precision, recall, F1, confusion matrices, urgent-legitimate false
  positives, holdout campaign details, and harmless perturbation drift.
- Joblib model bundle; probability estimate and disagreement/low-score abstention.
- Responsive dark dashboard with message cues, measured result table and temporary
  in-memory recent-analysis previews.
- No paid APIs, no URL fetching, and no permanent message storage.

## Requirements
- Windows 10/11, VS Code, and Python 3.10 or newer.
- Internet is required only to install dependencies; the runtime analysis is local.

## Windows VS Code setup
Open the `message risk` folder in VS Code. In the integrated PowerShell terminal:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If script activation is restricted, invoke the environment executable directly:
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`. In VS Code use
**Python: Select Interpreter** and choose `.venv`.

## Generate data, train, evaluate, test, launch

```powershell
.\.venv\Scripts\python.exe generate_dataset.py
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe evaluate_model.py
.\.venv\Scripts\python.exe run_tests.py
.\.venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:5000>. `train_model.py` performs training and evaluation;
`evaluate_model.py` reruns that reproducible full process and refreshes artifacts.
The test command writes actual outcomes to `reports/test_results.json`. If a command
fails, preserve and investigate the failure rather than describing it as a pass.

### Exact PowerShell commands (using activated environment)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python generate_dataset.py
python train_model.py
python evaluate_model.py
python run_tests.py
python app.py
```

## Evaluation design
The split assigns entire `campaign_group` values to holdout; a message-level random split
is not used. Duplicate template checks run before the split, and near matches between
training and holdout are removed from training. Cross-label exact text twins are retained
because their different simulated sender provenance is absent from text features; they
are an intentional example of why text cannot verify identity. This limitation and the
other synthetic feature rates are recorded in the cleaning report. The eight-variant
perturbation report records each transformed text and whether each model's prediction
changed. Results may change if code or dependencies change; inspect the generated
CSV/JSON and PNGs for measured values.

The rule baseline is a transparent keyword heuristic. Logistic Regression provides
class probability estimates used by the abstention threshold. Linear SVM's predictions
are compared for disagreement; the app does not label an SVM decision score as a
probability. Probabilities are model estimates, not calibrated guarantees.

### Why earlier scores were misleading

The earlier generator repeated a small set of class-specific templates. The saved
linear models assigned their strongest phishing weights to `hxxps`, `example`, and
`invalid`; link presence was also perfectly aligned with the label. Deduplication then
removed 226 near-duplicate rows, leaving only 254 messages and a 62-message holdout.
That combination made campaign-heldout scores look perfect without testing varied
wording. The revised generator balances link/channel rates, overlaps phrase families,
and includes same-text examples with different simulated sender provenance. The model
does not receive sender metadata, so those records appropriately expose a text-only
limitation. The current measured scores and false-positive rates are in
`results/model_comparison.csv`; they are modest/near chance on the current synthetic
holdout and should not be presented as strong detection performance.

## Outputs
- `data/synthetic_messages.csv` — generated data.
- `models/model_bundle.joblib` — fitted local classifiers and campaign metadata.
- `reports/cleaning_deduplication_report.json` — cleaning and leakage evidence.
- `reports/model_feature_audit.json` — largest positive/negative linear text weights.
- `reports/confusion_matrices.json` — per-model confusion matrices.
- `reports/error_analysis.json` and `results/error_analysis.csv` — actual false-positive
  and false-negative counts and messages.
- `reports/model_comparison.png`, `reports/confusion_matrix.png`,
  `reports/robustness_drift.png` — labelled experiment charts.
- `reports/acceptance_scenarios.json` and `reports/acceptance_perturbation_results.json`
  — actual AT-01/AT-02 predictions and eight controlled score comparisons.
- `reports/eight_perturbations.json` — per-model predictions for eight transformations
  of a held-out message, including a changed/not-changed flag.
- `results/model_comparison.csv` / `.json` and `results/holdout_predictions.csv`.
- `reports/test_results.json` — actual test run outcomes.

## Project structure

```text
app.py, config.py, generate_dataset.py, train_model.py, evaluate_model.py, run_tests.py
src/                 preprocessing, deduplication, training, prediction, safety, evaluation
data/                synthetic CSV and provenance manifest
models/              generated Joblib model
results/             measured experiment results
reports/             dedup evidence, test results, plots, project documentation
templates/            Flask HTML
static/               CSS and JavaScript
tests/                preprocessing, prediction, and acceptance tests
docs/                 use case, threat model, traceability, journal and report draft
```

## Safety, limitations, and privacy
Only fictional examples and `.invalid` defanged links are generated. The scanner reads
submitted text locally and never follows a URL. It has no URL, header, attachment,
sender-authentication, or live threat-intelligence analysis. Recent history retains
only a short preview in process memory and disappears when the server stops; no
database stores messages. The web server is intended for loopback/local educational
use only, not public deployment. Treat all predictions as fallible and verify using a
known independent channel.

See [data/source_manifest.md](data/source_manifest.md),
[docs/threat_model.md](docs/threat_model.md), [docs/use_case.md](docs/use_case.md),
[docs/requirement_traceability.md](docs/requirement_traceability.md),
[docs/experiment_journal_template.md](docs/experiment_journal_template.md),
[docs/final_report_draft.md](docs/final_report_draft.md), and
[docs/demo_script.md](docs/demo_script.md).
