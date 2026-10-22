# 3–5 minute demonstration script

**0:00–0:30 — Context.** Introduce PhishShift ML-07 as an educational, local-only
decision-support prototype. Read the persistent disclaimer aloud.

**0:30–1:10 — Reproducibility.** In the terminal show `python generate_dataset.py`, then
open the CSV and source manifest. Point out synthetic campaign IDs, the `.invalid`
defanged examples, and legitimate password/payroll/reset messages.

**1:10–2:00 — Training and evaluation.** Run `python train_model.py`. Show the cleaning
report, campaign holdout list, results CSV, and the three labelled plots. Explain that
these are measured synthetic-set results and discuss urgent legitimate false positives.

**2:00–3:20 — Web app.** Run `python app.py`, open `http://127.0.0.1:5000`, and analyse
one harmless simulated message. Demonstrate the result, cue list, score label,
abstention, and temporary recent preview. Do not submit real private messages.

**3:20–4:00 — Safety and tests.** Show that no URL is fetched and no user text is
permanently stored. Run `python run_tests.py`; if tests fail, report actual failures
and refer to `reports/test_results.json`.

**4:00–4:30 — Limitations.** Emphasize that model confidence is not a guarantee, the
examples are synthetic, and unexpected requests should be checked through a known
independent contact method.
