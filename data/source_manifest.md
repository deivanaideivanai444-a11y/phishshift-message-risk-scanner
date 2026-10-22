# Data provenance and sanitization

`synthetic_messages.csv` is generated locally by `generate_dataset.py`; it is not a
download or a copy of real communications. The fixed generator seed is recorded in
`config.py`. The generator creates 480 fictional examples across 20 balanced synthetic
campaign groups, with both SMS and email records, both labels, simulated sender types,
and dates used only as synthetic metadata.

All messages, sender descriptions, and campaign names are invented. Example URLs use
the reserved `.invalid` domain and defanged `hxxps`/`[.]` notation. No live URL, user
name, real address, credential, phone number, or personal message is included. Several
legitimate examples intentionally mention urgent payroll notices, passwords, and
account resets to measure false-positive impact.

The corpus is balanced by label, channel, and URL presence. For 4 of 12 message slots
in each campaign, paired phishing/legitimate records have the same visible text but
different simulated sender provenance. This intentionally demonstrates that text alone
cannot establish sender legitimacy; `sender_domain_type` is recorded as metadata but
excluded from model input. Remaining examples have overlapping action wording rather
than an all-or-nothing phrase-to-label mapping. The campaign holdout keeps each paired
set together. Cross-label exact text matches are therefore retained and counted, not
mistakenly dropped as duplicates. This is a synthetic limitation study, not evidence of
real-world detection quality.

Do not replace the data with personal or active-message data without a privacy and
licensing review.

To reproduce the CSV, run `python generate_dataset.py`. The generated CSV is a build
output; the generator is the source of truth.
