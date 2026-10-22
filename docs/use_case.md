# Use-case description — PhishShift (ML-07)

## Intended user and need
Students and security-awareness facilitators can inspect fictional SMS/email examples
and see how simple text classifiers react. A user pastes a message without following
its links, chooses SMS or Email, and receives a label, a probability estimate, visible
text cues, and a safe explanation. Ambiguous or model-disagreement cases abstain for
manual review.

## Scope and limits
This educational prototype analyses text locally. It does not verify sender identity,
inspect headers, resolve URLs, scan attachments, or establish that a message is safe.
Its synthetic examples cannot represent the full variety of real campaigns. The
appropriate action for suspicious or high-impact requests is independent verification
through a known contact path, not reliance on this label.

## Main flow
1. User selects SMS or Email and optionally enters a subject.
2. User pastes simulated text, with private details omitted.
3. The app checks input bounds and runs the saved TF-IDF classifiers locally.
4. The user sees the primary class estimate, cue matches, disagreement/abstention
   status, and a non-guarantee warning.
5. A short message preview is retained only in process memory during that server run.

## Success criteria
The reproducible workflow creates synthetic data, removes near duplicate templates
before partitioning, holds out complete campaigns, saves actual evaluation results,
generates labelled plots, and passes the automated tests. Project scores are descriptive,
not production-security evidence.
