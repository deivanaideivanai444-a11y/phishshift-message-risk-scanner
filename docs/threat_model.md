# Threat model

## Assets and boundaries
- User-provided message text (potentially sensitive): accepted by the local Flask form.
- Model and synthetic experiment artifacts: local files.
- Temporary recent-analysis previews: process memory only, removed when the server
  process stops. Flask's signed cookie contains only an opaque session token.

## Threats and mitigations
| Threat | Mitigation / remaining limitation |
|---|---|
| User follows a malicious link while investigating | UI warns not to click; submitted URLs are never fetched, opened, or executed. |
| Private message text is retained | No database/logging of body text; only a short preview in in-memory history. Avoid entering real secrets. |
| Classification mistaken for proof of safety | Persistent educational disclaimer, explanation, explicit score labeling, and abstention. |
| Urgent legitimate message is falsely flagged | Evaluation reports false-positive rate for synthetic legitimate urgent password/payroll/account-reset notices. |
| Campaign/template leakage inflates metrics | Deduplicate prior to splitting and hold out full campaign groups. |
| Crafted/novel message evades detection | Models are limited to synthetic text; manual verification remains necessary. |
| Malformed or excessive form input | Channel/body checks and length limits; no model call on empty/oversized text. |
| Model artifacts are corrupt/missing | Explicit model-loading error and health status; do not silently produce a verdict. |

## Out of scope
This prototype is not an email gateway, URL reputation service, malware scanner,
incident-response tool, or production authentication/session system. Run only on a
trusted local machine; do not expose the Flask development server to an untrusted
network.
