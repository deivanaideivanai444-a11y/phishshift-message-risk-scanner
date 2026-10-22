"""Generate a reproducible, harmless synthetic SMS/email dataset."""

import argparse
import random
from pathlib import Path

import pandas as pd

from config import DATASET_PATH, RANDOM_SEED

def make_rows(seed: int = RANDOM_SEED) -> list[dict[str, str]]:
    """Create 20 balanced campaigns × 24 messages = 480 fictional records.

    URL presence, channel, sender metadata, urgency and security-related words are
    balanced across labels so none is a direct label proxy.
    """
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    topics = (
        "parcel status", "club membership", "shared calendar", "monthly statement",
        "community event", "document access", "library account", "benefits portal",
        "service bulletin", "travel itinerary", "team schedule", "digital receipt",
        "training notice", "support ticket", "delivery window", "meeting notes",
        "account settings", "billing summary", "course reminder", "device notice",
    )
    contexts = (
        "blue timetable exercise with library opening hours and a quiet study room",
        "green roster exercise with shift notes, staff coverage and payroll dates",
        "amber calendar exercise with meeting minutes, room numbers and agenda notes",
        "violet statement exercise with monthly totals, receipt dates and budget labels",
        "silver gathering exercise with a public venue, start time and weather note",
        "copper document exercise with revision notes, page count and archive label",
        "indigo reading exercise with renewal dates, book titles and return desk hours",
        "teal benefits exercise with enrollment dates, plan names and office hours",
        "coral bulletin exercise with maintenance dates, service notes and status labels",
        "navy itinerary exercise with departure times, map notes and packing reminders",
        "gold schedule exercise with project names, team roles and calendar entries",
        "plum receipt exercise with item categories, issue dates and filing notes",
        "mint course exercise with lesson titles, classroom notes and completion dates",
        "rust support exercise with ticket topics, response windows and reference notes",
        "pearl delivery exercise with depot hours, route notes and collection windows",
        "orchid meeting exercise with discussion topics, attendee roles and action notes",
        "charcoal settings exercise with preference names, device notes and help hours",
        "sage billing exercise with account periods, invoice labels and payment dates",
        "ocean class exercise with module names, lecture notes and campus locations",
        "citrus device exercise with model labels, setup notes and service hours",
    )
    openings = (
        "A note about {topic} has a new status",
        "There is an update regarding {topic}",
        "The latest {topic} details are ready",
        "A request connected with {topic} was recorded",
        "Please review this {topic} message",
        "Your {topic} information has changed",
        "An account notice mentions {topic}",
        "The {topic} summary is available",
        "A service update covers {topic}",
        "This message concerns {topic}",
        "A new entry for {topic} is available",
        "The {topic} record needs attention",
    )
    phishing_actions = (
        "confirm whether you requested the change before the review window closes",
        "review the account details and respond if the request is yours",
        "verify the pending settings today to keep the request active",
        "check the password reset request and continue only if you recognize it",
        "confirm the payroll profile change before the stated deadline",
        "review the sign-in notice and use the service contact if it is unexpected",
        "respond to the account update promptly or ask support about the notice",
        "check the delivery preference change and confirm the selected option",
        "review the payment status before the next account update",
        "verify the shared document request using the provided reference",
        "confirm the account reset request or contact the help desk",
        "check the service notice today and decide whether the request is expected",
    )
    legitimate_actions = (
        "the change is recorded; contact support if you did not request it",
        "the reset is complete; use your saved bookmark if you need to sign in",
        "the regular payroll schedule is unchanged; review it in your usual portal",
        "the account notice is informational and no response is required",
        "the service window is scheduled; check your calendar for the time",
        "the payment summary is ready; contact the known service desk with questions",
        "the delivery preference was saved; review it through your usual service",
        "the password update was requested; use a known support channel if unsure",
        "the document was shared by your team; check your normal workspace",
        "the account reset was completed; no password reply is requested",
        "the appointment is confirmed; use your usual calendar for details",
        "the billing note is a receipt; contact the provider using saved details",
    )
    signoffs = (
        "A response may be needed.",
        "No password or code should be sent by reply.",
        "Contact the service using details you already have.",
        "Check the request independently if anything seems unexpected.",
        "This is a simulated example for offline model evaluation.",
        "Do not use the fictional address as a real service.",
    )
    url_forms = (
        " Reference only: hxxps://portal[.]example[.]invalid.",
        " Example address: hxxps://notice[.]example[.]invalid.",
        " No web address is included.",
        " Contact details are not included in this text.",
    )
    urgent_phrases = (
        "Please review when convenient.",
        "Please review today.",
        "This is time-sensitive.",
        "There is no immediate deadline.",
    )
    # Generate unique text-component combinations across the full dataset rather
    # than repeating a small set of templates with a changing sample number.
    combos = [
        (opening, action, signoff)
        for opening in openings
        for action in (*phishing_actions, *legitimate_actions)
        for signoff in signoffs
    ]
    rng.shuffle(combos)
    # Assign links/urgency evenly within each label and campaign, independently
    # of the label so those attributes cannot act as direct proxies.
    combo_index = 0
    for campaign_index, topic in enumerate(topics):
        campaign = f"synthetic-campaign-{campaign_index + 1:02d}"
        context_note = f"Practice context: {contexts[campaign_index]}."
        for offset in range(12):
            channel = "email" if offset % 2 == 0 else "sms"
            subject_options = (
                f"{topic.title()} service update",
                f"Notice: {topic.title()}",
                f"{topic.title()} status information",
                f"Your {topic.title()} summary",
            )
            message_subject = rng.choice(subject_options) if channel == "email" else ""
            for label in ("phishing", "legitimate"):
                opening, action, signoff = combos[combo_index]
                combo_index += 1
                urgency = urgent_phrases[offset % len(urgent_phrases)]
                url_form = url_forms[offset % len(url_forms)]
                # Both labels receive every style of metadata over the complete
                # corpus; the labels are not encoded in subject or sender fields.
                # Make wording cues probabilistic, not a deterministic encoding
                # of the assigned class. Each class uses the other class's
                # phrase family in 45% of non-ambiguous rows.
                own_action_probability = 0.55
                use_phishing_action = (
                    rng.random() < own_action_probability
                    if label == "phishing"
                    else rng.random() >= own_action_probability
                )
                action_pool = phishing_actions if use_phishing_action else legitimate_actions
                action = rng.choice(action_pool)
                if offset < 4:
                    # These paired messages intentionally have the same visible
                    # text but different simulated sender provenance. A text-only
                    # model cannot reliably infer that missing sender context.
                    opening = openings[(campaign_index + offset) % len(openings)]
                    action = (
                        "the account reset status is available; review it using a saved "
                        "bookmark or contact the service team"
                    )
                    signoff = "No password or verification code should be sent by reply."
                body = (
                    f"{opening.format(topic=topic)}. Example variation {offset + 1}. "
                    f"{action.capitalize()}. "
                    f"{urgency} {url_form} {signoff} {context_note}"
                )
                identifier = len(rows) + 1
                has_url = "hxxps://" in body
                sender_type = (
                    (
                        "unfamiliar synthetic domain"
                        if label == "phishing"
                        else "known service (simulated)"
                    )
                    if offset < 4
                    else (
                        "known service (simulated)"
                        if offset % 2 == 0
                        else "unfamiliar synthetic domain"
                    )
                )
                rows.append(
                    {
                        "message_id": f"MSG-{identifier:04d}",
                        "channel": channel,
                        "subject_or_title": message_subject,
                        "body_text": body,
                        "sanitized_url_features": (
                            "defanged example.invalid link" if has_url else "no link"
                        ),
                        "sender_domain_type": sender_type,
                        "label": label,
                        "campaign_group": campaign,
                        "collected_date": (
                            f"2025-{(campaign_index % 12) + 1:02d}-{(offset % 27) + 1:02d}"
                        ),
                    }
                )
    return rows


def generate_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(make_rows())
    frame.to_csv(path, index=False)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATASET_PATH)
    args = parser.parse_args()
    dataset = generate_dataset(args.output)
    print(f"Generated {len(dataset)} synthetic messages at {args.output}")
