"""Template deduplication performed before any train/test partitioning."""

from difflib import SequenceMatcher

import pandas as pd

from src.preprocessing import normalize_text


def deduplicate_templates(
    frame: pd.DataFrame, similarity_threshold: float = 0.92
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Keep the first record in each exact or near-duplicate text cluster.

    Similarity checks compare normalized subject + body text. The threshold is
    deliberately conservative; this is a template leakage safeguard, not a
    semantic plagiarism detector.
    """
    kept: list[int] = []
    signatures: list[tuple[str, str, str]] = []
    exact_removed = 0
    near_removed = 0
    conflicting_exact_duplicates = 0
    conflicting_near_duplicates = 0
    for index, row in frame.iterrows():
        signature = normalize_text(
            f"{row.get('channel', '')} {row.get('subject_or_title', '')} "
            f"{row.get('body_text', '')}"
        )
        if not signature:
            signature = f"empty:{index}"
        label = str(row.get("label", ""))
        exact_same_label = any(
            signature == earlier and label == earlier_label
            for earlier, earlier_label, _ in signatures
        )
        if exact_same_label:
            exact_removed += 1
            continue
        same_label_near_match = any(
            label == earlier_label
            and SequenceMatcher(None, signature, earlier).ratio() >= similarity_threshold
            for earlier, earlier_label, _ in signatures
        )
        if same_label_near_match:
            near_removed += 1
            continue
        if any(signature == earlier and label != earlier_label for earlier, earlier_label, _ in signatures):
            conflicting_exact_duplicates += 1
        elif any(
            label != earlier_label
            and signature != earlier
            and SequenceMatcher(None, signature, earlier).ratio() >= similarity_threshold
            for earlier, earlier_label, _ in signatures
        ):
            conflicting_near_duplicates += 1
        kept.append(index)
        signatures.append((signature, label, str(row.get("campaign_group", ""))))
    result = frame.loc[kept].copy().reset_index(drop=True)
    report: dict[str, object] = {
        "exact_duplicates_removed": exact_removed,
        "near_duplicates_removed": near_removed,
        "cross_label_exact_duplicates_preserved": conflicting_exact_duplicates,
        "cross_label_near_duplicates_preserved": conflicting_near_duplicates,
        "rows_after_deduplication": int(len(result)),
        "similarity_threshold": similarity_threshold,
        "comparison_basis": "NFKC-normalized, case-folded channel, subject and body",
    }
    return result, report


def remove_train_holdout_near_duplicates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    similarity_threshold: float = 0.92,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop training rows too similar to the untouched campaign holdout."""
    test_signatures = [
        normalize_text(
            f"{row.channel} {row.subject_or_title} {row.body_text}"
        )
        for row in test.itertuples()
    ]
    keep_indices: list[int] = []
    removed_exact = 0
    removed_near = 0
    for index, row in train.iterrows():
        signature = normalize_text(
            f"{row.get('channel', '')} {row.get('subject_or_title', '')} "
            f"{row.get('body_text', '')}"
        )
        scores = [
            SequenceMatcher(None, signature, heldout).ratio()
            for heldout in test_signatures
        ]
        highest = max(scores, default=0.0)
        if highest >= 1.0:
            removed_exact += 1
        elif highest >= similarity_threshold:
            removed_near += 1
        else:
            keep_indices.append(index)
    filtered = train.loc[keep_indices].copy().reset_index(drop=True)
    if set(filtered["campaign_group"]) & set(test["campaign_group"]):
        raise AssertionError("Campaign leakage detected after deduplication.")
    return filtered, {
        "train_holdout_exact_matches_removed": removed_exact,
        "train_holdout_near_matches_removed": removed_near,
        "train_holdout_similarity_threshold": similarity_threshold,
    }


def split_by_campaign(
    frame: pd.DataFrame, test_fraction: float = 0.25, seed: int = 37
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by complete campaign groups so no group can leak across partitions."""
    campaigns = sorted(frame["campaign_group"].astype(str).unique())
    if len(campaigns) < 4:
        raise ValueError("At least four campaign groups are required for holdout evaluation.")
    import numpy as np

    rng = np.random.default_rng(seed)
    shuffled = np.array(campaigns, dtype=object)
    rng.shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * test_fraction))
    test_groups = set(shuffled[:test_count])
    train = frame[~frame["campaign_group"].isin(test_groups)].copy()
    test = frame[frame["campaign_group"].isin(test_groups)].copy()
    if set(train["campaign_group"]) & set(test["campaign_group"]):
        raise AssertionError("Campaign leakage detected in split.")
    if set(train["label"]) != {"phishing", "legitimate"} or set(test["label"]) != {
        "phishing",
        "legitimate",
    }:
        raise ValueError("Campaign split must contain both labels in train and holdout.")
    return train.reset_index(drop=True), test.reset_index(drop=True)
