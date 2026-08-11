"""Classify semantic-view metric names by presentation role.

Names come from YAML `metrics[*].name` and downstream `explainer_metrics` keys,
which often carry a `table.` prefix (e.g. ``expense_detail.total_admissions`` or
``wgs_mad.claim_count``). The UI needs to pick a formatter (currency, integer,
percentage points) and the correlation agent needs to pick the right explainer
column when composing narrative text. Both used to keep private lookup tuples
that drifted apart; this module is the single source of truth.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple


CLAIM_COUNT_METRIC_CANDIDATES: Tuple[str, ...] = (
    "claim_count",
    "claims_count",
    "claim_line_count",
    "total_claims",
)

ADMISSION_COUNT_METRIC_CANDIDATES: Tuple[str, ...] = (
    "total_admissions",
    "admission_count",
    "admit_count",
    "admissions",
)

PAID_PER_ADMIT_METRIC_CANDIDATES: Tuple[str, ...] = (
    "avg_paid_per_admit",
    "paid_per_admit",
    "paid_per_admission",
    "average_paid_per_admit",
    "avg_paid_per_claim",
    "paid_per_claim",
)

ALLOWED_PER_ADMIT_METRIC_CANDIDATES: Tuple[str, ...] = (
    "avg_allowed_per_admit",
    "allowed_per_admit",
    "allowed_per_admission",
    "avg_allowed_per_claim",
)

PAID_RATIO_METRIC_CANDIDATES: Tuple[str, ...] = (
    "paid_ratio",
    "pay_ratio",
    "paid_to_allowed_ratio",
    "paid_to_allowed",
    "denial_rate",
    "approval_rate",
)


# Human labels for each role, used to build column headers when the YAML does
# not supply an explicit ``label`` for a metric.
ROLE_LABELS: dict = {
    "count": "Count",
    "admissions": "Admits",
    "currency": "Value",
    "ratio": "Ratio",
    "unknown": "Metric",
}


def _last_segment(metric_name: str) -> str:
    """Return the part after the last '.' — the bare metric identifier."""
    return metric_name.rsplit(".", 1)[-1] if "." in metric_name else metric_name


def _matches_candidate(bare_name: str, candidates: Iterable[str]) -> bool:
    lowered = bare_name.lower()
    for candidate in candidates:
        cand = candidate.lower()
        if lowered == cand or lowered.endswith(cand) or cand in lowered:
            return True
    return False


def classify_metric(metric_name: str) -> str:
    """Return one of ``count``, ``admissions``, ``currency``, ``ratio``, ``unknown``.

    Order matters — more specific patterns win. Admission-count metrics are a
    subclass of count semantically, but they render as their own column in the
    legacy UI, so we surface them separately for backwards compat.
    """
    if not metric_name:
        return "unknown"

    bare = _last_segment(metric_name)

    if _matches_candidate(bare, ADMISSION_COUNT_METRIC_CANDIDATES):
        return "admissions"
    if _matches_candidate(bare, CLAIM_COUNT_METRIC_CANDIDATES):
        return "count"
    if _matches_candidate(bare, PAID_RATIO_METRIC_CANDIDATES):
        return "ratio"
    if _matches_candidate(bare, PAID_PER_ADMIT_METRIC_CANDIDATES) or _matches_candidate(
        bare, ALLOWED_PER_ADMIT_METRIC_CANDIDATES
    ):
        return "currency"

    # Fallback heuristics on suffixes / substrings.
    lowered = bare.lower()
    if lowered.endswith("_count") or lowered.endswith("_cnt"):
        return "count"
    if lowered.endswith("_ratio") or lowered.endswith("_rate") or "ratio" in lowered:
        return "ratio"
    if lowered.startswith("total_") or "paid" in lowered or "allowed" in lowered or "billed" in lowered:
        return "currency"

    return "unknown"


def humanize_metric_name(metric_name: str) -> str:
    """Fallback display label when the YAML does not provide one.

    ``wgs_mad.avg_paid_per_claim`` → ``Avg Paid Per Claim``.
    """
    bare = _last_segment(metric_name)
    return " ".join(word.capitalize() for word in bare.replace("-", "_").split("_") if word)


def find_metric_key_by_role(
    available_keys: Iterable[str], role: str
) -> Optional[str]:
    """Return the first key in ``available_keys`` whose classified role matches.

    Useful when the correlation output uses a table-prefixed key like
    ``wgs_mad.claim_count`` and the caller wants a count-role metric without
    knowing the table name.
    """
    for key in available_keys:
        if classify_metric(key) == role:
            return key
    return None


__all__ = [
    "CLAIM_COUNT_METRIC_CANDIDATES",
    "ADMISSION_COUNT_METRIC_CANDIDATES",
    "PAID_PER_ADMIT_METRIC_CANDIDATES",
    "ALLOWED_PER_ADMIT_METRIC_CANDIDATES",
    "PAID_RATIO_METRIC_CANDIDATES",
    "ROLE_LABELS",
    "classify_metric",
    "humanize_metric_name",
    "find_metric_key_by_role",
]
