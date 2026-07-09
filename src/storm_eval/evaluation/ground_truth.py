"""Hard ground-truth checks against structured NOAA facts.

This is what separates the platform from a self-judging RAG demo: model answers
about storm intensity are checked against HURDAT2-derived truth (the same
``saffir_simpson`` field the parser computes). A category mismatch is a
*verifiable* hallucination, independent of any LLM judge.

The truth table is built in the silver/gold layers; here we load it and compare.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

_CAT = re.compile(r"categor(?:y|ies)\s*([1-5])", re.I)
_CAT_WORD = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
}
_CAT_WORD_RE = re.compile(r"categor(?:y|ies)\s+(one|two|three|four|five)", re.I)


def extract_landfall_category(answer: str) -> Optional[int]:
    """Pull a Saffir-Simpson category (1-5) out of free-text answer."""
    m = _CAT.search(answer)
    if m:
        return int(m.group(1))
    m = _CAT_WORD_RE.search(answer)
    if m:
        return _CAT_WORD[m.group(1).lower()]
    return None


@dataclass
class GroundTruthResult:
    checkable: bool          # did we have a structured fact to check?
    contradicted: bool       # did the answer contradict it?
    expected: Optional[int]
    found: Optional[int]
    field: str = "landfall_category"


def check_landfall_category(answer: str, expected_category: Optional[int]) -> GroundTruthResult:
    if expected_category is None:
        return GroundTruthResult(False, False, None, None)
    found = extract_landfall_category(answer)
    if found is None:
        return GroundTruthResult(True, False, expected_category, None)
    return GroundTruthResult(True, found != expected_category, expected_category, found)


def hallucination_flag(answer: str, truth: Dict) -> Optional[bool]:
    """Return True/False if a structured check applies, else None.

    ``truth`` carries the expected landfall category for the storm the question
    is about (looked up from the gold ground-truth table by the benchmark).
    """
    res = check_landfall_category(answer, truth.get("landfall_category"))
    return res.contradicted if res.checkable and res.found is not None else None
