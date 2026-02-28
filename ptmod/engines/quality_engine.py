from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from harmony_analyzer import MusicQualityChecker
    HARMONY_AVAILABLE = True
except Exception:
    HARMONY_AVAILABLE = False
    MusicQualityChecker = None  # type: ignore

@dataclass
class QualityResult:
    ralph_score: float          # (harmony + melody)/2, 0..100
    overall_score: float        # analyzer overall_score, 0..100
    harmony_score: float
    melody_score: float
    passed_analyzer: bool
    issues: list[str]

def evaluate_patterns_for_ralph(patterns, scale_mode: str, root_note: str) -> QualityResult:
    """
    Returns a Ralph score that is stable and aligned to the user's definition:
      ralph_score = (harmony_score + melody_score)/2.
    Melody score is taken from the analyzer's melody_score (voice-leading proxy).
    """
    if not HARMONY_AVAILABLE:
        return QualityResult(ralph_score=0.0, overall_score=0.0, harmony_score=0.0, melody_score=0.0,
                             passed_analyzer=False, issues=["Harmony analyzer not available"])

    checker = MusicQualityChecker()
    quality, passed = checker.full_quality_check(patterns, scale_mode.lower() if scale_mode else "major", root_note)

    harmony = float(getattr(quality, "harmony_score", 0.0))
    melody  = float(getattr(quality, "melody_score", 0.0))
    overall = float(getattr(quality, "overall_score", 0.0))
    issues  = list(getattr(quality, "issues", []) or [])
    ralph   = (harmony + melody) * 0.5

    return QualityResult(ralph_score=ralph, overall_score=overall, harmony_score=harmony,
                         melody_score=melody, passed_analyzer=bool(passed), issues=issues)
