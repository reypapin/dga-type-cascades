"""Dictionary rule for the type cascade: how much of a domain label is made of words."""

from __future__ import annotations

from pathlib import Path

MAX_WORD_LEN = 24


def load_words(path: Path, min_len: int = 3) -> frozenset[str]:
    words = set()
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        word = line.strip().lower()
        if len(word) >= min_len and word.isascii() and word.isalpha():
            words.add(word)
    return frozenset(words)


def main_label(domain: str) -> str:
    labels = [part for part in str(domain).strip().lower().split(".") if part]
    if not labels:
        return ""
    candidates = labels[:-1] or labels
    return max(candidates, key=len)


def coverage(label: str, words: frozenset[str]) -> float:
    """Fraction of the label's letters covered by non-overlapping dictionary words."""
    s = str(label).lower()
    letters = sum(c.isalpha() for c in s)
    if letters == 0:
        return 0.0
    best = [0] * (len(s) + 1)
    for i in range(1, len(s) + 1):
        best[i] = best[i - 1]
        for j in range(max(0, i - MAX_WORD_LEN), i):
            if i - j > best[i] - best[j] and s[j:i] in words:
                best[i] = best[j] + (i - j)
    return best[len(s)] / letters
