"""OPTIONAL — query embedding + expansion
Activate only if your data speciality or NFR requires it (files 18, 27). Off by default; CI does not require impl."""
from __future__ import annotations

# Tiny domain thesaurus; extend for your corpus (e.g. transliteration variants for Bangla names).
SYNONYMS = {
    "date": ["day", "when"], "begin": ["start", "commence"], "rain": ["monsoon", "rainfall"],
    "yield": ["harvest", "output"], "time": ["hour", "clock"], "festival": ["celebration", "puja"],
}


def expand(query: str, cfg: dict) -> list[str]:
    """Original query + keyword-only form + one variant per matching synonym."""
    from ..llm.client import content_words

    words = content_words(query)
    out = [query]
    if words and " ".join(words) != query:
        out.append(" ".join(words))
    for w in words:
        for syn in SYNONYMS.get(w, []):
            out.append(" ".join(syn if x == w else x for x in words))
    return list(dict.fromkeys(out))
