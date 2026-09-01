"""Corpus characterization pipeline for §5 of the NeurIPS 2026 JuDDGES-PL paper.

Computes descriptive statistics from the two Hugging Face datasets that feed
the paper:

    - JuDDGES/pl-court-raw-enriched   (Polish common-court judgments, ~100K-1M rows)
    - JuDDGES/pl-nsa-enriched         (Polish administrative-court judgments, ~1M-10M rows)

The script is streaming-safe: it never materialises the full dataset, and any
quantile computation that requires materialisation is taken over a bounded
sample. All outputs are written under ``eda/output/{tables,figures}`` with a
top-level ``summary.json`` aggregating headline numbers.

Field-name reference: the Croissant files at
``papers/neurips-2026-juddges-pl/croissant/{pl-court-raw-enriched,pl-nsa-enriched}.json``
are the source of truth for column names. Several list-valued columns are
exposed as JSON-encoded strings on Hugging Face (``legalBases``, ``judges``,
``references``, ``themePhrases``, ``text_legal_bases``, ``extracted_legal_bases``,
``extracted_keywords``, ``related_docket_numbers``); they are decoded
defensively in :func:`_as_list`.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tqdm import tqdm  # noqa: E402

LOG = logging.getLogger("juddges.eda")

RANDOM_STATE = 42

DATASET_IDS: Dict[str, str] = {
    "pl-court": "JuDDGES/pl-court-raw-enriched",
    "pl-nsa": "JuDDGES/pl-nsa-enriched",
}

# Field maps derived from the HF Croissant JSON-LDs. TODO(user): confirm against
# a live row dump; the fields below are taken verbatim from the Croissants.
FIELDS_COURT: Dict[str, str] = {
    "id": "_id",
    "date": "date",
    "year_source": "date",
    "court_level": "court_name",
    "court_id": "courtId",
    "type": "type",
    "content": "content",
    "text": "text",
    "thesis": "thesis",
    "legal_bases_raw": "legalBases",
    "text_legal_bases": "text_legal_bases",
    "references": "references",
    "theme_phrases": "themePhrases",
    "extracted_summary": "extracted_summary",
    "extracted_thesis": "extracted_thesis",
    "extracted_keywords": "extracted_keywords",
    "extracted_title": "extracted_title",
    "factual_state": "factual_state",
    "legal_state": "legal_state",
    "extracted_legal_references": "extracted_legal_references",
}

FIELDS_NSA: Dict[str, str] = {
    "id": "judgment_id",
    "date": None,  # TODO(user): NSA Croissant exposes no plain date column;
                   # confirm whether docket_number / judgment_id encodes year.
    "court_level": "court_type",
    "court_id": "court_name",
    "type": "judgment_type",
    "content": "full_text",
    "text": "full_text",
    "thesis": "thesis",
    "legal_bases_raw": "extracted_legal_bases",
    "extracted_legal_bases": "extracted_legal_bases",
    "extracted_summary": "extracted_summary",
    "extracted_thesis": "extracted_thesis",
    "extracted_keywords": "extracted_keywords",
    "factual_state": "extracted_factual_state",
    "legal_state": "extracted_legal_state",
    "case_type": "case_type_description",
    "keywords_native": "keywords",
    "sentence": "sentence",
    "reasons": "reasons_for_judgment",
    "dissent": "dissenting_opinion",
}

# §5.5 PII heuristics. All patterns are tuned for Polish judgments.
PESEL_RE = re.compile(r"\b(\d{11})\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"\b(?:\+?48[\s-]?)?(?:\d{3}[\s-]?){2}\d{3}\b")
# Heuristic full-name pattern: two consecutive Capitalised tokens with Polish
# diacritics. Will overcount (matches court names, place names, e.g.
# "Sąd Okręgowy"); §5.5 reports it explicitly as a heuristic upper bound.
FULL_NAME_RE = re.compile(
    r"\b[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{2,}\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]{2,}\b"
)

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
# Polish statute citation (e.g. "art. 415 k.c.", "art. 12 ust. 2 ustawy o ...").
STATUTE_RE = re.compile(
    r"art\.?\s*(\d+[a-z]?)(?:\s*§\s*\d+)?(?:\s*ust\.?\s*\d+)?\s*"
    r"(k\.\s*c\.|k\.\s*p\.\s*c\.|k\.\s*k\.|k\.\s*p\.\s*k\.|k\.\s*p\.|"
    r"u\.\s*p\.\s*d\.\s*o\.\s*f\.|ustaw[a-z]*\s+o\s+[\wąćęłńóśźż\s]{3,40}?)",
    re.IGNORECASE,
)
# Court case-citation patterns (signature like "II CSK 123/15").
CASE_CITE_RE = re.compile(r"\b[IVX]{1,4}\s+[A-ZĄĆ]{1,5}\s+\d+/\d{2,4}\b")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def pesel_checksum_ok(value: str) -> bool:
    """Return True if ``value`` is an 11-digit string passing the PESEL check digit."""
    if len(value) != 11 or not value.isdigit():
        return False
    weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
    s = sum(int(value[i]) * w for i, w in enumerate(weights))
    check = (10 - s % 10) % 10
    return check == int(value[10])


def _as_list(value: Any) -> List[Any]:
    """Decode HF list-typed columns that arrive as JSON strings.

    Several JuDDGES columns are exposed as ``sc:Text`` in the Croissant, but
    actually carry JSON-encoded list payloads. Returns ``[]`` for null/empty.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s[0] in "[{":
            try:
                decoded = json.loads(s)
            except json.JSONDecodeError:
                return [s]
            if isinstance(decoded, list):
                return decoded
            return [decoded]
        return [s]
    return [value]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


REFUSAL_MARKERS = (
    "i cannot", "i can't", "i am unable", "as an ai", "nie mogę", "nie jestem w stanie",
    "brak danych", "n/a", "null", "none",
)


def _looks_like_refusal(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return False
    if len(t) < 12 and t in {"n/a", "null", "none", "brak", "-"}:
        return True
    return any(m in t[:200] for m in REFUSAL_MARKERS)


def _extract_year(row: Dict[str, Any], date_field: Optional[str]) -> Optional[int]:
    if date_field and row.get(date_field):
        m = YEAR_RE.search(_as_text(row[date_field]))
        if m:
            return int(m.group(0))
    sig = row.get("signature") or row.get("docket_number") or row.get("judgment_id")
    if sig:
        m = YEAR_RE.search(_as_text(sig))
        if m:
            return int(m.group(0))
    return None


def _classify_court_level(row: Dict[str, Any], dataset_key: str) -> str:
    if dataset_key == "pl-court":
        name = _as_text(row.get("court_name")).lower()
        if "najwyż" in name:
            return "supreme"
        if "apelacyjn" in name:
            return "appellate"
        if "okręgow" in name:
            return "regional"
        if "rejonow" in name:
            return "district"
        return "other"
    name = _as_text(row.get("court_type") or row.get("court_name")).lower()
    if name.startswith("nsa") or "naczelny" in name:
        return "NSA"
    if name.startswith("wsa") or "wojewódzk" in name:
        return "WSA"
    return "other"


def _row_text(row: Dict[str, Any], fields: Dict[str, Optional[str]]) -> str:
    for key in ("text", "content"):
        col = fields.get(key)
        if col and not _is_missing(row.get(col)):
            return _as_text(row[col])
    return ""


def _approx_token_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


def iter_dataset(
    dataset_key: str,
    sample_size: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield rows from a HF dataset in streaming mode.

    Imports ``datasets`` lazily so the module can be syntax-checked without
    the dependency installed.
    """
    from datasets import load_dataset  # type: ignore

    repo_id = DATASET_IDS[dataset_key]
    LOG.info("Streaming %s ...", repo_id)
    ds = load_dataset(repo_id, split="train", streaming=True)
    for i, row in enumerate(ds):
        if sample_size is not None and i >= sample_size:
            break
        yield row


# ---------------------------------------------------------------------------
# §5.1 Coverage
# ---------------------------------------------------------------------------


@dataclass
class CoverageAccumulator:
    by_court_level: Counter = field(default_factory=Counter)
    by_year: Counter = field(default_factory=Counter)
    by_type: Counter = field(default_factory=Counter)
    total: int = 0


def coverage_stats(
    rows: Iterable[Dict[str, Any]],
    dataset_key: str,
    out_dir: Path,
) -> Dict[str, Any]:
    """Counts by court level, year, and type. Writes CSV tables + a year timeline figure."""
    fields_map = FIELDS_COURT if dataset_key == "pl-court" else FIELDS_NSA
    acc = CoverageAccumulator()
    date_field = fields_map.get("date")
    type_field = fields_map.get("type")

    for row in rows:
        acc.total += 1
        acc.by_court_level[_classify_court_level(row, dataset_key)] += 1
        year = _extract_year(row, date_field)
        if year is not None:
            acc.by_year[year] += 1
        if type_field:
            t = _as_text(row.get(type_field)).strip() or "unknown"
            acc.by_type[t] += 1

    tables = out_dir / "tables"
    figures = out_dir / "figures"
    pd.DataFrame(acc.by_court_level.most_common(), columns=["court_level", "count"]).to_csv(
        tables / f"{dataset_key}_coverage_by_court_level.csv", index=False
    )
    pd.DataFrame(sorted(acc.by_year.items()), columns=["year", "count"]).to_csv(
        tables / f"{dataset_key}_coverage_by_year.csv", index=False
    )
    pd.DataFrame(acc.by_type.most_common(50), columns=["type", "count"]).to_csv(
        tables / f"{dataset_key}_coverage_by_type.csv", index=False
    )

    if acc.by_year:
        years, counts = zip(*sorted(acc.by_year.items()))
        fig, ax = plt.subplots()
        ax.plot(years, counts, marker="o")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of judgments")
        ax.set_title(f"{dataset_key}: judgments per year")
        fig.tight_layout()
        _save_figure(fig, figures / f"{dataset_key}_year_timeline")

    return {
        "total": acc.total,
        "by_court_level": dict(acc.by_court_level),
        "year_min": min(acc.by_year) if acc.by_year else None,
        "year_max": max(acc.by_year) if acc.by_year else None,
        "type_top10": acc.by_type.most_common(10),
    }


# ---------------------------------------------------------------------------
# §5.2 Document structure
# ---------------------------------------------------------------------------


def document_structure_stats(
    rows: Iterable[Dict[str, Any]],
    dataset_key: str,
    out_dir: Path,
    sample_for_quantile: int = 50_000,
) -> Dict[str, Any]:
    """Length distributions (chars, approx tokens) and section-presence counts."""
    fields_map = FIELDS_COURT if dataset_key == "pl-court" else FIELDS_NSA
    char_lens: List[int] = []
    token_lens: List[int] = []
    section_present: Counter = Counter()
    sentencja_lens: List[int] = []
    uzasadnienie_lens: List[int] = []
    n = 0

    sentencja_re = re.compile(r"sentencj[aą]\b", re.IGNORECASE)
    uzasadnienie_re = re.compile(r"uzasadnieni[aeu]\b", re.IGNORECASE)
    dissent_re = re.compile(r"zdanie\s+odrębn", re.IGNORECASE)
    glosa_re = re.compile(r"\bglosa\b", re.IGNORECASE)

    rng = random.Random(RANDOM_STATE)
    reservoir: List[Tuple[int, int]] = []

    for row in rows:
        n += 1
        text = _row_text(row, fields_map)
        if not text:
            continue
        c, t = len(text), _approx_token_count(text)
        if len(reservoir) < sample_for_quantile:
            reservoir.append((c, t))
        else:
            j = rng.randrange(n)
            if j < sample_for_quantile:
                reservoir[j] = (c, t)

        if sentencja_re.search(text):
            section_present["sentencja"] += 1
            m = sentencja_re.search(text)
            end = uzasadnienie_re.search(text, m.end()) if m else None
            if m and end:
                sentencja_lens.append(end.start() - m.start())
        if uzasadnienie_re.search(text):
            section_present["uzasadnienie"] += 1
            m = uzasadnienie_re.search(text)
            if m:
                uzasadnienie_lens.append(len(text) - m.start())
        if dissent_re.search(text):
            section_present["dissent"] += 1
        if glosa_re.search(text):
            section_present["glosa"] += 1

    if reservoir:
        char_lens, token_lens = (list(x) for x in zip(*reservoir))

    quantiles = [0.5, 0.9, 0.95, 0.99]
    char_q = np.quantile(char_lens, quantiles).tolist() if char_lens else []
    token_q = np.quantile(token_lens, quantiles).tolist() if token_lens else []

    tables = out_dir / "tables"
    figures = out_dir / "figures"

    pd.DataFrame(
        {
            "metric": ["chars", "tokens"],
            "mean": [
                float(np.mean(char_lens)) if char_lens else 0.0,
                float(np.mean(token_lens)) if token_lens else 0.0,
            ],
            "p50": [char_q[0] if char_q else 0, token_q[0] if token_q else 0],
            "p90": [char_q[1] if char_q else 0, token_q[1] if token_q else 0],
            "p95": [char_q[2] if char_q else 0, token_q[2] if token_q else 0],
            "p99": [char_q[3] if char_q else 0, token_q[3] if token_q else 0],
        }
    ).to_csv(tables / f"{dataset_key}_length_distribution.csv", index=False)

    pd.DataFrame(section_present.most_common(), columns=["section", "count"]).to_csv(
        tables / f"{dataset_key}_section_presence.csv", index=False
    )

    if token_lens:
        fig, ax = plt.subplots()
        ax.hist(np.log10(np.clip(token_lens, 1, None)), bins=50)
        ax.set_xlabel("log10(approx. tokens)")
        ax.set_ylabel("Number of judgments (sampled)")
        ax.set_title(f"{dataset_key}: token-length distribution")
        fig.tight_layout()
        _save_figure(fig, figures / f"{dataset_key}_length_hist")

    ratio = None
    if sentencja_lens and uzasadnienie_lens:
        ratio = float(np.median(sentencja_lens)) / max(float(np.median(uzasadnienie_lens)), 1.0)

    return {
        "n_seen": n,
        "n_sampled": len(reservoir),
        "char_quantiles": dict(zip(("p50", "p90", "p95", "p99"), char_q)),
        "token_quantiles": dict(zip(("p50", "p90", "p95", "p99"), token_q)),
        "section_present": dict(section_present),
        "sentencja_to_uzasadnienie_ratio": ratio,
    }


# ---------------------------------------------------------------------------
# §5.3 Citation patterns
# ---------------------------------------------------------------------------


def citation_patterns(
    rows: Iterable[Dict[str, Any]],
    dataset_key: str,
    out_dir: Path,
) -> Dict[str, Any]:
    """Top-30 cited statutes and case-citation rate."""
    fields_map = FIELDS_COURT if dataset_key == "pl-court" else FIELDS_NSA
    statute_counter: Counter = Counter()
    case_cite_counter: Counter = Counter()
    n = 0
    n_with_case_cite = 0

    structured_field = fields_map.get("legal_bases_raw")

    for row in rows:
        n += 1
        if structured_field:
            for entry in _as_list(row.get(structured_field)):
                key = _statute_key(entry)
                if key:
                    statute_counter[key] += 1
        text = _row_text(row, fields_map)
        if text:
            for m in STATUTE_RE.finditer(text):
                code = re.sub(r"\s+", "", m.group(2)).lower()[:60]
                statute_counter[f"art.{m.group(1)} {code}"] += 1
            cites = CASE_CITE_RE.findall(text)
            if cites:
                n_with_case_cite += 1
                for c in cites:
                    case_cite_counter[c] += 1

    tables = out_dir / "tables"
    figures = out_dir / "figures"
    top_statutes = statute_counter.most_common(30)
    pd.DataFrame(top_statutes, columns=["statute", "count"]).to_csv(
        tables / f"{dataset_key}_top30_statutes.csv", index=False
    )
    pd.DataFrame(case_cite_counter.most_common(50), columns=["case", "count"]).to_csv(
        tables / f"{dataset_key}_case_citations_top50.csv", index=False
    )

    if top_statutes:
        labels, counts = zip(*top_statutes)
        fig, ax = plt.subplots(figsize=(7, 8))
        ax.barh(range(len(labels)), counts)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Citations")
        ax.set_title(f"{dataset_key}: top-30 cited statutes")
        fig.tight_layout()
        _save_figure(fig, figures / f"{dataset_key}_top30_statutes")

    return {
        "n_docs_seen": n,
        "n_unique_statutes": len(statute_counter),
        "case_citation_rate": (n_with_case_cite / n) if n else 0.0,
        "top10_statutes": top_statutes[:10],
    }


def _statute_key(entry: Any) -> Optional[str]:
    """Best-effort canonicalisation of a structured ``legalBases`` entry."""
    if entry is None:
        return None
    if isinstance(entry, str):
        s = entry.strip()
        return s[:120] if s else None
    if isinstance(entry, dict):
        title = (
            entry.get("title")
            or entry.get("name")
            or entry.get("act")
            or entry.get("source")
        )
        article = entry.get("article") or entry.get("art") or entry.get("artykul")
        if title and article:
            return f"art.{article} {title}"[:120]
        if title:
            return _as_text(title)[:120]
        return json.dumps(entry, ensure_ascii=False, sort_keys=True)[:120]
    return _as_text(entry)[:120]


# ---------------------------------------------------------------------------
# §5.4 Extraction-field statistics
# ---------------------------------------------------------------------------

EXTRACTED_FIELDS = (
    "extracted_summary",
    "extracted_thesis",
    "extracted_keywords",
    "extracted_title",
    "factual_state",
    "legal_state",
)


def extraction_field_stats(
    rows: Iterable[Dict[str, Any]],
    dataset_key: str,
    out_dir: Path,
) -> Dict[str, Any]:
    """Length distribution + missingness/refusal rate per LLM-extracted field."""
    fields_map = FIELDS_COURT if dataset_key == "pl-court" else FIELDS_NSA
    counters: Dict[str, Dict[str, Any]] = {
        f: {"n": 0, "missing": 0, "refusal": 0, "lengths": [], "kw_set": set()}
        for f in EXTRACTED_FIELDS
    }
    n = 0
    for row in rows:
        n += 1
        for logical in EXTRACTED_FIELDS:
            col = fields_map.get(logical)
            if not col:
                continue
            value = row.get(col)
            c = counters[logical]
            c["n"] += 1
            if _is_missing(value):
                c["missing"] += 1
                continue
            text = _as_text(value)
            if _looks_like_refusal(text):
                c["refusal"] += 1
            c["lengths"].append(len(text))
            if logical == "extracted_keywords":
                for kw in _as_list(value):
                    if isinstance(kw, str) and kw.strip():
                        c["kw_set"].add(kw.strip().lower())

    rows_out = []
    for f, c in counters.items():
        lens = c["lengths"]
        rows_out.append(
            {
                "field": f,
                "n": c["n"],
                "missing": c["missing"],
                "missing_rate": (c["missing"] / c["n"]) if c["n"] else 0.0,
                "refusal": c["refusal"],
                "refusal_rate": (c["refusal"] / c["n"]) if c["n"] else 0.0,
                "mean_chars": float(np.mean(lens)) if lens else 0.0,
                "p50_chars": float(np.quantile(lens, 0.5)) if lens else 0.0,
                "p95_chars": float(np.quantile(lens, 0.95)) if lens else 0.0,
                "unique_keywords": len(c["kw_set"]) if f == "extracted_keywords" else None,
            }
        )

    df = pd.DataFrame(rows_out)
    df.to_csv(out_dir / "tables" / f"{dataset_key}_extraction_field_stats.csv", index=False)
    return {"n_seen": n, "per_field": rows_out}


# ---------------------------------------------------------------------------
# §5.5 Pseudonymization audit
# ---------------------------------------------------------------------------


def pseudonymization_audit(
    rows: Iterable[Dict[str, Any]],
    dataset_key: str,
    out_dir: Path,
    n: int = 1000,
) -> Dict[str, Any]:
    """Stratified-by-year sample of size ``n``, report PII pattern hit-rates."""
    fields_map = FIELDS_COURT if dataset_key == "pl-court" else FIELDS_NSA
    rng = random.Random(RANDOM_STATE)

    per_year: Dict[Optional[int], List[Dict[str, Any]]] = defaultdict(list)
    seen = 0
    for row in rows:
        seen += 1
        per_year[_extract_year(row, fields_map.get("date"))].append(row)
        if seen >= max(n * 50, 5000):
            break

    selected: List[Dict[str, Any]] = []
    keys = [k for k in per_year.keys() if k is not None] or [None]
    per_bucket = max(1, n // len(keys))
    for k in keys:
        bucket = per_year.get(k, [])
        rng.shuffle(bucket)
        selected.extend(bucket[:per_bucket])
    selected = selected[:n]

    counters: Counter = Counter()
    by_court: Dict[str, Counter] = defaultdict(Counter)
    by_year: Dict[int, Counter] = defaultdict(Counter)

    for row in selected:
        text = _row_text(row, fields_map)
        if not text:
            continue
        court_level = _classify_court_level(row, dataset_key)
        year = _extract_year(row, fields_map.get("date")) or -1
        flags = {
            "pesel": any(pesel_checksum_ok(m) for m in PESEL_RE.findall(text)),
            "email": bool(EMAIL_RE.search(text)),
            "phone": bool(PHONE_RE.search(text)),
            "fullname": bool(FULL_NAME_RE.search(text)),
        }
        for k, v in flags.items():
            if v:
                counters[k] += 1
                by_court[court_level][k] += 1
                by_year[year][k] += 1

    pd.DataFrame(
        [{"pattern": k, "doc_count": v, "rate": v / max(len(selected), 1)} for k, v in counters.items()]
    ).to_csv(out_dir / "tables" / f"{dataset_key}_pseudonym_audit.csv", index=False)

    pd.DataFrame(
        [
            {"court_level": court, **dict(c)}
            for court, c in by_court.items()
        ]
    ).to_csv(out_dir / "tables" / f"{dataset_key}_pseudonym_by_court.csv", index=False)

    pd.DataFrame(
        [
            {"year": yr, **dict(c)}
            for yr, c in sorted(by_year.items())
        ]
    ).to_csv(out_dir / "tables" / f"{dataset_key}_pseudonym_by_year.csv", index=False)

    return {
        "sample_size": len(selected),
        "totals": dict(counters),
        "rates": {k: v / max(len(selected), 1) for k, v in counters.items()},
    }


# ---------------------------------------------------------------------------
# §5.6 Cross-branch comparison
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", re.UNICODE)


def _top_tokens(rows: Iterable[Dict[str, Any]], dataset_key: str, k: int, max_docs: int) -> Counter:
    fields_map = FIELDS_COURT if dataset_key == "pl-court" else FIELDS_NSA
    c: Counter = Counter()
    for i, row in enumerate(rows):
        if i >= max_docs:
            break
        text = _row_text(row, fields_map).lower()
        if not text:
            continue
        c.update(_TOKEN_RE.findall(text))
    return Counter(dict(c.most_common(k)))


def cross_branch_comparison(
    out_dir: Path,
    sample_size: Optional[int],
    top_k: int = 10_000,
    max_docs_per_branch: int = 5_000,
) -> Dict[str, Any]:
    """Vocabulary + statute-citation overlap between common and administrative branches."""
    docs_cap = min(sample_size or max_docs_per_branch, max_docs_per_branch)
    court_tokens = _top_tokens(iter_dataset("pl-court", docs_cap), "pl-court", top_k, docs_cap)
    nsa_tokens = _top_tokens(iter_dataset("pl-nsa", docs_cap), "pl-nsa", top_k, docs_cap)

    overlap = set(court_tokens) & set(nsa_tokens)
    jaccard = len(overlap) / max(len(set(court_tokens) | set(nsa_tokens)), 1)

    court_stats = pd.read_csv(out_dir / "tables" / "pl-court_top30_statutes.csv") \
        if (out_dir / "tables" / "pl-court_top30_statutes.csv").exists() else pd.DataFrame()
    nsa_stats = pd.read_csv(out_dir / "tables" / "pl-nsa_top30_statutes.csv") \
        if (out_dir / "tables" / "pl-nsa_top30_statutes.csv").exists() else pd.DataFrame()
    statute_overlap = (
        len(set(court_stats["statute"]) & set(nsa_stats["statute"]))
        if not court_stats.empty and not nsa_stats.empty
        else None
    )

    pd.DataFrame(
        [
            {"branch": "pl-court", "vocab_size": len(court_tokens)},
            {"branch": "pl-nsa", "vocab_size": len(nsa_tokens)},
            {"branch": "overlap", "vocab_size": len(overlap)},
        ]
    ).to_csv(out_dir / "tables" / "cross_branch_vocab.csv", index=False)

    return {
        "vocab_jaccard_top10k": jaccard,
        "vocab_overlap_size": len(overlap),
        "statute_overlap_top30": statute_overlap,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=200)
    try:
        fig.savefig(base.with_suffix(".pgf"))
    except Exception as exc:  # noqa: BLE001
        LOG.debug("PGF export skipped for %s: %s", base.name, exc)
    plt.close(fig)


def _safe(name: str, fn: Callable[..., Dict[str, Any]], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        LOG.error("Stat group %s failed: %s", name, exc)
        LOG.debug("%s", traceback.format_exc())
        return {"error": str(exc)}


def _materialise(dataset_key: str, sample_size: Optional[int]) -> List[Dict[str, Any]]:
    """Stream the dataset once into memory (bounded by ``sample_size``)."""
    rows: List[Dict[str, Any]] = []
    for row in tqdm(
        iter_dataset(dataset_key, sample_size),
        desc=f"stream:{dataset_key}",
        unit="row",
    ):
        rows.append(row)
    return rows


def run_dataset(dataset_key: str, out_dir: Path, sample_size: Optional[int]) -> Dict[str, Any]:
    """Run all per-dataset stat groups against a single dataset.

    The streamed rows are materialised once (bounded by ``sample_size``) so
    that each stat function gets a fresh iterator without re-streaming.
    """
    rows = _materialise(dataset_key, sample_size)
    LOG.info("Materialised %d rows for %s", len(rows), dataset_key)

    summary: Dict[str, Any] = {"dataset": DATASET_IDS[dataset_key], "rows_seen": len(rows)}
    summary["coverage"] = _safe("coverage", coverage_stats, iter(rows), dataset_key, out_dir)
    summary["document_structure"] = _safe(
        "document_structure", document_structure_stats, iter(rows), dataset_key, out_dir
    )
    summary["citations"] = _safe("citations", citation_patterns, iter(rows), dataset_key, out_dir)
    summary["extraction_fields"] = _safe(
        "extraction_fields", extraction_field_stats, iter(rows), dataset_key, out_dir
    )
    summary["pseudonymization"] = _safe(
        "pseudonymization", pseudonymization_audit, iter(rows), dataset_key, out_dir
    )
    return summary


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (5.5, 3.5),
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "savefig.bbox": "tight",
        }
    )


def _print_versions() -> None:
    import platform

    LOG.info("python=%s", platform.python_version())
    LOG.info("numpy=%s pandas=%s matplotlib=%s", np.__version__, pd.__version__, matplotlib.__version__)
    try:
        import datasets  # type: ignore

        LOG.info("datasets=%s", datasets.__version__)
    except Exception:  # noqa: BLE001
        LOG.info("datasets=<not installed>")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_IDS.keys()),
        help="Run statistics on a single dataset.",
    )
    parser.add_argument("--all", action="store_true", help="Run both datasets + cross-branch.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Cap inspected docs per dataset (default: full streaming).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "output",
        help="Where tables/figures/summary.json are written.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    _configure_matplotlib()
    _print_versions()
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    out_dir: Path = args.output_dir
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    if not args.dataset and not args.all:
        parser.error("must pass --dataset {pl-court,pl-nsa} or --all")

    targets = sorted(DATASET_IDS.keys()) if args.all else [args.dataset]
    summary: Dict[str, Any] = {}
    for key in targets:
        summary[key] = _safe(f"dataset:{key}", run_dataset, key, out_dir, args.sample_size)

    if args.all:
        summary["cross_branch"] = _safe(
            "cross_branch", cross_branch_comparison, out_dir, args.sample_size
        )

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    LOG.info("Wrote %s", out_dir / "summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
