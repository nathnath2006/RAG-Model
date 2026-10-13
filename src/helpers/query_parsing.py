from __future__ import annotations

import re
from dataclasses import dataclass, field

# `strong` aliases identify the company on their own; `weak` ones are common words ("block") and are used only
# when no strong alias matched anywhere in the question.
ALIASES: dict[str, dict[str, list[str]]] = {
    "3M": {"strong": ["3m"]},
    "AES Corporation": {"strong": ["aes corporation", "aes corp", "aes"]},
    "AMD": {"strong": ["amd", "advanced micro devices"]},
    "Activision Blizzard": {"strong": ["activision blizzard", "activision"]},
    "Adobe": {"strong": ["adobe"]},
    "Amazon": {"strong": ["amazon"]},
    "Amcor": {"strong": ["amcor"]},
    "American Express": {"strong": ["american express", "amex"]},
    "American Water Works": {"strong": ["american water works", "american water"]},
    "Best Buy": {"strong": ["best buy"]},
    "Block": {"strong": ["block inc"], "weak": ["block"]},
    "Boeing": {"strong": ["boeing"]},
    "CVS Health": {"strong": ["cvs health", "cvs"]},
    "Coca-Cola": {"strong": ["coca cola"], "weak": ["coke"]},
    "Corning": {"strong": ["corning"]},
    "Costco": {"strong": ["costco"]},
    "Foot Locker": {"strong": ["foot locker", "footlocker"]},
    "General Mills": {"strong": ["general mills"]},
    "JPMorgan": {"strong": ["jpmorgan", "jp morgan", "jpmorgan chase", "jpm"]},
    "Johnson & Johnson": {"strong": ["johnson and johnson", "j and j", "jnj"]},
    "Kraft Heinz": {"strong": ["kraft heinz"]},
    "Lockheed Martin": {"strong": ["lockheed martin", "lockheed"]},
    "MGM Resorts": {"strong": ["mgm resorts", "mgm"]},
    "Microsoft": {"strong": ["microsoft"]},
    "Netflix": {"strong": ["netflix"]},
    "Nike": {"strong": ["nike"]},
    "Paypal": {"strong": ["paypal"]},
    "PepsiCo": {"strong": ["pepsico", "pepsi"]},
    "Pfizer": {"strong": ["pfizer"]},
    "Ulta Beauty": {"strong": ["ulta beauty", "ulta"]},
    "Verizon": {"strong": ["verizon"]},
    "Walmart": {"strong": ["walmart", "wal mart"]},
}

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def normalize(text: str) -> str:
    """Lowercase, '&' -> 'and', possessives dropped, everything else non-alphanumeric -> one space, padded."""
    text = text.lower().replace("&", " and ")
    text = re.sub(r"['’]s\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return f" {text.strip()} "


def _compile(tier: str) -> list[tuple[str, re.Pattern]]:
    out = []
    for company, tiers in ALIASES.items():
        for alias in tiers.get(tier, []):
            out.append((company, re.compile(rf"(?<![a-z0-9]){re.escape(normalize(alias).strip())}(?![a-z0-9])")))
    return out


_STRONG, _WEAK = _compile("strong"), _compile("weak")


def extract_companies(question: str) -> list[str]:
    """Canonical company names mentioned, in alphabetical order. Empty when none is recognised."""
    text = normalize(question)
    found = {c for c, pat in _STRONG if pat.search(text)}
    if not found:
        found = {c for c, pat in _WEAK if pat.search(text)}
    return sorted(found)


def extract_years(question: str) -> list[int]:
    """Years mentioned, ascending: 2022, FY2022 and FY22 / FY '22 all give 2022."""
    years = {int(y) for y in re.findall(r"(?<!\d)(20[0-2]\d)(?!\d)", question)}
    years |= {2000 + int(y) for y in re.findall(r"\bFY\s?['’]?(\d{2})(?!\d)", question, flags=re.I)}
    return sorted(years)


def extract_fiscal_years(question: str) -> list[int]:
    """Years explicitly tagged as fiscal ('FY2022', 'FY 22', 'fiscal year 2022'), ascending."""
    tagged = re.findall(r"\b(?:FY|fiscal(?: year)?)\s?['’]?((?:20)?\d{2})(?!\d)", question, flags=re.I)
    return sorted({int(y) if len(y) == 4 else 2000 + int(y) for y in tagged})


def extract_quarter(question: str) -> int | None:
    """Quarter number from 'Q2' or 'second quarter'; None if absent. Several quarters -> the latest."""
    quarters = {int(q) for q in re.findall(r"\bQ([1-4])\b", question, flags=re.I)}
    quarters |= {ORDINALS[w.lower()] for w in re.findall(r"\b(first|second|third|fourth) quarter\b", question, flags=re.I)}
    return max(quarters) if quarters else None


@dataclass
class ParsedQuery:
    companies: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    quarter: int | None = None
    fiscal_years: list[int] = field(default_factory=list)

    @property
    def year(self) -> int | None:
        """Best single guess of the filing year: the latest fiscal-tagged year, else the latest year mentioned."""
        years = self.fiscal_years or self.years
        return years[-1] if years else None


def parse_query(question: str) -> ParsedQuery:
    return ParsedQuery(extract_companies(question), extract_years(question), extract_quarter(question),
                       extract_fiscal_years(question))


def build_where(parsed: ParsedQuery, use_year: bool = False) -> dict | None:
    """
    Chroma filter from a parsed query; None means "search everything".
    The company is a hard filter. The year is optional because later filings repeat earlier years, so it can
    exclude valid evidence: with use_year the filing year or the one after it is accepted.
    """
    clauses = []
    if parsed.companies:
        clauses.append({"company_name": parsed.companies[0] if len(parsed.companies) == 1
                        else {"$in": parsed.companies}})
    if use_year and parsed.year:
        clauses.append({"year": {"$in": [parsed.year, parsed.year + 1]}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
