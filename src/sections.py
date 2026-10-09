from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import groupby

from langchain_core.documents import Document

# Items a company may leave empty: "[Reserved]", "(Removed and Reserved)", "Intentionally omitted", "Not applicable".
OMITTED = r"[\[(]?(?:removedand)?reserved|intentionallyomitted|omitted|notapplicable"

# Each form is an ordered list of (key, part, item id, canonical title, title regex).
# The title regex is matched against the text right after "Item <id>", lowercased and with ALL whitespace
# removed: pypdf splits words ("B\nUSINESS", "DISCUSSI ON"), so whitespace can't be trusted.
# Requiring the title is what separates a heading from a cross-reference ("see Item 8 of this report"),
# and it is the only way to tell 10-Q "Part I Item 2 (MD&A)" from "Part II Item 2 (Unregistered Sales)".
FORM_10K = [
    ("1",  "I",   "1",  "Business",                                          r"business"),
    ("1A", "I",   "1A", "Risk Factors",                                      r"riskfactors"),
    ("1B", "I",   "1B", "Unresolved Staff Comments",                         r"unresolved"),
    ("1C", "I",   "1C", "Cybersecurity",                                     r"cybersecurity"),
    ("2",  "I",   "2",  "Properties",                                        r"propert"),
    ("3",  "I",   "3",  "Legal Proceedings",                                 r"legalproceeding"),
    ("4",  "I",   "4",  "Mine Safety Disclosures",                           r"minesafety|submissionofmatters|" + OMITTED),
    ("5",  "II",  "5",  "Market for Registrant's Common Equity",             r"market"),
    ("6",  "II",  "6",  "Selected Financial Data",                           r"selected|" + OMITTED),
    ("7",  "II",  "7",  "Management's Discussion and Analysis",              r"management"),
    ("7A", "II",  "7A", "Quantitative and Qualitative Disclosures About Market Risk", r"quantitative"),
    ("8",  "II",  "8",  "Financial Statements and Supplementary Data",       r"financialstatements|consolidatedfinancial"),
    ("9",  "II",  "9",  "Changes in and Disagreements with Accountants",     r"changes"),
    ("9A", "II",  "9A", "Controls and Procedures",                           r"controls"),
    ("9B", "II",  "9B", "Other Information",                                 r"otherinformation"),
    ("9C", "II",  "9C", "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections", r"disclosureregarding"),
    ("10", "III", "10", "Directors, Executive Officers and Corporate Governance", r"directors"),
    ("11", "III", "11", "Executive Compensation",                            r"executivecompensation"),
    ("12", "III", "12", "Security Ownership of Certain Beneficial Owners",   r"securityownership"),
    ("13", "III", "13", "Certain Relationships and Related Transactions",    r"certainrelationships"),
    ("14", "III", "14", "Principal Accountant Fees and Services",            r"principalaccount"),
    ("15", "IV",  "15", "Exhibits and Financial Statement Schedules",        r"exhibit|financialstatementschedule"),
    ("16", "IV",  "16", "Form 10-K Summary",                                 r"form10-?ksummary|summary"),
]

FORM_10Q = [
    ("I-1",   "I",  "1",  "Financial Statements",                            r"(?:condensed)?(?:consolidated)?financialstatements|condensedconsolidated|consolidatedstatements"),
    ("I-2",   "I",  "2",  "Management's Discussion and Analysis",            r"management"),
    ("I-3",   "I",  "3",  "Quantitative and Qualitative Disclosures About Market Risk", r"quantitative"),
    ("I-4",   "I",  "4",  "Controls and Procedures",                         r"controls"),
    ("II-1",  "II", "1",  "Legal Proceedings",                               r"legalproceeding"),
    ("II-1A", "II", "1A", "Risk Factors",                                    r"riskfactors"),
    ("II-2",  "II", "2",  "Unregistered Sales of Equity Securities and Use of Proceeds", r"unregistered|issuerpurchases|purchasesofequity|changesinsecurities"),
    ("II-3",  "II", "3",  "Defaults Upon Senior Securities",                 r"defaults"),
    ("II-4",  "II", "4",  "Mine Safety Disclosures",                         r"minesafety|submissionofmatters|" + OMITTED),
    ("II-5",  "II", "5",  "Other Information",                               r"otherinformation"),
    ("II-6",  "II", "6",  "Exhibits",                                        r"exhibit"),
]

FORMS = {"10-K": FORM_10K, "10-Q": FORM_10Q}

# "Item" + id anywhere in the text, not only at line start: pypdf often glues a heading to the end of the
# previous line ("...Virginia.Item 3.\nLegal Proceedings"). \s also covers \n and \u00a0 ("ITEM\n1A.").
# "1[0-6]" comes before "[1-9]" so "Item 12" isn't read as "Item 1"; the lookahead rejects "Item 17",
# "Item 9(e)" and 8-K style "Item 8.01". The start must not be inside a word ("Itemized", "systemitem"),
# except a capital I right after a lowercase letter: pypdf can drop the separator ("Financial OfficerITEM 9B").
WORD_START = r"(?:(?<![A-Za-z])|(?<=[a-z])(?-i:(?=I)))"
ITEM = re.compile(WORD_START + r"items?\s*(1[0-6]|[1-9])(?:\s?([A-Ca-c]))?(?![0-9A-Za-z(]|\.\d)", re.I)
ITEM_8K = re.compile(WORD_START + r"items?\s*([1-9]\.\d{2})(?!\d)", re.I)

# A cross-reference is announced by what comes before it: "see Item 7", "in Part II, Item 8",
# 'under "Item 1A', "(Item 7)". Checked on the last characters before the match, whitespace collapsed.
CROSS_REF_CUE = re.compile(
    r"(?:\b(?:see|in|to|under|of|and|or|by|within|at|from|refer|also|including|per|our|this|the|"
    r"caption|captioned|entitled|herein|both|with|on)|[,;(\[“‘]|(?:^|\s)[\"'])\s*$",
    re.I,
)
# "Items 1 and 2. Business and Properties", "ITEMS 10, 11, 12 and 13. DIRECTORS...": skip the extra ids
# so the title check still applies; the section is keyed by the first id.
# (whitespace is already stripped, so the letter must not eat the "a" of "12and13")
JOINED_ID = re.compile(r"^(?:(?:and|&|,|through|-|–)(?:1[0-6]|[1-9])(?:[a-c](?=[.:,&\-–—]|and))?)+")
LEADING_PUNCT = re.compile(r"^[.:\-–—]+")
WHITESPACE = re.compile(r"\s+")

COVER = "cover"  # section label for the text before the first heading (cover page, table of contents)


@dataclass(frozen=True)
class Heading:
    key: str        # unique within the form: "7A" for a 10-K, "II-1A" for a 10-Q, "2.02" for an 8-K
    part: str | None
    item: str
    title: str
    page: int       # index in the page list
    offset: int     # character offset of the match in that page
    rank: int       # position of the key in the form's canonical order


def form_of(doc_name: str, doc_type: str | None = None) -> str | None:
    """Map financebench doc_type ("10k", "10q", "8k", "Earnings") or the file name to a form."""
    label = (doc_type or doc_name).upper().replace("-", "")
    for form in ("10K", "10Q", "8K"):
        if re.search(rf"(?:^|_){form}(?:_|$)", label) or label == form:
            return f"{form[:-1]}-{form[-1]}"
    return None


def _is_cross_ref(text: str, start: int) -> bool:
    before = WHITESPACE.sub(" ", text[max(0, start - 40):start])
    return bool(CROSS_REF_CUE.search(before))


def _title_tail(text: str, end: int) -> str:
    tail = WHITESPACE.sub("", text[end:end + 80]).lower()
    tail = JOINED_ID.sub("", tail)
    return LEADING_PUNCT.sub("", tail)


def _candidates(pages: list[str], form: str) -> list[Heading]:
    """Every 'Item <id>' that is not a cross-reference and is followed by its canonical title."""
    out = []
    if form == "8-K":
        for p, text in enumerate(pages):
            for m in ITEM_8K.finditer(text):
                if not _is_cross_ref(text, m.start()):
                    title = text[m.end():m.end() + 120].strip(" .:\u00a0").split("\n")[0].strip()
                    out.append(Heading(m.group(1), None, m.group(1), title, p, m.start(),
                                       int(m.group(1).replace(".", ""))))
        return out

    schema = FORMS[form]
    for p, text in enumerate(pages):
        for m in ITEM.finditer(text):
            if _is_cross_ref(text, m.start()):
                continue
            item = m.group(1) + (m.group(2) or "").upper()
            tail = _title_tail(text, m.end())
            for rank, (key, part, sid, title, title_re) in enumerate(schema):
                if sid == item and re.match(title_re, tail):
                    out.append(Heading(key, part, item, title, p, m.start(), rank))
                    break
    return out


def _select(cands: list[Heading], before_cost: float = 0.5) -> list[Heading]:
    """
    Pick the headings that form the body of the filing: a run of strictly increasing items that best
    explains every candidate. The table of contents, running page headers ("PART II Item 7" on every page)
    and stray matches are what is left out.

    Score of a run = number of headings kept
                     - candidates inside a kept section whose key differs from that section's key
                       (a TOC entry or a misplaced match; a repeat of the section's own heading is free)
                     - `before_cost` * candidates before the first kept heading (front matter: the TOC)
    Picking a TOC entry as a section start drags the rest of the TOC into that section, which costs more
    than leaving the TOC in front matter, so the body run wins even when both runs are complete.
    Solved exactly with an O(n^2) dynamic program; ties go to the earliest headings.
    """
    n = len(cands)
    if n == 0:
        return []
    ranks = sorted({c.rank for c in cands})
    # prefix[r][k] = number of candidates with rank r among cands[:k]
    prefix = {r: [0] * (n + 1) for r in ranks}
    for k, c in enumerate(cands):
        for r in ranks:
            prefix[r][k + 1] = prefix[r][k] + (c.rank == r)

    def misplaced(i: int, j: int) -> int:  # candidates strictly between i and j not sharing i's key
        return (j - i - 1) - (prefix[cands[i].rank][j] - prefix[cands[i].rank][i + 1])

    best = [0.0] * n
    nxt: list[int | None] = [None] * n
    for i in range(n - 1, -1, -1):
        best[i] = 1 - misplaced(i, n)
        for j in range(i + 1, n):
            if cands[j].rank > cands[i].rank:
                score = 1 + best[j] - misplaced(i, j)
                if score > best[i]:
                    best[i], nxt[i] = score, j

    start = max(range(n), key=lambda i: (best[i] - before_cost * i, -i))
    chosen, i = [], start
    while i is not None:
        chosen.append(cands[i])
        i = nxt[i]
    return chosen


def find_headings(pages: list[str], form: str | None) -> list[Heading]:
    """Section headings of one filing, given its page texts in order. Empty for unsupported forms."""
    if form not in FORMS and form != "8-K":
        return []
    cands = sorted(_candidates(pages, form), key=lambda h: (h.page, h.offset))
    headings = _select(cands)
    # A run packed on one or two pages of a long filing is a table of contents or a cross-reference index,
    # not the body. It can win the selection when few body headings were recognised; drop it and select
    # again rather than tag the whole body with the TOC's last item. 8-Ks are short and have no TOC.
    if form != "8-K" and len(pages) > 10 and _compact(headings):
        toc = {(h.page, h.offset) for h in headings}
        headings = _select([c for c in cands if (c.page, c.offset) not in toc])
        if _compact(headings):
            return []
    # Filings that only list items in an index at the end (Intel 10-K, McDonald's 10-K, JPMorgan 10-Q)
    # have no usable structure: unstructured, rather than the whole filing tagged "cover".
    if headings and headings[0].page > len(pages) / 2:
        return []
    return headings


def _compact(headings: list[Heading]) -> bool:
    return bool(headings) and headings[-1].page - headings[0].page <= 2


def split_by_section(documents: list[Document]) -> list[Document]:
    """
    Cut page Documents at section headings and tag every piece with section metadata.

    Expects one Document per page with "doc_name", "page" and optionally "doc_type" metadata (what
    load_pdfs returns). A page holding several headings is cut into several Documents, so later chunks
    never straddle two items and keep their page number. Text before the first heading is tagged "cover";
    filings without a detectable structure (earnings releases) are tagged section=None.
    """
    out = []
    for doc_name, group in groupby(documents, key=lambda d: d.metadata["doc_name"]):
        pages = sorted(group, key=lambda d: d.metadata["page"])
        form = form_of(doc_name, pages[0].metadata.get("doc_type"))
        headings = find_headings([d.page_content for d in pages], form)
        cuts = {}  # page index -> [(offset, heading)]
        for h in headings:
            cuts.setdefault(h.page, []).append((h.offset, h))

        current = None if not headings else COVER
        for idx, page in enumerate(pages):
            text = page.page_content
            bounds = [(0, current)] + [(off, h) for off, h in cuts.get(idx, [])]
            for (start, sec), (end, _) in zip(bounds, bounds[1:] + [(len(text), None)]):
                piece = text[start:end].strip()
                if piece:
                    out.append(Document(page_content=piece, metadata=page.metadata | _section_meta(sec, form)))
            current = bounds[-1][1]
    return out


def _section_meta(sec: Heading | str | None, form: str | None) -> dict:
    if isinstance(sec, Heading):
        return {"form": form, "section": sec.key, "part": sec.part, "item": sec.item, "section_title": sec.title}
    return {"form": form, "section": sec, "part": None, "item": None, "section_title": None}
