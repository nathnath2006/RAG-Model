"""
Tests for src/sections.py (heading detection) and src/ingestion.py (parallel loading).

    python tests/test_sections.py          # no pytest needed
    pytest tests/test_sections.py          # also works

The unit tests use synthetic text built from real pypdf output quirks. The golden test loads real
financebench PDFs and checks the exact heading sequence ("<key>@<page index>"); it is skipped when the
PDFs are not there. First run parses ~8,000 pages (~70s on 8 cores), later runs read Data/cache/pages.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.documents import Document  # noqa: E402

import sections as S  # noqa: E402

PDF_DIR = ROOT / "Data" / "financebench" / "pdfs"
CACHE_DIR = ROOT / "Data" / "cache" / "pages"


def keys(pages: list[str], form: str = "10-K") -> list[str]:
    return [h.key for h in S.find_headings(pages, form)]


def candidate_keys(text: str, form: str = "10-K") -> list[str]:
    return [h.key for h in S._candidates([text], form)]


# --- heading recognition -------------------------------------------------------------------------------

def test_heading_glued_to_previous_sentence():
    # pypdf often drops the line break before a heading; the old ^-anchored regex missed all of these
    assert candidate_keys("headquarters in Arlington, Virginia.Item 3.\nLegal Proceedings See Note 7") == ["3"]
    assert candidate_keys("webcasts.Item\xa01A.\xa0Risk\xa0Factors\nYou should carefully") == ["1A"]
    assert candidate_keys("pension cost by $154 million . Item 7A. Quantitative and Qualitative") == ["7A"]


def test_heading_glued_to_a_word():
    assert candidate_keys("Executive Vice President and Chief Financial OfficerITEM\n9B.\nOTHER\nINFORMATION None.") == ["9B"]


def test_title_split_by_whitespace():
    assert candidate_keys("ITEM\n1A.\nRISK\nFACTORS The risks described below") == ["1A"]
    assert candidate_keys("PART I\nITEM 1. B\nUSINESS\nGENERAL") == ["1"]
    assert candidate_keys("Item 1. \nBusines\n \ns.\n \n3M Company was incorporated") == ["1"]
    assert candidate_keys("ITEM 1A    Risk Factors\nOur business is subject") == ["1A"]  # no punctuation


def test_cross_references_are_rejected():
    for text in [
        "See Item 8 of Part II, “Financial Statements and Supplementary Data”",
        "Refer to Item 1A. Risk Factors for additional information.",
        'we may discuss under "Part I, Item 1A. Risk Factors," and',
        'segments under " Item 1. Business ." and',
        "refer to Item 1A risk Factors and Item 7 Management’s Discussion",
        "Results of Operations (Item 7) included elsewhere",
        "Consolidated Financial Statements in Item 8 of this report.",
        "The information required by this Item 10 is incorporated by reference",
        "In accordance with Item 601(b)(4)(iii)(A) of Regulation S-K",
        "pursuant to Item 9(e) of Schedule 14A",
        "Items 405, 407(d)(4) and 407(d)(5) of Regulation S-K",
    ]:
        assert candidate_keys(text) == [], text


def test_running_page_header_without_title_is_ignored():
    # Microsoft prints "PART II / Item 7" at the top of every page of Item 7
    assert candidate_keys("PART II\nItem 7\n \nrevenue increased 7%") == []


def test_combined_items_keyed_by_first_id():
    assert candidate_keys("PART III\nITEMS 10, 11, 12 and 13. DIRECTORS, EXECUTIVE OFFICERS") == ["10"]
    assert candidate_keys("Items 1 and 2. Business and Properties") == ["1"]


def test_reserved_and_omitted_items():
    assert candidate_keys("ITEM 6.  INTENTIONALLY OMITTED\n") == ["6"]
    assert candidate_keys("Item 6. [Reserved]\n") == ["6"]
    assert candidate_keys("Item 4. (Removed and Reserved)\n") == ["4"]


def test_10q_title_decides_the_part():
    text = ("Item 2. Management’s Discussion and Analysis\n...\n"
            "PART II\nItem 2. Unregistered Sales of Equity Securities and Use of Proceeds\n")
    assert candidate_keys(text, "10-Q") == ["I-2", "II-2"]


def test_8k_items():
    text = "Item 2.02 Results of Operations and Financial Condition.\nItem 9.01 Financial Statements and Exhibits."
    assert candidate_keys(text, "8-K") == ["2.02", "9.01"]
    assert candidate_keys("Item 8.01 Other Events", "10-K") == []  # 8-K ids are not 10-K "Item 8"


# --- selection: table of contents, running headers, order ----------------------------------------------

TITLES = {k: t for k, _, _, t, _ in S.FORM_10K}


def toc_page() -> str:
    return "TABLE OF CONTENTS\n" + "\n".join(f"Item {k}. {TITLES[k]} {n + 3}" for n, k in enumerate(TITLES))


def test_table_of_contents_is_skipped():
    body = [f"Item {k}. {TITLES[k]}\n" + "Body text. " * 50 for k in TITLES]
    hs = S.find_headings(["cover page", toc_page(), *body], "10-K")
    assert [h.key for h in hs] == list(TITLES)
    assert [h.page for h in hs] == list(range(2, 2 + len(TITLES)))


FILLER = ["Body text. " * 50] * 10  # filings are long; the TOC check only applies past 10 pages


def test_repeated_heading_keeps_first_occurrence():
    pages = [toc_page(),
             "Item 1. Business\nWe make things.",
             "Item 1. Business (continued)\nMore things.",  # repeated heading on the next page
             *FILLER,
             "Item 1A. Risk Factors\nRisks."]
    hs = S.find_headings(pages, "10-K")
    assert [(h.key, h.page) for h in hs] == [("1", 1), ("1A", 13)]


def test_complete_toc_does_not_beat_a_weak_body():
    # only 2 body headings recognised vs 23 TOC entries: the TOC run scores higher but is one page long
    pages = [toc_page(), "Item 1. Business\nWe make things.", *FILLER, "Item 1A. Risk Factors\nRisks."]
    assert [(h.key, h.page) for h in S.find_headings(pages, "10-K")] == [("1", 1), ("1A", 12)]


def test_cross_reference_index_only_is_unstructured():
    # Intel / McDonald's: no Item headings in the body, only an index on the last pages
    pages = ["Annual report text. " * 100] * 20 + [toc_page()]
    assert keys(pages) == []


def test_form_of():
    assert S.form_of("APPLE_2022_10K") == "10-K"
    assert S.form_of("APPLE_2023Q3_10Q") == "10-Q"
    assert S.form_of("COSTCO_2023_8K_dated-2023-01-19") == "8-K"
    assert S.form_of("MCDONALDS_8K_dated-2023-02-13") == "8-K"
    assert S.form_of("PEPSICO_2023Q1_EARNINGS") is None
    assert S.form_of("AMD_2022_annualreport", "10k") == "10-K"  # financebench metadata wins


def test_split_by_section_cuts_pages_at_headings():
    texts = [toc_page(),
             "Item 1. Business\nWe sell widgets.",
             "More business.\nItem 1A. Risk Factors\nWidgets may fail.\nItem 1B. Unresolved Staff Comments\nNone.",
             *FILLER,
             "Item 2. Properties\nA warehouse."]
    docs = [Document(page_content=t, metadata={"doc_name": "ACME_2022_10K", "page": i}) for i, t in enumerate(texts)]
    out = S.split_by_section(docs)
    assert [(d.metadata["page"], d.metadata["section"]) for d in out] == [
        (0, "cover"), (1, "1"), (2, "1"), (2, "1A"), (2, "1B"), *[(i, "1B") for i in range(3, 13)], (13, "2")]
    assert out[2].page_content == "More business."
    assert out[3].metadata["section_title"] == "Risk Factors" and out[3].metadata["part"] == "I"
    no_ws = lambda t: "".join(t.split())  # pieces are stripped; no other text may be lost or duplicated
    assert no_ws("".join(d.page_content for d in out)) == no_ws("".join(texts))


def test_split_by_section_unstructured_filing():
    docs = [Document(page_content="Revenue grew 10%.", metadata={"doc_name": "PEPSICO_2023Q1_EARNINGS", "page": 0})]
    assert [d.metadata["section"] for d in S.split_by_section(docs)] == [None]


# --- golden test on real filings -------------------------------------------------------------------------
# GOLDEN_VERIFIED: every heading checked by hand in context (before/after text) to be the body heading,
# not a TOC line or a cross-reference. GOLDEN_HOLDOUT: companies not used while tuning the regex; checked by
# sequence and spot-checked. "" means the filing has no usable Item structure.

GOLDEN_VERIFIED = {
    "3M_2018_10K": "1@3 1A@9 1B@11 2@11 3@11 4@11 5@12 6@13 7@14 7A@50 8@51 9@127 9A@127 9B@127 10@128 11@128 12@129 13@129 14@129 15@130 16@132",
    "APPLE_2022_10K": "1@3 1A@7 1B@19 2@19 3@19 4@19 5@20 6@21 7@22 7A@28 8@30 9@55 9A@55 9B@56 9C@56 10@56 11@56 12@56 13@56 14@56 15@57 16@59",
    "AMAZON_2019_10K": "1@2 1A@5 1B@14 2@15 3@15 4@15 5@16 6@17 7@18 7A@31 8@33 9@70 9A@70 9B@72 10@72 11@72 12@72 13@72 14@72 15@73 16@74",
    "INTEL_2020_10K": "",  # cross-reference index only
    "JPMORGAN_2022_10K": "1@2 1A@10 1B@34 2@34 3@34 4@34 5@35 6@35 7@35 7A@35 8@36 9@36 9A@36 9B@36 9C@36 10@37 11@38 12@38 13@38 14@38 15@39",
    "PFIZER_2017_10K": "1@4 1A@23 1B@36 2@36 3@36 4@36 5@39 6@39 7@39 7A@39 8@39 9@39 9A@40 9B@40 10@41 11@41 12@41 13@41 14@41 15@42 16@46",
    "MICROSOFT_2023_10K": "1@3 1A@22 1B@36 2@36 3@36 4@36 5@37 6@38 7@39 7A@56 8@57 9@98 9A@98 9B@100 9C@100 10@100 11@100 12@100 13@100 14@100 15@101 16@107",
    "COCACOLA_2021_10K": "1@3 1A@12 1B@25 2@26 3@26 4@29 5@31 6@33 7@33 7A@58 8@60 9@130 9A@130 9B@130 9C@130 10@130 11@130 12@130 13@131 14@131 15@131 16@139",
    "NIKE_2021_10K": "1@2 1A@11 1B@25 2@25 3@25 4@25 5@26 6@28 7@29 7A@52 8@54 9@95 9A@95 9B@95 10@96 11@96 12@96 13@96 14@96 15@97 16@101",
    "BOEING_2018_10K": "1@2 1A@6 1B@15 2@15 3@16 4@16 5@17 6@18 7@19 7A@47 8@48 9@120 9A@120 9B@120 10@121 11@123 12@124 13@124 14@124 15@125 16@127",
    "VERIZON_2016_10K": "1@3 1A@19 1B@22 2@22 3@22 4@22 5@23 6@23 7@23 7A@23 8@23 9@23 9A@24 9B@24 10@24 11@25 12@25 13@25 14@25 15@26",
    "CVSHEALTH_2018_10K": "1@2 1A@29 1B@60 2@60 3@61 4@61 5@62 6@63 7@63 7A@63 8@63 9@63 9A@63 9B@64 10@64 11@65 12@65 13@66 14@66 15@67 16@71",
    "PG_E_2020_10K": "1@13 1A@32 1B@55 2@55 3@55 4@56 5@59 6@59 7@59 7A@107 8@108 9@213 9A@213 9B@213 10@214 11@214 12@214 13@215 14@215 15@216 16@231",
    "WALMART_2019_10K": "1@6 1A@14 1B@22 2@23 3@25 4@26 5@27 6@28 7@29 7A@42 8@44 9@80 9A@80 9B@81 10@82 11@82 12@82 13@82 14@82 15@83 16@83",
    "AMERICANEXPRESS_2022_10K": "1@3 1A@24 1B@39 2@39 3@39 4@39 5@40 6@41 7@42 7A@90 8@90 9@157 9A@157 9B@157 9C@157 10@158 14@159 15@160 16@165",
    "ADOBE_2016_10K": "1@2 1A@19 1B@30 2@31 3@32 4@32 5@33 6@35 7@36 7A@56 8@59 9@102 9A@102 9B@102 10@103 11@103 12@103 13@103 14@103 15@103",
    "ORACLE_2020_10K": "1@4 1A@18 1B@34 2@34 3@34 4@34 5@35 6@37 7@38 7A@60 8@62 9@62 9A@62 9B@64 10@65 11@65 12@65 13@65 14@65 15@66",
    "GENERALMILLS_2019_10K": "1@2 1A@7 1B@14 2@14 3@16 4@16 5@16 6@17 7@18 7A@47 8@49 9@108 9A@108 9B@109 10@109 11@110 12@110 13@110 14@110 15@111 16@115",
    "APPLE_2023Q3_10Q": "I-1@3 I-2@16 I-3@21 I-4@21 II-1@22 II-1A@22 II-2@23 II-3@23 II-4@23 II-5@23 II-6@24",
    "JPMORGAN_2023Q1_10Q": "",  # cross-reference index only
    "COSTCO_2023_8K_dated-2023-01-19": "8.01@1 9.01@1",
    "PEPSICO_2023Q1_EARNINGS": "",
}

GOLDEN_HOLDOUT = {
    "ACTIVISIONBLIZZARD_2019_10K": "1@3 1A@9 1B@26 2@26 3@26 4@26 5@27 6@29 7@30 7A@59 8@61 9@61 9A@61 9B@62 10@63 11@63 12@63 13@63 14@63 15@64 16@64",
    "AES_2022_10K": "1@5 1A@59 1B@76 2@76 3@76 4@80 5@81 6@82 7@84 7A@122 8@125 9@203 9A@203 9B@205 9C@205 10@206 11@206 12@206 13@207 14@207 15@208",
    "AMCOR_2020_10K": "1@5 1A@9 1B@19 2@19 3@19 4@19 5@20 6@22 7@23 7A@41 8@43 9@112 9A@112 9B@113 10@113 11@114 12@114 13@114 14@114 15@115 16@119",
    "AMD_2018_10K": "1@3 1A@16 1B@31 2@31 3@31 4@34 5@35 6@37 7@38 7A@48 8@50 9@94 9A@94 9B@95 10@96 11@96 12@96 13@96 14@96 15@97 16@106",
    "AMERICANWATERWORKS_2020_10K": "1@5 1A@26 1B@41 2@41 3@41 4@48 5@49 6@49 7@50 7A@76 8@78 9@131 9A@131 9B@131 10@132 11@132 12@132 13@132 14@132 15@133",
    "BESTBUY_2019_10K": "1@3 1A@6 1B@15 2@16 3@19 4@19 5@23 6@25 7@26 7A@46 8@48 9@91 9A@91 9B@91 10@92 11@92 12@93 13@93 14@93 15@94 16@96",
    "BLOCK_2020_10K": "1@4 1A@17 1B@45 2@45 3@46 4@47 5@48 6@51 7@56 7A@76 8@78 9@138 9A@139 9B@140 10@141 11@141 12@141 13@141 14@141 15@142 16@145",
    "BOSTONPROPERTIES_2015_10K": "1@5 1A@17 1B@33 2@34 3@40 4@40 5@41 6@43 7@47 7A@105 8@107 9@173 9A@173 9B@174 10@175 11@175 12@175 13@176 14@176 15@177",
    "COSTCO_2017_10K": "1@2 1A@7 1B@14 2@15 3@15 4@15 5@16 6@17 7@18 7A@29 8@31 9@31 9A@31 9B@32 10@32 11@32 12@33 13@33 14@33 15@33 16@35",
    "CORNING_2021_10K": "1@1 1A@13 1B@20 2@20 3@20 4@20 5@21 6@22 7@23 7A@49 8@49 9@49 9A@50 9B@50 9C@50 10@51 11@53 12@53 13@54 14@54 15@55 16@58",
    "EBAY_2016_10K": "1@3 1A@8 1B@26 2@27 3@28 4@29 5@30 6@33 7@34 7A@53 8@55 9@55 9A@55 9B@56 10@57 11@57 12@57 13@57 14@57 15@58",
    "FEDEX_2023_10K": "1@4 1A@27 1B@40 2@40 3@44 4@44 5@47 6@47 7@47 7A@73 8@74 9@116 9A@116 9B@116 9C@116 10@117 11@117 12@117 13@117 14@117 15@118 16@128",
    "FOOTLOCKER_2022_10K": "1@3 1A@6 1B@17 2@17 3@18 4@18 5@19 6@21 7@22 7A@37 8@37 9@79 9A@79 9B@81 10@81 11@81 12@81 13@81 14@81 15@81 16@81",
    "JOHNSON_JOHNSON_2019_10K": "1@6 1A@11 1B@16 2@16 3@17 4@17 5@20 6@21 7@22 7A@38 8@39 9@114 9A@114 9B@114 10@114 11@115 12@115 13@115 14@115 15@116 16@116",
    "KRAFTHEINZ_2017_10K": "1@5 1A@10 1B@20 2@20 3@20 4@20 5@21 6@22 7@23 7A@50 8@51 9@113 9A@113 9B@114 10@114 11@114 12@114 13@115 14@115 15@115 16@119",
    "LOCKHEEDMARTIN_2020_10K": "1@2 1A@9 1B@23 2@23 3@24 4@24 5@27 6@29 7@31 7A@62 8@63 9@106 9A@106 9B@108 10@108 11@108 12@109 13@109 14@109 15@110 16@113",
    "MCDONALDS_2022_10K": "",  # cross-reference index only
    "MGMRESORTS_2018_10K": "1@2 1A@12 1B@25 2@26 3@27 4@27 5@28 6@30 7@32 7A@52 8@53 9@103 9A@103 9B@104 10@104 11@104 12@104 13@104 14@104 15@105 16@111",
    "NETFLIX_2017_10K": "1@2 1A@4 1B@14 2@15 3@15 4@15 5@16 6@18 7@20 7A@32 8@32 9@32 9A@33 9B@35 10@36 11@36 12@36 13@36 14@36 15@37 16@38",
    "PAYPAL_2022_10K": "1@3 1A@16 1B@30 2@30 3@31 4@31 5@31 6@32 7@32 7A@52 8@55 9@55 9A@55 9B@55 9C@55 10@55 11@55 12@55 13@56 14@56 15@57 16@125",
    "PEPSICO_2015_10K": "1@2 1A@11 1B@25 2@26 3@27 4@27 5@31 6@34 7@38 7A@128 8@128 9@128 9A@128 9B@129 10@129 11@129 12@129 13@130 14@130 15@131",
    "SALESFORCE_2023_10K": "1@5 1A@11 1B@34 2@34 3@34 4@34 5@37 6@38 7@38 7A@52 8@55 9@93 9A@93 9B@94 9C@94 10@95 11@95 12@95 13@95 14@95 15@96",
    "ULTABEAUTY_2023_10K": "1@4 1A@16 1B@27 2@28 3@29 4@29 5@30 6@33 7@33 7A@45 8@45 9@45 9A@45 9B@46 9C@46 10@46 11@46 12@47 13@47 14@47 15@48 16@81",
    "3M_2023Q2_10Q": "I-1@2 I-2@53 I-3@75 I-4@75 II-1@76 II-1A@76 II-2@84 II-3@84 II-4@84 II-5@84 II-6@85",
    "AMCOR_2023Q2_10Q": "I-1@4 I-2@32 I-3@48 I-4@49 II-1@50 II-1A@50 II-2@50 II-3@50 II-4@50 II-5@50 II-6@51",
    "BESTBUY_2024Q2_10Q": "I-1@2 I-2@13 I-3@23 I-4@23 II-1@23 II-2@24 II-5@24 II-6@24",
    "LOCKHEEDMARTIN_2022Q2_10Q": "I-1@2 I-2@27 I-3@47 I-4@47 II-1@49 II-1A@49 II-2@50 II-6@52",
    "NETFLIX_2023Q2_10Q": "I-2@18 I-3@30 I-4@30 II-1@31 II-1A@32 II-2@32 II-5@32 II-6@32",  # no "Item 1" in the body
    "PAYPAL_2023Q2_10Q": "I-1@3 I-2@48 I-3@66 I-4@68 II-1@69 II-1A@69 II-2@75 II-3@75 II-4@75 II-5@75 II-6@76",
    "Pfizer_2023Q2_10Q": "I-1@7 I-2@40 I-3@62 I-4@62 II-1@63 II-1A@63 II-2@64 II-5@64 II-6@64",
    "WALMART_2024Q1_10Q": "I-1@2 I-2@16 I-3@26 I-4@26 II-1@27 II-1A@30 II-2@30 II-5@30 II-6@34",
    "FOOTLOCKER_2022_10Q": "I-1@3 I-2@22 I-4@33 II-1@33 II-1A@33 II-2@34 II-6@35",  # the filing has no Item 3
    "BESTBUY_2023_8K_dated-2023-04-12": "1.01@1 1.02@1 9.01@1",
    "PEPSICO_2023_8K_dated-2023-05-30": "8.01@1 9.01@2",
    "EBAY_2023Q2_8K": "5.02@1",
}


def test_golden_filings():
    if not PDF_DIR.exists():
        print("  skipped: no PDFs in", PDF_DIR)
        return
    import ingestion

    golden = GOLDEN_VERIFIED | GOLDEN_HOLDOUT
    docs = ingestion.load_pdfs([PDF_DIR / f"{name}.pdf" for name in golden], cache_dir=CACHE_DIR)
    pages: dict[str, list[str]] = {}
    for d in docs:
        pages.setdefault(d.metadata["doc_name"], []).append(d.page_content)

    failures = []
    for name, expected in golden.items():
        got = " ".join(f"{h.key}@{h.page}" for h in S.find_headings(pages[name], S.form_of(name)))
        if got != expected:
            failures.append(f"{name}\n    expected: {expected}\n    got:      {got}")
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}\n{e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
