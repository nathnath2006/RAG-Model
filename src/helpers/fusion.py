from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb
from google import genai
from google.genai import types

from helpers import embedding as eb
from helpers import query_parsing as qp

LLM_MODEL = "gemini-3.5-flash"
RECIPR_RANK_F_K = 60

logger = logging.getLogger(__name__)

def generate_queries(client: genai.Client, question: str, n: int = 3) -> list[str]:
    prompt = f"""
    ROLE AND TASK
    You rewrite questions about SEC filings (10-K, 10-Q, 8-K) so that a search engine
    finds the page that answers them. Write {n} different rewrites of the question below.

    DO NOT CHANGE
    Keep exactly as written: company names, years, fiscal periods (FY2018, Q2 2023...),
    numbers, and units (USD millions, percent, billions...). Never add a company, year,
    number or fact that is not in the original question.

    WHAT TO CHANGE
    - Use the vocabulary the filings themselves use. For example, "capital expenditure"
    becomes "purchases of property, plant and equipment", "net PPNE" becomes
    "property, plant and equipment, net", and "sales" becomes "net sales" or "revenue".
    - Where it applies, name the statement likely to hold the answer: income statement,
    balance sheet, cash flow statement, or MD&A.
    - If the question asks for a ratio or a computed metric, name the line items needed
    to compute it (for example operating income and net sales for operating margin).
    - Make the rewrites differ from one another: vary the wording, the statement named,
    or the line item, so each one can retrieve a different page. Do not return
    near-duplicates.

    EXAMPLES
    Question: What was Acme Corp's FY2021 inventory turnover ratio?
    Output: ["Acme Corp FY2021 cost of sales and inventories, consolidated income statement and balance sheet", "Acme Corp 2021 inventory turnover, average inventories, cost of goods sold, MD&A", "Acme Corp fiscal 2021 inventories net, cost of products sold, days inventory outstanding"]

    Question: How much interest expense did Acme Corp report for Q3 2022, in USD millions?
    Output: ["Acme Corp Q3 2022 interest expense, condensed consolidated statement of income, USD millions", "Acme Corp third quarter 2022 interest expense net, other income (expense), in millions", "Acme Corp 2022 three months ended September 30 interest expense, MD&A results of operations"]

    OUTPUT
    Return a JSON list of exactly {n} strings and nothing else: no explanation,
    no markdown, no code fences.

    Question: {question}
    """

    try:
        response = client.models.generate_content(
            model = LLM_MODEL,
            contents= prompt,
            config= types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
        )
        variants = json.loads(response.text)
    except Exception:
        # A failed generation must not stop the run: with no variants, fusion falls back to the plain search
        # it will still be an okay answer.
        logger.exception("variant generation failed for %r", question)
        return []
    if not isinstance(variants, list):
        logger.warning("expected a JSON list of variants for %r, got %s", question, type(variants).__name__)
        return []

    # Keep a rewrite only if it is a new string that still names every year of the original question
    years = set(qp.extract_years(question))
    kept, seen = [], {question.strip().lower()}
    for v in variants:
        if not isinstance(v, str) or not v.strip() or v.strip().lower() in seen:
            continue
        if not years <= set(qp.extract_years(v)):
            logger.warning("variant dropped a year of %r: %r", question, v)
            continue
        seen.add(v.strip().lower())
        kept.append(v.strip())
    return kept[:n]


def reciprocal_ranked_fusion(ranked_lists: list[list[dict]], k: int = RECIPR_RANK_F_K) -> list[dict]:
    """
    Merge several ranked lists of hits (what eb.search returns, best first) into one. A piece scores
    sum(1 / (k + rank)) over the lists it appears in, rank starting at 1, so a piece that is consistently
    well placed beats one that is first in a single list. Ties keep the order of first appearance.
    Returns copies of the hits with a "score" key; "distance" is the one of the first list holding the piece
    and is not comparable across queries.
    """
    scores: dict[str, float] = {}
    first_hit: dict[str, dict] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1 / (k + rank)
            first_hit.setdefault(hit["id"], hit)
    order = sorted(scores, key=scores.get, reverse=True)  # stable: ties stay in first-seen order
    return [{**first_hit[id_], "score": scores[id_]} for id_ in order]


def _variants(client: genai.Client, question: str, n: int, cache_path: str | Path | None) -> list[str]:
    """Variants of a question, read from the JSON cache when present and written to it otherwise."""
    cache: dict[str, list[str]] = {}
    if cache_path and Path(cache_path).exists():
        cache = json.loads(Path(cache_path).read_text())
    if question not in cache:
        variants = generate_queries(client, question, n)
        if not variants:                      # a failed generation is not cached, so a rerun retries it
            return []
        cache[question] = variants
        if cache_path:
            Path(cache_path).write_text(json.dumps(cache, indent=1))
    return cache[question][:n]


def search_fusion(
    collection: chromadb.Collection,
    client: genai.Client,
    question: str,
    k: int = 5,
    n: int = 3,
    depth: int = 20,
    where: dict | None = None,
    cache_path: str | Path | None = None,
) -> list[dict]:
    """
    Top-k pieces for a question by RAG-Fusion: the original question and n rewrites are each searched
    (depth results, same `where` filter, so a rewrite cannot change the company or the year), then merged
    with reciprocal rank fusion. With no rewrites available it is a plain search.
    """
    queries = [question, *_variants(client, question, n, cache_path)]
    ranked_lists = [eb.search(collection, client, q, k=depth, where=where) for q in queries]
    return reciprocal_ranked_fusion(ranked_lists)[:k]
