from __future__ import annotations

import json
from pathlib import Path

import chromadb
from google import genai

import helpers.embedding as eb

# Where to look: "none" searches everything, "company" filters on the question's company, "doc" on its exact filing.
SCOPES = ("none", "company", "doc")


def gold_pages(row: dict) -> set[tuple[str, int]]:
    """(doc_name, page) pairs holding the evidence. evidence_page_num is 0-indexed, like the stored `page`."""
    return {(e["doc_name"], e["evidence_page_num"]) for e in row["evidence"]}


def _where(row: dict, scope: str) -> dict | None:
    if scope == "company":
        return {"company_name": row["company"]}
    if scope == "doc":
        return {"doc_name": row["doc_name"]}
    return None


def retrieval_eval(
    rows: list[dict],
    collection: chromadb.Collection,
    client: genai.Client,
    scope: str = "doc",
    k: int = 10,
    out_path: str | Path | None = None,
) -> tuple[dict, list[dict]]:
    """
    For each question, search and find the rank of the first retrieved piece that sits on a gold evidence page
    (None if absent from the top k). Returns (summary, per-question results); results are also dumped as JSON
    when out_path is given, so they can be inspected without paying for the queries again.
    """
    results = []
    for row in rows:
        gold = gold_pages(row)
        hits = eb.search(collection, client, row["question"], k=k, where=_where(row, scope))
        pages = [(h["metadata"]["doc_name"], h["metadata"]["page"]) for h in hits]
        rank = next((i + 1 for i, p in enumerate(pages) if p in gold), None)
        results.append({
            "id": row["financebench_id"], "question": row["question"], "doc_name": row["doc_name"],
            "question_type": row["question_type"], "gold": sorted(gold), "retrieved": pages, "rank": rank,
        })

    n = len(results)
    summary = {"scope": scope, "n": n, "mrr": sum(1 / r["rank"] for r in results if r["rank"]) / n}
    for cut in (1, 3, 5, k):
        summary[f"recall@{cut}"] = sum(1 for r in results if r["rank"] and r["rank"] <= cut) / n
    if out_path:
        Path(out_path).write_text(json.dumps({"summary": summary, "results": results}, indent=1))
    return summary, results
