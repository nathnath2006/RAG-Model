from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb
from google import genai
from tqdm.auto import tqdm

import helpers.embedding as eb
from helpers import fusion
from helpers import query_parsing as qp
from helpers import rag

logger = logging.getLogger(__name__)

# Where to look: "none" searches everything, "company" filters on the question's company, "doc" on its exact filing.
SCOPES = ("none", "company", "doc")
# Filters built from the question text alone (query_parsing), as a real user query would be:
# "rules" is the company only, "rules_year" adds the filing year (or the one after it).
RULES_SCOPES = ("rules", "rules_year")
# "fusion": RAG-Fusion (original question + LLM rewrites merged by reciprocal rank fusion), with the filter of "rules_year".
FUSION_SCOPES = ("fusion",)
ALL_SCOPES = SCOPES + RULES_SCOPES + FUSION_SCOPES


def gold_pages(row: dict) -> set[tuple[str, int]]:
    """(doc_name, page) pairs holding the evidence. evidence_page_num is 0-indexed, like the stored `page`."""
    return {(e["doc_name"], e["evidence_page_num"]) for e in row["evidence"]}


def _where(row: dict, scope: str) -> dict | None:
    if scope == "company":
        return {"company_name": row["company"]}
    if scope == "doc":
        return {"doc_name": row["doc_name"]}
    if scope in RULES_SCOPES + FUSION_SCOPES:
        return qp.build_where(qp.parse_query(row["question"]), use_year=scope != "rules")
    return None


def retrieval_eval(
    rows: list[dict],
    collection: chromadb.Collection,
    client: genai.Client,
    scope: str = "doc",
    k: int = 10,
    out_path: str | Path | None = None,
    refresh: bool = False,
) -> tuple[dict, list[dict]]:
    """
    For each question, search and find the rank of the first retrieved piece that sits on a gold evidence page
    (None if absent from the top k). Returns (summary, per-question results).

    Results are saved as JSON at out_path. A later call with the same scope, k and questions loads that file
    instead of searching again, which is free but ignores any change made since (parser, vector store, fusion
    settings): pass refresh=True, or delete the file, to run the queries again.
    """
    if out_path and not refresh and Path(out_path).exists():
        saved = json.loads(Path(out_path).read_text())
        if (saved["summary"].get("scope") == scope and saved["summary"].get("n") == len(rows)
                and f"recall@{k}" in saved["summary"]
                and [r["id"] for r in saved["results"]] == [row["financebench_id"] for row in rows]):
            return saved["summary"], saved["results"]
    results = []
    for row in rows:
        gold = gold_pages(row)
        where = _where(row, scope)
        if scope in FUSION_SCOPES:  # the rewrites are cached next to the results, so a rerun does not pay again
            cache = Path(out_path).parent / "fusion_variants.json" if out_path else None
            hits = fusion.search_fusion(collection, client, row["question"], k=k, where=where, cache_path=cache)
        else:
            hits = eb.search(collection, client, row["question"], k=k, where=where)
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


JUDGE_PROMPT = """
You grade an answer to a question about a company's SEC filings against a reference answer.

Mark the answer correct if it reaches the same result as the reference:
- numbers: the same value once rounding to the precision of the reference and the unit or format are
  allowed for (1,577 million = 1577 USD millions = $1.577 billion);
- yes/no or comparison questions: the same conclusion;
- open-ended reference answers: it conveys the key points of the reference and does not contradict it.
  Extra detail is fine.

Mark it incorrect if the number is wrong or missing, if it contradicts the reference, or if it says that the
context does not contain the answer.

Question: {question}
Reference answer: {gold}
Answer to grade: {answer}

Return JSON only: {{"correct": true or false, "reason": "<one short sentence>"}}
"""


def _load(path: str | Path | None) -> dict[str, dict]:
    return {r["id"]: r for r in json.loads(Path(path).read_text())} if path and Path(path).exists() else {}


def _save(path: str | Path | None, records: dict[str, dict]) -> None:
    if path:
        Path(path).write_text(json.dumps(list(records.values()), indent=1))


def answer_eval(
    rows: list[dict],
    collection: chromadb.Collection,
    client: genai.Client,
    k: int = rag.K,
    use_fusion: bool = False,
    out_path: str | Path | None = None,
    limit: int | None = None,
) -> list[dict]:
    """
    Run the RAG system on the questions (the first `limit` of them, for a trial) and keep, per question, the
    answer, the pages it was built from and whether a gold evidence page was among them. Questions already in
    out_path are skipped, so a run that stops can be resumed and a rerun costs nothing.
    """
    records = _load(out_path)
    selected = rows[:limit]
    cache = Path(out_path).parent / "fusion_variants.json" if out_path and use_fusion else None
    for row in tqdm(selected, desc="answering"):
        if row["financebench_id"] in records:
            continue
        try:
            out = rag.answer(collection, client, row["question"], k=k, use_fusion=use_fusion, cache_path=cache)
        except Exception:
            logger.exception("answering failed for %s", row["financebench_id"])
            continue
        records[row["financebench_id"]] = {
            "id": row["financebench_id"], "question": row["question"], "question_type": row["question_type"],
            "doc_name": row["doc_name"], "gold_answer": row["answer"], "answer": out["answer"],
            "pages": out["pages"], "gold_in_context": bool(gold_pages(row) & set(map(tuple, out["pages"]))),
        }
        _save(out_path, records)
    return [records[r["financebench_id"]] for r in selected if r["financebench_id"] in records]


def judge_answers(records: list[dict], client: genai.Client, out_path: str | Path | None = None) -> list[dict]:
    """
    Add "correct" (True/False, None if the verdict could not be read) and "reason" to each record, using the
    model as a judge against the gold answer. Records already judged in out_path are reused.
    """
    judged = _load(out_path)
    for rec in tqdm(records, desc="judging"):
        if rec["id"] in judged and judged[rec["id"]].get("answer") == rec["answer"]:
            rec.update(correct=judged[rec["id"]]["correct"], reason=judged[rec["id"]]["reason"])
            continue
        prompt = JUDGE_PROMPT.format(question=rec["question"], gold=rec["gold_answer"], answer=rec["answer"])
        try:
            verdict = json.loads(rag.generate(client, prompt, as_json=True))
            rec["correct"], rec["reason"] = bool(verdict["correct"]), str(verdict.get("reason", ""))
        except Exception:
            logger.exception("judging failed for %s", rec["id"])
            rec["correct"], rec["reason"] = None, "verdict could not be read"
        judged[rec["id"]] = rec
        _save(out_path, judged)
    return records
