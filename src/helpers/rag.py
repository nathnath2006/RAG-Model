from __future__ import annotations

import chromadb
from google import genai
from google.genai import errors, types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from helpers import embedding as eb
from helpers import fusion
from helpers import query_parsing as qp

LLM_MODEL = "gemini-3.5-flash"
K = 10  # pieces handed to the model as context

TEMPLATE = """
Answer the question based on the context below:
{context}
Question: {question}.

In case the answer is not contained within the context, you have to mention it.
"""


def _is_retryable(exc: BaseException) -> bool:
    """Rate limit (429) and server-side errors are worth retrying; a bad request (400) is not."""
    return isinstance(exc, errors.APIError) and (exc.code == 429 or exc.code >= 500)


@retry(retry=retry_if_exception(_is_retryable), wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(6), reraise=True)
def generate(client: genai.Client, prompt: str, as_json: bool = False, model: str = LLM_MODEL) -> str:
    """One model call at temperature 0; the text of the reply ('' when the model returned none)."""
    config = types.GenerateContentConfig(temperature=0, response_mime_type="application/json" if as_json else None)
    return client.models.generate_content(model=model, contents=prompt, config=config).text or ""


def retrieve_hits(
    collection: chromadb.Collection,
    client: genai.Client,
    question: str,
    k: int = K,
    use_fusion: bool = False,
    cache_path=None,
) -> list[dict]:
    """
    Top-k pieces for a question. The company and filing year found in the question restrict the search;
    a filter that matches nothing is relaxed step by step: company and year, company alone, whole collection.
    """
    parsed = qp.parse_query(question)
    tried, hits = [], []
    for where in (qp.build_where(parsed, use_year=True), qp.build_where(parsed), None):
        if where in tried:
            continue
        tried.append(where)
        if use_fusion:
            hits = fusion.search_fusion(collection, client, question, k=k, where=where, cache_path=cache_path)
        else:
            hits = eb.search(collection, client, question, k=k, where=where)
        if hits:
            break
    return hits


def answer(
    collection: chromadb.Collection,
    client: genai.Client,
    question: str,
    k: int = K,
    use_fusion: bool = False,
    cache_path=None,
) -> dict:
    """Answer a question from the indexed filings. Returns the answer and the (doc_name, page) of the context."""
    hits = retrieve_hits(collection, client, question, k, use_fusion, cache_path)
    context = "\n\n".join(h["text"] for h in hits)
    text = generate(client, TEMPLATE.format(context=context, question=question))
    return {"answer": text, "pages": [(h["metadata"]["doc_name"], h["metadata"]["page"]) for h in hits]}
