from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from google import genai
from google.genai import errors, types
from langchain_core.documents import Document
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from tqdm.auto import tqdm

MODEL = "gemini-embedding-001"
DIM = 1536
BATCH_SIZE = 20 #to infer depending on your plan

COLLECTION = "financebench"
CLEANING_VERSION = "1"  # bump when the cleaning or header rules change, so old vectors can be told apart
MIN_CHARS = 50          # pages that are just "Table of Contents" or a lone page number carry no meaning
MAX_CHARS = 6000        # ~2000 tokens at 3 chars/token, the model's input limit; longer pieces get a warning

logger = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    """Rate limit (429) and server-side errors are worth retrying; a bad request (400) is not."""
    return isinstance(exc, errors.APIError) and (exc.code == 429 or exc.code >= 500)


@retry(retry=retry_if_exception(_is_retryable), wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(6), reraise=True)
def _embed_batch(client: genai.Client, batch: list[str], task: str) -> list[list[float]]:
    result = client.models.embed_content(
        model=MODEL,
        contents=batch,
        config=types.EmbedContentConfig(task_type=task, output_dimensionality=DIM),
    )

    if result.embeddings is None:
        raise ValueError("API returned no embeddings.")
    else:
        vectors = [e.values for e in result.embeddings]
        if len(vectors) != len(batch):
            raise ValueError(f"Expected {len(batch)} embeddings, got {len(vectors)}")

        return vectors

def _normalize(vector: list[float]) -> list[float]:
    """Scale to unit length: Gemini only returns normalized vectors at its native 3072 dimensions."""
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


def embed_texts(client: genai.Client, texts: list[str], task: str) -> list[list[float]]:
    """
    Embed texts with Gemini, in the same order as the input: out[i] is the vector of texts[i].
    task is "RETRIEVAL_DOCUMENT" for the pieces you index and "RETRIEVAL_QUERY" for a user question.
    """
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        vectors.extend(_embed_batch(client, texts[start:start + BATCH_SIZE], task))
    return [_normalize(v) for v in vectors]


@dataclass(frozen=True)
class Piece:
    id: str
    text: str        # cleaned text, what you show to the user and the LLM
    embed_text: str  # header + text, what is actually embedded
    metadata: dict


def _clean(text: str) -> str:
    """
    Light cleaning only. Line breaks are kept: financial tables rely on them to keep rows apart.
    """
    text = text.replace("\xa0", " ")
    text = re.sub(r"^\s*Table of Contents\s*", "", text, flags=re.I)  # running header repeated on every page
    text = re.sub(r"[ \t]+", " ", text)                               # runs of spaces
    text = re.sub(r"\n[ \t]*(?=\n)", "\n", text)                     # whitespace-only lines
    text = re.sub(r"\n{3,}", "\n\n", text)                            # long runs of blank lines
    return text.strip()


def _header(md: dict) -> str:
    """One line of context so that look-alike chunks of different companies and years stay distinguishable."""
    parts = [md.get("company_name"), f"{md.get('doc_type', '')} {md.get('year', '')}".strip()]
    if md.get("item"):
        parts.append(f"Item {md['item']}. {md.get('section_title') or ''}".strip())
    elif md.get("section_title"):
        parts.append(md["section_title"])
    return " | ".join(p for p in parts if p)


def prepare(docs: list[Document]) -> list[Piece]:
    """
    Turn section pieces (what sections.split_by_section returns) into records ready to embed and store.
    The id depends only on where the piece is (doc, page, n-th piece of that page), so a rerun targets
    the same record. The text_hash and prepared_at metadata tell you what text, under which rules, and
    when, a stored vector was made from.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    seen: dict[tuple[str, int], int] = {}
    pieces = []
    for d in docs:
        text = _clean(d.page_content)
        if len(text) < MIN_CHARS:
            continue
        md = d.metadata
        key = (md["doc_name"], md["page"])
        n = seen[key] = seen.get(key, -1) + 1

        header = _header(md)
        embed_text = f"{header}\n\n{text}" if header else text
        if len(embed_text) > MAX_CHARS:
            logger.warning("%s page %s piece %s is %d chars, may exceed the model input limit", *key, n, len(embed_text))

        # Chroma metadata accepts only str/int/float/bool, so None values are dropped; "source" is an absolute path
        meta = {k: md[k] for k in ("doc_name", "company_name", "doc_type", "year", "page", "page_label",
                                   "form", "section", "part", "item", "section_title") if md.get(k) is not None}
        meta |= {
            "text_hash": hashlib.sha256(embed_text.encode()).hexdigest(),
            "cleaning_version": CLEANING_VERSION,
            "prepared_at": now,
        }
        pieces.append(Piece(id=f"{key[0]}:p{key[1]}:{n}", text=text, embed_text=embed_text, metadata=meta))
    return pieces


def get_collection(path: str | Path, name: str = COLLECTION) -> chromadb.Collection:
    """
    Open (or create) the persistent collection. It uses cosine distance and records which model and
    dimension filled it: vectors from different models or sizes live in different spaces and must never mix.
    """
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(
        name,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"model": MODEL, "dim": DIM},
    )
    stored = collection.metadata or {}
    if (stored.get("model"), stored.get("dim")) != (MODEL, DIM):
        raise ValueError(f"Collection {name!r} was built with {stored}, not {MODEL} / {DIM}: use a new collection")
    return collection


def index(collection: chromadb.Collection, client: genai.Client, pieces: list[Piece], batch_size: int = BATCH_SIZE) -> int:
    """
    Embed and store pieces, batch by batch. A piece is skipped when its id is already stored with the same
    text_hash, so a rerun after a crash (or with unchanged rules) only does the missing work. Each batch is
    saved before the next one starts. Returns the number of pieces embedded.
    """
    done = 0
    for start in tqdm(range(0, len(pieces), batch_size), desc="indexing"):
        batch = pieces[start:start + batch_size]
        stored = collection.get(ids=[p.id for p in batch], include=["metadatas"])
        known = {id_: (md or {}).get("text_hash") for id_, md in zip(stored["ids"], stored["metadatas"] or [])}
        todo = [p for p in batch if known.get(p.id) != p.metadata["text_hash"]]
        if not todo:
            continue
        vectors = embed_texts(client, [p.embed_text for p in todo], "RETRIEVAL_DOCUMENT")
        collection.upsert(
            ids=[p.id for p in todo],
            embeddings=vectors,
            documents=[p.text for p in todo],
            metadatas=[p.metadata for p in todo],
        )
        done += len(todo)
    return done


def search(collection: chromadb.Collection, client: genai.Client, question: str, k: int = 5, where: dict | None = None) -> list[dict]:
    """
    Top-k pieces for a question, closest first. `where` filters on metadata, e.g. {"company_name": "3M"}.
    The question is embedded here, with the query task type.
    """
    vector = embed_texts(client, [question], "RETRIEVAL_QUERY")[0]
    res = collection.query(query_embeddings=[vector], n_results=k, where=where,
                           include=["documents", "metadatas", "distances"])
    return [
        {"id": id_, "text": text, "metadata": md, "distance": dist}
        for id_, text, md, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
