from __future__ import annotations

import json
import logging
import os
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from pathlib import Path

from langchain_core.documents import Document
from tqdm.auto import tqdm

import helpers.pdf_logging as pf

logger = logging.getLogger(__name__)


def _init_worker(log_file: str | None) -> None:
    """
    Runs once in every worker process to attach the logging handler.
    """
    logger = logging.getLogger("pypdf")
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    if log_file and not logger.handlers:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(pf.JsonFormatter())
        logger.addHandler(handler)


def extract_pdf(path: str) -> list[tuple[str, dict]]:
    """
    Extract one PDF into (text, metadata) pairs, one per page.
    Same text as PyPDFLoader (plain extraction_mode, stripped), but only pypdf is imported in the
    worker, and plain tuples are cheaper to send back to the parent process than Document objects.
    """
    from pypdf import PdfReader

    reader = PdfReader(path)
    labels = reader.page_labels
    total = len(reader.pages)
    return [
        (page.extract_text(extraction_mode="plain").strip(),
         {"source": path, "page": i, "page_label": labels[i], "total_pages": total})
        for i, page in enumerate(reader.pages)
    ]


def _read_cache(cache_dir: Path, pdf: Path) -> list[tuple[str, dict]] | None:
    """Cached pages of `pdf`, or None when missing or stale (the PDF changed since it was cached)."""
    try:
        data = json.loads((cache_dir / f"{pdf.stem}.json").read_text())
    except (OSError, ValueError):
        return None
    stat = pdf.stat()
    if (data.get("size"), data.get("mtime_ns")) != (stat.st_size, stat.st_mtime_ns):
        return None
    return [(text, md) for text, md in data["pages"]]


def _write_cache(cache_dir: Path, pdf: Path, pages: list[tuple[str, dict]]) -> None:
    stat = pdf.stat()
    data = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "pages": pages}
    (cache_dir / f"{pdf.stem}.json").write_text(json.dumps(data))


def load_pdfs(
    paths: list[Path],
    meta: dict[str, dict] | None = None,
    max_workers: int | None = None,
    log_file: Path | None = None,
    cache_dir: Path | None = None,
    executor: Executor | None = None,
) -> list[Document]:
    """
    Load PDFs in parallel, one PDF per task, and return one Document per page in input order.
    
    Largest files are submitted first so a 400-page 10-K does not
    start last and leave the other workers idle at the end.
    `meta` maps a doc_name (file stem) to extra metadata merged into each of its pages.
    `cache_dir` keeps the extracted text as one JSON file per PDF, so re-running the notebook only
    parses PDFs that are new or changed.
    """
    paths = [Path(p) for p in paths]
    meta = meta or {}
    results: list[list[tuple[str, dict]]] = [[] for _ in paths]
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for i, path in enumerate(paths):
            results[i] = _read_cache(cache_dir, path) or []
    todo = sorted((i for i in range(len(paths)) if not results[i]),
                  key=lambda i: paths[i].stat().st_size, reverse=True)

    if todo:  # spawning the pool costs ~1s, skip it on a fully cached run
        pool = executor or ProcessPoolExecutor(
            max_workers=max_workers or os.cpu_count(),
            initializer=_init_worker,
            initargs=(str(log_file) if log_file else None,),
        )
        try:
            futures = {pool.submit(extract_pdf, str(paths[i])): i for i in todo}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="loading pdfs"):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:  # one unreadable PDF (e.g. a truncated download) must not sink the batch
                    logger.error("skipped %s: %s: %s", paths[i].name, type(e).__name__, e)
                    continue
                if cache_dir:
                    _write_cache(cache_dir, paths[i], results[i])
        finally:
            if executor is None:
                pool.shutdown()

    documents = []
    for path, pages in zip(paths, results):
        extra = {"doc_name": path.stem, **meta.get(path.stem, {})}
        documents.extend(Document(page_content=text, metadata=md | extra) for text, md in pages)
    return documents
