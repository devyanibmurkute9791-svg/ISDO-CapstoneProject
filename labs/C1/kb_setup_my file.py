#!/usr/bin/env python3
"""
ISDO Knowledge Base Loader & Search Test
------------------------------------------
1. Reads all .md files from data/kb/
2. Splits each file at '## ' headings into chunks
3. Stores chunks in a ChromaDB collection called 'isdo_kb'
4. Runs 4 sample queries and prints the best-matching article + confidence

Requires only the `chromadb` package (chromadb's bundled default embedding
function is used, so no extra embedding library is needed):
    pip install chromadb
"""

import os
import re
import glob
import chromadb

KB_DIR = "data/kb"
PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "isdo_kb"

SAMPLE_QUERIES = [
    "How do I reset my VPN connection?",
    "My password is not working, what should I do?",
    "Application keeps crashing on startup",
    "How to request access to a shared drive?",
]


def split_into_chunks(text: str, filename: str):
    """
    Split a markdown document at '## ' headings.
    Returns (doc_title, [{heading, text}, ...]).
    Any content before the first '##' (e.g. the '# Title' + intro) becomes
    its own chunk so nothing is lost.
    """
    title_match = re.match(r"^#\s+(.+)", text.strip())
    doc_title = title_match.group(1).strip() if title_match else filename

    parts = re.split(r"(?m)^##\s+", text)

    chunks = []
    intro = parts[0].strip()
    if intro:
        chunks.append({"heading": doc_title, "text": intro})

    for part in parts[1:]:
        lines = part.split("\n", 1)
        heading = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        chunks.append({"heading": heading, "text": f"## {heading}\n\n{body}".strip()})

    return doc_title, chunks


def load_kb(kb_dir: str):
    """Read every .md file in kb_dir and return (ids, documents, metadatas)."""
    ids, documents, metadatas = [], [], []

    md_files = sorted(glob.glob(os.path.join(kb_dir, "*.md")))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in '{kb_dir}'")

    for filepath in md_files:
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        doc_title, chunks = split_into_chunks(text, filename)

        for i, chunk in enumerate(chunks):
            ids.append(f"{filename}::chunk{i}")
            documents.append(chunk["text"])
            metadatas.append({
                "source_file": filename,
                "article_title": doc_title,
                "heading": chunk["heading"],
            })

    return ids, documents, metadatas


def build_collection():
    """Create (or reset) the 'isdo_kb' collection and load all KB chunks into it."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    # Start clean each run so re-running the script doesn't duplicate chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # cosine space makes the confidence-score math below intuitive
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids, documents, metadatas = load_kb(KB_DIR)
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    n_articles = len(set(m["source_file"] for m in metadatas))
    print(f"Loaded {len(documents)} chunks from {n_articles} article(s) into '{COLLECTION_NAME}'.\n")
    return collection


def run_sample_queries(collection):
    print("Running sample queries:")
    print("-" * 50)
    for query in SAMPLE_QUERIES:
        result = collection.query(query_texts=[query], n_results=1)

        best_meta = result["metadatas"][0][0]
        distance = result["distances"][0][0]
        # cosine distance = 1 - cosine similarity, so similarity is the confidence
        confidence = max(0.0, 1 - distance)

        print(f"Query      : {query}")
        print(f"Best match : {best_meta['article_title']}  ({best_meta['source_file']})")
        print(f"Section    : {best_meta['heading']}")
        print(f"Confidence : {confidence:.2%}")
        print()


if __name__ == "__main__":
    kb_collection = build_collection()
    run_sample_queries(kb_collection)