"""
ISDO - Knowledge Base Ingestion & Retrieval Test
--------------------------------------------------
1. Reads all .md files from data/kb/
2. Splits each file at '## ' headings into chunks
3. Stores all chunks in a ChromaDB collection called 'isdo_kb'
4. Runs 4 sample queries and prints the best-matching article + confidence score

Dependencies: chromadb only (pip install chromadb --break-system-packages)
"""

import re
from pathlib import Path

import chromadb

KB_DIR = Path("data/kb")
CHROMA_DIR = Path("chroma_db")          # persistent on-disk store
COLLECTION_NAME = "isdo_kb"

SAMPLE_QUERIES = [
    "My VPN keeps disconnecting on wifi",
    "I forgot my password and got locked out",
    "How do I renew an expired certificate",
    "Self service portal for resetting credentials",
]


# --------------------------------------------------------------------------- #
# 1 & 2. Read + chunk markdown files
# --------------------------------------------------------------------------- #
def parse_markdown(filepath: Path):
    """
    Returns (article_title, chunks) where chunks is a list of dicts:
    {"heading": str, "text": str}

    Splits on lines starting with '## '. Content before the first '## '
    (e.g. the '# Title' line and any intro text) is kept as an 'Overview'
    chunk if it has substantive text; otherwise it's dropped as just a title.
    Files with no '## ' headings at all become a single chunk.
    """
    text = filepath.read_text(encoding="utf-8")

    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    article_title = title_match.group(1).strip() if title_match else filepath.stem

    parts = re.split(r"(?m)^##\s+(.+)$", text)
    preamble = parts[0].strip()

    chunks = []

    if len(parts) == 1:
        # No '##' headings found -> whole file is one chunk
        if text.strip():
            chunks.append({"heading": article_title, "text": text.strip()})
        return article_title, chunks

    # Preamble text (after stripping the '# Title' line) becomes an Overview chunk
    preamble_body = re.sub(r"^#\s+.+$", "", preamble, count=1, flags=re.MULTILINE).strip()
    if preamble_body:
        chunks.append({"heading": "Overview", "text": preamble_body})

    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        chunk_text = f"## {heading}\n\n{body}" if body else f"## {heading}"
        chunks.append({"heading": heading, "text": chunk_text})

    return article_title, chunks


def load_kb_chunks(kb_dir: Path):
    """Reads every .md file in kb_dir and returns flat lists ready for Chroma."""
    ids, documents, metadatas = [], [], []

    md_files = sorted(kb_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {kb_dir.resolve()}")

    for filepath in md_files:
        article_title, chunks = parse_markdown(filepath)
        for idx, chunk in enumerate(chunks):
            ids.append(f"{filepath.stem}::{idx}")
            documents.append(chunk["text"])
            metadatas.append(
                {
                    "article": article_title,
                    "source_file": filepath.name,
                    "heading": chunk["heading"],
                }
            )

    return ids, documents, metadatas


# --------------------------------------------------------------------------- #
# 3. Store chunks in ChromaDB
# --------------------------------------------------------------------------- #
def build_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Fresh start each run so re-ingesting doesn't duplicate/stale data
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine distance -> easy similarity score
    )

    ids, documents, metadatas = load_kb_chunks(KB_DIR)
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"Ingested {len(ids)} chunks from {KB_DIR} into collection '{COLLECTION_NAME}'.\n")
    return collection


# --------------------------------------------------------------------------- #
# 4. Test with sample queries
# --------------------------------------------------------------------------- #
def run_sample_queries(collection):
    print("=" * 70)
    print("SAMPLE QUERY RESULTS")
    print("=" * 70)

    for query in SAMPLE_QUERIES:
        result = collection.query(query_texts=[query], n_results=1)

        print(f"\nQuery: \"{query}\"")

        if not result["ids"][0]:
            print("  No matches found.")
            continue

        best_metadata = result["metadatas"][0][0]
        best_distance = result["distances"][0][0]

        # Cosine distance is in [0, 2]; convert to an intuitive 0-100% similarity
        similarity = max(0.0, 1 - best_distance)
        confidence_pct = similarity * 100

        print(f"  Best match article : {best_metadata['article']}")
        print(f"  Matched section    : {best_metadata['heading']}")
        print(f"  Source file        : {best_metadata['source_file']}")
        print(f"  Confidence score   : {confidence_pct:.1f}%  (cosine distance={best_distance:.4f})")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    collection = build_collection()
    run_sample_queries(collection)