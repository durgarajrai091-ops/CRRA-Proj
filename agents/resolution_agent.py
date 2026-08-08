"""
ISDO Lab C4 — Resolution / KB Agent
Searches ChromaDB for matching KB articles and drafts a resolution.
For high-confidence L1 matches, auto-resolves. For low confidence, flags for HITL.

Requires: Lab C1 KB setup must be run first (or run kb_setup.py before this).
"""

import anthropic
import chromadb
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── LOAD CHROMADB KB (built in Lab C1) ───────────────────────────────────────

KB_DIR = Path("data/kb")

def chunk_article(text: str, filename: str) -> list[dict]:
    """Split a markdown article at ## headings. Each section = one chunk.
    (Matches the chunking logic from Lab C1's kb_setup.py exactly.)"""
    chunks = []
    current_lines = []
    current_heading = "Introduction"

    for line in text.split("\n"):
        if line.startswith("## ") and current_lines:
            chunks.append({
                "content": "\n".join(current_lines).strip(),
                "heading": current_heading,
                "filename": filename
            })
            current_lines = []
            current_heading = line[3:].strip()
        current_lines.append(line)

    if current_lines:
        chunks.append({
            "content": "\n".join(current_lines).strip(),
            "heading": current_heading,
            "filename": filename
        })
    return chunks

def build_kb():
    """Load KB articles into ChromaDB, chunked at ## headings (replicates Lab C1 setup)."""
    db = chromadb.Client()
    try:
        kb = db.get_collection("isdo_kb")
        print("KB already loaded.")
        return kb
    except Exception:
        pass

    kb = db.create_collection("isdo_kb")
    docs, ids, metas = [], [], []

    chunk_id = 0
    for md_file in sorted(KB_DIR.glob("*.md")):
        text = md_file.read_text()
        for chunk in chunk_article(text, md_file.name):
            docs.append(chunk["content"])
            ids.append(f"kb_{chunk_id}")
            metas.append({"filename": md_file.name, "heading": chunk["heading"], "source": str(md_file)})
            chunk_id += 1

    if docs:
        kb.add(documents=docs, ids=ids, metadatas=metas)
        print(f"KB loaded: {len(docs)} chunks from {len(list(KB_DIR.glob('*.md')))} articles")
    return kb

KB = build_kb()

# ── TOOL DEFINITIONS ──────────────────────────────────────────────────────────

tools = [
    {
        "name": "search_kb",
        "description": "Search the knowledge base for articles matching the ticket description. Returns top 2 matching articles with similarity scores.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query based on the ticket's short description and symptoms"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "draft_resolution",
        "description": "Draft a resolution message to send to the user based on KB article content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "priority": {"type": "string", "description": "The ticket's priority (P1-P4), as given in the ticket details"},
                "resolution_text": {
                    "type": "string",
                    "description": "Plain-English resolution steps to send to the requester"
                },
                "auto_resolve": {
                    "type": "boolean",
                    "description": "True if this is an L1 issue that can be resolved without human escalation"
                },
                "confidence": {
                    "type": "string",
                    "enum": ["HIGH", "MEDIUM", "LOW"],
                    "description": "HIGH = exact KB match found; MEDIUM = partial match; LOW = no clear match, needs human review"
                },
                "kb_article_used": {"type": "string", "description": "Name of the KB article used"}
            },
            "required": ["ticket_number", "priority", "resolution_text", "auto_resolve", "confidence", "kb_article_used"]
        }
    }
]

# ── TOOL IMPLEMENTATION ───────────────────────────────────────────────────────

def search_kb(query):
    # Search across all chunks, but group results by source article — a ticket's
    # best chunk match (e.g. "Symptoms") often isn't the same chunk that contains
    # the fix (e.g. "Resolution Steps"). Once we know which article is the best
    # match, hand the model the FULL article rather than two arbitrary chunks,
    # so it has complete resolution steps to work with, not just the header.
    raw = KB.query(query_texts=[query], n_results=10)
    best_per_file = {}
    for i, doc in enumerate(raw["documents"][0]):
        meta = raw["metadatas"][0][i] if raw["metadatas"] else {}
        distance = raw["distances"][0][i] if raw.get("distances") else 1.0
        fname = meta.get("filename", f"unknown_{i}")
        if fname not in best_per_file or distance < best_per_file[fname]:
            best_per_file[fname] = distance

    ranked = sorted(best_per_file.items(), key=lambda kv: kv[1])[:2]
    articles = []
    for fname, distance in ranked:
        confidence_score = max(0, 1 - distance)
        full_path = KB_DIR / fname
        full_text = full_path.read_text() if full_path.exists() else ""
        articles.append({
            "article": fname,
            "content_preview": full_text,  # whole article — these are ~2KB, fits easily
            "confidence_score": round(confidence_score, 2)
        })
    return {"articles": articles, "query": query}

def handle_tool(name, inp):
    if name == "search_kb":
        return search_kb(inp["query"])
    elif name == "draft_resolution":
        return inp
    return {"error": "Unknown tool"}

# ── RESOLUTION AGENT ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the ISDO Resolution Agent for Zensar's IT Service Desk.

For each ticket:
1. Use search_kb to find matching knowledge base articles
2. Use draft_resolution to create the resolution with correct confidence level

Confidence rules:
- HIGH confidence + auto_resolve=True: L1 issue, KB article fully covers it, send fix directly
- MEDIUM confidence + auto_resolve=False: KB article partially matches, human should review
- LOW confidence + auto_resolve=False: No clear match, escalate to L2

Priority gate (applies on top of the confidence rules above):
- auto_resolve may ONLY be True when confidence is HIGH *and* ticket priority is P3 or P4.
- P1 and P2 tickets are NEVER auto-resolved, even at HIGH confidence — set
  auto_resolve=False and note in resolution_text that the fix is suggested but requires
  human sign-off because of ticket priority.

Always be specific — include exact steps from the KB article, not vague advice.

Search discipline: call search_kb at most ONCE. Use the ticket's short_description
and description text directly as your query — do not rephrase or retry with a
different phrasing if the first search doesn't return a strong match. A LOW-confidence
result is a valid, expected, correct outcome for tickets with no matching KB article
(e.g. printer, hardware, or unsupported-topic issues) — it means the system is working
as designed, not that you should search again. After ONE search, immediately call
draft_resolution with whatever confidence level the result actually supports."""

def resolve_ticket(ticket_number, short_description, description, category, priority):
    print(f"\n{'='*55}")
    print(f"Resolving: {ticket_number} | Category: {category} | Priority: {priority}")
    print(f"{'='*55}")
    print(f"Issue: {short_description}")

    messages = [{
        "role": "user",
        "content": f"Find a resolution for this ticket:\n\nTicket: {ticket_number}\nCategory: {category}\nPriority: {priority}\nSummary: {short_description}\nDetails: {description}"
    }]

    MAX_ROUNDS = 4  # hard safety cap: never loop forever if the model won't converge
    rounds = 0
    while True:
        rounds += 1
        if rounds > MAX_ROUNDS:
            print(f"  ⚠️  Stopped after {MAX_ROUNDS} tool rounds without a decision — treating as LOW confidence.")
            break

        response = client.messages.create(
            model="claude-opus-5",
            max_tokens=800,
            output_config={"effort": "low"},
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(block.text)
            break

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    result = handle_tool(block.name, block.input)

                    if block.name == "search_kb":
                        print(f"  → KB search: '{block.input.get('query')}'")
                        for art in result.get("articles", []):
                            print(f"     [{art['confidence_score']:.0%}] {art['article']}")

                    elif block.name == "draft_resolution":
                        conf = block.input.get("confidence")
                        auto = block.input.get("auto_resolve")
                        print(f"\n  → Confidence: {conf}  |  Auto-resolve: {auto}")
                        print(f"  → KB Article: {block.input.get('kb_article_used')}")
                        print(f"\n  RESOLUTION DRAFT:")
                        print(f"  {block.input.get('resolution_text')[:500]}")
                        if not auto:
                            print(f"\n  ⚠️  HITL FLAG: Low confidence — human review required before sending.")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })

            messages.append({"role": "user", "content": tool_results})

# ── RUN ON SAMPLE TICKETS ─────────────────────────────────────────────────────

if __name__ == "__main__":
    test_tickets = [
        ("INC0001001", "VPN not connecting after password change",
         "User reports VPN client fails to connect after AD password was reset.", "Network", "P3"),
        ("INC0001006", "Password reset request",
         "User locked out of AD account after 5 failed attempts.", "Access", "P3"),
        ("INC0001002", "Cannot access ERP system - login error",
         "Multiple Finance users unable to login to SAP. Error: DBCON_FAIL.", "Application", "P1"),
    ]

    for number, short_desc, desc, cat, pri in test_tickets:
        resolve_ticket(number, short_desc, desc, cat, pri)