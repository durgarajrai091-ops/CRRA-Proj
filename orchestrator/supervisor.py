"""
ISDO Lab C6 — LangGraph Orchestrator / Supervisor
Wires all agents into a StateGraph: Triage → Resolution → SLA → Communication.
Conditional routing: auto-resolve if HIGH confidence, else escalate.
Lab C7 adds the HITL gate (already included here as a conditional node).
"""

import anthropic
import chromadb
import json
import os
import re
import requests
import sys
from pathlib import Path

# Running as `python .\orchestrator\supervisor.py` puts only the orchestrator/
# folder on Python's import path, not the project root — so guardrails/ isn't
# found without this. Add the project root explicitly, before any project-local
# import, so the script works the same way no matter where it's launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datetime import datetime
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from guardrails.pii_redactor import redact, restore

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def extract_text(response):
    """Pull the text content from a response, skipping any ThinkingBlocks that
    adaptive thinking may place before the text block."""
    for block in response.content:
        if hasattr(block, "text"):
            return block.text.strip()
    return ""

# ══════════════════════════════════════════════════════
# STATE SCHEMA — shared across all agent nodes
# ══════════════════════════════════════════════════════

class TicketState(TypedDict):
    # Input
    ticket_number: str
    short_description: str
    description: str
    category: str
    priority: str
    sla_due: str
    request_type: str  # e.g. "Access Grant" — only populated on REQ- tickets

    # PII redaction — populated by triage_node, consumed by every node
    # that sends ticket text to Claude, restored just before the final
    # message goes to the requester / ServiceNow.
    short_description_clean: str
    description_clean: str
    pii_mapping: dict

    # Triage Agent outputs
    triage_category: str
    triage_priority: str
    triage_assignment_group: str
    pii_detected: bool
    triage_reasoning: str

    # Resolution Agent outputs
    kb_article: str
    resolution_text: str
    auto_resolve: bool
    confidence: str

    # SLA Agent outputs
    sla_breach_risk: str
    sla_minutes_remaining: int
    escalation_required: bool

    # Communication Agent outputs
    user_message: str

    # Workflow control
    hitl_required: bool
    hitl_reason: str  # Lab C7 — human-readable reason shown at the HITL gate
    hitl_approved: bool
    final_status: str
    audit_log: list

# ══════════════════════════════════════════════════════
# HELPER: CHROMADB KB
# ══════════════════════════════════════════════════════

def chunk_article(text: str, filename: str) -> list[dict]:
    """Split a markdown article at ## headings (matches Lab C1's chunking)."""
    chunks = []
    current_lines = []
    current_heading = "Introduction"
    for line in text.split("\n"):
        if line.startswith("## ") and current_lines:
            chunks.append({"content": "\n".join(current_lines).strip(),
                            "heading": current_heading, "filename": filename})
            current_lines = []
            current_heading = line[3:].strip()
        current_lines.append(line)
    if current_lines:
        chunks.append({"content": "\n".join(current_lines).strip(),
                        "heading": current_heading, "filename": filename})
    return chunks

def load_kb():
    db = chromadb.Client()
    kb_dir = Path("data/kb")
    kb = db.create_collection("isdo_kb_supervisor")
    docs, ids, metas = [], [], []
    chunk_id = 0
    for f in sorted(kb_dir.glob("*.md")):
        for chunk in chunk_article(f.read_text(), f.name):
            docs.append(chunk["content"])
            ids.append(f"kb_{chunk_id}")
            metas.append({"filename": f.name, "heading": chunk["heading"]})
            chunk_id += 1
    if docs:
        kb.add(documents=docs, ids=ids, metadatas=metas)
    return kb

KB = load_kb()
KB_DIR = Path("data/kb")

SLA_HOURS = {"P1": 1, "P2": 4, "P3": 8, "P4": 24}

def log(state, agent, action, detail):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "action": action,
        "detail": detail
    }
    print(f"  [AUDIT] {agent}: {action} — {detail}")
    return entry

# ══════════════════════════════════════════════════════
# NODE 1 — TRIAGE AGENT
# ══════════════════════════════════════════════════════

def triage_node(state: TicketState) -> TicketState:
    print(f"\n▶ TRIAGE AGENT — {state['ticket_number']}")

    # Redact PII before anything reaches Claude. Both fields get their own
    # mapping, merged into one — a name could plausibly show up in either.
    clean_short, map_short = redact(state["short_description"])
    clean_desc, map_desc = redact(state["description"])
    pii_mapping = {**map_short, **map_desc}

    print(f"  Sending to Claude: {clean_desc[:100]}")
    if pii_mapping:
        masked = ", ".join(f"{k}={v}" for k, v in pii_mapping.items())
        print(f"  (PII redacted: {masked})")

    prompt = (f"Classify this IT ticket:\n\nSummary: {clean_short}\n"
              f"Details: {clean_desc}\n\n"
              f"Return JSON with: category, priority, assignment_group, pii_detected, reasoning. "
              f"Category must be exactly ONE of: Network, Application, Hardware, Access, Email, Server, Software. "
              f"Priority: P1=critical/many users, P2=significant, P3=single user, P4=request. "
              f"Reasoning: one short sentence, under 15 words. "
              f"Only return the JSON, no other text.")

    response = client.messages.create(
        model="claude-opus-5", max_tokens=600, output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}]
    )
    text = extract_text(response)

    try:
        # Extract JSON from response
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        result = json.loads(text)
    except Exception as e:
        # Fallback: the model may have returned JSON without fencing it, or
        # added a stray sentence before/after it — search for the object directly
        # instead of giving up immediately.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        result = None
        if match:
            try:
                result = json.loads(match.group(0))
            except Exception:
                result = None
        if result is None:
            print(f"  ⚠️  Triage JSON parse failed ({e}) — raw output: {text[:200]!r}")
            result = {"category": state.get("category", "Unknown"), "priority": state.get("priority", "P3"),
                      "assignment_group": "Service-Desk", "pii_detected": False, "reasoning": "Parse error — defaults used"}

    audit_log = state.get("audit_log", [])
    audit_log.append(log(state, "TriageAgent", "classify_ticket",
                         f"{result.get('category')}/{result.get('priority')} — {result.get('reasoning', '')[:60]}"))

    print(f"  Category: {result.get('category')}  Priority: {result.get('priority')}")
    print(f"  Assign To: {result.get('assignment_group')}  PII: {result.get('pii_detected')}")

    return {**state,
            "short_description_clean": clean_short,
            "description_clean": clean_desc,
            "pii_mapping": pii_mapping,
            "triage_category": result.get("category", state.get("category", "")),
            "triage_priority": result.get("priority", state.get("priority", "P3")),
            "triage_assignment_group": result.get("assignment_group", "Service-Desk"),
            "pii_detected": result.get("pii_detected", False),
            "triage_reasoning": result.get("reasoning", ""),
            "audit_log": audit_log}

# ══════════════════════════════════════════════════════
# NODE 2 — RESOLUTION / KB AGENT
# ══════════════════════════════════════════════════════

def resolution_node(state: TicketState) -> TicketState:
    print(f"\n▶ RESOLUTION AGENT — searching KB")

    results = KB.query(query_texts=[state.get("short_description_clean", state["short_description"])], n_results=10)
    best_per_file = {}
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i] if results["metadatas"] else {}
        distance = results["distances"][0][i] if results.get("distances") else 1.0
        fname = meta.get("filename", "Unknown")
        if fname not in best_per_file or distance < best_per_file[fname]:
            best_per_file[fname] = distance

    if best_per_file:
        article_name, distance = min(best_per_file.items(), key=lambda kv: kv[1])
        full_path = KB_DIR / article_name
        article_content = full_path.read_text() if full_path.exists() else ""
    else:
        article_name, distance, article_content = "None", 1.0, ""

    confidence_score = max(0, 1 - distance)

    confidence = "HIGH" if confidence_score > 0.6 else ("MEDIUM" if confidence_score > 0.35 else "LOW")
    auto_resolve = confidence == "HIGH" and state.get("triage_priority") in ["P3", "P4"]

    prompt = (f"Based on this KB article, draft a clear resolution for the user.\n\n"
              f"KB Article ({article_name}):\n{article_content}\n\n"
              f"Ticket: {state.get('short_description_clean', state['short_description'])}\n"
              f"Write 3-4 steps the user can follow. Plain English. Be specific.")

    response = client.messages.create(
        model="claude-opus-5", max_tokens=400, output_config={"effort": "medium"},
        messages=[{"role": "user", "content": prompt}]
    )
    resolution = extract_text(response)

    # ── A2A: LOW confidence → ask the Knowledge Specialist for a deeper look ──
    if confidence == "LOW":
        print(f"  → Confidence LOW — calling A2A Knowledge Specialist")
        try:
            post_resp = requests.post(
                "http://localhost:8001/tasks",
                json={"query": state.get("short_description_clean", state["short_description"]), "ticket_number": state["ticket_number"]},
                timeout=15
            )
            post_resp.raise_for_status()
            task_id = post_resp.json()["task_id"]
            print(f"  → POST http://localhost:8001/tasks  task_id: {task_id}")

            get_resp = requests.get(f"http://localhost:8001/tasks/{task_id}", timeout=15)
            get_resp.raise_for_status()
            a2a_result = get_resp.json()["result"]
            print(f"  → GET  http://localhost:8001/tasks/{task_id}")

            confidence = a2a_result.get("confidence", confidence)
            confidence_score = a2a_result.get("confidence_score", confidence_score)
            resolution = a2a_result.get("resolution", resolution)
            article_name = a2a_result.get("best_match", article_name)
            auto_resolve = confidence == "HIGH" and state.get("triage_priority") in ["P3", "P4"]

            print(f"  A2A RESULT: Best Match: {article_name}  |  Confidence: {confidence} "
                  f"({a2a_result.get('confidence_score', 0):.0%})")
            print(f"  Updated confidence: {confidence}")

            audit_log = state.get("audit_log", [])
            audit_log.append(log(state, "ResolutionAgent", "a2a_knowledge_specialist",
                                 f"A2A returned {confidence} confidence, article: {article_name}"))
            state = {**state, "audit_log": audit_log}

        except requests.exceptions.RequestException as e:
            print(f"  ⚠️  A2A Knowledge Specialist unreachable ({type(e).__name__}) — "
                  f"falling back to local LOW-confidence result. HITL will handle this ticket.")
            audit_log = state.get("audit_log", [])
            audit_log.append(log(state, "ResolutionAgent", "a2a_knowledge_specialist",
                                 f"A2A call failed ({type(e).__name__}) — kept local LOW confidence"))
            state = {**state, "audit_log": audit_log}

    audit_log = state.get("audit_log", [])
    audit_log.append(log(state, "ResolutionAgent", "final_decision",
                         f"Article: {article_name}  Confidence: {confidence} ({confidence_score:.0%})  AutoResolve: {auto_resolve}"))

    print(f"  FINAL — KB Article: {article_name}")
    print(f"  FINAL — Confidence: {confidence} ({confidence_score:.0%})  |  Auto-resolve: {auto_resolve}")

    return {**state,
            "kb_article": article_name,
            "resolution_text": resolution,
            "auto_resolve": auto_resolve,
            "confidence": confidence,
            "audit_log": audit_log}

# ══════════════════════════════════════════════════════
# NODE 3 — SLA AGENT
# ══════════════════════════════════════════════════════

def sla_node(state: TicketState) -> TicketState:
    print(f"\n▶ SLA AGENT — checking deadline")

    sla_due = state.get("sla_due", "")
    priority = state.get("triage_priority", state.get("priority", "P3"))

    try:
        due_dt = datetime.strptime(sla_due, "%Y-%m-%d %H:%M:%S")
        now = datetime(2024, 1, 15, 10, 30)
        minutes_remaining = int((due_dt - now).total_seconds() / 60)
    except Exception:
        minutes_remaining = 999

    sla_hours = SLA_HOURS.get(priority, 8)
    total_minutes = sla_hours * 60

    if minutes_remaining < 0:
        risk = "BREACHED"
    elif minutes_remaining < total_minutes * 0.2:
        risk = "CRITICAL"
    elif minutes_remaining < total_minutes * 0.5:
        risk = "AT_RISK"
    else:
        risk = "ON_TRACK"

    escalation_required = risk in ["BREACHED", "CRITICAL"] and priority in ["P1", "P2"]

    # ── HITL trigger conditions (Lab C7) ──────────────────────────────────
    # Three independent triggers; any one is sufficient to require a human
    # approval before the ticket reaches the Communication node. Checked in
    # this order only to pick the most relevant reason to display — all three
    # are equally binding, none overrides another.
    triage_category = state.get("triage_category", state.get("category", ""))
    request_type = state.get("request_type", "")
    confidence = state.get("confidence", "")

    hitl_required = False
    hitl_reason = ""

    if escalation_required and priority == "P1":
        hitl_required = True
        hitl_reason = f"P1 SLA {risk} — escalation requires human approval before proceeding"
    elif triage_category == "Access" and request_type == "Access Grant":
        hitl_required = True
        hitl_reason = "ACCESS GRANT — security-sensitive request requires human approval"
    elif confidence == "LOW":
        hitl_required = True
        hitl_reason = "LOW KB confidence — Resolution Agent could not find a clear fix, human review required"

    audit_log = state.get("audit_log", [])
    audit_log.append(log(state, "SLAAgent", "get_sla_status",
                         f"Risk: {risk}  Minutes remaining: {minutes_remaining}  Escalate: {escalation_required}"))

    print(f"  SLA Risk: {risk}  |  Minutes remaining: {minutes_remaining}")
    print(f"  Escalation needed: {escalation_required}  |  HITL required: {hitl_required}")
    if hitl_required:
        print(f"  HITL reason: {hitl_reason}")

    return {**state,
            "sla_breach_risk": risk,
            "sla_minutes_remaining": minutes_remaining,
            "escalation_required": escalation_required,
            "hitl_required": hitl_required or state.get("hitl_required", False),
            "hitl_reason": hitl_reason or state.get("hitl_reason", ""),
            "audit_log": audit_log}

# ══════════════════════════════════════════════════════
# NODE 4 — HITL GATE (Lab C7)
# ══════════════════════════════════════════════════════

def hitl_node(state: TicketState) -> TicketState:
    print(f"\n▶ HITL GATE — human approval required")
    print(f"  {'⚠️  ' * 8}")
    reason = state.get("hitl_reason") or f"SLA {state.get('sla_breach_risk')} — escalation pending"
    print(f"  Ticket:  {state['ticket_number']}  |  Priority: {state.get('triage_priority')}")
    print(f"  Reason:  {reason}")
    print(f"  KB Confidence: {state.get('confidence')}  |  Auto-resolve: {state.get('auto_resolve')}")
    print(f"  {'⚠️  ' * 8}")

    decision = input("  Approve action? [y/n]: ").strip().lower()
    approved = decision == "y"

    audit_log = state.get("audit_log", [])
    audit_log.append(log(state, "HITLGate", "approval_decision",
                         f"Decision: {'APPROVED' if approved else 'REJECTED'} by human operator"))

    print(f"  Decision: {'✅ APPROVED' if approved else '❌ REJECTED'}")
    return {**state, "hitl_approved": approved, "audit_log": audit_log}

# ══════════════════════════════════════════════════════
# NODE 5 — COMMUNICATION AGENT
# ══════════════════════════════════════════════════════

def communication_node(state: TicketState) -> TicketState:
    print(f"\n▶ COMMUNICATION AGENT — drafting user message")

    if state.get("hitl_required") and not state.get("hitl_approved", True):
        msg = (f"Dear User,\n\nYour ticket {state['ticket_number']} has been reviewed. "
               f"Due to the nature of this request, it requires additional approval. "
               f"You will be contacted by the senior support team shortly.")
    elif state.get("auto_resolve"):
        msg = (f"Dear User,\n\nRegarding your ticket {state['ticket_number']}:\n\n"
               f"{state.get('resolution_text', 'Please see the resolution below.')}\n\n"
               f"Please try these steps and let us know if the issue persists.\n\nIT Support Team")
    else:
        msg = (f"Dear User,\n\nYour ticket {state['ticket_number']} has been received and assigned "
               f"to our {state.get('triage_assignment_group', 'support team')}. "
               f"Expected resolution: within {SLA_HOURS.get(state.get('triage_priority', 'P3'), 8)} hours.\n\n"
               f"We will keep you updated. Reference: {state['ticket_number']}\n\nIT Support Team")

    # Restore any PII tokens that made it into the drafted message (e.g. if
    # Claude echoed a [NAME_1]-style token from the redacted resolution_text)
    # back to real values — this is the system-of-record channel, so the
    # requester/ServiceNow are meant to see the real data, unlike Claude.
    msg = restore(msg, state.get("pii_mapping", {}))

    audit_log = state.get("audit_log", [])
    audit_log.append(log(state, "CommunicationAgent", "post_comment",
                         f"Message drafted ({len(msg)} chars). Auto-resolve: {state.get('auto_resolve')}"))

    print(f"\n  USER MESSAGE:\n  {'-'*40}")
    print(f"  {msg[:300]}")
    print(f"  {'-'*40}")

    final_status = "RESOLVED" if state.get("auto_resolve") else "ESCALATED" if state.get("escalation_required") else "IN_PROGRESS"
    return {**state, "user_message": msg, "final_status": final_status, "audit_log": audit_log}

# ══════════════════════════════════════════════════════
# ROUTING FUNCTIONS
# ══════════════════════════════════════════════════════

def route_after_sla(state: TicketState) -> Literal["hitl", "communication"]:
    if state.get("hitl_required"):
        return "hitl"
    return "communication"

def route_after_hitl(state: TicketState) -> Literal["communication"]:
    return "communication"

# ══════════════════════════════════════════════════════
# BUILD GRAPH
# ══════════════════════════════════════════════════════

def build_graph():
    g = StateGraph(TicketState)

    g.add_node("triage", triage_node)
    g.add_node("resolution", resolution_node)
    g.add_node("sla", sla_node)
    g.add_node("hitl", hitl_node)
    g.add_node("communication", communication_node)

    g.set_entry_point("triage")
    g.add_edge("triage", "resolution")
    g.add_edge("resolution", "sla")
    g.add_conditional_edges("sla", route_after_sla, {"hitl": "hitl", "communication": "communication"})
    g.add_edge("hitl", "communication")
    g.add_edge("communication", END)

    return g.compile()

GRAPH = build_graph()

# ══════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    test_tickets = [
        # P3 — should auto-resolve if KB confidence is HIGH
        {"ticket_number": "INC0001001", "short_description": "VPN not connecting after password change",
         "description": "VPN client fails after password reset. Auth failed error.",
         "category": "Network", "priority": "P2", "sla_due": "2024-01-15 14:00:00",
         "audit_log": []},
        # P1 — HITL required
        {"ticket_number": "INC0001002", "short_description": "ERP system down - SAP login failing",
         "description": "Multiple Finance users cannot login to SAP. Error DBCON_FAIL.",
         "category": "Application", "priority": "P1", "sla_due": "2024-01-15 11:00:00",
         "audit_log": []},
        # No KB match — should trigger A2A fallback to Knowledge Specialist
        {"ticket_number": "INC0001099", "short_description": "Cisco Webex not launching on M2 Mac",
         "description": "Webex app fails to open after macOS update on Apple Silicon.",
         "category": "Software", "priority": "P3", "sla_due": "2024-01-15 18:00:00",
         "audit_log": []},
        # Lab C7 — Access Grant request: HITL required regardless of priority/SLA
        {"ticket_number": "REQ-1002", "short_description": "VPN access for new contractor joining project Phoenix",
         "description": "New contractor starting Monday needs VPN access to the Phoenix project network segment. Manager approval attached.",
         "category": "Access", "priority": "P2", "sla_due": "2024-01-15 16:00:00",
         "request_type": "Access Grant", "audit_log": []},
    ]

    for ticket in test_tickets:
        print(f"\n{'═'*55}")
        print(f"PROCESSING TICKET: {ticket['ticket_number']}")
        print(f"{'═'*55}")
        result = GRAPH.invoke(ticket)
        print(f"\n✅ FINAL STATUS: {result.get('final_status')}")
        print(f"   Audit entries: {len(result.get('audit_log', []))}")