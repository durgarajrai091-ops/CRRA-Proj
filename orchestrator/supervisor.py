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
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

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
    hitl_approved: bool
    final_status: str
    audit_log: list

# ══════════════════════════════════════════════════════
# HELPER: CHROMADB KB
# ══════════════════════════════════════════════════════

def load_kb():
    db = chromadb.Client()
    kb_dir = Path("data/kb")
    kb = db.create_collection("isdo_kb_supervisor")
    docs, ids, metas = [], [], []
    for i, f in enumerate(sorted(kb_dir.glob("*.md"))):
        docs.append(f.read_text())
        ids.append(f"kb_{i}")
        metas.append({"filename": f.name})
    if docs:
        kb.add(documents=docs, ids=ids, metadatas=metas)
    return kb

KB = load_kb()

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

    prompt = (f"Classify this IT ticket:\n\nSummary: {state['short_description']}\n"
              f"Details: {state['description']}\n\n"
              f"Return JSON with: category, priority, assignment_group, pii_detected, reasoning. "
              f"Priority: P1=critical/many users, P2=significant, P3=single user, P4=request. "
              f"Only return the JSON, no other text.")

    response = client.messages.create(
        model="claude-opus-4-5", max_tokens=300, temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text.strip()

    try:
        # Extract JSON from response
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        result = json.loads(text)
    except Exception:
        result = {"category": state.get("category", "Unknown"), "priority": state.get("priority", "P3"),
                  "assignment_group": "Service-Desk", "pii_detected": False, "reasoning": "Parse error — defaults used"}

    audit_log = state.get("audit_log", [])
    audit_log.append(log(state, "TriageAgent", "classify_ticket",
                         f"{result.get('category')}/{result.get('priority')} — {result.get('reasoning', '')[:60]}"))

    print(f"  Category: {result.get('category')}  Priority: {result.get('priority')}")
    print(f"  Assign To: {result.get('assignment_group')}  PII: {result.get('pii_detected')}")

    return {**state,
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

    results = KB.query(query_texts=[state["short_description"]], n_results=1)
    article_content = results["documents"][0][0] if results["documents"][0] else ""
    article_name = results["metadatas"][0][0].get("filename", "Unknown") if results["metadatas"][0] else "None"
    distance = results["distances"][0][0] if results.get("distances") else 0.5
    confidence_score = max(0, 1 - distance)

    confidence = "HIGH" if confidence_score > 0.6 else ("MEDIUM" if confidence_score > 0.35 else "LOW")
    auto_resolve = confidence == "HIGH" and state.get("triage_priority") in ["P3", "P4"]

    prompt = (f"Based on this KB article, draft a clear resolution for the user.\n\n"
              f"KB Article ({article_name}):\n{article_content[:600]}\n\n"
              f"Ticket: {state['short_description']}\n"
              f"Write 3-4 steps the user can follow. Plain English. Be specific.")

    response = client.messages.create(
        model="claude-opus-4-5", max_tokens=400, temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    resolution = response.content[0].text.strip()

    audit_log = state.get("audit_log", [])
    audit_log.append(log(state, "ResolutionAgent", "search_kb",
                         f"Article: {article_name}  Confidence: {confidence} ({confidence_score:.0%})  AutoResolve: {auto_resolve}"))

    print(f"  KB Article: {article_name}")
    print(f"  Confidence: {confidence} ({confidence_score:.0%})  |  Auto-resolve: {auto_resolve}")

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
    hitl_required = escalation_required and priority == "P1"

    audit_log = state.get("audit_log", [])
    audit_log.append(log(state, "SLAAgent", "get_sla_status",
                         f"Risk: {risk}  Minutes remaining: {minutes_remaining}  Escalate: {escalation_required}"))

    print(f"  SLA Risk: {risk}  |  Minutes remaining: {minutes_remaining}")
    print(f"  Escalation needed: {escalation_required}  |  HITL required: {hitl_required}")

    return {**state,
            "sla_breach_risk": risk,
            "sla_minutes_remaining": minutes_remaining,
            "escalation_required": escalation_required,
            "hitl_required": hitl_required or state.get("hitl_required", False),
            "audit_log": audit_log}

# ══════════════════════════════════════════════════════
# NODE 4 — HITL GATE (Lab C7)
# ══════════════════════════════════════════════════════

def hitl_node(state: TicketState) -> TicketState:
    print(f"\n▶ HITL GATE — human approval required")
    print(f"  {'⚠️  ' * 8}")
    print(f"  Ticket:  {state['ticket_number']}  |  Priority: {state.get('triage_priority')}")
    print(f"  Reason:  SLA {state.get('sla_breach_risk')} — escalation pending")
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
    ]

    for ticket in test_tickets:
        print(f"\n{'═'*55}")
        print(f"PROCESSING TICKET: {ticket['ticket_number']}")
        print(f"{'═'*55}")
        result = GRAPH.invoke(ticket)
        print(f"\n✅ FINAL STATUS: {result.get('final_status')}")
        print(f"   Audit entries: {len(result.get('audit_log', []))}")
