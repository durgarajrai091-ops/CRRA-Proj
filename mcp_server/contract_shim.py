"""
CRRA Lab C2 — Mock Contract API

Serves the vendor contract portfolio from data/contracts.csv over a small
Flask REST surface, computing the derived risk fields (notice window state,
approval band, licence utilisation) the procurement policy depends on.

All derived-field arithmetic happens here, against a fixed simulated review
date, so the agent in later labs never has to do date math and everyone's
output is identical regardless of when they run it.
"""

import csv
import os
from datetime import datetime, timedelta

from flask import Flask, jsonify, request

app = Flask(__name__)

SIMULATED_TODAY = datetime(2025, 4, 1).date()
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "contracts.csv")

contracts = {}


def parse_date(value):
    return datetime.strptime(value, "%Y-%m-%d").date()


def approval_band(annual_value_inr):
    if annual_value_inr < 1_000_000:
        return "A"
    if annual_value_inr <= 5_000_000:
        return "B"
    return "C"


def derive_fields(row):
    renewal_date = parse_date(row["renewal_date"])
    notice_days = int(row["notice_days"])
    notice_deadline = renewal_date - timedelta(days=notice_days)
    days_to_notice_deadline = (notice_deadline - SIMULATED_TODAY).days

    if SIMULATED_TODAY > renewal_date:
        notice_state = "EXPIRED"
    elif SIMULATED_TODAY > notice_deadline:
        notice_state = "INSIDE_WINDOW"
    elif days_to_notice_deadline <= 30:
        notice_state = "APPROACHING"
    else:
        notice_state = "OPEN"

    seats_purchased = int(row.get("seats_purchased") or 0)
    seats_active = int(row.get("seats_active") or 0)
    utilisation_pct = (
        round(seats_active / seats_purchased * 100) if seats_purchased > 0 else None
    )

    annual_value_inr = int(row["annual_value_inr"])
    days_to_renewal = (renewal_date - SIMULATED_TODAY).days

    return {
        "contract_id": row["contract_id"],
        "vendor": row["vendor"],
        "category": row["category"],
        "business_unit": row.get("business_unit"),
        "owner": row.get("owner") or "UNASSIGNED",
        "annual_value_inr": annual_value_inr,
        "approval_band": approval_band(annual_value_inr),
        "renewal_date": row["renewal_date"],
        "days_to_renewal": days_to_renewal,
        "notice_deadline": notice_deadline.isoformat(),
        "notice_state": notice_state,
        "days_to_notice_deadline": days_to_notice_deadline,
        "auto_renew": row.get("auto_renew", "").strip().upper() == "Y",
        "seats_purchased": seats_purchased or None,
        "seats_active": seats_active or None,
        "utilisation_pct": utilisation_pct,
        "proposed_uplift_pct": float(row["proposed_uplift_pct"])
        if row.get("proposed_uplift_pct")
        else None,
        "status": row.get("status") or "Active",
    }


def load_contracts():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            c = derive_fields(row)
            contracts[c["contract_id"]] = c


@app.route("/health")
def health():
    return jsonify({"status": "ok", "contract_count": len(contracts)})


@app.route("/api/contracts")
def list_contracts():
    results = list(contracts.values())
    category = request.args.get("category")
    band = request.args.get("band")
    notice_state = request.args.get("notice_state")
    if category:
        results = [c for c in results if c["category"] == category]
    if band:
        results = [c for c in results if c["approval_band"] == band]
    if notice_state:
        results = [c for c in results if c["notice_state"] == notice_state]
    return jsonify(results)


@app.route("/api/contracts/expiring")
def expiring_contracts():
    days = request.args.get("days", default=90, type=int)
    results = [c for c in contracts.values() if c["days_to_renewal"] <= days]
    results.sort(key=lambda c: c["days_to_renewal"])
    return jsonify(results)


@app.route("/api/categories")
def categories():
    grouped = {}
    for c in contracts.values():
        cat = grouped.setdefault(
            c["category"], {"vendors": [], "count": 0, "total_value_inr": 0}
        )
        cat["vendors"].append(c["vendor"])
        cat["count"] += 1
        cat["total_value_inr"] += c["annual_value_inr"]
    return jsonify(grouped)


@app.route("/api/contracts/<contract_id>")
def get_contract(contract_id):
    contract = contracts.get(contract_id)
    if contract is None:
        return jsonify({"error": f"contract {contract_id} not found"}), 404
    return jsonify(contract)


@app.route("/api/contracts/<contract_id>", methods=["PATCH"])
def update_contract(contract_id):
    contract = contracts.get(contract_id)
    if contract is None:
        return jsonify({"error": f"contract {contract_id} not found"}), 404
    payload = request.get_json(silent=True) or {}
    for field in ("status", "owner", "proposed_uplift_pct"):
        if field in payload:
            contract[field] = payload[field]
    return jsonify(contract)


if __name__ == "__main__":
    load_contracts()
    print(f"Loaded {len(contracts)} contracts from {CSV_PATH}")
    app.run(port=5001, debug=True)