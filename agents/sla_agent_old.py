"""
ISDO Lab C5 — SLA & Escalation Agent
Monitors SLA deadlines, predicts breach risk, and triggers escalation for P1 tickets.
Includes HITL pause before any escalation action is taken.
"""

import anthropic
import csv
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── TOOL DEFINITIONS ──────────────────────────────────────────────────────────

tools = [
    {
        "name": "get_sla_status",
        "description": "Check the SLA status of a ticket. Returns time remaining, breach risk level, and whether the SLA has already been breached.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "sla_due": {
                    "type": "string",
                    "description": "SLA due datetime in format YYYY-MM-DD HH:MM:SS"
                },
                "priority": {
                    "type": "string",
                    "enum": ["P1", "P2", "P3", "P4"]
                }
            },
            "required": ["ticket_number", "sla_due", "priority"]
        }
    },
    {
        "name": "update_ticket",
        "description": "Update the ticket state or add escalation notes in ServiceNow.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "action": {
                    "type": "string",
                    "enum": ["escalate", "add_note", "update_state"],
                    "description": "Action to perform on the ticket"
                },
                "escalation_team": {
                    "type": "string",
                    "description": "Team to escalate to (e.g. L2-Network, Senior-Engineer)"
                },
                "note": {
                    "type": "string",
                    "description": "Work note to add to the ticket"
                },
                "new_state": {
                    "type": "string",
                    "description": "New state for the ticket e.g. In Progress, Escalated, Resolved"
                }
            },
            "required": ["ticket_number", "action"]
        }
    }
]

# ── TOOL IMPLEMENTATION ───────────────────────────────────────────────────────

# SLA thresholds by priority (hours to resolve)
SLA_HOURS = {"P1": 1, "P2": 4, "P3": 8, "P4": 24}

def get_sla_status(ticket_number, sla_due, priority):
    """Calculate SLA status and breach risk."""
    try:
        due_dt = datetime.strptime(sla_due, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return {"error": f"Invalid sla_due format: {sla_due}"}

    # Simulate "now" as 2024-01-15 10:30 for demo consistency
    now = datetime(2024, 1, 15, 10, 30)
    time_remaining = due_dt - now
    minutes_remaining = int(time_remaining.total_seconds() / 60)

    sla_hours = SLA_HOURS.get(priority, 8)
    total_minutes = sla_hours * 60

    if minutes_remaining < 0:
        risk = "BREACHED"
        status_msg = f"SLA BREACHED by {abs(minutes_remaining)} minutes"
    elif minutes_remaining < total_minutes * 0.2:  # Less than 20% time left
        risk = "CRITICAL"
        status_msg = f"Only {minutes_remaining} minutes remaining — breach imminent"
    elif minutes_remaining < total_minutes * 0.5:  # Less than 50% time left
        risk = "AT_RISK"
        status_msg = f"{minutes_remaining} minutes remaining — at risk"
    else:
        risk = "ON_TRACK"
        status_msg = f"{minutes_remaining} minutes remaining — on track"

    return {
        "ticket_number": ticket_number,
        "sla_due": sla_due,
        "priority": priority,
        "minutes_remaining": minutes_remaining,
        "breach_risk": risk,
        "status_message": status_msg,
        "requires_escalation": risk in ["BREACHED", "CRITICAL"]
    }

def update_ticket(ticket_number, action, escalation_team=None, note=None, new_state=None):
    """Simulate updating a ticket in ServiceNow."""
    result = {"ticket_number": ticket_number, "action": action, "success": True}

    if action == "escalate":
        result["message"] = f"Ticket {ticket_number} escalated to {escalation_team}"
        print(f"  [ServiceNow Mock] ESCALATED {ticket_number} → {escalation_team}")
    elif action == "add_note":
        result["message"] = f"Work note added to {ticket_number}: {note[:50]}..."
        print(f"  [ServiceNow Mock] NOTE ADDED to {ticket_number}")
    elif action == "update_state":
        result["message"] = f"Ticket {ticket_number} state changed to: {new_state}"
        print(f"  [ServiceNow Mock] STATE CHANGED {ticket_number} → {new_state}")

    return result

# ── ESCALATION GUARDRAIL (enforced in code, not just the system prompt) ──────
# Step 4 / Compare table: "Only CRITICAL/BREACHED + P1/P2 triggers escalation."
# The system prompt asks the model to follow this; this function makes it true
# regardless of what the model decides to call.
ESCALATABLE_PRIORITIES = {"P1", "P2"}

def escalation_allowed(sla_result, priority):
    """Return (allowed, reason). sla_result is the last get_sla_status() output
    for this ticket, or None if the model tried to escalate without checking SLA first."""
    if sla_result is None:
        return False, "escalate called before get_sla_status — risk unknown"
    if priority not in ESCALATABLE_PRIORITIES:
        return False, f"priority {priority} is not eligible for automatic escalation (P3/P4 are monitored only)"
    if not sla_result.get("requires_escalation"):
        return False, f"breach_risk is {sla_result.get('breach_risk')} — escalation not required"
    return True, ""

def handle_tool(name, inp):
    if name == "get_sla_status":
        return get_sla_status(inp["ticket_number"], inp["sla_due"], inp["priority"])
    elif name == "update_ticket":
        return update_ticket(
            inp["ticket_number"],
            inp["action"],
            inp.get("escalation_team"),
            inp.get("note"),
            inp.get("new_state")
        )
    return {"error": "Unknown tool"}

# ── HITL GATE ────────────────────────────────────────────────────────────────

def hitl_approve(ticket_number, action, detail):
    """Pause and ask for human approval before escalating."""
    print(f"\n  {'⚠️ '*10}")
    print(f"  HITL APPROVAL REQUIRED")
    print(f"  Ticket:  {ticket_number}")
    print(f"  Action:  {action}")
    print(f"  Detail:  {detail}")
    print(f"  {'⚠️ '*10}")
    decision = input("  Approve escalation? [y/n]: ").strip().lower()
    return decision == "y"

# ── SLA AGENT ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the ISDO SLA & Escalation Agent for Zensar's IT Service Desk.

For each ticket:
1. Use get_sla_status to check breach risk
2. If breach risk is CRITICAL or BREACHED and priority is P1 or P2, use update_ticket to escalate
3. For P1 tickets with CRITICAL/BREACHED status, always escalate — these cannot wait

Escalation teams by category:
- Network issues → L2-Network-Ops
- Application issues → L2-App-Support
- Server issues → L2-Server-Ops
- Access/Security issues → L2-Security-Ops
- General → L2-Service-Desk"""

def monitor_ticket(ticket_number, short_description, category, priority, sla_due):
    """Run SLA monitoring for a single ticket.
    Returns a decision dict for the caller (e.g. the Lab C6 orchestrator) with:
    breach_risk, requires_escalation, escalated, escalation_team, hitl_required, hitl_approved.
    """
    require_hitl = (priority == "P1")   # code-derived, not caller-supplied — a P1 ticket
                                          # always gets the HITL gate, whoever calls this function.
    sla_result = None    # set once get_sla_status runs; guards the escalate branch below
    decision = {
        "ticket_number": ticket_number, "breach_risk": None, "requires_escalation": None,
        "escalated": False, "escalation_team": None, "hitl_required": False, "hitl_approved": None,
    }
    print(f"\n{'='*55}")
    print(f"SLA Check: {ticket_number} | {priority} | Category: {category}")
    print(f"{'='*55}")

    messages = [{
        "role": "user",
        "content": (f"Monitor SLA for this ticket and escalate if needed:\n\n"
                    f"Ticket: {ticket_number}\nDescription: {short_description}\n"
                    f"Category: {category}\nPriority: {priority}\nSLA Due: {sla_due}")
    }]

    while True:
        response = client.messages.create(
            model="claude-opus-5",
            max_tokens=500,
            output_config={"effort": "medium"},
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
                    if block.name == "update_ticket" and block.input.get("action") == "escalate":
                        allowed, reason = escalation_allowed(sla_result, priority)
                        if not allowed:
                            result = {"success": False, "message": f"Escalation blocked: {reason}"}
                            print(f"  🛡️  Guardrail blocked escalation: {reason}")
                            tool_results.append({
                                "type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)
                            })
                            continue

                        # HITL gate: pause before escalation on P1 tickets — derived from priority,
                        # so it fires no matter who calls monitor_ticket or what they pass in.
                        if require_hitl:
                            decision["hitl_required"] = True
                            approved = hitl_approve(
                                block.input.get("ticket_number"),
                                "Escalate ticket",
                                f"Escalate to {block.input.get('escalation_team', 'L2 team')}"
                            )
                            decision["hitl_approved"] = approved
                            if not approved:
                                result = {"success": False, "message": "Escalation rejected by human approver"}
                                print("  Escalation cancelled.")
                                tool_results.append({
                                    "type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)
                                })
                                continue

                    result = handle_tool(block.name, block.input)
                    if block.name == "update_ticket" and block.input.get("action") == "escalate" and result.get("success"):
                        decision["escalated"] = True
                        decision["escalation_team"] = block.input.get("escalation_team")

                    if block.name == "get_sla_status":
                        sla_result = result
                        decision["breach_risk"] = result.get("breach_risk")
                        decision["requires_escalation"] = result.get("requires_escalation")
                        print(f"  → Risk Level: {result.get('breach_risk')}")
                        print(f"  → Status:     {result.get('status_message')}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })

            messages.append({"role": "user", "content": tool_results})

    return decision

# ── RUN SLA MONITORING ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test tickets with various SLA situations
    # (using simulated "now" = 2024-01-15 10:30)
    test_tickets = [
        # P1 with SLA due in 30 min — CRITICAL
        ("INC0001002", "Cannot access ERP - SAP login failure", "Application", "P1",
         "2024-01-15 11:00:00"),
        # P2 with SLA due in 3.5 hours
        ("INC0001001", "VPN not connecting", "Network", "P2",
         "2024-01-15 14:00:00"),
        # P1 already BREACHED
        ("INC0001010", "Exchange server high CPU", "Server", "P1",
         "2024-01-15 09:30:00"),
        # P3 ON_TRACK
        ("INC0001003", "Laptop running slowly", "Hardware", "P3",
         "2024-01-17 09:00:00"),
    ]

    for ticket_number, desc, category, priority, sla_due in test_tickets:
        monitor_ticket(ticket_number, desc, category, priority, sla_due)