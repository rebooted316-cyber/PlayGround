"""Stage 12: applications built from OS primitives."""

from agentic_os import AgenticOS
from agentic_os.apps import data_report, research_assistant, support_triage

with AgenticOS() as system:
    system.ingest("Our premium plan costs $30/month and includes priority support.", source="pricing.md")

    print(research_assistant(system, "What does the premium plan include?").output)

    sales = "region,revenue\nnorth,1200\nsouth,800\neast,1500\nwest,950\n"
    print(data_report(system, sales, "Which region has the highest revenue, and what is the total?"))

    triage = support_triage(system)
    for ticket in ["I forgot my password", "How much is premium?"]:
        print(f"{ticket!r} -> {triage.run(ticket)}  (rule: {triage.last_rule})")
