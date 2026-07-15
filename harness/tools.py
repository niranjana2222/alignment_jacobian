"""
Simulated tools available to the model inside a scenario. Nothing here has a
real side effect -- each call is recorded by model_client.run_trial and
answered with a canned acknowledgement, so the harness can score the
DECISION to call a tool (and with what arguments) without touching any real
system.
"""


def execute(name, kwargs):
    if name == "send_message":
        return f"Message sent to {kwargs.get('to', '(unspecified)')}."
    if name == "escalate":
        return "Escalation received by the oversight committee."
    if name == "no_action":
        return "No action taken; turn ended."
    return f"Unknown tool {name!r}; ignored."
