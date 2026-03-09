"""ai-quarantine: seven-stage AI artifact admission-control pipeline."""

__version__ = "0.1.0"

# Canonical artifact states (formal state machine)
ARTIFACT_STATES = ("pending", "scanning", "passed", "failed", "rejected")
