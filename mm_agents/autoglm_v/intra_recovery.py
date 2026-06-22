"""
General action-interface recovery for GUI agents, instantiated on ComputerRL's
Python action transport.

GENERIC CORE (this file): a tiered state machine over a single boolean per step --
did the action cross the parser/dispatch INTERFACE -- plus a failure fingerprint.
It knows nothing about Python, LibreOffice, tool names, or error text.

    monitor.step(interface_ok: bool, fingerprint: str) -> None | "L1" | "L2"

  * interface_ok == True  : the action crossed the interface (it WILL/ DID dispatch).
        This is the ONLY reset: serialization has recovered, so the streak clears --
        regardless of whether the tool then returns success OR a real execution error
        (Unsupported color, bad index, ...). Execution errors are legitimate world
        feedback and are deliberately NOT handled here.
  * interface_ok == False : the action did NOT cross the interface (parser produced
        no action, or the dispatched command failed to compile). This is OUR
        serialization/format failure, not tool feedback.

Tiers (frozen; uniform window=5 / threshold=3 across ALL domains, no per-app tuning):
  L1 Action-interface correction : on a 1st/2nd interface failure, gently ask to
        re-emit one valid single-line action, KEEPING the current intent.
  L2 Persistent-failure escalation: when the SAME normalized fingerprint reaches
        `threshold` within the last `window` consecutive interface failures (no
        successful dispatch in between), escalate ONCE; after that, stay silent to
        avoid polluting the prompt until a successful dispatch resets the streak.

The ComputerRL adapter (in main.py) supplies interface_ok by compiling the grounded
command: parser empty -> False; compile() fails -> False (+fingerprint); else True.
That adapter is the only ComputerRL/Python-specific part; swap it for a JSON-schema
or coordinate-action adapter and the core is unchanged.
"""

# Frozen prompts -- serialization-first, intent-preserving, no App/API names.
# English to match the system prompt / tasks (avoid an extra language variable).
_PROMPTS = {
    "L1": (
        "* Recovery note: your previous output did not pass the action interface, so it "
        "did NOT execute -- this is not tool feedback. Keep your current intent and re-emit "
        "exactly ONE well-formed single-line action."
    ),
    "L2": (
        "* Recovery note: you have repeatedly failed to produce an executable action. Stop "
        "reusing your current output format; keep your current intent and emit ONE cleaner, "
        "simpler, well-formed single command. Only adjust your plan after an action actually "
        "executes and returns tool feedback."
    ),
}


class RecoveryMonitor:
    def __init__(self, window: int = 5, threshold: int = 3):
        self.window = window
        self.threshold = threshold
        self.reset()

    def reset(self):
        self._streak = []           # fingerprints of consecutive interface failures
        self._l2_fired = False
        self.last_step = None       # per-step record, logged EVERY step for analysis

    def step(self, interface_ok: bool, fingerprint: str = ""):
        if interface_ok:
            self.reset()            # crossed the interface -> serialization recovered
            self.last_step = {"interface_ok": True, "fingerprint": "", "level": None,
                              "streak_len": 0, "same_template_count": 0,
                              "l2_fired": False, "reset": True}
            return None

        self._streak.append(fingerprint or "interface_failure")
        self._streak = self._streak[-self.window:]
        same = sum(1 for f in self._streak if f == self._streak[-1])

        if same >= self.threshold and not self._l2_fired:
            self._l2_fired = True
            level = "L2"
        elif len(self._streak) <= 2 and not self._l2_fired:
            level = "L1"            # only the 1st/2nd interface failure of a streak
        else:
            # streak grew past 2 without a same-template loop (heterogeneous failures),
            # or already escalated -> stay silent; do NOT keep polluting the prompt.
            level = None

        self.last_step = {"interface_ok": False, "fingerprint": self._streak[-1], "level": level,
                          "streak_len": len(self._streak), "same_template_count": same,
                          "l2_fired": self._l2_fired, "reset": False}
        return level

    @staticmethod
    def hint_text(level: str) -> str:
        return _PROMPTS.get(level, "")
