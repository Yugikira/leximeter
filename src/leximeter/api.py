"""Public facade.

Each public function constructs a task config and delegates to a task.  The
awkward ``**cfg_kwargs`` / ``inspect.signature`` magic from GABRIEL is gone:
parameters are explicit, and provider selection happens via ``client=`` /
``embedding_client=`` / ``search_tool=`` (or environment defaults).

Populated as tasks are ported (milestones M3–M5).
"""

# from .tasks.rate import Rate, RateConfig  # etc., as tasks land
