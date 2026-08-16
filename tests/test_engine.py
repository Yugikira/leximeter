"""Engine-level protections: bounded loops on direct-call tasks."""

from __future__ import annotations

import asyncio

from conftest import DictClient

from leximeter import seed


def run(coro):
    return asyncio.run(coro)


def test_seed_generation_loop_has_budget(tmp_path):
    """A provider that keeps returning partial results must not loop forever."""
    client = DictClient({"entity 1": "only one"})  # always yields 1 entity
    res = run(
        seed(
            "generate people",
            save_dir=str(tmp_path / "seed"),
            client=client,
            num_entities=8,
            entities_per_generation=4,
        )
    )
    # budget = ceil(8/4) + 2 = 4 batches, then give up gracefully
    assert len(res) == 4
    assert "entity_id" in res.columns
