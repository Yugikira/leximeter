"""受控对照：同一样本交替测 compare 耗时（tag 区分 HEAD / baseline）。"""

import asyncio
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import leximeter  # noqa: E402


async def main() -> None:
    df = pd.DataFrame({
        "circle": [
            "The new CEO announced a 20% R&D budget cut at the all-hands.",
            "Our team ships weekly; velocity is up 30% since Q1.",
        ],
        "square": [
            "The CFO proposed reallocating funds to marketing instead.",
            "Sprint cycles are now two weeks; scope creep remains an issue.",
        ],
    })
    tag = sys.argv[1]
    t0 = time.monotonic()
    res = await leximeter.compare(
        df, "circle", "square",
        save_dir=f"runs/cmp_bench_{tag}",
        model="deepseek-v4-flash",
        differentiate=True,
        reset_files=True,
    )
    dt = time.monotonic() - t0
    print(f"[{tag}] elapsed={dt:.1f}s rows={len(res)}", flush=True)


asyncio.run(main())
