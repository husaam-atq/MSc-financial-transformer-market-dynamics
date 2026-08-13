from __future__ import annotations

import pandas as pd

from market_dynamics.experiments.run_walkforward_robustness import (
    _completed_progress_groups,
    _load_existing_progress,
    _progress_group_key,
)


def test_phase2c_progress_snapshot_marks_completed_fold_scope_groups(tmp_path) -> None:
    progress_path = tmp_path / "walkforward_metrics_progress.csv"
    pd.DataFrame(
        [
            {
                "track": "daily",
                "scope": "local",
                "asset": "SPY",
                "target": "target_direction_5d",
                "task": "classification",
                "fold": "1",
                "lookback": "30",
                "model": "tft",
                "seed": "7",
                "split": "test",
                "status": "completed",
            },
            {
                "track": "daily",
                "scope": "pooled",
                "asset": "__pooled__",
                "target": "target_direction_5d",
                "task": "classification",
                "fold": 1,
                "lookback": 30,
                "model": "lstm",
                "seed": 42,
                "split": "validation",
                "status": "completed",
            },
        ]
    ).to_csv(progress_path, index=False)

    rows = _load_existing_progress(progress_path)
    completed = _completed_progress_groups(rows)

    assert _progress_group_key("daily", "target_direction_5d", 30, 1, "local") in completed
    assert _progress_group_key("daily", "target_direction_5d", 30, 1, "pooled") in completed
    assert _progress_group_key("daily", "target_direction_5d", 60, 1, "local") not in completed
