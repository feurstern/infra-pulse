"""Quick start: validate raw samples, read the report, and put them on a regular grid.

Run it with ``uv run python examples/quickstart.py``.
"""

import pandas as pd

import infrapulse as ip

raw = pd.DataFrame(
    {
        "timestamp": [
            "2026-01-05T09:00:00+07:00",  # the same instant as 02:00:00Z
            "2026-01-05T02:01:00Z",
            "2026-01-05T02:02:10Z",
            "2026-01-05T02:02:50Z",  # two samples in one minute
            "2026-01-05T02:06:00Z",  # nothing from 02:03 to 02:05
            "2026-01-05T02:00:00Z",
            "2026-01-05T02:01:00Z",
        ],
        "metric": ["cpu_usage"] * 7,
        "host": ["vps-01"] * 5 + ["vps-02"] * 2,
        "value": [12.0, 14.5, 13.0, 15.0, 80.0, 30.0, None],  # vps-02 stops after 02:00
        "unit": ["percent"] * 7,
    }
)

frame = ip.MetricFrame.from_dataframe(raw)
print(frame)
print(frame.report.summary())

grid = ip.regularize(frame, "1min")
print(grid.data[["timestamp", "host", "value", "count"]].to_string(index=False))
print(grid.gaps()[["host", "start", "end", "missing_intervals"]].to_string(index=False))
