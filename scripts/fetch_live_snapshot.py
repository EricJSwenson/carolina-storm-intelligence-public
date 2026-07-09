"""Fetch a snapshot of live NOAA data and write it to data/live/.

Run on a schedule (see .github/workflows/live-data.yml) so the repo carries a
recent snapshot of active storms even when the live APIs aren't reachable at
view time. During quiet periods there are simply no active storms -- that's
expected, not an error.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from storm_eval.live import feeds

OUT = Path(__file__).resolve().parents[1] / "data" / "live"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    storms = feeds.get_active_storms()
    snapshot = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "active_storm_count": len(storms),
        "storms": [s.__dict__ for s in storms],
    }
    (OUT / "active_storms.json").write_text(json.dumps(snapshot, indent=2))
    print(f"wrote snapshot: {len(storms)} active storm(s) at {snapshot['fetched_at']}")


if __name__ == "__main__":
    main()
