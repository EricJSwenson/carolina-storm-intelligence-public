"""Local warehouse access (DuckDB).

DuckDB stands in for the Databricks SQL warehouse during local development: the
DDL in ``sql/ddl`` applies unchanged, and the analytics queries in
``sql/analytics`` run against it. In production the same SQL targets Databricks.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from storm_eval.config import ROOT, settings

DDL_DIR = ROOT / "sql" / "ddl"


def connect(path: Path | str | None = None) -> "duckdb.DuckDBPyConnection":
    path = Path(path or settings.warehouse_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def init_schema(con: "duckdb.DuckDBPyConnection") -> None:
    """Apply every DDL file (idempotent CREATE ... IF NOT EXISTS)."""
    for sql_file in sorted(DDL_DIR.glob("*.sql")):
        con.execute(sql_file.read_text(encoding="utf-8"))
