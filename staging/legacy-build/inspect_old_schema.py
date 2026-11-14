"""
inspect_db_schema.py — Full schema dump for any SQLite database

Prints every table's columns (name/type/notnull/default/pk), row count,
one sample row, all indexes, and all foreign key relationships. Also
writes the same report to a text file so it can be pasted into chat.

Run:
    python inspect_db_schema.py --db battle_scenes.db
    python inspect_db_schema.py --db battle_scenes.db --out old_schema_dump.txt
"""

import sqlite3
import argparse
from pathlib import Path


def dump_schema(db_path: Path, out_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    emit("=" * 80)
    emit(f"  SCHEMA DUMP — {db_path}")
    emit("=" * 80)

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row["name"] for row in cur.fetchall()]

    if not tables:
        emit("  No tables found.")
        conn.close()
        return

    for table in tables:
        emit(f"\n{'-' * 80}")
        emit(f"  TABLE: {table}")
        emit(f"{'-' * 80}")

        # ── Columns ──────────────────────────────────────────────
        cur.execute(f"PRAGMA table_info('{table}')")
        cols = cur.fetchall()
        emit("  COLUMNS:")
        for col in cols:
            # cid, name, type, notnull, dflt_value, pk
            pk_flag = " PRIMARY KEY" if col["pk"] else ""
            notnull_flag = " NOT NULL" if col["notnull"] else ""
            default = f" DEFAULT {col['dflt_value']}" if col["dflt_value"] is not None else ""
            emit(f"    {col['name']:<25} {col['type']:<15} {notnull_flag}{default}{pk_flag}")

        # ── Row count ────────────────────────────────────────────
        cur.execute(f"SELECT COUNT(*) AS c FROM '{table}'")
        row_count = cur.fetchone()["c"]
        emit(f"\n  ROW COUNT: {row_count:,}")

        # ── Sample row ───────────────────────────────────────────
        if row_count > 0:
            cur.execute(f"SELECT * FROM '{table}' LIMIT 1")
            sample = cur.fetchone()
            emit("\n  SAMPLE ROW:")
            for key in sample.keys():
                val = sample[key]
                if isinstance(val, str) and len(val) > 120:
                    val = f"{val[:120]}... [{len(val)} chars total]"
                emit(f"    {key:<20} = {val!r}")

        # ── Indexes ──────────────────────────────────────────────
        cur.execute(f"PRAGMA index_list('{table}')")
        indexes = cur.fetchall()
        if indexes:
            emit("\n  INDEXES:")
            for idx in indexes:
                emit(f"    {idx['name']}  (unique={bool(idx['unique'])})")

        # ── Foreign keys ─────────────────────────────────────────
        cur.execute(f"PRAGMA foreign_key_list('{table}')")
        fks = cur.fetchall()
        if fks:
            emit("\n  FOREIGN KEYS:")
            for fk in fks:
                emit(f"    {table}.{fk['from']}  ->  {fk['table']}.{fk['to']}")

    emit(f"\n{'=' * 80}")
    emit(f"  {len(tables)} table(s) total")
    emit("=" * 80)

    conn.close()

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nFull dump written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Dump full SQLite schema details")
    parser.add_argument("--db", type=Path, required=True, help="Path to .db file")
    parser.add_argument("--out", type=Path, default=Path("schema_dump.txt"), help="Output text file")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"ERROR: {args.db} not found")
        return

    dump_schema(args.db, args.out)


if __name__ == "__main__":
    main()