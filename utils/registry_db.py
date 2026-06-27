# registry_db.py
# Plan-B registry schema (SQLite) + helper writers
from __future__ import annotations
import sqlite3, hashlib, datetime
from typing import Optional
import pandas as pd

# ---------------- Schema ----------------
def ensure_schema_planB(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")

    # 1) datasets
    conn.execute("""
    CREATE TABLE IF NOT EXISTS datasets(
        dataset_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        name         TEXT,
        dataset_type TEXT NOT NULL,
        format       TEXT,
        created_at   TEXT DEFAULT (datetime('now')),
        file_hash    TEXT,
        row_count    INTEGER,
        UNIQUE(dataset_type, file_hash)
    )
    """)

    # 2) rules
    conn.execute("""
    CREATE TABLE IF NOT EXISTS rules(
        rule_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        rule_key  TEXT UNIQUE,
        name      TEXT,
        applies_to TEXT
    )
    """)

    # 3) validation_run
    conn.execute("""
    CREATE TABLE IF NOT EXISTS validation_run(
        run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id   INTEGER NOT NULL,
        triggered_by TEXT NOT NULL,
        status       TEXT NOT NULL CHECK (status IN ('running','succeeded','failed')),
        started_at   TEXT DEFAULT (datetime('now')),
        completed_at TEXT,
        FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
    )
    """)

    # 4) violations
    conn.execute("""
    CREATE TABLE IF NOT EXISTS violations(
        violation_id  INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id        INTEGER NOT NULL,
        rule_id       INTEGER NOT NULL,
        feature_index TEXT,
        feature_hash  TEXT,
        error_type    TEXT,
        message       TEXT,
        error_count   INTEGER DEFAULT 1,
        latitude      REAL,
        longitude     REAL,
        created_at    TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(run_id)  REFERENCES validation_run(run_id) ON DELETE CASCADE,
        FOREIGN KEY(rule_id) REFERENCES rules(rule_id),
        UNIQUE(run_id, rule_id, feature_hash)
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS violation_status(
        status_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        violation_id INTEGER NOT NULL,
        status       TEXT NOT NULL CHECK (status IN ('invalid','waived')),
        note         TEXT,
        set_at       TEXT DEFAULT (datetime('now')),
        set_by       TEXT,
        FOREIGN KEY(violation_id) REFERENCES violations(violation_id) ON DELETE CASCADE
    )
    """)

    conn.execute("""
    CREATE TRIGGER IF NOT EXISTS trg_violation_init_status
    AFTER INSERT ON violations
    BEGIN
        INSERT INTO violation_status(violation_id, status, note, set_by)
        VALUES (NEW.violation_id, 'invalid', '', 'system');
    END;
    """)

    conn.execute("""
    CREATE VIEW IF NOT EXISTS v_violation_latest AS
    WITH ranked AS (
        SELECT
            v.violation_id, v.run_id, v.rule_id, v.feature_hash,
            v.error_type, v.message, v.error_count,
            v.latitude, v.longitude,
            vs.status, vs.note, vs.set_at, vs.set_by,
            ROW_NUMBER() OVER (
                PARTITION BY v.violation_id
                ORDER BY datetime(vs.set_at) DESC, vs.status_id DESC
            ) AS rn
        FROM violations v
        JOIN violation_status vs ON vs.violation_id = v.violation_id
    )
    SELECT * FROM ranked WHERE rn = 1
    """)
    conn.commit()

# ---------------- Helpers ----------------
def _md5_bytes(b: bytes) -> str:
    return hashlib.md5(b or b"").hexdigest()

def upsert_dataset_row(
        conn: sqlite3.Connection,
        dtype: str,
        fmt: str,
        file_name: str,
        file_bytes: bytes,
        row_count: int
) -> int:
    file_hash = _md5_bytes(file_bytes or b"")
    cur = conn.execute(
        "SELECT dataset_id FROM datasets WHERE dataset_type=? AND file_hash=?",
        (dtype, file_hash)
    )
    row = cur.fetchone()
    if row:
        return int(row[0])

    name = f"{dtype}:{file_name}" if file_name else dtype
    cur = conn.execute("""
        INSERT INTO datasets(name, dataset_type, format, file_hash, row_count)
        VALUES (?, ?, ?, ?, ?)
    """, (name, dtype, fmt, file_hash, int(row_count or 0)))
    conn.commit()
    return int(cur.lastrowid)

def _upsert_rule(conn: sqlite3.Connection, rule_key: str, applies_to: Optional[str] = None, name: Optional[str] = None) -> int:
    rule_key = rule_key or "unknown"
    cur = conn.execute("SELECT rule_id FROM rules WHERE rule_key=?", (rule_key,))
    row = cur.fetchone()
    if row:
        return int(row[0])
    cur = conn.execute(
        "INSERT INTO rules(rule_key, name, applies_to) VALUES(?,?,?)",
        (rule_key, name or rule_key, applies_to)
    )
    conn.commit()
    return int(cur.lastrowid)

def begin_run(conn: sqlite3.Connection, dataset_id: int, triggered_by: str = "user") -> int:
    cur = conn.execute("""
        INSERT INTO validation_run(dataset_id, triggered_by, status)
        VALUES (?, ?, 'running')
    """, (int(dataset_id), triggered_by))
    conn.commit()
    return int(cur.lastrowid)

def _compute_feature_hash(row: pd.Series, dtype: str) -> str:
    parts = [
        dtype,
        str(row.get("index", "")),
        str(row.get("error_type", "")),
        str(row.get("message", "")),
    ]
    try:
        g = row.get("geometry", None)
        if g is not None:
            parts.append(getattr(g, "wkb_hex", ""))
    except Exception:
        pass
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()

def write_violations(conn: sqlite3.Connection, run_id: int, df: pd.DataFrame, dtype: str) -> int:
    if df is None or df.empty:
        return 0

    if "error_count" not in df.columns:
        df["error_count"] = 1
    if "rule_key" not in df.columns:
        df["rule_key"] = df.get("error_type", "unknown")

    if "lat" not in df.columns or "lon" not in df.columns:
        try:
            if "geometry" in df.columns:
                df = df.copy()
                df["lon"] = df["lon"] if "lon" in df.columns else df["geometry"].centroid.x
                df["lat"] = df["lat"] if "lat" in df.columns else df["geometry"].centroid.y
        except Exception:
            pass

    inserted = 0
    for _, r in df.iterrows():
        rule_id = _upsert_rule(conn, str(r.get("rule_key") or r.get("error_type") or "unknown"),
                               applies_to=dtype, name=str(r.get("rule_key") or r.get("error_type")))
        feature_hash = str(r.get("feature_hash")) if "feature_hash" in df.columns else _compute_feature_hash(r, dtype)
        feature_index = str(r.get("index", ""))
        lon = r.get("lon", None); lat = r.get("lat", None)

        try:
            conn.execute("""
                INSERT OR IGNORE INTO violations(
                    run_id, rule_id, feature_index, feature_hash,
                    error_type, message, error_count, latitude, longitude
                ) VALUES(?,?,?,?,?,?,?,?,?)
            """, (int(run_id), int(rule_id), feature_index, feature_hash,
                  str(r.get("error_type","")), str(r.get("message","")),
                  int(r.get("error_count", 1)),
                  float(lat) if pd.notna(lat) else None,
                  float(lon) if pd.notna(lon) else None))
            inserted += 1
        except Exception:
            pass

    conn.commit()
    return inserted

def finish_run(conn: sqlite3.Connection, run_id: int, status: str = "succeeded") -> None:
    conn.execute("""
        UPDATE validation_run
        SET status=?, completed_at=datetime('now')
        WHERE run_id=?
    """, (status, int(run_id)))
    conn.commit()
