"""
Forensic Case Database (SQLite) — Persistent case numbers and inspection history.
Guarantees identical files receive the exact same Case ID upon repeated inspection.
"""
import hashlib
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "cases.db"


def _get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the persistent forensic cases table."""
    with _get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS forensic_cases (
                file_hash TEXT PRIMARY KEY,
                case_number TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                media_type TEXT NOT NULL,
                verdict TEXT NOT NULL,
                confidence REAL NOT NULL,
                first_inspected TEXT NOT NULL,
                last_inspected TEXT NOT NULL,
                inspection_count INTEGER DEFAULT 1
            )
        """)
        conn.commit()
    logger.info(f"✅ Forensic Case Database initialized at {DB_PATH}")


def compute_file_hash(content: bytes) -> str:
    """Compute SHA-256 fingerprint for forensic integrity."""
    return hashlib.sha256(content).hexdigest()


def get_or_create_case(
    content: bytes,
    filename: str,
    media_type: str,
    verdict: str,
    confidence: float
) -> Tuple[str, int]:
    """
    Returns (case_number, inspection_count).
    If the file has been analyzed before, returns the existing case number.
    Otherwise generates a new permanent Case ID.
    """
    init_db()
    file_hash = compute_file_hash(content)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with _get_db() as conn:
        row = conn.execute(
            "SELECT case_number, inspection_count FROM forensic_cases WHERE file_hash = ?",
            (file_hash,)
        ).fetchone()

        if row:
            case_no = row["case_number"]
            count = row["inspection_count"] + 1
            conn.execute("""
                UPDATE forensic_cases
                SET inspection_count = ?, last_inspected = ?, verdict = ?, confidence = ?, filename = ?
                WHERE file_hash = ?
            """, (count, now, verdict, confidence, filename, file_hash))
            conn.commit()
            logger.info(f"Existing case {case_no} retrieved for {filename} (Inspection #{count})")
            return case_no, count
        else:
            # Generate deterministic permanent case number from file hash:
            # 5-digit sequence + year code e.g. DF-74291-26
            numeric_seed = int(file_hash[:6], 16) % 90000 + 10000
            case_no = f"DF-{numeric_seed}-26"

            conn.execute("""
                INSERT INTO forensic_cases
                (file_hash, case_number, filename, media_type, verdict, confidence, first_inspected, last_inspected, inspection_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (file_hash, case_no, filename, media_type, verdict, confidence, now, now))
            conn.commit()
            logger.info(f"New case registered: {case_no} for {filename}")
            return case_no, 1
