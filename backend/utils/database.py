"""
Forensic Case Database (SQLite) — Persistent case numbers and inspection history.
Guarantees identical files receive the exact same Case ID upon repeated inspection.
Provides history retrieval, filtering, and case management.
"""
import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent.parent / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "cases.db"


MEDIA_DIR = DB_DIR / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


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
                inspection_count INTEGER DEFAULT 1,
                details TEXT,
                media_path TEXT
            )
        """)
        # Ensure 'details' and 'media_path' columns exist for existing tables
        cursor = conn.execute("PRAGMA table_info(forensic_cases)")
        columns = [row["name"] for row in cursor.fetchall()]
        if "details" not in columns:
            conn.execute("ALTER TABLE forensic_cases ADD COLUMN details TEXT")
        if "media_path" not in columns:
            conn.execute("ALTER TABLE forensic_cases ADD COLUMN media_path TEXT")
        conn.commit()
    logger.info(f"✅ Forensic Case Database initialized at {DB_PATH}")


def compute_file_hash(content: bytes) -> str:
    """Compute SHA-256 fingerprint for forensic integrity."""
    return hashlib.sha256(content).hexdigest()


def save_media_file(file_hash: str, filename: str, content: bytes) -> str:
    """Save media content locally and return relative URL path."""
    ext = Path(filename).suffix.lower() or ".bin"
    file_on_disk = MEDIA_DIR / f"{file_hash[:16]}{ext}"
    try:
        if not file_on_disk.exists():
            with open(file_on_disk, "wb") as f:
                f.write(content)
        return f"/media/{file_on_disk.name}"
    except Exception as e:
        logger.error(f"Failed to save media file: {e}")
        return ""


def get_or_create_case(
    content: bytes,
    filename: str,
    media_type: str,
    verdict: str,
    confidence: float,
    details: Optional[dict] = None
) -> Tuple[str, int, str]:
    """
    Returns (case_number, inspection_count, media_path).
    If the file has been analyzed before, returns existing case number and updates details.
    Otherwise generates a new permanent Case ID.
    """
    init_db()
    file_hash = compute_file_hash(content)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    media_path = save_media_file(file_hash, filename, content)
    
    if details is not None:
        details["media_url"] = media_path
    details_json = json.dumps(details) if details else "{}"

    with _get_db() as conn:
        row = conn.execute(
            "SELECT case_number, inspection_count, media_path FROM forensic_cases WHERE file_hash = ?",
            (file_hash,)
        ).fetchone()

        if row:
            case_no = row["case_number"]
            count = row["inspection_count"] + 1
            existing_media_path = row["media_path"] or media_path
            conn.execute("""
                UPDATE forensic_cases
                SET inspection_count = ?, last_inspected = ?, verdict = ?, confidence = ?, filename = ?, details = ?, media_path = ?
                WHERE file_hash = ?
            """, (count, now, verdict, confidence, filename, details_json, existing_media_path, file_hash))
            conn.commit()
            logger.info(f"Existing case {case_no} retrieved for {filename} (Inspection #{count})")
            return case_no, count, existing_media_path
        else:
            # Generate deterministic permanent case number from file hash:
            # 5-digit sequence + year code e.g. DF-74291-26
            numeric_seed = int(file_hash[:6], 16) % 90000 + 10000
            case_no = f"DF-{numeric_seed}-26"

            conn.execute("""
                INSERT INTO forensic_cases
                (file_hash, case_number, filename, media_type, verdict, confidence, first_inspected, last_inspected, inspection_count, details, media_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """, (file_hash, case_no, filename, media_type, verdict, confidence, now, now, details_json, media_path))
            conn.commit()
            logger.info(f"New case registered: {case_no} for {filename}")
            return case_no, 1, media_path


def get_all_cases(limit: int = 100) -> List[dict]:
    """Retrieve history of forensic cases ordered by last inspection."""
    init_db()
    cases = []
    with _get_db() as conn:
        cursor = conn.execute("""
            SELECT file_hash, case_number, filename, media_type, verdict, confidence,
                   first_inspected, last_inspected, inspection_count, details, media_path
            FROM forensic_cases
            ORDER BY datetime(last_inspected) DESC
            LIMIT ?
        """, (limit,))
        for row in cursor.fetchall():
            item = dict(row)
            try:
                item["details"] = json.loads(item["details"]) if item.get("details") else {}
            except Exception:
                item["details"] = {}
            if not item.get("media_path") and item.get("details", {}).get("media_url"):
                item["media_path"] = item["details"]["media_url"]
            cases.append(item)
    return cases


def delete_case_by_number(case_number: str) -> bool:
    """Delete a specific case record."""
    init_db()
    with _get_db() as conn:
        res = conn.execute("DELETE FROM forensic_cases WHERE case_number = ?", (case_number,))
        conn.commit()
        return res.rowcount > 0


def clear_all_cases() -> bool:
    """Clear all forensic inspection history."""
    init_db()
    with _get_db() as conn:
        conn.execute("DELETE FROM forensic_cases")
        conn.commit()
        return True
