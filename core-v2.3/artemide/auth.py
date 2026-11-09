"""SQLite-backed developer API keys, revocation, rate limits and usage records."""
from __future__ import annotations
import hashlib
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


class APIKeyError(Exception): pass
class RateLimitExceeded(APIKeyError): pass
class QuotaExceeded(APIKeyError): pass


def _now():
    return datetime.now(timezone.utc)


class APIKeyStore:
    def __init__(self, path="artemide_api_keys.sqlite3"):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS api_keys (
              id TEXT PRIMARY KEY, label TEXT NOT NULL, prefix TEXT NOT NULL,
              key_hash TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL,
              revoked_at TEXT, requests_per_minute INTEGER NOT NULL,
              monthly_token_limit INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS api_usage (
              id INTEGER PRIMARY KEY AUTOINCREMENT, key_id TEXT NOT NULL,
              created_at TEXT NOT NULL, prompt_tokens INTEGER NOT NULL DEFAULT 0,
              completion_tokens INTEGER NOT NULL DEFAULT 0,
              FOREIGN KEY(key_id) REFERENCES api_keys(id)
            );
            CREATE INDEX IF NOT EXISTS idx_usage_key_time ON api_usage(key_id, created_at);
            """)
    def _connect(self):
        db=sqlite3.connect(self.path, timeout=15)
        db.row_factory=sqlite3.Row
        db.execute("PRAGMA busy_timeout=15000")
        return db
    @staticmethod
    def _hash(raw): return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    def create_key(self, label, *, requests_per_minute=60, monthly_token_limit=100000):
        raw="art_"+secrets.token_urlsafe(32)
        import uuid
        key_id=uuid.uuid4().hex
        now=_now().isoformat()
        with self._connect() as db:
            db.execute("""INSERT INTO api_keys
              (id,label,prefix,key_hash,created_at,requests_per_minute,monthly_token_limit)
              VALUES(?,?,?,?,?,?,?)""",
              (key_id,label,raw[:12],self._hash(raw),now,requests_per_minute,monthly_token_limit))
        return {"id":key_id,"label":label,"prefix":raw[:12],"api_key":raw,
                "created_at":now,"requests_per_minute":requests_per_minute,
                "monthly_token_limit":monthly_token_limit}
    def has_active_keys(self):
        with self._connect() as db:
            return db.execute("SELECT 1 FROM api_keys WHERE revoked_at IS NULL LIMIT 1").fetchone() is not None
    def authenticate(self, raw):
        if not raw: return None
        digest=self._hash(raw)
        now=_now()
        minute=(now-timedelta(seconds=60)).isoformat()
        month=now.strftime("%Y-%m-01T00:00:00+00:00")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row=db.execute("SELECT * FROM api_keys WHERE key_hash=?",(digest,)).fetchone()
            if row is None or row["revoked_at"] is not None:
                db.rollback(); return None
            recent=db.execute("SELECT COUNT(*) n FROM api_usage WHERE key_id=? AND created_at>=?",
                              (row["id"],minute)).fetchone()["n"]
            if recent>=row["requests_per_minute"]:
                db.rollback(); raise RateLimitExceeded("API key request-per-minute limit exceeded")
            used=db.execute("""SELECT COALESCE(SUM(prompt_tokens+completion_tokens),0) n
                FROM api_usage WHERE key_id=? AND created_at>=?""",(row["id"],month)).fetchone()["n"]
            if used>=row["monthly_token_limit"]:
                db.rollback(); raise QuotaExceeded("API key monthly token limit reached")
            cur=db.execute("INSERT INTO api_usage(key_id,created_at) VALUES(?,?)",
                           (row["id"],now.isoformat()))
            usage_id=cur.lastrowid
            db.commit()
        return {"key_id":row["id"],"usage_id":usage_id,"label":row["label"],
                "monthly_token_limit":row["monthly_token_limit"]}
    def record_usage(self, identity, prompt_tokens, completion_tokens):
        if not identity or identity.get("key_id") is None or identity.get("usage_id") is None: return
        with self._connect() as db:
            db.execute("""UPDATE api_usage SET prompt_tokens=?,completion_tokens=? WHERE id=? AND key_id=?""",
                       (int(prompt_tokens),int(completion_tokens),identity["usage_id"],identity["key_id"]))
    def list_keys(self):
        now=_now(); month=now.strftime("%Y-%m-01T00:00:00+00:00")
        with self._connect() as db:
            rows=db.execute("""SELECT k.id,k.label,k.prefix,k.created_at,k.revoked_at,
                k.requests_per_minute,k.monthly_token_limit,
                COALESCE((SELECT SUM(u.prompt_tokens+u.completion_tokens) FROM api_usage u
                  WHERE u.key_id=k.id AND u.created_at>=?),0) AS tokens_used
                FROM api_keys k ORDER BY k.created_at DESC""",(month,)).fetchall()
        return [dict(row) for row in rows]
    def revoke(self, key_id):
        with self._connect() as db:
            cur=db.execute("UPDATE api_keys SET revoked_at=? WHERE id=? AND revoked_at IS NULL",
                           (_now().isoformat(),key_id))
            return cur.rowcount==1
    def usage(self, key_id):
        now=_now(); month=now.strftime("%Y-%m-01T00:00:00+00:00")
        with self._connect() as db:
            row=db.execute("""SELECT COUNT(*) requests,
                COALESCE(SUM(prompt_tokens),0) prompt_tokens,
                COALESCE(SUM(completion_tokens),0) completion_tokens
                FROM api_usage WHERE key_id=? AND created_at>=?""",(key_id,month)).fetchone()
        return {"period_start":month,**dict(row),"total_tokens":row["prompt_tokens"]+row["completion_tokens"]}
