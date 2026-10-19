"""本地 SQLite，事件复核留痕；每次操作独立连接以支持界面/视频线程。"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid

KINDS = {"suspected_fall": "疑似跌倒", "low_activity": "持续低活动", "zone_entry": "进入指定区域"}
STATUSES = {"pending": "待确认", "confirmed": "已确认", "false_positive": "误报"}


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, source TEXT NOT NULL, source_kind TEXT NOT NULL,
                    started_utc TEXT NOT NULL, config_json TEXT NOT NULL, summary_json TEXT);
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                    kind TEXT NOT NULL, time_s REAL NOT NULL, onset_s REAL NOT NULL,
                    evidence TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                    note TEXT NOT NULL DEFAULT '', snapshot TEXT, clip TEXT,
                    clip_info_json TEXT, reviewed_utc TEXT);
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY, event_id TEXT NOT NULL REFERENCES events(id),
                    status TEXT NOT NULL, note TEXT NOT NULL, reviewed_utc TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        try:
            with con:
                yield con
        finally:
            con.close()

    def session(self, source, source_kind, config):
        session_id = uuid.uuid4().hex
        with self.connect() as con:
            con.execute("INSERT INTO sessions(id,source,source_kind,started_utc,config_json) VALUES(?,?,?,?,?)",
                        (session_id, str(source), source_kind, now(), json.dumps(config, ensure_ascii=False)))
        return session_id

    def finish(self, session_id, summary):
        with self.connect() as con:
            con.execute("UPDATE sessions SET summary_json=? WHERE id=?", (json.dumps(summary, ensure_ascii=False), session_id))

    def add(self, session_id, event):
        if event["kind"] not in KINDS:
            raise ValueError("未知事件类型")
        event_id = uuid.uuid4().hex
        with self.connect() as con:
            con.execute("INSERT INTO events(id,session_id,kind,time_s,onset_s,evidence) VALUES(?,?,?,?,?,?)",
                        (event_id, session_id, event["kind"], event["time_s"], event["onset_s"], event["evidence"]))
        return event_id

    def media(self, event_id, snapshot=None, clip=None, clip_info=None):
        with self.connect() as con:
            con.execute("UPDATE events SET snapshot=COALESCE(?,snapshot),clip=COALESCE(?,clip),clip_info_json=COALESCE(?,clip_info_json) WHERE id=?",
                        (snapshot, clip, json.dumps(clip_info) if clip_info is not None else None, event_id))

    def review(self, event_id, status, note=""):
        if status not in STATUSES:
            raise ValueError("未知确认状态")
        if len(note) > 2000:
            raise ValueError("备注不能超过 2000 字")
        timestamp = now()
        with self.connect() as con:
            cursor = con.execute("UPDATE events SET status=?,note=?,reviewed_utc=? WHERE id=?", (status, note, timestamp, event_id))
            if cursor.rowcount != 1:
                raise ValueError("事件不存在")
            con.execute("INSERT INTO reviews(event_id,status,note,reviewed_utc) VALUES(?,?,?,?)", (event_id, status, note, timestamp))

    def events(self, status=None, session_id=None):
        clauses, parameters = [], []
        if status:
            clauses.append("e.status=?")
            parameters.append(status)
        if session_id:
            clauses.append("e.session_id=?")
            parameters.append(session_id)
        query = "SELECT e.*,s.source,s.source_kind,s.started_utc FROM events e JOIN sessions s ON e.session_id=s.id"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with self.connect() as con:
            return [dict(row) for row in con.execute(query + " ORDER BY s.started_utc DESC,e.time_s DESC", parameters)]

    def export(self, path):
        # 导出前由调用者选择新路径，独占创建避免无意覆盖。
        with self.connect() as con:
            sessions = [dict(row) for row in con.execute("SELECT * FROM sessions")]
            reviews = [dict(row) for row in con.execute("SELECT * FROM reviews ORDER BY id")]
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump({"sessions": sessions, "events": self.events(), "reviews": reviews}, stream, ensure_ascii=False, indent=2)
