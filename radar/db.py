import sqlite3, os, json, time
from . import config
SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots(ts INTEGER, symbol TEXT, price REAL, features TEXT, PRIMARY KEY(ts,symbol));
CREATE TABLE IF NOT EXISTS book_history(ts INTEGER, symbol TEXT, side TEXT, price REAL, qty REAL);
CREATE INDEX IF NOT EXISTS ix_book ON book_history(symbol, ts);
CREATE TABLE IF NOT EXISTS signals(id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, symbol TEXT, kind TEXT,
  score REAL, prob REAL, price REAL, atr REAL, features TEXT, reasons TEXT, label INTEGER, ret REAL, labeled_ts INTEGER);
CREATE TABLE IF NOT EXISTS alerts(symbol TEXT, kind TEXT, ts INTEGER, PRIMARY KEY(symbol,kind));
"""
def connect():
    os.makedirs(os.path.dirname(config.DB_PATH) or ".", exist_ok=True)
    c = sqlite3.connect(config.DB_PATH); c.executescript(SCHEMA); return c
def prune(c, days=21):
    cut = int(time.time()) - days*86400
    c.execute("DELETE FROM book_history WHERE ts<?", (int(time.time())-6*3600,))
    c.execute("DELETE FROM snapshots WHERE ts<?", (cut,)); c.commit()
def save_snapshot(c, ts, sym, price, feats):
    c.execute("INSERT OR REPLACE INTO snapshots VALUES(?,?,?,?)", (ts, sym, price, json.dumps(feats)))
