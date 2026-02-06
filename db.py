import sqlite3
from datetime import datetime

DB_PATH = "news.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            link TEXT UNIQUE,
            publish_time TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_news(df):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for _, row in df.iterrows():
        try:
            c.execute("""
                INSERT OR IGNORE INTO news
                (title, content, link, publish_time, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                row.get("标题", ""),
                row.get("内容", ""),
                row.get("链接", ""),
                row.get("发布时间", ""),
                datetime.now().isoformat()
            ))
        except Exception:
            pass

    # 只保留最新1500条
    c.execute("""
        DELETE FROM news
        WHERE id NOT IN (
            SELECT id FROM news
            ORDER BY publish_time DESC
            LIMIT 1500
        )
    """)
    conn.commit()
    conn.close()

def load_news(keyword=None, limit=1500):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    if keyword:
        kw = f"%{keyword}%"
        c.execute("""
            SELECT title, content, link, publish_time
            FROM news
            WHERE title LIKE ? OR content LIKE ?
            ORDER BY publish_time DESC
            LIMIT ?
        """, (kw, kw, limit))
    else:
        c.execute("""
            SELECT title, content, link, publish_time
            FROM news
            ORDER BY publish_time DESC
            LIMIT ?
        """, (limit,))

    rows = c.fetchall()
    conn.close()

    return rows
