from __future__ import annotations
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AnalyticsService:
    DATABASE_NAME = "analytics.db"
    def __init__(
        self,
        database_path: str = "./database",
    ):
        self.database_path = Path(database_path)
        self.database_path.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.database = (
            self.database_path /
            self.DATABASE_NAME
        )

        self.connection = sqlite3.connect(
            self.database,
            check_same_thread=False,
        )

        self.connection.row_factory = sqlite3.Row

        self._initialize_database()

        logger.info(
            "AnalyticsService initialized."
        )

    # =====================================================
    # Database
    # =====================================================

    def _initialize_database(self):
        cursor = self.connection.cursor()
        cursor.execute("""

        CREATE TABLE IF NOT EXISTS chat_events(

            event_id TEXT PRIMARY KEY,

            conversation_id TEXT,

            user_id TEXT,

            provider TEXT,

            model TEXT,

            created_at TEXT

        )

        """)

        cursor.execute("""

        CREATE TABLE IF NOT EXISTS token_usage(

            usage_id TEXT PRIMARY KEY,

            conversation_id TEXT,

            prompt_tokens INTEGER,

            completion_tokens INTEGER,

            total_tokens INTEGER,

            created_at TEXT

        )

        """)

        cursor.execute("""

        CREATE TABLE IF NOT EXISTS latency(

            latency_id TEXT PRIMARY KEY,

            conversation_id TEXT,

            duration_ms REAL,

            created_at TEXT

        )

        """)

        self.connection.commit()

    # =====================================================
    # Chat Event
    # =====================================================

    def log_chat_event(
        self,
        conversation_id: str,
        user_id: str,
        provider: str,
        model: str,
    ):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT INTO chat_events
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                conversation_id,
                user_id,
                provider,
                model,
                datetime.utcnow().isoformat(),
            ),
        )

        self.connection.commit()

    # =====================================================
    # Token Usage
    # =====================================================

    def log_token_usage(
        self,
        conversation_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ):

        total = (
            prompt_tokens +
            completion_tokens
        )

        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT INTO token_usage
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                conversation_id,
                prompt_tokens,
                completion_tokens,
                total,
                datetime.utcnow().isoformat(),
            ),
        )

        self.connection.commit()

    # =====================================================
    # Response Latency
    # =====================================================

    def log_latency(
        self,
        conversation_id: str,
        duration_ms: float,
    ):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT INTO latency
            VALUES (?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                conversation_id,
                duration_ms,
                datetime.utcnow().isoformat(),
            ),
        )

        self.connection.commit()

    # =====================================================
    # Total Chats
    # =====================================================

    def total_chats(self):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM chat_events
            """
        )

        return cursor.fetchone()[0]

    # =====================================================
    # Total Tokens
    # =====================================================

    def total_tokens(self):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT SUM(total_tokens)
            FROM token_usage
            """
        )

        result = cursor.fetchone()[0]

        return result or 0

    # =====================================================
    # Average Latency
    # =====================================================

    def average_latency(self):

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT AVG(duration_ms)
            FROM latency
            """
        )

        value = cursor.fetchone()[0]

        return round(
            value or 0,
            2,
        )

    # =====================================================
    # Dashboard
    # =====================================================

    def dashboard(self):

        return {

            "total_chats":

                self.total_chats(),

            "total_tokens":

                self.total_tokens(),

            "average_latency_ms":

                self.average_latency(),

        }

    # =====================================================
    # Health
    # =====================================================

    def health(self):

        return {

            "status":

                "healthy",

            "database":

                str(self.database),

            "dashboard":

                self.dashboard(),

        }

    # =====================================================
    # Close
    # =====================================================

    def close(self):

        self.connection.close()

        logger.info(
            "AnalyticsService closed."
        )

  # =====================================================
# Model Usage
# =====================================================

cursor.execute("""

CREATE TABLE IF NOT EXISTS model_usage(

    usage_id TEXT PRIMARY KEY,

    provider TEXT,

    model TEXT,

    conversation_id TEXT,

    user_id TEXT,

    created_at TEXT

)

""")

# =====================================================
# Cost Tracking
# =====================================================

cursor.execute("""

CREATE TABLE IF NOT EXISTS costs(

    cost_id TEXT PRIMARY KEY,

    conversation_id TEXT,

    provider TEXT,

    model TEXT,

    input_tokens INTEGER,

    output_tokens INTEGER,

    total_cost REAL,

    created_at TEXT

)

""")

# =====================================================
# User Activity
# =====================================================

cursor.execute("""

CREATE TABLE IF NOT EXISTS user_activity(

    activity_id TEXT PRIMARY KEY,

    user_id TEXT,

    action TEXT,

    conversation_id TEXT,

    created_at TEXT

)

""")

self.connection.commit()

# =====================================================
# Model Usage
# =====================================================

def log_model_usage(
    self,
    conversation_id: str,
    user_id: str,
    provider: str,
    model: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        INSERT INTO model_usage
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            provider,
            model,
            conversation_id,
            user_id,
            datetime.utcnow().isoformat(),
        ),
    )

    self.connection.commit()

# =====================================================
# User Activity
# =====================================================

def log_user_activity(
    self,
    user_id: str,
    action: str,
    conversation_id: Optional[str] = None,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        INSERT INTO user_activity
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            user_id,
            action,
            conversation_id,
            datetime.utcnow().isoformat(),
        ),
    )

    self.connection.commit()

# =====================================================
# Cost Tracking
# =====================================================

def log_cost(
    self,
    conversation_id: str,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    total_cost: float,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        INSERT INTO costs
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            conversation_id,
            provider,
            model,
            input_tokens,
            output_tokens,
            total_cost,
            datetime.utcnow().isoformat(),
        ),
    )

    self.connection.commit()

# =====================================================
# Model Statistics
# =====================================================

def model_statistics(self):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT
            provider,
            model,
            COUNT(*) AS usage_count
        FROM model_usage
        GROUP BY provider, model
        ORDER BY usage_count DESC
        """
    )

    rows = cursor.fetchall()

    return [dict(row) for row in rows]

# =====================================================
# Provider Statistics
# =====================================================

def provider_statistics(self):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT
            provider,
            COUNT(*) AS requests
        FROM model_usage
        GROUP BY provider
        ORDER BY requests DESC
        """
    )

    return [
        dict(row)
        for row in cursor.fetchall()
    ]

# =====================================================
# Conversation Statistics
# =====================================================

def conversation_statistics(
    self,
    conversation_id: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT
            COUNT(*) AS requests
        FROM chat_events
        WHERE conversation_id=?
        """,
        (conversation_id,),
    )

    requests = cursor.fetchone()["requests"]

    cursor.execute(
        """
        SELECT
            SUM(total_tokens) AS tokens
        FROM token_usage
        WHERE conversation_id=?
        """,
        (conversation_id,),
    )

    row = cursor.fetchone()

    return {

        "conversation_id": conversation_id,

        "requests": requests,

        "tokens": row["tokens"] or 0,

    }


# =====================================================
# User Activity Summary
# =====================================================

def user_statistics(
    self,
    user_id: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT
            COUNT(*) AS activity
        FROM user_activity
        WHERE user_id=?
        """,
        (user_id,),
    )

    activity = cursor.fetchone()["activity"]

    cursor.execute(
        """
        SELECT
            COUNT(DISTINCT conversation_id)
            AS conversations
        FROM chat_events
        WHERE user_id=?
        """,
        (user_id,),
    )

    conversations = cursor.fetchone()["conversations"]

    return {

        "user_id": user_id,

        "activities": activity,

        "conversations": conversations,

    }


# =====================================================
# Daily Report
# =====================================================

def daily_report(
    self,
    date: Optional[str] = None,
):

    if date is None:

        date = datetime.utcnow().strftime("%Y-%m-%d")

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT
            COUNT(*) AS requests
        FROM chat_events
        WHERE DATE(created_at)=?
        """,
        (date,),
    )

    requests = cursor.fetchone()["requests"]

    cursor.execute(
        """
        SELECT
            SUM(total_tokens)
            AS tokens
        FROM token_usage
        WHERE DATE(created_at)=?
        """,
        (date,),
    )

    tokens = cursor.fetchone()["tokens"]

    return {

        "date": date,

        "requests": requests,

        "tokens": tokens or 0,

    }

def dashboard(self):

    return {

        "total_chats":

            self.total_chats(),

        "total_tokens":

            self.total_tokens(),

        "average_latency_ms":

            self.average_latency(),

        "providers":

            self.provider_statistics(),

        "models":

            self.model_statistics(),

    }

# =====================================================
# RAG Analytics
# =====================================================

cursor.execute("""

CREATE TABLE IF NOT EXISTS rag_queries(

    query_id TEXT PRIMARY KEY,

    conversation_id TEXT,

    user_id TEXT,

    query TEXT,

    retrieved_chunks INTEGER,

    similarity_score REAL,

    retrieval_time_ms REAL,

    created_at TEXT

)

""")

# =====================================================
# Memory Analytics
# =====================================================

cursor.execute("""

CREATE TABLE IF NOT EXISTS memory_usage(

    memory_id TEXT PRIMARY KEY,

    conversation_id TEXT,

    user_id TEXT,

    memories_used INTEGER,

    memory_tokens INTEGER,

    created_at TEXT

)

""")

self.connection.commit()

# =====================================================
# Log RAG Query
# =====================================================

def log_rag_query(
    self,
    conversation_id: str,
    user_id: str,
    query: str,
    retrieved_chunks: int,
    similarity_score: float,
    retrieval_time_ms: float,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        INSERT INTO rag_queries
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            conversation_id,
            user_id,
            query,
            retrieved_chunks,
            similarity_score,
            retrieval_time_ms,
            datetime.utcnow().isoformat(),
        ),
    )

    self.connection.commit()

# =====================================================
# Memory Usage
# =====================================================

def log_memory_usage(
    self,
    conversation_id: str,
    user_id: str,
    memories_used: int,
    memory_tokens: int,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        INSERT INTO memory_usage
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            conversation_id,
            user_id,
            memories_used,
            memory_tokens,
            datetime.utcnow().isoformat(),
        ),
    )

    self.connection.commit()

# =====================================================
# RAG Statistics
# =====================================================

def rag_statistics(self):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT

            COUNT(*) AS queries,

            AVG(retrieved_chunks)
                AS average_chunks,

            AVG(similarity_score)
                AS average_similarity,

            AVG(retrieval_time_ms)
                AS average_latency

        FROM rag_queries
        """
    )

    row = cursor.fetchone()

    return dict(row)

# =====================================================
# Memory Statistics
# =====================================================

def memory_statistics(self):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT

            COUNT(*) AS requests,

            AVG(memories_used)
                AS average_memories,

            AVG(memory_tokens)
                AS average_tokens

        FROM memory_usage
        """
    )

    return dict(cursor.fetchone())

# =====================================================
# Top Users
# =====================================================

def top_users(
    self,
    limit: int = 10,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT

            user_id,

            COUNT(*) AS requests

        FROM chat_events

        GROUP BY user_id

        ORDER BY requests DESC

        LIMIT ?
        """,
        (limit,),
    )

    return [
        dict(row)
        for row in cursor.fetchall()
    ]

# =====================================================
# Top Models
# =====================================================

def top_models(
    self,
    limit: int = 10,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT

            provider,

            model,

            COUNT(*) AS usage

        FROM model_usage

        GROUP BY provider, model

        ORDER BY usage DESC

        LIMIT ?
        """,
        (limit,),
    )

    return [
        dict(row)
        for row in cursor.fetchall()
    ]

# =====================================================
# Performance
# =====================================================

def performance_report(self):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT

            MIN(duration_ms)
                AS fastest,

            MAX(duration_ms)
                AS slowest,

            AVG(duration_ms)
                AS average

        FROM latency
        """
    )

    return dict(cursor.fetchone())

# =====================================================
# Daily Trend
# =====================================================

def daily_trend(
    self,
    days: int = 30,
):

    cursor = self.connection.cursor()

    cursor.execute(
        f"""
        SELECT

            DATE(created_at) AS day,

            COUNT(*) AS requests

        FROM chat_events

        GROUP BY DATE(created_at)

        ORDER BY day DESC

        LIMIT {days}
        """
    )

    return [
        dict(row)
        for row in cursor.fetchall()
    ]

def dashboard(self):

    return {

        "chat": {

            "total":

                self.total_chats(),

            "tokens":

                self.total_tokens(),

            "latency":

                self.average_latency(),

        },

        "rag":

            self.rag_statistics(),

        "memory":

            self.memory_statistics(),

        "providers":

            self.provider_statistics(),

        "models":

            self.model_statistics(),

        "performance":

            self.performance_report(),

        "top_users":

            self.top_users(),

        "top_models":

            self.top_models(),

    }

# =====================================================
# Export Dashboard
# =====================================================

import json

def export_dashboard_json(
    self,
    output_path: str,
):

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            self.dashboard(),
            file,
            indent=4,
        )

    logger.info(
        "Dashboard exported to %s",
        output_path,
    )

# =====================================================
# Export Chat Events
# =====================================================

import csv

def export_chat_events_csv(
    self,
    output_path: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM chat_events
        """
    )

    rows = cursor.fetchall()

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(rows[0].keys() if rows else [])

        for row in rows:

            writer.writerow(row)

    logger.info(
        "Chat events exported."
    )

# =====================================================
# Export Token Usage
# =====================================================

def export_token_usage_csv(
    self,
    output_path: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM token_usage
        """
    )

    rows = cursor.fetchall()

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(rows[0].keys() if rows else [])

        for row in rows:

            writer.writerow(row)


# =====================================================
# Database Statistics
# =====================================================

def database_statistics(self):

    cursor = self.connection.cursor()

    tables = [

        "chat_events",
        "token_usage",
        "latency",
        "model_usage",
        "costs",
        "user_activity",
        "rag_queries",
        "memory_usage",

    ]

    stats = {}

    for table in tables:

        cursor.execute(

            f"SELECT COUNT(*) FROM {table}"

        )

        stats[table] = cursor.fetchone()[0]

    return stats

# =====================================================
# Diagnostics
# =====================================================

def diagnostics(self):

    return {

        "database":

            str(self.database),

        "database_statistics":

            self.database_statistics(),

        "dashboard":

            self.dashboard(),

        "status":

            "healthy",

    }

# =====================================================
# Cleanup
# =====================================================

def cleanup(
    self,
    days: int = 90,
):

    cursor = self.connection.cursor()

    tables = [

        "chat_events",
        "token_usage",
        "latency",
        "model_usage",
        "costs",
        "user_activity",
        "rag_queries",
        "memory_usage",

    ]

    for table in tables:

        cursor.execute(

            f"""
            DELETE FROM {table}

            WHERE DATE(created_at)
            < DATE('now', ?)
            """,

            (f"-{days} day",),

        )

    self.connection.commit()

    logger.info(
        "Old analytics removed."
    )

# =====================================================
# Reset
# =====================================================

def reset(self):

    cursor = self.connection.cursor()

    tables = [

        "chat_events",
        "token_usage",
        "latency",
        "model_usage",
        "costs",
        "user_activity",
        "rag_queries",
        "memory_usage",

    ]

    for table in tables:

        cursor.execute(

            f"DELETE FROM {table}"

        )

    self.connection.commit()

    logger.warning(
        "Analytics database reset."
    )

# =====================================================
# Service Info
# =====================================================

def info(self):

    return {

        "service":

            "AnalyticsService",

        "version":

            "1.0.0",

        "database":

            str(self.database),

        "dashboard":

            self.dashboard(),

    }

# =====================================================
# Health
# =====================================================

def health(self):

    return {

        "status":

            "healthy",

        "database":

            str(self.database),

        "tables":

            self.database_statistics(),

    }

# =====================================================
# Close
# =====================================================

def close(self):

    self.connection.close()

    logger.info(

        "AnalyticsService shutdown."

    )

