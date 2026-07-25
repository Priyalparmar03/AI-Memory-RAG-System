"""
services/session_service.py

Production Session Service

Responsibilities
----------------
- User Session Management
- JWT Session Tracking
- Active Session Management
- User Activity Tracking
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class SessionService:

    DATABASE_NAME = "sessions.db"

    def __init__(
        self,
        database_path: str = "./database",
        session_duration_hours: int = 24,
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

        self.session_duration = session_duration_hours

        self.connection = sqlite3.connect(
            self.database,
            check_same_thread=False,
        )

        self.connection.row_factory = sqlite3.Row

        self._initialize_database()

        logger.info(
            "SessionService initialized."
        )

  def _initialize_database(self):

    cursor = self.connection.cursor()

    cursor.execute("""

    CREATE TABLE IF NOT EXISTS sessions(

        session_id TEXT PRIMARY KEY,

        user_id TEXT NOT NULL,

        access_token TEXT,

        refresh_token TEXT,

        device TEXT,

        ip_address TEXT,

        user_agent TEXT,

        created_at TEXT,

        last_activity TEXT,

        expires_at TEXT,

        is_active INTEGER DEFAULT 1

    )

    """)

    self.connection.commit()

def create_session(
    self,
    user_id: str,
    access_token: str,
    refresh_token: str,
    device: str = "Unknown",
    ip_address: str = "",
    user_agent: str = "",
):

    session_id = str(uuid.uuid4())

    now = datetime.utcnow()

    expires = now + timedelta(
        hours=self.session_duration
    )

    cursor = self.connection.cursor()

    cursor.execute(
        """
        INSERT INTO sessions
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            user_id,
            access_token,
            refresh_token,
            device,
            ip_address,
            user_agent,
            now.isoformat(),
            now.isoformat(),
            expires.isoformat(),
            1,
        ),
    )

    self.connection.commit()

    logger.info(
        "Session created: %s",
        session_id,
    )

    return session_id

def get_session(
    self,
    session_id: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM sessions
        WHERE session_id=?
        """,
        (session_id,),
    )

    row = cursor.fetchone()

    if row is None:

        return None

    return dict(row)

def update_activity(
    self,
    session_id: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        UPDATE sessions
        SET last_activity=?
        WHERE session_id=?
        """,
        (
            datetime.utcnow().isoformat(),
            session_id,
        ),
    )

    self.connection.commit()

def delete_session(
    self,
    session_id: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        DELETE FROM sessions
        WHERE session_id=?
        """,
        (session_id,),
    )

    self.connection.commit()

    logger.info(
        "Session deleted: %s",
        session_id,
    )

def active_sessions(
    self,
    user_id: Optional[str] = None,
):

    cursor = self.connection.cursor()

    if user_id:

        cursor.execute(
            """
            SELECT *
            FROM sessions
            WHERE user_id=?
            AND is_active=1
            """,
            (user_id,),
        )

    else:

        cursor.execute(
            """
            SELECT *
            FROM sessions
            WHERE is_active=1
            """
        )

    rows = cursor.fetchall()

    return [
        dict(row)
        for row in rows
    ]

def total_sessions(self):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM sessions
        """
    )

    return cursor.fetchone()[0]

def dashboard(self):

    return {

        "active_sessions":

            len(
                self.active_sessions()
            ),

        "total_sessions":

            self.total_sessions(),

    }

def health(self):

    return {

        "status":

            "healthy",

        "database":

            str(self.database),

        "dashboard":

            self.dashboard(),

    }

def close(self):

    self.connection.close()

    logger.info(
        "SessionService shutdown."
    )


# =====================================================
# Indexes
# =====================================================

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_session_user
ON sessions(user_id)
""")

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_access_token
ON sessions(access_token)
""")

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_refresh_token
ON sessions(refresh_token)
""")

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_expiry
ON sessions(expires_at)
""")

cursor.execute("""
CREATE INDEX IF NOT EXISTS idx_active
ON sessions(is_active)
""")

# =====================================================
# Validate Access Token
# =====================================================

def validate_access_token(
    self,
    access_token: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM sessions
        WHERE access_token=?
        AND is_active=1
        """,
        (access_token,),
    )

    row = cursor.fetchone()

    if row is None:

        return None

    session = dict(row)

    expires = datetime.fromisoformat(
        session["expires_at"]
    )

    if expires < datetime.utcnow():

        self.expire_session(
            session["session_id"]
        )

        return None

    return session

# =====================================================
# Validate Refresh Token
# =====================================================

def validate_refresh_token(
    self,
    refresh_token: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM sessions
        WHERE refresh_token=?
        AND is_active=1
        """,
        (refresh_token,),
    )

    row = cursor.fetchone()

    if row is None:

        return None

    return dict(row)
  # =====================================================
# Refresh Token Rotation
# =====================================================

def rotate_refresh_token(
    self,
    session_id: str,
    new_refresh_token: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        UPDATE sessions
        SET refresh_token=?
        WHERE session_id=?
        """,
        (
            new_refresh_token,
            session_id,
        ),
    )

    self.connection.commit()

    logger.info(
        "Refresh token rotated: %s",
        session_id,
    )

# =====================================================
# Expire Session
# =====================================================

def expire_session(
    self,
    session_id: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        UPDATE sessions
        SET is_active=0
        WHERE session_id=?
        """,
        (session_id,),
    )

    self.connection.commit()

    logger.info(
        "Session expired: %s",
        session_id,
    )

# =====================================================
# Logout
# =====================================================

def logout(
    self,
    session_id: str,
):

    self.expire_session(
        session_id
    )

    logger.info(
        "User logged out."
    )

# =====================================================
# Logout All Devices
# =====================================================

def logout_all(
    self,
    user_id: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        UPDATE sessions
        SET is_active=0
        WHERE user_id=?
        """,
        (user_id,),
    )

    self.connection.commit()

    logger.info(
        "Logged out all sessions for %s",
        user_id,
    )

# =====================================================
# Heartbeat
# =====================================================

def heartbeat(
    self,
    session_id: str,
):

    self.update_activity(
        session_id
    )

    return {

        "status": "alive",

        "timestamp":

            datetime.utcnow().isoformat(),

    }

# =====================================================
# Idle Time
# =====================================================

def idle_time(
    self,
    session_id: str,
):

    session = self.get_session(
        session_id
    )

    if session is None:

        return None

    last = datetime.fromisoformat(
        session["last_activity"]
    )

    idle = (

        datetime.utcnow()

        - last

    ).total_seconds()

    return round(idle, 2)

# =====================================================
# Session Duration
# =====================================================

def session_duration(
    self,
    session_id: str,
):

    session = self.get_session(
        session_id
    )

    if session is None:

        return None

    created = datetime.fromisoformat(
        session["created_at"]
    )

    duration = (

        datetime.utcnow()

        - created

    ).total_seconds()

    return round(duration, 2)

# =====================================================
# Cleanup Expired
# =====================================================

def cleanup_expired_sessions(self):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        UPDATE sessions
        SET is_active=0
        WHERE expires_at < ?
        """,
        (
            datetime.utcnow().isoformat(),
        ),
    )

    self.connection.commit()

    logger.info(
        "Expired sessions cleaned."
    )

# =====================================================
# Cleanup Expired
# =====================================================

def cleanup_expired_sessions(self):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        UPDATE sessions
        SET is_active=0
        WHERE expires_at < ?
        """,
        (
            datetime.utcnow().isoformat(),
        ),
    )

    self.connection.commit()

    logger.info(
        "Expired sessions cleaned."
    )

def dashboard(self):

    return {

        "active_sessions":

            len(
                self.active_sessions()
            ),

        "total_sessions":

            self.total_sessions(),

        "expired_sessions":

            self.total_sessions()

            - len(
                self.active_sessions()
            ),

    }


# =====================================================
# Active Conversations
# =====================================================

cursor.execute("""

CREATE TABLE IF NOT EXISTS active_conversations(

    session_id TEXT PRIMARY KEY,

    conversation_id TEXT,

    updated_at TEXT

)

""")

# =====================================================
# Login History
# =====================================================

cursor.execute("""

CREATE TABLE IF NOT EXISTS login_history(

    login_id TEXT PRIMARY KEY,

    user_id TEXT,

    session_id TEXT,

    ip_address TEXT,

    device TEXT,

    browser TEXT,

    login_time TEXT

)

""")

# =====================================================
# Trusted Devices
# =====================================================

cursor.execute("""

CREATE TABLE IF NOT EXISTS trusted_devices(

    device_id TEXT PRIMARY KEY,

    user_id TEXT,

    device_name TEXT,

    browser TEXT,

    operating_system TEXT,

    ip_address TEXT,

    trusted INTEGER DEFAULT 1,

    created_at TEXT

)

""")

self.connection.commit()


# =====================================================
# Active Conversation
# =====================================================

def set_active_conversation(
    self,
    session_id: str,
    conversation_id: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE
        INTO active_conversations
        VALUES (?, ?, ?)
        """,
        (
            session_id,
            conversation_id,
            datetime.utcnow().isoformat(),
        ),
    )

    self.connection.commit()

def get_active_conversation(
    self,
    session_id: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT conversation_id
        FROM active_conversations
        WHERE session_id=?
        """,
        (session_id,),
    )

    row = cursor.fetchone()

    if row is None:
        return None

    return row["conversation_id"]

def clear_active_conversation(
    self,
    session_id: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        DELETE
        FROM active_conversations
        WHERE session_id=?
        """,
        (session_id,),
    )

    self.connection.commit()

# =====================================================
# Trusted Device
# =====================================================

def register_device(
    self,
    user_id: str,
    device_name: str,
    browser: str,
    operating_system: str,
    ip_address: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        INSERT INTO trusted_devices
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            user_id,
            device_name,
            browser,
            operating_system,
            ip_address,
            1,
            datetime.utcnow().isoformat(),
        ),
    )

    self.connection.commit()

def trusted_devices(
    self,
    user_id: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM trusted_devices
        WHERE user_id=?
        AND trusted=1
        """,
        (user_id,),
    )

    return [
        dict(row)
        for row in cursor.fetchall()
    ]

# =====================================================
# Login History
# =====================================================

def log_login(
    self,
    user_id: str,
    session_id: str,
    ip_address: str,
    device: str,
    browser: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        INSERT INTO login_history
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            user_id,
            session_id,
            ip_address,
            device,
            browser,
            datetime.utcnow().isoformat(),
        ),
    )

    self.connection.commit()

def login_history(
    self,
    user_id: str,
    limit: int = 20,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM login_history
        WHERE user_id=?
        ORDER BY login_time DESC
        LIMIT ?
        """,
        (
            user_id,
            limit,
        ),
    )

    return [
        dict(row)
        for row in cursor.fetchall()
    ]

# =====================================================
# Device Statistics
# =====================================================

def device_statistics(self):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT

            device,

            COUNT(*) AS users

        FROM login_history

        GROUP BY device

        ORDER BY users DESC
        """
    )

    return [
        dict(row)
        for row in cursor.fetchall()
    ]

def browser_statistics(self):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT

            browser,

            COUNT(*) AS users

        FROM login_history

        GROUP BY browser

        ORDER BY users DESC
        """
    )

    return [
        dict(row)
        for row in cursor.fetchall()
    ]

# =====================================================
# Session Analytics
# =====================================================

def session_statistics(self):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT

            COUNT(*) AS sessions,

            COUNT(
                DISTINCT user_id
            ) AS users

        FROM sessions

        WHERE is_active=1
        """
    )

    return dict(
        cursor.fetchone()
    )

def dashboard(self):

    return {

        "active_sessions":

            len(
                self.active_sessions()
            ),

        "total_sessions":

            self.total_sessions(),

        "session_statistics":

            self.session_statistics(),

        "device_statistics":

            self.device_statistics(),

        "browser_statistics":

            self.browser_statistics(),

    }

# =====================================================
# Failed Login Attempts
# =====================================================

cursor.execute("""

CREATE TABLE IF NOT EXISTS failed_logins(

    failure_id TEXT PRIMARY KEY,

    user_id TEXT,

    ip_address TEXT,

    reason TEXT,

    created_at TEXT

)

""")

# =====================================================
# Security Alerts
# =====================================================

cursor.execute("""

CREATE TABLE IF NOT EXISTS security_alerts(

    alert_id TEXT PRIMARY KEY,

    user_id TEXT,

    severity TEXT,

    description TEXT,

    created_at TEXT

)

""")

self.connection.commit()

# =====================================================
# Failed Login Attempts
# =====================================================

cursor.execute("""

CREATE TABLE IF NOT EXISTS failed_logins(

    failure_id TEXT PRIMARY KEY,

    user_id TEXT,

    ip_address TEXT,

    reason TEXT,

    created_at TEXT

)

""")

# =====================================================
# Security Alerts
# =====================================================

cursor.execute("""

CREATE TABLE IF NOT EXISTS security_alerts(

    alert_id TEXT PRIMARY KEY,

    user_id TEXT,

    severity TEXT,

    description TEXT,

    created_at TEXT

)

""")

self.connection.commit()

# =====================================================
# Security Alert
# =====================================================

def create_security_alert(
    self,
    user_id: str,
    severity: str,
    description: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        INSERT INTO security_alerts
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            user_id,
            severity,
            description,
            datetime.utcnow().isoformat(),
        ),
    )

    self.connection.commit()

# =====================================================
# Suspicious Activity
# =====================================================

def suspicious_activity(
    self,
    user_id: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM failed_logins
        WHERE user_id=?
        AND DATE(created_at)=DATE('now')
        """,
        (user_id,),
    )

    failures = cursor.fetchone()[0]

    if failures >= 5:

        self.create_security_alert(
            user_id=user_id,
            severity="HIGH",
            description="Too many failed logins",
        )

        return True

    return False

# =====================================================
# Revoke Device
# =====================================================

def revoke_device(
    self,
    device_id: str,
):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        UPDATE trusted_devices
        SET trusted=0
        WHERE device_id=?
        """,
        (device_id,),
    )

    self.connection.commit()

# =====================================================
# Security Statistics
# =====================================================

def security_statistics(self):

    cursor = self.connection.cursor()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM failed_logins
        """
    )

    failed = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM security_alerts
        """
    )

    alerts = cursor.fetchone()[0]

    return {

        "failed_logins": failed,

        "security_alerts": alerts,

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

        "login_history",

        "failed_logins",

        "security_alerts",

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

    self.cleanup_expired_sessions()

    self.connection.commit()

    logger.info(
        "Session cleanup completed."
    )

# =====================================================
# Diagnostics
# =====================================================

def diagnostics(self):

    return {

        "database":

            str(self.database),

        "dashboard":

            self.dashboard(),

        "security":

            self.security_statistics(),

        "status":

            "healthy",

    }

# =====================================================
# Reset
# =====================================================

def reset(self):

    cursor = self.connection.cursor()

    tables = [

        "sessions",

        "active_conversations",

        "trusted_devices",

        "login_history",

        "failed_logins",

        "security_alerts",

    ]

    for table in tables:

        cursor.execute(

            f"DELETE FROM {table}"

        )

    self.connection.commit()

    logger.warning(
        "SessionService reset."
    )

# =====================================================
# Info
# =====================================================

def info(self):

    return {

        "service":

            "SessionService",

        "version":

            "1.0.0",

        "database":

            str(self.database),

        "dashboard":

            self.dashboard(),

        "security":

            self.security_statistics(),

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

        "sessions":

            self.total_sessions(),

        "security":

            self.security_statistics(),

    }

# =====================================================
# Close
# =====================================================

def close(self):

    self.connection.close()

    logger.info(
        "SessionService shutdown."
    )

