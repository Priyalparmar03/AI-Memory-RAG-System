"""
Creates realistic demo data for the
AI Memory RAG System.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker

from database.database import Database

fake = Faker()

db = Database()

# ==========================================================
# Seeder
# ==========================================================


class DatabaseSeeder:

    def __init__(self):

        self.db = db

    # ======================================================
    # Random Helpers
    # ======================================================

    def random_date(
        self,
        days: int = 90,
    ):

        return datetime.now() - timedelta(

            days=random.randint(0, days),

            hours=random.randint(0, 23),

            minutes=random.randint(0, 59),

        )

    def random_uuid(self):

        return str(uuid.uuid4())

    # ======================================================
    # Users
    # ======================================================

    def seed_users(
        self,
        count: int = 10,
    ):

        print(f"Creating {count} users...")

        users = []

        for _ in range(count):

            username = fake.user_name()

            email = fake.email()

            password = (
                fake.sha256()
            )

            role = random.choice(

                [

                    "user",

                    "admin",

                ]

            )

            created = self.random_date()

            user_id = self.db.insert(

                """
                INSERT INTO users(

                    username,

                    email,

                    password_hash,

                    role,

                    created_at

                )

                VALUES(

                    ?,?,?,?,?

                )
                """,

                (

                    username,

                    email,

                    password,

                    role,

                    created.isoformat(),

                ),

            )

            users.append(

                {

                    "id": user_id,

                    "username": username,

                }

            )

        print(

            f"Created {len(users)} users."

        )

        return users

    # ======================================================
    # Clear Database
    # ======================================================

    def clear(self):

        tables = [

            "messages",

            "conversations",

            "memory",

            "sessions",

            "login_history",

            "trusted_devices",

            "failed_logins",

            "users",

        ]

        for table in tables:

            if self.db.table_exists(table):

                self.db.execute(

                    f"DELETE FROM {table}"

                )

        print("Database cleared.")

    # ======================================================
    # Run
    # ======================================================

    def run(self):

        print("=" * 60)

        print("AI Memory RAG Seeder")

        print("=" * 60)

        self.db.initialize()

        self.clear()

        users = self.seed_users(10)

        print()

        print(

            f"Users Created : {len(users)}"

        )

        print()

        print("Done.")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    DatabaseSeeder().run()

# ======================================================
# Sessions
# ======================================================

def seed_sessions(
    self,
    users: list[dict],
):

    print("Creating sessions...")

    sessions = []

    for user in users:

        session_id = self.random_uuid()

        access_token = fake.sha256()

        refresh_token = fake.sha256()

        created = self.random_date()

        expires = created + timedelta(days=7)

        self.db.insert(

            """
            INSERT INTO sessions(

                session_id,

                user_id,

                access_token,

                refresh_token,

                device,

                ip_address,

                user_agent,

                created_at,

                last_activity,

                expires_at,

                is_active

            )

            VALUES(

                ?,?,?,?,?,?,?,?,?,?,?

            )

            """,

            (

                session_id,

                user["id"],

                access_token,

                refresh_token,

                fake.word(),

                fake.ipv4(),

                fake.user_agent(),

                created.isoformat(),

                created.isoformat(),

                expires.isoformat(),

                1,

            ),

        )

        sessions.append(session_id)

    print(f"Created {len(sessions)} sessions.")

    return sessions


# ======================================================
# Sessions
# ======================================================

def seed_sessions(
    self,
    users: list[dict],
):

    print("Creating sessions...")

    sessions = []

    for user in users:

        session_id = self.random_uuid()

        access_token = fake.sha256()

        refresh_token = fake.sha256()

        created = self.random_date()

        expires = created + timedelta(days=7)

        self.db.insert(

            """
            INSERT INTO sessions(

                session_id,

                user_id,

                access_token,

                refresh_token,

                device,

                ip_address,

                user_agent,

                created_at,

                last_activity,

                expires_at,

                is_active

            )

            VALUES(

                ?,?,?,?,?,?,?,?,?,?,?

            )

            """,

            (

                session_id,

                user["id"],

                access_token,

                refresh_token,

                fake.word(),

                fake.ipv4(),

                fake.user_agent(),

                created.isoformat(),

                created.isoformat(),

                expires.isoformat(),

                1,

            ),

        )

        sessions.append(session_id)

    print(f"Created {len(sessions)} sessions.")

    return sessions

# ======================================================
# Conversations
# ======================================================

def seed_conversations(

    self,

    users,

    per_user: int = 3,

):

    print("Creating conversations...")

    conversations = []

    topics = [

        "Machine Learning",

        "Python",

        "Docker",

        "Flask",

        "RAG",

        "Vector Database",

        "LangChain",

        "Deep Learning",

    ]

    for user in users:

        for _ in range(per_user):

            conversation_id = self.random_uuid()

            title = random.choice(topics)

            created = self.random_date()

            self.db.insert(

                """
                INSERT INTO conversations(

                    conversation_id,

                    user_id,

                    title,

                    created_at,

                    updated_at

                )

                VALUES(

                    ?,?,?,?,?

                )

                """,

                (

                    conversation_id,

                    user["id"],

                    title,

                    created.isoformat(),

                    created.isoformat(),

                ),

            )

            conversations.append(

                {

                    "id": conversation_id,

                    "user": user["id"],

                }

            )

    print(

        f"Created {len(conversations)} conversations."

    )

    return conversations

# ======================================================
# Messages
# ======================================================

def seed_messages(

    self,

    conversations,

):

    print("Creating messages...")

    prompts = [

        "Explain AI",

        "What is Docker?",

        "How does RAG work?",

        "Explain Transformers",

        "What is Python?",

        "How do embeddings work?",

    ]

    replies = [

        "Artificial Intelligence is...",

        "Docker is a container platform...",

        "RAG combines retrieval with LLMs...",

        "Transformers use attention...",

        "Python is a programming language...",

        "Embeddings are vector representations...",

    ]

    total = 0

    for conversation in conversations:

        count = random.randint(4, 10)

        for _ in range(count):

            user_message = random.choice(prompts)

            assistant_message = random.choice(replies)

            self.db.insert(

                """
                INSERT INTO messages(

                    conversation_id,

                    role,

                    content,

                    created_at

                )

                VALUES(

                    ?,?,?,?

                )

                """,

                (

                    conversation["id"],

                    "user",

                    user_message,

                    self.random_date().isoformat(),

                ),

            )

            self.db.insert(

                """
                INSERT INTO messages(

                    conversation_id,

                    role,

                    content,

                    created_at

                )

                VALUES(

                    ?,?,?,?

                )

                """,

                (

                    conversation["id"],

                    "assistant",

                    assistant_message,

                    self.random_date().isoformat(),

                ),

            )

            total += 2

    print(f"Created {total} messages.")

# ======================================================
# Memory
# ======================================================

def seed_memory(

    self,

    users,

):

    print("Creating memories...")

    memories = [

        "Interested in AI",

        "Prefers Python",

        "Learning Docker",

        "Uses Flask",

        "Researching RAG",

        "Likes Deep Learning",

        "Working on NLP",

    ]

    total = 0

    for user in users:

        for _ in range(random.randint(2, 6)):

            self.db.insert(

                """
                INSERT INTO memory(

                    user_id,

                    content,

                    importance,

                    created_at

                )

                VALUES(

                    ?,?,?,?

                )

                """,

                (

                    user["id"],

                    random.choice(memories),

                    random.randint(1, 10),

                    self.random_date().isoformat(),

                ),

            )

            total += 1

    print(

        f"Created {total} memories."

    )

