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

# ======================================================
# Documents
# ======================================================

def seed_documents(
    self,
    users,
    per_user: int = 3,
):

    print("Creating documents...")

    documents = []

    categories = [

        "Research Paper",
        "Documentation",
        "Tutorial",
        "Manual",
        "Guide",
        "Notes",

    ]

    for user in users:

        for _ in range(per_user):

            document_id = self.random_uuid()

            filename = fake.file_name(
                extension="pdf"
            )

            created = self.random_date()

            self.db.insert(

                """
                INSERT INTO documents(

                    document_id,
                    user_id,
                    filename,
                    category,
                    created_at

                )

                VALUES(?,?,?,?,?)

                """,

                (

                    document_id,
                    user["id"],
                    filename,
                    random.choice(categories),
                    created.isoformat(),

                ),

            )

            documents.append(

                {

                    "id": document_id,
                    "user": user["id"],

                }

            )

    print(f"Created {len(documents)} documents.")

    return documents

# ======================================================
# Document Chunks
# ======================================================

def seed_document_chunks(
    self,
    documents,
):

    print("Creating document chunks...")

    total = 0

    for document in documents:

        chunk_count = random.randint(5, 15)

        for index in range(chunk_count):

            self.db.insert(

                """
                INSERT INTO document_chunks(

                    document_id,
                    chunk_index,
                    content,
                    created_at

                )

                VALUES(?,?,?,?)

                """,

                (

                    document["id"],
                    index,

                    fake.paragraph(
                        nb_sentences=6
                    ),

                    self.random_date().isoformat(),

                ),

            )

            total += 1

    print(f"Created {total} chunks.")

# ======================================================
# Chat Events
# ======================================================

def seed_chat_events(
    self,
    users,
):

    print("Creating chat events...")

    events = [

        "message_sent",
        "conversation_created",
        "document_uploaded",
        "search",
        "memory_saved",

    ]

    total = 0

    for user in users:

        for _ in range(random.randint(15, 40)):

            self.db.insert(

                """
                INSERT INTO chat_events(

                    user_id,
                    event_type,
                    created_at

                )

                VALUES(?,?,?)

                """,

                (

                    user["id"],

                    random.choice(events),

                    self.random_date().isoformat(),

                ),

            )

            total += 1

    print(f"Created {total} chat events.")

# ======================================================
# Token Usage
# ======================================================

def seed_token_usage(
    self,
    users,
):

    print("Creating token usage...")

    total = 0

    for user in users:

        for _ in range(random.randint(25, 60)):

            prompt = random.randint(
                100,
                1500,
            )

            completion = random.randint(
                50,
                1200,
            )

            self.db.insert(

                """
                INSERT INTO token_usage(

                    user_id,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    created_at

                )

                VALUES(?,?,?,?,?)

                """,

                (

                    user["id"],

                    prompt,

                    completion,

                    prompt + completion,

                    self.random_date().isoformat(),

                ),

            )

            total += 1

    print(f"Created {total} token records.")

# ======================================================
# Latency
# ======================================================

def seed_latency(
    self,
    users,
):

    print("Creating latency...")

    total = 0

    for user in users:

        for _ in range(random.randint(20, 40)):

            self.db.insert(

                """
                INSERT INTO latency(

                    user_id,
                    latency_ms,
                    created_at

                )

                VALUES(?,?,?)

                """,

                (

                    user["id"],

                    round(
                        random.uniform(
                            50,
                            2500,
                        ),
                        2,
                    ),

                    self.random_date().isoformat(),

                ),

            )

            total += 1

    print(f"Created {total} latency records.")

# ======================================================
# Model Usage
# ======================================================

def seed_model_usage(
    self,
    users,
):

    print("Creating model usage...")

    models = [

        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gpt-4.1",
        "llama3",
        "mistral",

    ]

    total = 0

    for user in users:

        for _ in range(random.randint(15, 40)):

            self.db.insert(

                """
                INSERT INTO model_usage(

                    user_id,
                    model_name,
                    created_at

                )

                VALUES(?,?,?)

                """,

                (

                    user["id"],

                    random.choice(models),

                    self.random_date().isoformat(),

                ),

            )

            total += 1

    print(f"Created {total} model usage records.")

# ======================================================
# RAG Queries
# ======================================================

def seed_rag_queries(
    self,
    users,
):

    print("Creating RAG queries...")

    queries = [

        "Explain RAG",
        "Vector databases",
        "LangChain",
        "Docker",
        "Machine Learning",
        "Deep Learning",
        "Embeddings",

    ]

    total = 0

    for user in users:

        for _ in range(random.randint(8, 25)):

            self.db.insert(

                """
                INSERT INTO rag_queries(

                    user_id,
                    query,
                    similarity_score,
                    created_at

                )

                VALUES(?,?,?,?)

                """,

                (

                    user["id"],

                    random.choice(queries),

                    round(
                        random.uniform(
                            0.65,
                            0.99,
                        ),
                        3,
                    ),

                    self.random_date().isoformat(),

                ),

            )

            total += 1

    print(f"Created {total} RAG queries.")

# ======================================================
# Export History
# ======================================================

def seed_exports(
    self,
    users,
):
    """
    Seed export history.
    """

    print("Creating export history...")

    formats = [
        "json",
        "pdf",
        "docx",
        "html",
        "csv",
        "zip",
    ]

    total = 0

    for user in users:

        for _ in range(random.randint(5, 15)):

            self.db.insert(

                """
                INSERT INTO exports(

                    user_id,
                    filename,
                    format,
                    created_at

                )

                VALUES(
                    ?,?,?,?
                )
                """,

                (

                    user["id"],

                    fake.file_name(),

                    random.choice(formats),

                    self.random_date().isoformat(),

                ),

            )

            total += 1

    print(f"Created {total} exports.")

# ======================================================
# Cost Analytics
# ======================================================

def seed_costs(
    self,
    users,
):
    """
    Seed API cost analytics.
    """

    print("Creating cost analytics...")

    total = 0

    for user in users:

        for _ in range(random.randint(15, 40)):

            self.db.insert(

                """
                INSERT INTO costs(

                    user_id,
                    provider,
                    model_name,
                    amount,
                    created_at

                )

                VALUES(
                    ?,?,?,?,?
                )
                """,

                (

                    user["id"],

                    random.choice(
                        [
                            "OpenAI",
                            "Google",
                            "Anthropic",
                        ]
                    ),

                    random.choice(
                        [
                            "gpt-4.1",
                            "gemini-2.5-pro",
                            "claude-4",
                        ]
                    ),

                    round(
                        random.uniform(
                            0.001,
                            2.5,
                        ),
                        4,
                    ),

                    self.random_date().isoformat(),

                ),

            )

            total += 1

    print(f"Created {total} cost records.")


# ======================================================
# Memory Usage
# ======================================================

def seed_memory_usage(
    self,
    users,
):
    """
    Seed memory usage analytics.
    """

    print("Creating memory usage...")

    total = 0

    for user in users:

        for _ in range(random.randint(8, 20)):

            self.db.insert(

                """
                INSERT INTO memory_usage(

                    user_id,
                    memory_count,
                    storage_bytes,
                    created_at

                )

                VALUES(
                    ?,?,?,?
                )
                """,

                (

                    user["id"],

                    random.randint(
                        5,
                        200,
                    ),

                    random.randint(
                        1024,
                        100000,
                    ),

                    self.random_date().isoformat(),

                ),

            )

            total += 1

    print(f"Created {total} memory usage records.")


# ======================================================
# Summary
# ======================================================

def summary(self):

    print()

    print("=" * 70)

    print("DATABASE SUMMARY")

    print("=" * 70)

    tables = [

        "users",

        "sessions",

        "conversations",

        "messages",

        "memory",

        "documents",

        "document_chunks",

        "chat_events",

        "token_usage",

        "latency",

        "model_usage",

        "rag_queries",

        "exports",

        "costs",

        "memory_usage",

    ]

    for table in tables:

        if self.db.table_exists(table):

            rows = self.db.count(table)

            print(

                f"{table:<25}{rows:>10}"

            )

    print("=" * 70)

documents = self.seed_documents(users)

self.seed_document_chunks(
    documents
)

self.seed_chat_events(users)

self.seed_token_usage(users)

self.seed_latency(users)

self.seed_model_usage(users)

self.seed_rag_queries(users)

self.seed_exports(users)

self.seed_costs(users)

self.seed_memory_usage(users)

self.summary()

print()

print(
    "Database seeded successfully."
)

# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    try:

        DatabaseSeeder().run()

    except KeyboardInterrupt:

        print()

        print("Seeder cancelled.")

    except Exception as exc:

        print()

        print("Seeder failed:")

        print(exc)

        raise
