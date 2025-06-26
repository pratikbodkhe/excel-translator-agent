import time

import psycopg2

from config.config import config


def create_user_and_database():
    """Create database user and database if they don't exist"""
    admin_conn = None
    try:
        # Connect to default postgres database using postgres superuser
        admin_conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            user="postgres",
            password="",
            dbname="postgres"
        )
        admin_conn.autocommit = True
        cursor = admin_conn.cursor()

        # Create user if not exists
        cursor.execute(f"SELECT 1 FROM pg_roles WHERE rolname='{config.POSTGRES_USER}';")
        if not cursor.fetchone():
            cursor.execute(f"CREATE USER {config.POSTGRES_USER} WITH PASSWORD '{config.POSTGRES_PASSWORD}';")
            print(f"Created user: {config.POSTGRES_USER}")

        # Create database if not exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname='{config.POSTGRES_DB}';")
        if not cursor.fetchone():
            cursor.execute(f"CREATE DATABASE {config.POSTGRES_DB} OWNER {config.POSTGRES_USER};")
            print(f"Created database: {config.POSTGRES_DB}")

    except Exception as e:
        print(f"Error creating user/database: {e}")
        # Try to connect directly to the database if it already exists
        try:
            test_conn = psycopg2.connect(config.postgres_uri)
            test_conn.close()
            print("Database connection successful, proceeding...")
        except Exception as e2:
            print(f"Database connection also failed: {e2}")
            raise
    finally:
        if admin_conn:
            admin_conn.close()

def create_tables():
    """Create database tables if they don't exist"""
    max_retries = 5
    retry_delay = 3

    for attempt in range(max_retries):
        conn = None
        try:
            conn = psycopg2.connect(config.postgres_uri)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS translations (
                    id SERIAL PRIMARY KEY,
                    original TEXT NOT NULL,
                    context TEXT NOT NULL DEFAULT '',
                    translation TEXT NOT NULL,
                    last_used TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT unique_translation UNIQUE (original, context)
                )
            """)

            conn.commit()
            print("Database tables created successfully")
            return
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries exceeded. Failed to create tables.")
                raise
        finally:
            if conn:
                conn.close()

if __name__ == '__main__':
    create_user_and_database()
    create_tables()
