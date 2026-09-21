import sqlite3
from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Database directory
DATABASE_DIR = PROJECT_ROOT / "databases"

# SQLite database file
DATABASE_PATH = DATABASE_DIR / "ids_predictions.db"


def get_connection():
    """
    Create and return a SQLite database connection.
    """
    connection = sqlite3.connect(DATABASE_PATH)

    # Return rows as dictionary-like objects
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    """
    Create the predictions table if it does not already exist.
    """

    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            predicted_class_id INTEGER NOT NULL,
            predicted_class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            time_steps INTEGER NOT NULL,
            features_per_step INTEGER NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()

def save_prediction(
    timestamp,
    predicted_class_id,
    predicted_class_name,
    confidence,
    time_steps,
    features_per_step
):
    """
    Save a prediction into the database.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO predictions (
            timestamp,
            predicted_class_id,
            predicted_class_name,
            confidence,
            time_steps,
            features_per_step
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            timestamp,
            predicted_class_id,
            predicted_class_name,
            confidence,
            time_steps,
            features_per_step
        )
    )

    prediction_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return prediction_id


def get_predictions(limit=100):
    """
    Retrieve recent predictions from the database.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            timestamp,
            predicted_class_id,
            predicted_class_name,
            confidence,
            time_steps,
            features_per_step
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )

    rows = cursor.fetchall()

    connection.close()

    return [dict(row) for row in rows]
if __name__ == "__main__":
    initialize_database()

    print("=" * 60)
    print("DATABASE INITIALIZATION")
    print("=" * 60)
    print(f"Database path: {DATABASE_PATH}")
    print("Database initialized successfully.")
    print("Table created: predictions")
    print("=" * 60)