import sqlite3
from datetime import datetime
from loguru import logger

logger.add("./logs/store_history.log", rotation="10 MB")

class ConversationDatabase:
    def __init__(self, db_file="conversation_history.db"):
        self.db_file = db_file
        self._initialize_database()

    def _get_connection(self):
        return sqlite3.connect(self.db_file)

    def _initialize_database(self):
        """Create the database table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS conversation_history (
            user_id TEXT PRIMARY KEY, -- Ensure one entry per user_id
            history TEXT NOT NULL,    -- Stores the complete conversation as a single text blob
            timestamp TEXT NOT NULL   -- Updated with the latest message timestamp
        );
        """
        with self._get_connection() as conn:
            conn.execute(query)
            conn.commit()

    def save_or_update_message(self, user_id, message):
        """
        Save a new message to the database or update the existing conversation.
        """
        timestamp = datetime.now().isoformat()
        query_select = "SELECT history FROM conversation_history WHERE user_id = ?;"
        query_insert = "INSERT INTO conversation_history (user_id, history, timestamp) VALUES (?, ?, ?);"
        query_update = "UPDATE conversation_history SET history = ?, timestamp = ? WHERE user_id = ?;"

        with self._get_connection() as conn:
            cursor = conn.execute(query_select, (user_id,))
            row = cursor.fetchone()

            if row:
                # Append new message to existing history
                updated_history = row[0] + f"\n{message}"
                conn.execute(query_update, (updated_history, timestamp, user_id))
            else:
                # Create a new entry
                new_history = f"[{timestamp}] {message}"
                conn.execute(query_insert, (user_id, new_history, timestamp))
            conn.commit()

    def fetch_history(self, user_id):
        """
        Fetch the conversation history for a given user.
        """
        query = "SELECT history, timestamp FROM conversation_history WHERE user_id = ?;"
        with self._get_connection() as conn:
            cursor = conn.execute(query, (user_id,))
            return cursor.fetchone()

    def delete_user_history(self, user_id):
        """
        Delete a user's conversation history.
        """
        query = "DELETE FROM conversation_history WHERE user_id = ?;"
        with self._get_connection() as conn:
            cursor = conn.execute(query, (user_id,))
            conn.commit()
            return cursor.rowcount  # Number of rows deleted
    
    def clear_table(self):
        """
        Delete all rows from the conversation_history table while keeping its structure intact.
        """
        query = "DELETE FROM conversation_history;"
        with self._get_connection() as conn:
            conn.execute(query)
            conn.commit()
            logger.info("All records have been cleared, table structure retained.")


# For testing
if __name__ == "__main__":
    db = ConversationDatabase()

    user_id = "user123"
    
    # Save or update messages
    db.save_or_update_message(user_id, "Message 1")
    db.save_or_update_message(user_id, "Message 2")
    
    # Fetch history
    history = db.fetch_history(user_id)
    if history:
        print("Conversation History:")
        print(history[0])
        print(f"Last Updated: {history[1]}")
    else:
        print("No history found.")

    # Delete user history
    rows_deleted = db.delete_user_history(user_id)
    if rows_deleted > 0:
        print(f"Deleted {rows_deleted} row(s) for user {user_id}.")
    else:
        print(f"No records found for user {user_id}.")
    
    # Try fetching again to verify deletion
    history = db.fetch_history(user_id)
    if history:
        print("Conversation History after deletion:")
        print(history[0])
    else:
        print("No history found after deletion.")
