import psycopg2
from psycopg2.extras import RealDictCursor

class DatabaseManager:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname="anomaly_detection",
            user="postgres",
            password="tokentimeismine",  # Replace with your PostgreSQL password
            host="localhost",
            port="5432"
        )
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        self.create_table()

    def create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                image_name VARCHAR(255) NOT NULL,
                score FLOAT NOT NULL,
                result VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    def save_prediction(self, image_name, score, result):
        self.cursor.execute("""
            INSERT INTO predictions (image_name, score, result)
            VALUES (%s, %s, %s)
            RETURNING id;
        """, (image_name, score, result))
        self.conn.commit()
        return self.cursor.fetchone()['id']

    def get_all_predictions(self):
        self.cursor.execute("""
            SELECT id, image_name, score, result, timestamp
            FROM predictions
            ORDER BY timestamp DESC;
        """)
        return self.cursor.fetchall()

    def close(self):
        self.cursor.close()
        self.conn.close()