import pyodbc
from datetime import datetime
import time
import os

def insert_record(data):
    """Insert record into the database."""
    db_username = os.getenv('DB_USERNAME')
    db_password = os.getenv('DB_PASSWORD')

    # Adjusted connection string with the correct ODBC driver
    connection_string = (
        r'DRIVER={SQL Server};'
        r'SERVER=127.0.0.1\SQLEXPRESS;'
        r'DATABASE=ragdb;'
        r'UID=' + db_username + ';'
        r'PWD=' + db_password
    )

    try:
        # Connect to the database
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        # SQL INSERT statement
        insert_stmt = """
        INSERT INTO RAGLOG (question, answer, date_enter, time_elape, temperature, topp, topk, model_name, local_rag, con_score, misc) 
        VALUES (?,?,?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        values = (
            data['question'], data['answer'],data['date_enter'], data['time_elape'], data['temperature'], data['topp'],
            data['topk'], data['model_name'], data['local_rag'], data['con_score'], data['misc']
        )

        # Execute the insert statement
        cursor.execute(insert_stmt, values)
        conn.commit()

        print("Record inserted successfully.")

    except pyodbc.Error as e:
        print("Error in connection", e)

    finally:
        # Ensure the connection is closed
        if conn:
            cursor.close()
            conn.close()