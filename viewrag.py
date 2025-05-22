import os
import pyodbc

# Retrieve database credentials from environment variables
db_username = os.getenv('DB_USERNAME')
db_password = os.getenv('DB_PASSWORD')

# SQL Server connection string
connection_string = (
    'DRIVER={SQL Server};'  # Or use the specific driver version you have, like 'ODBC Driver 17 for SQL Server'
    'SERVER=127.0.0.1\SQLEXPRESS;'
    'DATABASE=ragdb;'
    'UID=' + db_username + ';'
    'PWD=' + db_password
)

# SQL query to select records from RAGLOG ordered by RID descending
sql_query = 'SELECT * FROM RAGLOG ORDER BY RID DESC'

try:
    # Connect to the database
    with pyodbc.connect(connection_string) as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql_query)

            # Print column headers
            print(f"{'rid':<15}{'question':<25}{'answer':<25}{'date_enter':<20}{'time_elape':<15}"
                  f"{'temperature':<15}{'topp':<10}{'topk':<10}{'model_name':<20}"
                  f"{'local_rag':<15}{'con_score':<15}{'misc':<30}")
            print("-" * 150)
            
            # Fetch and print each row data
            for row in cursor.fetchall():
                # Access row fields by index since it's generally more reliable across different drivers
                print(f"{row[0]:<15}{str(row[1])[:24]:<25}{str(row[2])[:24]:<25}{str(row[3])[:19]:<20}"
                      f"{row[4]:<15}{row[5]:<15}{row[6]:<10}{row[7]:<10}{str(row[8])[:19]:<20}"
                      f"{str(row[9])[:14]:<15}{row[10]:<15}{str(row[11])[:29]:<30}")
except pyodbc.Error as e:
    print("Error in connection", e)
