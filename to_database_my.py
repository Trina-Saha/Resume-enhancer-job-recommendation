import pandas as pd
import streamlit as st
import mysql.connector

# Establish MySQL connection
mysql_conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='job_database'
)
print("Connection successful")

# Create a cursor object to execute SQL queries
cursor = mysql_conn.cursor()

# Create the table if it doesn't exist
cursor.execute('CREATE TABLE IF NOT EXISTS links_table (JobID INT PRIMARY KEY, JobLink TEXT, Location VARCHAR(30), WorkType VARCHAR(30), Role VARCHAR(30), Skills VARCHAR(255), Company VARCHAR(255))')

# Read the DataFrame from CSV
df = pd.read_csv('Job_db2.csv')

# Define the SQL INSERT query template
insert_query = "INSERT INTO links_table (JobID, JobLink, Location, WorkType, Role, Skills, Company) VALUES (%s, %s, %s, %s, %s, %s, %s)"

# Iterate over each row in the DataFrame and insert data into the database
for index, row in df.iterrows():
    data = (row['JobID'], row['JobLink'], row['Location'], row['WorkType'], row['Role'], row['Skills'], row['Company'])
    cursor.execute(insert_query, data)

# Commit the transaction to save the changes
mysql_conn.commit()

# Close the cursor and MySQL connection
cursor.close()
mysql_conn.close()

print("Data inserted successfully into MySQL database.")
