import mysql.connector
from mysql.connector import errorcode

#need to change this config to connect to student machine, i don't know how yet since it has to go through a vpn
config = {
    'user': 'root',
    'password': '',
    'host': 'localhost',
    'database': 'SuperReader'
}

# Command string to create a Characters tabel for a text

create_characters_table = """
CREATE TABLE IF NOT EXISTS characters (
    character_id INT AUTO_INCREMENT PRIMARY KEY,
    character_name VARCHAR(255) NOT NULL,
    traits TEXT -- or JSON if you're using MySQL 5.7+
);
"""
# Command string to create a Lines tabel for a text
create_lines_table = """
CREATE TABLE IF NOT EXISTS lines (
    line_id INT AUTO_INCREMENT PRIMARY KEY,
    line_text TEXT NOT NULL,
    order_in_text INT NOT NULL,
    character_id INT,
    FOREIGN KEY (character_id) REFERENCES characters(character_id)
);
"""

# Function to create tables per text analyzed
def create_tables():
    try:
        # Establish connection to MySQL
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()

        # Execute table creation commands
        cursor.execute(create_characters_table)
        print("Created 'characters' table successfully.")

        cursor.execute(create_lines_table)
        print("Created 'lines' table successfully.")

        # Commit changes
        connection.commit()

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
    else:
        # Close connection
        cursor.close()
        connection.close()

    # Call the function to create tables
    if __name__ == "__main__":
        create_tables()