import sqlite3
from sqlite3 import Error
import pandas as pd


class NotConnectedError(Exception):
    def info(self):
        return 'Error: "You are not connected to the database"\n'


class Server:
    def __init__(self, db_name):
        self._db_name = db_name
        self.connection = None

    def connect(self):
        try:
            self.connection = sqlite3.connect(self._db_name)
            print("Connected")
        except Error as err:
            print(err)

    def query_script(self, query_script_name: str) -> None:
        query_string = None

        with open(query_script_name, 'r') as file:
            query_string = ''.join(file.readlines())
        
        if query_string:
            self.query(query_string)
        else:
            print("An problem occured")

    def query(self, query_code):
        if self.connection:
            cur = self.connection.cursor()
            cur.execute(query_code)
            rows = cur.fetchall()
            for row in rows:
                print(row)
        else:
            raise NotConnectedError

    def load_xlsx(self, xlsx_name="database.xlsx"):
        excel_file = pd.ExcelFile(xlsx_name)
        for i in excel_file.sheet_names:
            excel_file.parse(sheet_name=i).to_sql(i, self.connection, if_exists="replace")

    def disconnect(self):
        if self.connection:
            self.connection.close()
            print("Disconnected")

    def __delete__(self):
        self.disconnect()


if __name__ == '__main__':
    server = Server("cancer.db").connect()
    

