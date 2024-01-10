import pymysql
from typing import *


def preprocess(sqls: str) -> None:
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='root', database='leetcode')
    cur = conn.cursor()
    sql_list = sqls.split("\n")
    sql_list = [s.strip() for s in sql_list]
    for s in sql_list:
        cur.execute(s)
    cur.close()
    conn.commit()
    conn.close()


def execute(sql: str) -> Any:
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='root', database='leetcode')
    cur = conn.cursor()
    cur.execute(sql)
    data = cur.fetchall()
    cur.close()
    conn.commit()
    conn.close()
    return data
