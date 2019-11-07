# -*- coding: utf-8 -*-
import pymysql

# get the config DB connection
def  getConfigDBConn():
    user = "root"
    password = 'root'
    write_db = 'pp_conf'
    host = 'localhost'
    conn_conf = pymysql.connect(user=user, password=password, database=write_db, host=host, charset='utf8')

    return conn_conf
