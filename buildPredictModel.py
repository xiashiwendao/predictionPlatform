# -*- coding: utf-8 -*-
# from Logger import MyLogger
import logging
import sys, os
import pymysql
import pandas as pd
import json
from utils import DataUtils
# from utils import Logger
from utils.Logger import MyLogger
from utils import Logger
from utils import Utils

logger = Utils.getLogger()
logger.debug("get logger!")
def buildModel(sourceId):
    conn_conf = DataUtils.getConfigDBConn()

    sourceId=14
    sql = 'select use_factor from isf_forecast_factor where ds_conf_id=' + str(sourceId)
    df = pd.read_sql(sql, conn_conf)
    factor_raw = df['use_factor'][0]
    factors = json.loads(factor_raw)
    factors_join = ','.join(factors)

    sql = 'select sqld from isf_data_source_conf where id=' + str(sourceId)
    df_source = pd.read_sql(sql, conn_conf)
    conn_conf.close()

    # build the query sql
    sqld = df['sqld'][0]
    from_part  = sql.split('from', maxsplit=1)
    sql_execute = 'select ' + factors_join + from_part

    # get the history biz data
    data_conn = DataUtils.getDataDBConn(sourceId)
    df_sqld = pd.read_sql(sql_execute)
    data_conn.close()

    # train the model
    

    



# buildModel(14)

