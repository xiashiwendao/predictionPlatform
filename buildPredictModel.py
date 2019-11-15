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
from optimizerModel import RandomforestOptimizer

logger = Utils.getLogger()
logger.debug("get logger!")
def buildModel(sourceId):
    conn_conf = DataUtils.getConfigDBConn()

    sourceId=14
    sql = 'select use_factor, seq from isf_forecast_factor where ds_conf_id=' + str(sourceId)
    df = pd.read_sql(sql, conn_conf)
    # get all the factors
    factor_raw = df['use_factor'][0]
    factors = json.loads(factor_raw)
    factors_join = ','.join(factors)
    # get the forecast column(which is y column)
    seq = df['seq'][0]
    json_obj = json.loads(seq)
    forecast_column = json_obj['key'=='fForecast']['value']
    logger.debug('seq is: ', seq, 'forecast_column: ', forecast_column)
    
    # the execute sql which will be executed in customer biz database
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

    y = df_sqld[forecast_column] # get the forcast value as y
    del df_sqld[forecast_column] # delete the y column, then get the X data
    X = df_sqld
    # train the model
    rfo = RandomforestOptimizer.RandomforestOptimizer(X, y)
    rfo.getOptimizedModel(X, y)

    



# buildModel(14)

