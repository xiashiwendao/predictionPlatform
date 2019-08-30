import numpy as np
import pandas as pd
import os, sys
import pymysql
datasetPath = os.path.abspath(".\\dataset")
sys.path.append(os.path.abspath("."))
from data.DataExtractor import DataExtractor



# 
csvFileForGoods = os.path.join(datasetPath, "banner_carr.csv")
# conn = pymysql.connect(user='root',password='root',database='pp',host='localhost', port=3307,charset='utf8')
# query = "select * from banner_carr"
# df = pd.read_sql(query,conn)
# df.to_csv(csvFileForGoods)

# get good daily info
df_banner=pd.read_csv(csvFileForGoods)
# change the data format, consistent with "SaleTrend.csv", for later join
df_banner["REPORT_DATE"] = df_banner["REPORT_DATE"].apply(lambda x:x.replace('-','/'))
# df_banner["REPORT_DATE"].head(5)
df_banner_daily_agg = df_banner.drop("ID", axis=1).groupby("REPORT_DATE").mean().reset_index()
df_banner_daily_agg.head(5)
df_banner_daily_agg["REPORT_DATE"].values

# get sales daily info
csvFileForSalesTrend = os.path.join(datasetPath, "SalesTrend.csv")
df_trend = pd.read_csv(csvFileForSalesTrend)
df_trend_carre = df_trend[df_trend.BANNER_NAME == 'Carrefour']
df_trend_carre_daily_agg = df_trend_carre.groupby('REPORT_DATE').mean().reset_index()
df_trend_carre_daily_agg["REPORT_DATE"].values
# df_trend_carre_daily_agg.head(5)
# join good daily info & sales daily info
df_merge = df_banner_daily_agg.merge(df_trend_carre_daily_agg, on=["REPORT_DATE"], how="inner")
len(df_merge)
df_merge.to_csv(os.path.join(datasetPath, "banner_carr_daily.csv"))
df_carre_month = df_merge.groupby(["YEAR_OF_WEEK","MONTH_OF_YEAR"]).sum().reset_index()
df_carre_month.head(5)
df_carre_month.to_csv(os.path.join(datasetPath, "banner_carr_month.csv"))


path = os.path.abspath(".\\data")
path
sys.path.append(path)
from data import Test
Test.hello_print()
dataGetter = DataExtractor()
df = dataGetter.getDataByMonth(True)
df.head(5)