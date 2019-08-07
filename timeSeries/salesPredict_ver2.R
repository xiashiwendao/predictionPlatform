library(randomForest)
library(sqldf)
library(CrossR)
library(caret)
df = read.csv("dataset\\SalesTrend.csv")
head(df, 1)
trend = sqldf("select REPORT_DATE, sum(QLI) as QLI
,avg(LAST_1_MONTH_AVG) as LAST_1_MONTH_AVG
,avg(LAST_1_MONTH_MIN) as LAST_1_MONTH_MIN
,avg(LAST_1_MONTH_MAX) as LAST_1_MONTH_MAX
,avg(LAST_1_MONTH_SUM) as LAST_1_MONTH_SUM
,avg(LAST_1_MONTH_MID) as LAST_1_MONTH_MID
,avg(LAST_1_MONTH_MID25) as LAST_1_MONTH_MID25
,avg(LAST_1_MONTH_MID75) as LAST_1_MONTH_MID75
,avg(LAST_2_MONTH_AVG) as LAST_2_MONTH_AVG
,avg(LAST_2_MONTH_MIN) as LAST_2_MONTH_MIN
,avg(LAST_2_MONTH_MAX) as LAST_2_MONTH_MAX
,avg(LAST_2_MONTH_SUM) as LAST_2_MONTH_SUM
,avg(LAST_2_MONTH_MID) as LAST_2_MONTH_MID
,avg(LAST_2_MONTH_MID25) as LAST_2_MONTH_MID25
,avg(LAST_2_MONTH_MID75) as LAST_2_MONTH_MID75
,avg(LAST_3_MONTH_AVG) as LAST_3_MONTH_AVG
,avg(LAST_3_MONTH_MIN) as LAST_3_MONTH_MIN
,avg(LAST_3_MONTH_MAX) as LAST_3_MONTH_MAX
,avg(LAST_3_MONTH_SUM) as LAST_3_MONTH_SUM
,avg(LAST_3_MONTH_MID) as LAST_3_MONTH_MID
,avg(LAST_3_MONTH_MID25) as LAST_3_MONTH_MID25
,avg(LAST_3_MONTH_MID75) as LAST_3_MONTH_MID75
,avg(LAST_4_MONTH_AVG) as LAST_4_MONTH_AVG
,avg(LAST_4_MONTH_MIN) as LAST_4_MONTH_MIN
,avg(LAST_4_MONTH_MAX) as LAST_4_MONTH_MAX
,avg(LAST_4_MONTH_SUM) as LAST_4_MONTH_SUM
,avg(LAST_4_MONTH_MID) as LAST_4_MONTH_MID
,avg(LAST_4_MONTH_MID25) as LAST_4_MONTH_MID25
,avg(LAST_4_MONTH_MID75) as LAST_4_MONTH_MID75
,avg(LAST_5_MONTH_AVG) as LAST_5_MONTH_AVG
,avg(LAST_5_MONTH_MIN) as LAST_5_MONTH_MIN
,avg(LAST_5_MONTH_MAX) as LAST_5_MONTH_MAX
,avg(LAST_5_MONTH_SUM) as LAST_5_MONTH_SUM
,avg(LAST_5_MONTH_MID) as LAST_5_MONTH_MID
,avg(LAST_5_MONTH_MID25) as LAST_5_MONTH_MID25
,avg(LAST_5_MONTH_MID75) as LAST_5_MONTH_MID75
,avg(LAST_6_MONTH_AVG) as LAST_6_MONTH_AVG
,avg(LAST_6_MONTH_MIN) as LAST_6_MONTH_MIN
,avg(LAST_6_MONTH_MAX) as LAST_6_MONTH_MAX
,avg(LAST_6_MONTH_SUM) as LAST_6_MONTH_SUM
,avg(LAST_6_MONTH_MID) as LAST_6_MONTH_MID
,avg(LAST_6_MONTH_MID25) as LAST_6_MONTH_MID25
,avg(LAST_6_MONTH_MID75) as LAST_6_MONTH_MID75
,avg(LAST_12_MONTH_AVG) as LAST_12_MONTH_AVG
,avg(LAST_12_MONTH_MIN) as LAST_12_MONTH_MIN
,avg(LAST_12_MONTH_MAX) as LAST_12_MONTH_MAX
,avg(LAST_12_MONTH_SUM) as LAST_12_MONTH_SUM
,avg(LAST_12_MONTH_MID) as LAST_12_MONTH_MID
,avg(LAST_12_MONTH_MID25) as LAST_12_MONTH_MID25
,avg(LAST_12_MONTH_MID75) as LAST_12_MONTH_MID75
 from df where BANNER_NAME='Carrefour' group by REPORT_DATE")

banner_group_raw = read.csv("dataset\\banner_group.csv")

banner_group=banner_group_raw[-2]
head(banner_group)
banner_trend = merge(trend, banner_group, by="REPORT_DATE")
#head(df_trend, 3)
#y = df_trend[["QLI"]]
#X = df_trend[-2]
#splitSet = train_test_split(X, y, test_size=0.2)
#X_train = splitSet$X_train
#y_train = splitSet$y_train
#X_test = splitSet$X_test
#y_test = splitSet$y_test
#head(X_train, 1)
trainIndex = createDataPartition(banner_trend$QLI,
                                 p=0.8, list=FALSE,times=1)

train = banner_trend[trainIndex,]
test = banner_trend[-trainIndex,]
rf = randomForest(QLI~., data=train,mtry=3,
                  importance=TRUE, na.action=na.omit)
banner_trend["REPORT_DATE"]
