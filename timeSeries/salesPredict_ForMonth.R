library(randomForest)
library(sqldf)
#library(CrossR)
library(caret)
library(TSA)
library(Metrics)
banner_trend = read.csv("dataset\\banner_carr_month.csv")


# get the sequence Data
ts.raw = banner_trend[["QLI"]]
# get the TS Data
ts.ts = ts(ts.raw, start = c(2016,9),frequency = 12)
plot(ts.ts)
start(ts.ts)
end(ts.ts)
frequency(ts.ts)
ts.ts
fit = stl(ts.ts, s.window = "period")
fit
library(forecast)
forecast(fit, 1)
ts.ma=arima(ts.ts, order=c(1,0,1), seasonal = list(order=c(1,0,0), period=12))
#plot(ts.ts)
#plot(ts.ts, n1=c(2016,9), n.ahead=36, type='o')
#plot(ts.ma)
pred_val =predict(ts.ma, 6)
pred_val$pred
rmse(banner_trend[["QLI"]][-(1:99)], as.vector(pred_val$se))
# ???Իع�???
# train_lm = train[[,-1]]
# as.vector(train[,-1])
sa.lm = lm(QLI~., data=train)
sa.pred = predict(sa.lm, test)
rmse(test[["QLI"]], as.vector(sa.pred))



data("AirPassengers")
plot(AirPassengers)
AirPassengers
lap = log(AirPassengers)
plot(lap)
fit = stl(lap, s.window = "period")
fit$time.series
