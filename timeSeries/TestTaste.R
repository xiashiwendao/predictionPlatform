women.text='height weight
58    115
59    117
60    120
61    123
62    126
63    129
64    132
65    135
66    139
67    142
68    146
69    150
70    154
71    159
72    164'
women.df <- read.table(text = women.text, header = TRUE)
plot(women.df,col=2)
fit<-lm(weight~height, data = women.df)
summary(fit)
plot(women.df$height, women.df$weight,col=3)
abline(fit)
fit
weight_hat = predict(fit, women.df,interval="prediction")
dim(weight_hat)
length(women.df$height)
plot(women.df$height, women.df$weight,col=3)
abline(women)
cor(women.df$height, women.df$weight)



par(mfrow=c(2,2))
plot(fit)
fit2<-lm(weight~height+I(height^2), data=women.df)
par(mfrow=c(2,2))
plot(fit2)

summary(fit2)












require(graphics)

## Predictions
x <- rnorm(15)
y <- x + rnorm(15)
predict(lm(y ~ x))
new <- data.frame(x = seq(-3, 3, 0.5))
predict(lm(y ~ x), new, se.fit = TRUE)
pred.w.plim <- predict(lm(y ~ x), new, interval = "prediction")
pred.w.clim <- predict(lm(y ~ x), new, interval = "confidence")
matplot(new$x, cbind(pred.w.clim, pred.w.plim[,-1]),
        lty = c(1,2,2,3,3), type = "l", ylab = "predicted y")


seq(1, 10, by = 2)  # diff between adj elements is 2
seq(1, 10, length=25)  # length of the vector is 25


library(randomForest)
library(caret)
library(e1071)
library(dplyr)
# randomForest taste
data_train = read.csv("dataset\\train_rf.csv")
data_train$Survived = as.factor(data_train$Survived)
data_train <- na.omit(data_train)

head(data_train, 2)
dim(data_train)
data_test = read.csv("dataset\\test_rf.csv")

glimpse(data_train)
glimpse(data_test)
trControl = trainControl(method = "cv", number=10, search = "grid")
set.seed(1234)
# Run the model
rf_default <- train(Survived~.,
                    data = data_train,
                    method = "rf",
                    metric = "Accuracy",
                    trControl = trControl)
# Print the results
print(rf_default)
