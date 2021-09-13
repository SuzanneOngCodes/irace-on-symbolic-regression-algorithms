# Install packages
# install.packages("caret", dependencies = TRUE)
# install.packages("randomForest")

library(tidyverse)
library(dplyr)
library(randomForest)
library(caret)
library(e1071)
data_train <- read.csv("https://raw.githubusercontent.com/guru99-edu/R-Programming/master/train.csv")
glimpse(data_train)
data_test <- read.csv("https://raw.githubusercontent.com/guru99-edu/R-Programming/master/test.csv") 
glimpse(data_test)

# Detected NaN in default datasets
data_train <- data_train[, colSums(is.na(data_train)) == 0]

# Define the control
trControl <- trainControl(method = "csv",
                          number = 10,
                          search = "grid")
set.seed(1234)
# Run the model
data_train$Survived <-as.factor(data_train$Survived)
rf_default <- randomForest(Survived~.,
                    data = data_train,
                    trControl = trControl, importance = TRUE, na.action = na.omit)
# Print the results
print(rf_default)
# Plot the error vs the number of trees graph
plot(rf_default)