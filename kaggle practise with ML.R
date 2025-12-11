str(train_csv)
library(dplyr)
library(caret)
library(VIM)
library(dplyr)
library(data.table)
library(mltools)
train_csv <- train_csv %>%
  mutate(
    temperature = as.numeric(temperature),
    has_children = as.numeric(has_children),
    ID = as.numeric(ID),
    across(-c(temperature, has_children, ID), as.factor)
  )
test <- test %>%
  mutate(
    temperature = as.numeric(temperature),
    has_children = as.numeric(has_children),
    ID = as.numeric(ID),
    across(-c(temperature, has_children, ID), as.factor)
  )
sum(is.na(train_csv))
test_imputed <- test
cols_with_missing <- colnames(test_imputed)[apply(test_imputed, 2, anyNA)]
test_imputed <- kNN(test, k = 5)
test_cleaned <- test_imputed[, !grepl("_imp$", names(test_imputed))]
train_imputed <- train_csv
cols_with_missing <- colnames(train_imputed)[apply(train_imputed, 2, anyNA)]
train_imputed <- kNN(train_csv, k = 5)
any(is.na(train_imputed))
train_cleaned <- train_imputed[, !grepl("_imp$", names(train_imputed))]
train_cleaned <- train_cleaned[, !names(train_cleaned) %in% "toCoupon_GEQ5min"]
test_cleaned <- test_cleaned[, !names(test_cleaned) %in% "toCoupon_GEQ5min"]


#random forest 500 tree
X <- train_cleaned[, -c(1, ncol(train_cleaned))]
Y <- train_cleaned$Y
model_rf <- randomForest(X, Y, ntree = 500)
print(model_rf)
predictions_test <- predict(model_rf, test_cleaned)
predictions_test_df <- data.frame(ID = test$ID, Predicted_Y = predictions_test)
write.csv(predictions_test_df, file = "C:\\Users\\Efe/Desktop/predictions_test.csv", row.names = FALSE)

#random forest 300 tree
X <- train_cleaned[, -c(1, ncol(train_cleaned))]
Y <- train_cleaned$Y
model_rf <- randomForest(X, Y, ntree = 300)
print(model_rf)
predictions_test <- predict(model_rf, test_cleaned)
predictions_test_df <- data.frame(ID = test$ID, Predicted_Y = predictions_test)
write.csv(predictions_test_df, file = "C:\\Users\\Efe/Desktop/random_forest_300tree.csv", row.names = FALSE)

#random forest 300 tree mtry 2
X <- train_cleaned[, -c(1, ncol(train_cleaned))]
Y <- train_cleaned$Y
model_rf <- randomForest(X, Y, ntree = 300, mtry=2)
print(model_rf)
predictions_test <- predict(model_rf, test_cleaned)
predictions_test_df <- data.frame(ID = test$ID, Predicted_Y = predictions_test)
write.csv(predictions_test_df, file = "C:\\Users\\Efe/Desktop/random_forest_300treemtry2.csv", row.names = FALSE)

#random forest 500 tree mtry 10
X <- train_cleaned[, -c(1, ncol(train_cleaned))]
Y <- train_cleaned$Y
model_rf <- randomForest(X, Y, ntree = 500, mtry=10)
print(model_rf)
predictions_test <- predict(model_rf, test_cleaned)
predictions_test_df <- data.frame(ID = test$ID, Predicted_Y = predictions_test)
write.csv(predictions_test_df, file = "C:\\Users\\Efe/Desktop/random_forest_500treemtry10.csv", row.names = FALSE)

#random forest 500 tree mtry 8
X <- train_cleaned[, -c(1, ncol(train_cleaned))]
Y <- train_cleaned$Y
model_rf <- randomForest(X, Y, ntree = 500, mtry=8)
print(model_rf)
predictions_test <- predict(model_rf, test_cleaned)
predictions_test_df <- data.frame(ID = test$ID, Predicted_Y = predictions_test)
write.csv(predictions_test_df, file = "C:\\Users\\Efe/Desktop/random_forest_500treemtry8.csv", row.names = FALSE)

#random forest 500 tree mtry 6
str(train_cleaned)
X <- train_cleaned[, -c(1, ncol(train_cleaned))]
Y <- train_cleaned$Y
model_rf <- randomForest(X, Y, ntree = 500, mtry=6)
print(model_rf)
predictions_test <- predict(model_rf, test_cleaned)
predictions_test_df <- data.frame(ID = test$ID, Predicted_Y = predictions_test)
write.csv(predictions_test_df, file = "C:\\Users\\Efe/Desktop/random_forest_500treemtry6.csv", row.names = FALSE)

#random forest 500 tree mtry 4
X <- train_cleaned[, -c(1, ncol(train_cleaned))]
Y <- train_cleaned$Y
model_rf <- randomForest(X, Y, ntree = 500, mtry=4)
print(model_rf)
predictions_test <- predict(model_rf, test_cleaned)
predictions_test_df <- data.frame(ID = test$ID, Predicted_Y = predictions_test)
write.csv(predictions_test_df, file = "C:\\Users\\Efe/Desktop/random_forest_500treemtry4.csv", row.names = FALSE)


library(nnet)
#neural network
factor_vars <- sapply(train_data, is.factor)
for (var in names(factor_vars)[factor_vars]) {
  if (length(unique(train_data[[var]])) < 2) {
    stop(paste("Factor variable", var, "has less than 2 levels."))
  }
}



output_path <- "C:\\Users\\Efe\\Desktop\\"
train_data <- train_cleaned[, -1]
test_data <- test_cleaned[, -1]
train_data$Y <- as.factor(train_data$Y)
grid <- expand.grid(size = c(2, 4, 6), decay = c(0.1, 0.01))
for (i in 1:nrow(grid)) {
  size <- grid$size[i]
  decay <- grid$decay[i]
  nn_model <- nnet(Y ~ ., data = train_data, size = size, decay = decay, linout = FALSE, trace = FALSE)
  predictions <- predict(nn_model, test_data, type = "class")
  output_df <- data.frame(ID = test_cleaned$ID, Predictions = predictions)
  file_name <- paste0("neuralnetwork_size", size, "decay", decay, ".csv")
  write.csv(output_df, file.path(output_path, file_name), row.names = FALSE)
}

#Support Vector machines
library(e1071)
library(kernlab)
library(doParallel)
cl <- makeCluster(detectCores() - 1) # Use all but one core
registerDoParallel(cl)
train_cleaned$Y <- as.factor(make.names(train_cleaned$Y))
predictors <- train_cleaned[, -c(1, which(names(train_cleaned) == "Y"))] # Exclude ID and Y columns
response <- train_cleaned$Y
dummies <- dummyVars(~ ., data = predictors)
train_predictors <- predict(dummies, newdata = predictors)
test_predictors <- predict(dummies, newdata = test_cleaned[, -1])
train_predictors <- as.data.frame(train_predictors)
preProcess_normalize <- preProcess(train_predictors, method = c("center", "scale"))
train_predictors <- predict(preProcess_normalize, train_predictors)
test_predictors <- predict(preProcess_normalize, test_predictors)
sigma_estimate <- sigest(as.matrix(train_predictors))
sigma_values <- seq(sigma_estimate[1], sigma_estimate[3], length.out = 5)
train_control <- trainControl(method = "cv", number = 5, savePredictions = "final", classProbs = TRUE)
tune_grid <- expand.grid(sigma = sigma_values, C = seq(0.1, 1, by = 0.2))
svm_model <- train(x = train_predictors, y = response, 
                   method = "svmRadial", 
                   trControl = train_control, 
                   tuneGrid = tune_grid,
                   preProcess = c("center", "scale"))
print(svm_model$bestTune)
predictions <- predict(svm_model, newdata = test_predictors)
predictions <- ifelse(predictions == "X1", 1, 0)
results <- data.frame(ID = test_cleaned$ID, Prediction = predictions)
write.csv(results, "C:\\Users\\Efe\\Desktop\\svm_predictions.csv", row.names = FALSE)
top_3_models <- svm_model$results[order(svm_model$results$Accuracy, decreasing = TRUE), ][1:3, ]
for (i in 1:nrow(top_3_models)) {
  best_sigma <- top_3_models$sigma[i]
  best_C <- top_3_models$C[i]
  
  model <- svm(x = as.matrix(train_predictors), y = response, 
               type = "C-classification", 
               kernel = "radial", 
               cost = best_C,
               gamma = 1 / (2 * (best_sigma^2)),
               scale = TRUE)
  
  preds <- predict(model, newdata = as.matrix(test_predictors))
  
  result <- data.frame(ID = test_cleaned$ID, Prediction = preds)
  file_name <- paste0("C:\\Users\\Efe\\Desktop\\svm_sigma", best_sigma, "_C", best_C, ".csv")
  
  write.csv(result, file_name, row.names = FALSE)
}
stopCluster(cl)
registerDoSEQ()

#random forest 500 tree mtry 6 with one hoc encoding
library(randomForest)
X <- train_cleaned[, -c(1, ncol(train_cleaned))]
Y <- train_cleaned$Y
dummies <- dummyVars(~ ., data = X)
X_encoded <- predict(dummies, newdata = X)
X_encoded <- as.data.frame(X_encoded)
model_rf <- randomForest(X_encoded, Y, ntree = 500, mtry = 6)
print(model_rf)
test_X <- test_cleaned[, -1] # Remove the 'ID' column
test_X_encoded <- predict(dummies, newdata = test_X)
predictions_test <- predict(model_rf, test_X_encoded)
predictions_test <- ifelse(predictions_test == levels(predictions_test)[2], "1", "0")
predictions_test_df <- data.frame(ID = test_cleaned$ID, Predicted_Y = predictions_test)
write.csv(predictions_test_df, file = "C:\\Users\\Efe\\Desktop\\random_forest_500treemtry6.csv", row.names = FALSE)

#Logistic Regression
library(caret)
library(tidyverse)
predictors <- train_cleaned %>% select(-ID, -Y)
response <- train_cleaned$Y
log_model <- train(predictors, response, method = "glm", family = binomial, trControl = trainControl(method = "cv"))
test_predictors <- test_cleaned %>% select(-ID)
test$predictions <- predict(log_model, test_predictors)
test$predictions <- ifelse(test$predictions == "X1", "1", "0")
predictions_df <- test %>% select(ID, predictions)
log_mse <- mean((test$predictions - ifelse(test$Y == levels(test$Y)[1], 1, 0))^2)
write.csv(predictions_df, "C:\\Users\\Efe\\Desktop\\logistic_regression_predictions.csv", row.names = FALSE)

#XGboost
library(xgboost)
library(doParallel)
library(caret)
library(dplyr)
library(Matrix)
cl <- makeCluster(detectCores() - 1) # Use one less than the total number of cores
registerDoParallel(cl)
train_data <- train_cleaned %>% select(-ID)
test_data <- test_cleaned %>% select(-ID)
train_data$Y <- as.factor(train_data$Y)
test_data <- test_data[, !names(test_data) %in% "toCoupon_GEQ5min"]
train_data <- train_data[, !names(train_data) %in% "toCoupon_GEQ5min"]
train_data_matrix <- sparse.model.matrix(Y ~ . - 1, data = train_data)
test_data_matrix <- sparse.model.matrix(~ . - 1, data = test_data)
train_labels <- train_data$Y


train_control <- trainControl(method = "cv", 
                              number = 5, 
                              summaryFunction = twoClassSummary, 
                              classProbs = TRUE, 
                              allowParallel = TRUE)
xgb_grid <- expand.grid(nrounds = c(50, 100, 150),
                        max_depth = c(3, 6, 9),
                        eta = c(0.01, 0.1, 0.3),
                        gamma = 0,
                        colsample_bytree = 1,
                        min_child_weight = 1,
                        subsample = 1)
xgb_model <- train(x = train_data_matrix, 
                   y = train_labels, 
                   method = "xgbTree", 
                   trControl = train_control, 
                   tuneGrid = xgb_grid, 
                   metric = "ROC")
predictions <- predict(xgb_model, newdata = test_data_matrix, type = "raw")
predictions <- ifelse(predictions == "X1", "1", "0")
predictions_df <- data.frame(ID = test$ID, Prediction = predictions)
write.csv(predictions_df, "C:\\Users\\Efe\\Desktop\\xgboost_predictions.csv", row.names = FALSE)
top_3_models <- xgb_model$results %>% arrange(desc(ROC)) %>% head(3)

for (i in 1:nrow(top_3_models)) {
  model <- top_3_models[i,]
  model_name <- paste("xgboost_nrounds", model$nrounds, "maxdepth", model$max_depth, "eta", model$eta, sep = "_")
  model_predictions <- predict(xgb_model, newdata = test_data_matrix, type = "raw")
  model_predictions <- ifelse(model_predictions == "X1", "1", "0")
  model_predictions_df <- data.frame(ID = test$ID, Prediction = model_predictions)
  write.csv(model_predictions_df, paste0("C:\\Users\\Efe\\Desktop\\", model_name, ".csv"), row.names = FALSE)
}
stopCluster(cl)
registerDoSEQ()

#Best model (probably)
library(randomForest)
library(caret)
library(dplyr)
library(fastDummies)
train_cleaned$temperature <- as.factor(train_cleaned$temperature)
train_cleaned$has_children <- as.factor(train_cleaned$has_children)
test_cleaned$temperature <- as.factor(test_cleaned$temperature)
test_cleaned$has_children <- as.factor(test_cleaned$has_children)

one_hot_columns <- c("destination", "passenger", "weather", "coupon", "gender", "maritalStatus",
                     "occupation", "car", "Bar", "toCoupon_GEQ15min", "toCoupon_GEQ25min", 
                     "direction_same", "direction_opp")

existing_one_hot_columns <- one_hot_columns[one_hot_columns %in% names(train_cleaned)]
train_cleaned <- dummy_cols(train_cleaned, select_columns = existing_one_hot_columns, remove_first_dummy = FALSE, remove_selected_columns = FALSE)
test_cleaned <- dummy_cols(test_cleaned, select_columns = existing_one_hot_columns, remove_first_dummy = FALSE, remove_selected_columns = FALSE)

all_columns <- names(train_cleaned)
ordinal_columns <- setdiff(all_columns, c(existing_one_hot_columns, "ID", "Y"))

for (col in ordinal_columns) {
  train_cleaned[[col]] <- as.integer(as.factor(train_cleaned[[col]]))
  test_cleaned[[col]] <- as.integer(as.factor(test_cleaned[[col]]))
}
train_cleaned <- train_cleaned %>%
  select(-one_of(existing_one_hot_columns))
test_cleaned <- test_cleaned %>%
  select(-one_of(existing_one_hot_columns))
X <- train_cleaned[, -c(1, ncol(train_cleaned))]
Y <- train_cleaned$Y
model_rf <- randomForest(X, Y, ntree = 500, mtry = 6)
print(model_rf)
predictions_test <- predict(model_rf, test_cleaned)
predictions_test_df <- data.frame(ID = test_cleaned$ID, Predicted_Y = predictions_test)
write.csv(predictions_test_df, file = "C:\\Users\\Efe\\Desktop\\random_forest_500treemtry6.csv", row.names = FALSE)












# Load necessary libraries
library(randomForest)
library(caret)

# Define the columns for one-hot encoding
one_hot_columns <- c('destination', 'passanger', 'weather', 'coupon', 'gender', 'maritalStatus', 'occupation', 'car', 'toCoupon_GEQ15min', 'toCoupon_GEQ25min', 'direction_same', 'direction_opp')

# Apply one-hot encoding to the specified columns
train_cleaned_one_hot <- train_cleaned
test_cleaned_one_hot <- test_cleaned

train_cleaned_one_hot <- dummyVars(~ ., data = train_cleaned_one_hot[, one_hot_columns])
train_cleaned_one_hot <- predict(train_cleaned_one_hot, train_cleaned)
train_cleaned_one_hot <- data.frame(train_cleaned_one_hot)

test_cleaned_one_hot <- dummyVars(~ ., data = test_cleaned_one_hot[, one_hot_columns])
test_cleaned_one_hot <- predict(test_cleaned_one_hot, test_cleaned)
test_cleaned_one_hot <- data.frame(test_cleaned_one_hot)

# Define the columns for ordinal encoding (all columns except ID, Y, and one-hot encoded columns)
ordinal_columns <- setdiff(colnames(train_cleaned), c(one_hot_columns, 'ID', 'Y'))

# Apply ordinal encoding to the rest of the columns
train_cleaned_ordinal <- train_cleaned[, ordinal_columns]
test_cleaned_ordinal <- test_cleaned[, ordinal_columns]

# Combine one-hot and ordinal encoded columns
train_encoded <- cbind(train_cleaned_one_hot, train_cleaned_ordinal, Y = train_cleaned$Y)
test_encoded <- cbind(test_cleaned_one_hot, test_cleaned_ordinal)

# Separate predictors and target variable
X <- train_encoded[, -which(names(train_encoded) == 'Y')]
Y <- train_encoded$Y

# Train the Random Forest model
model_rf <- randomForest(X, Y, ntree = 500, mtry = 6)
print(model_rf)

# Make predictions on the test dataset
predictions_test <- predict(model_rf, test_encoded)

# Create a dataframe with IDs and predictions
predictions_test_df <- data.frame(ID = test_cleaned$ID, Predicted_Y = predictions_test)

# Save the predictions to a CSV file
write.csv(predictions_test_df, file = "C:\\Users\\Efe\\Desktop\\random_forest_500treemtry6.csv", row.names = FALSE)

