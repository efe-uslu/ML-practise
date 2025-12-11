#Creating the missing values for practise purposes.
library(missForest)
data = prodNA(Spotify_Youtube,noNA=0.1)


#Setting the seed to 412
set.seed(412)


#Implementing the libraries
library(readxl)
library(readr)
library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(cluster)
library(mice)
library(tidyr)
library(corrplot)
library(car)
library(VIM)
library(lattice)
library(mosaic)
library(reshape2)
library(gridExtra)
library(VIM)
library(naniar)
library(MissMech)
library(caret)
library(tensorflow)
library(randomForest)
library(caTools)
library(pdp)
library(keras)
library(RSNNS)
library(glmnet)
library(mgcv)
library(reshape2)
library(ggcorrplot)
library(FSA)
library(rcompanion)
library(parallel)
library(doParallel)
library(purrr)
library(tibble)
library(nnet)
library(mlbench)
library(e1071)
library(tidyverse)
library(xgboost)


#Adjusting the dataframe
data <- data[,-1 ] #Was an extra column for data no


#Examining variables and their data types
str(data) #Every type seems correct


#Checking head, tail and the existence of NAs
head(data)
tail(data)
sum(is.na(data))


#Every column name have the same format so no adjusting is needed here.


#Identifying outliers
identify_outliers <- function(data) {
  detect_outliers <- function(x) {
    Q1 <- quantile(x, 0.25, na.rm = TRUE)
    Q3 <- quantile(x, 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    lower_bound <- Q1 - 1.5 * IQR
    upper_bound <- Q3 + 1.5 * IQR
    return(x < lower_bound | x > upper_bound)
  }
#Apply the outlier detection function to each numeric column
  outlier_matrix <- sapply(data, function(column) {
    if (is.numeric(column)) {
      return(detect_outliers(column))
    } else {
      return(rep(FALSE, length(column)))
    }
  })
#Combine the results into the original data frame with an outlier flag
  outliers <- as.data.frame(outlier_matrix)
  colnames(outliers) <- paste0(colnames(data), "_outlier")
  result <- cbind(data, outliers)
  return(result)
}
data_with_outliers <- identify_outliers(data)
#View the data with outlier flags
print(data_with_outliers)
#After identifying the outliers, it is concluded that removal of the outliers from the columns: Speechiness, Tempo, Duration were removed due to them not generally being considered a song. Or because they are remixes/compilations.
remove_outliers <- function(data, cols) {
  for (col in cols) {
    if (col %in% colnames(data)) {
      Q1 <- quantile(data[[col]], 0.25, na.rm = TRUE)
      Q3 <- quantile(data[[col]], 0.75, na.rm = TRUE)
      IQR <- Q3 - Q1
      lower_bound <- Q1 - 1.5 * IQR
      upper_bound <- Q3 + 1.5 * IQR
      
      # Remove outliers
      data <- data %>%
        filter(data[[col]] >= lower_bound & data[[col]] <= upper_bound)
    } else {
      warning(paste("Column", col, "is not in the dataset."))
    }
  }
  return(data)
}
#Columns to check for outliers
cols_to_check <- c("Speechiness", "Tempo", "Duration_ms")
data_cleaned <- remove_outliers(data, cols_to_check)
#View the cleaned data
print(data_cleaned)


#Determining Missingness mechanism
gg_miss_case(data_cleaned) + 
  theme_minimal() +
  scale_fill_manual(values = c("orange", "red")) +
  labs(title = "Missing Values by Case", x = "Cases", y = "Number of Missing Values")
vis_miss(data_cleaned) +
  theme_minimal() +
  scale_fill_manual(values = c("navyblue", "red")) +
  labs(title = "Missing Data Matrix", x = "Variables", y = "Cases") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))



#Checking for duplicates&NAs and removing
sum(duplicated(data_cleaned)) #No duplicates.
total_rows <- nrow(data_cleaned)
missing_percentages <- numeric(length = ncol(data_cleaned))
for (i in 1:ncol(data_cleaned)) {
  missing_count <- sum(is.na(data_cleaned[, i]))
  missing_percentages[i] <- (missing_count / total_rows) * 100
}
missing_data_summary <- data.frame(
  Column = names(data_cleaned),
  MissingPercentage = missing_percentages
)
print(missing_data_summary) #Missing percentages for columns is in the range of 7-15%.


#Dropping unimportant columns regarding the analysis
columns_to_remove <- c("Comments","Instrumentalness","Artist", "Url_spotify", "Track", "Album", "Uri", "Url_youtube", "Title", "Channel", "Description","Views","Likes")
data_cleaned <- data_cleaned %>% select(-all_of(columns_to_remove))
print(data_cleaned)


#Checking the correlation matrix and VIF values
#Correlation matrix for numerical variables
num_data <- data_cleaned %>% select_if(is.numeric) %>% drop_na()
cor_matrix <- cor(num_data, use = "complete.obs")
#Plot correlation matrix
corrplot(cor_matrix, method = "circle") #Correlation plot indicates a small chance of multicollinearity so its better to check VIF values. Also some of the correlations are expected (Loudness&Energy, Acousticness&energy)
lm_model <- lm(Stream ~ ., data = data_cleaned)
vif(lm_model) #The adjusted GVIF values are not too high, this does not suggest the existence of multicollinearity.
ks.test(data_cleaned$Key, data_final$Key)

#Checking the plots.
numeric_columns <- sapply(data_cleaned, is.numeric)
numeric_data <- data_cleaned[, numeric_columns]
print("Numeric columns in the dataset:")
print(names(numeric_data))
par(mfrow = c(ceiling(sqrt(sum(numeric_columns))), ceiling(sqrt(sum(numeric_columns)))))
for (column_name in names(numeric_data)) {
  hist(numeric_data[[column_name]], main = paste("Distribution of", column_name), xlab = column_name, col = 'blue', border = 'black')
}#We see that most columns have skewed distributions while valence, tempo and duration_ms has approx normal. Also comments and streams have low frequencies for very high values as expected (popular songs are too popular).
data_melted <- melt(data_cleaned, id.vars = "Stream")
#Generate regression lines
all_vars <- names(data_cleaned)
#Create a list to store the plots
plots <- list()
#Loop through each variable and create plots
for (var in all_vars) {
  if (var != "Stream") {  # Exclude the Stream variable itself
    if (is.numeric(data_cleaned[[var]])) {
      # Numeric variable: scatter plot with linear regression line
      p <- ggplot(data_cleaned, aes_string(x = "Stream", y = var)) + 
        geom_point() +
        geom_smooth(method = "lm") +
        labs(title = paste("Scatter plot of", var, "vs Stream"),
             x = "Stream",
             y = var) +
        theme_minimal()
    } else {
      # Categorical variable: jitter plot
      p <- ggplot(data_cleaned, aes_string(x = "Stream", y = var)) + 
        geom_jitter() +
        labs(title = paste("Jitter plot of", var, "vs Stream"),
             x = "Stream",
             y = var) +
        theme_minimal()
    }
    plots[[var]] <- p
  }
}
#Arrange the plots on a single page
do.call("grid.arrange", c(plots, ncol = 2))
#This indicates some relation with certain variables against the response
densityplot(~ Stream|Album_type , data=data_cleaned, groups = Licensed, col=c("yellowgreen","darkblue"),par.settings = list(superpose.line = list(col=c("yellowgreen", "darkblue"))), plot.points = FALSE, auto.key = list(col=c("yellowgreen","darkblue")),lwd=2, main="Density-plot of Stream by Album type and License")
#This plot shows a similar relationship on streams depending on the album type, but as there are many more data for the "album" we might conduct a related test to see the real relation
#Also there might be a slight difference for license T/F's effect on album typed streamings.

#Plot of the response variable
hist(data_cleaned$Stream,main="Histogram of Stream",xlab="Stream",col="skyblue",breaks="Scott")

#For categorical variables
palette2<-"Set2"
ggplot(data_cleaned, aes(x = "", fill = Album_type)) +
  geom_bar(width = 1) +
  coord_polar(theta = "y") +
  labs(title = "Distribution of Album types", fill = "Album types") +
  scale_fill_brewer(palette = palette2)+
  theme_void()
palette1<-"Set1"
ggplot(data_cleaned, aes(x = "", fill = official_video)) +
  geom_bar(width = 1) +
  coord_polar(theta = "y") +
  labs(title = "Distribution of Offical videos", fill = "Official videos") +
  scale_fill_brewer(palette = palette1)+
  theme_void()
palette3<-"Set3"
ggplot(data_cleaned, aes(x = "", fill = Licensed)) +
  geom_bar(width = 1) +
  coord_polar(theta = "y") +
  labs(title = "Distribution of License status", fill = "License status") +
  scale_fill_brewer(palette = palette3)+
  theme_void()



#Used Pmm method for numeric and k-nearest for categorical imputation
#Define the columns to exclude from imputation
exclude_cols <- c("Album_type", "Licensed", "official_video")
#Separate the columns to be imputed and the columns to exclude
data_to_impute <- data_cleaned[, !(names(data_cleaned) %in% exclude_cols)]
data_excluded <- data_cleaned[, exclude_cols]
#Standardize the data to be imputed and save scale attributes
data_scaled <- scale(data_to_impute)
scale_attr <- attr(data_scaled, "scaled:scale")
center_attr <- attr(data_scaled, "scaled:center")
#Perform PMM imputation on the standardized dataset
imputed_data <- mice(data_scaled, method = 'pmm', m = 5, maxit = 50, seed = 500)
#To see the summary of the imputation
summary(imputed_data)
#To get the complete dataset after imputation
completed_data <- complete(imputed_data)
#Rescale the imputed data back to original scale
data_imputed <- as.data.frame(completed_data)
#Apply the scale and center attributes correctly to each numeric column
for (i in seq_along(data_to_impute)) {
  if (is.numeric(data_to_impute[[i]])) {
    data_imputed[[i]] <- (data_imputed[[i]] * scale_attr[i]) + center_attr[i]
  }
}
#Combine the imputed data with the excluded columns
data_final <- cbind(data_imputed, data_excluded)
#View the final dataset
head(data_final)
#For categorical variables
data_final$Album_type <- data_cleaned$Album_type
data_final$Licensed <- data_cleaned$Licensed
data_final$official_video <- data_cleaned$official_video
data_final_knn <- kNN(data_final, variable = c("Album_type", "Licensed", "official_video"), k = 5)
data_final$Album_type <- data_final_knn$Album_type
data_final$Licensed <- data_final_knn$Licensed
data_final$official_video <- data_final_knn$official_video

#Checking whether imputation changed the distribution of the data.
variables <- colnames(data_cleaned)
create_density_plots <- function(data1, data2, dataset_name1, dataset_name2) {
  plots <- list()
  for (var in variables) {
    if (is.numeric(data1[[var]]) && is.numeric(data2[[var]])) {
      p1 <- ggplot(data1, aes_string(x = var)) +
        geom_density(fill = "blue", alpha = 0.5) +
        ggtitle(paste("Density plot of", var, "before imputation"))
      p2 <- ggplot(data2, aes_string(x = var)) +
        geom_density(fill = "red", alpha = 0.5) +
        ggtitle(paste("Density plot of", var, "after imputation"))
      plots <- c(plots, list(p1, p2))
    }
  }
  grid.arrange(grobs = plots, ncol = 2)
}
create_density_plots(data_cleaned, data_final, "data_cleaned", "data_final")
perform_ks_test <- function(data1, data2) {
  ks_results <- data.frame(Variable = character(), Statistic = numeric(), P_Value = numeric(), stringsAsFactors = FALSE)
  
  for (var in variables) {
    if (is.numeric(data1[[var]]) && is.numeric(data2[[var]])) {
      ks_test <- ks.test(data1[[var]], data2[[var]])
      ks_results <- rbind(ks_results, data.frame(Variable = var, Statistic = ks_test$statistic, P_Value = ks_test$p.value))
    }
  }
  return(ks_results)
}
ks_results <- perform_ks_test(data_cleaned, data_final)
print(ks_results)


#Log transformation on Stream variable
data_final <- data_final %>%
  mutate(Stream = log(Stream + 1))
hist(data_final$Stream,main="Histogram of Stream",xlab="Stream",col="skyblue",breaks="Scott")

#EDA Q1 codes
data_to_visualize <- data_final %>% select(-Stream)
create_plot <- function(data, var) {
  if (is.numeric(data[[var]])) {
    ggplot(data, aes_string(x = var)) +
      geom_density(fill = 'blue', color = 'black', alpha = 0.7) +
      theme_minimal() +
      labs(title = paste("Density plot of", var), x = var, y = "Density")
  } else {
    ggplot(data, aes_string(x = var)) +
      geom_bar(fill = 'blue', color = 'black', alpha = 0.7) +
      theme_minimal() +
      labs(title = paste("Distribution of", var), x = var, y = "Count") +
      theme(axis.text.x = element_text(angle = 90, hjust = 1))
  }
}
plot_list <- lapply(names(data_to_visualize), function(var) {
  create_plot(data_to_visualize, var)
})
do.call(grid.arrange, c(plot_list, ncol = 3))

#EDA Q2 codes.
numeric_vars <- data_final %>% select_if(is.numeric)
linear_models <- lapply(names(numeric_vars), function(var) {
  lm(as.formula(paste(var, "~ Stream")), data = data_final)
})
nonlinear_models <- lapply(names(numeric_vars), function(var) {
  gam(as.formula(paste(var, "~ s(Stream, bs='re')")), data = data_final)
})
model_comparison <- data.frame(
  Variable = names(numeric_vars),
  Linear_AIC = sapply(linear_models, AIC),
  Nonlinear_AIC = sapply(nonlinear_models, AIC)
)
model_comparison <- model_comparison %>%
  mutate(Better_Model = ifelse(Linear_AIC < Nonlinear_AIC, "Linear", "Nonlinear"))
print(model_comparison)


#EDA Q3 codes.
numeric_vars <- data_final %>% select_if(is.numeric)
correlation_matrix <- cor(numeric_vars, use = "complete.obs")
ggcorrplot(correlation_matrix, 
           method = "circle", 
           type = "lower", 
           lab = TRUE, 
           lab_col = "black", 
           title = "Correlation Matrix of Numeric Variables",
           ggtheme = ggplot2::theme_minimal(),
           colors = c("skyblue", "white", "#E46726"))
lm_model <- lm(Stream ~ ., data = numeric_vars)
vif_values <- vif(lm_model)
print(vif_values)


#EDA Q4 codes..
ggplot(data_final, aes(x=Licensed, y=Stream, fill=Licensed))+
  geom_violin(trim=FALSE)+ 
  geom_boxplot(width=0.2,fill="lightgray") +  
  labs(
    title = "Violin Plot: License status vs Stream",
    x = "License status",
    y = "Stream"
  )
wilcox_test <- wilcox.test(Stream ~ Licensed, data = data_final)
print(wilcox_test)


#EDA Q5 codes.
ggplot(data_final, aes(x=Album_type, y=Stream, fill=Album_type))+
  geom_violin(trim=FALSE)+ 
  geom_boxplot(width=0.2,fill="lightgray") +  
  labs(
    title = "Violin Plot: Album type vs Stream",
    x = "Album type",
    y = "Stream"
  )
kruskal_test <- kruskal.test(Stream ~ Album_type, data = data_final)
print(kruskal_test)
dunn_test <- dunnTest(Stream ~ Album_type, data = data_final, method = "bh")
print(dunn_test)
dunn_test_res <- dunn_test$res
print(dunn_test_res)

#Standardization before creating the models.
standardize_data <- function(data) {
  numeric_vars <- data %>% select_if(is.numeric)
  means <- sapply(numeric_vars, mean, na.rm = TRUE)
  sds <- sapply(numeric_vars, sd, na.rm = TRUE)
  numeric_standardized <- sweep(numeric_vars, 2, means, FUN = "-")
  numeric_standardized <- sweep(numeric_standardized, 2, sds, FUN = "/")
  non_numeric_vars <- data %>% select_if(negate(is.numeric))
  data_standardized <- cbind(non_numeric_vars, numeric_standardized)
  list(
    standardized_data = data_standardized,
    means = means,
    sds = sds
  )
}
result <- standardize_data(data_final)
data_final <- result$standardized_data
means <- result$means
sds <- result$sds

  
#Final part (CV and ML)
#Statistical model creation

numeric_data <- data_final %>% select_if(is.numeric)
train_control <- trainControl(method = "cv", number = 10)
linear_cv <- train(Stream ~ ., data = numeric_data, method = "lm", trControl = train_control)
predictor_names <- names(numeric_data)[names(numeric_data) != "Stream"]
poly_terms <- paste("poly(", predictor_names, ", 2)", collapse = " + ")
poly_formula <- as.formula(paste("Stream ~", poly_terms))
poly_cv <- train(poly_formula, data = numeric_data, method = "lm", trControl = train_control)
print(linear_cv)
print(poly_cv)
linear_res <- linear_cv$resample
poly_res <- poly_cv$resample
res <- data.frame(
  Model = rep(c("Linear", "Polynomial"), each = nrow(linear_res)),
  RMSE = c(linear_res$RMSE, poly_res$RMSE),
  Rsquared = c(linear_res$Rsquared, poly_res$Rsquared)
)
rmse_plot <- ggplot(res, aes(x = Model, y = RMSE, fill = Model)) +
  geom_boxplot() +
  theme_minimal() +
  ggtitle("RMSE Comparison") +
  ylab("RMSE")
rsquared_plot <- ggplot(res, aes(x = Model, y = Rsquared, fill = Model)) +
  geom_boxplot() +
  theme_minimal() +
  ggtitle("R-squared Comparison") +
  ylab("R-squared")
grid.arrange(rmse_plot, rsquared_plot, ncol = 2)

poly_model <- poly_cv$finalModel
poly_coefficients <- coef(poly_model)
print(poly_coefficients)

create_summary_table <- function(coefficients) {
  terms <- names(coefficients)
  effects <- data.frame(Variable = character(), Linear = character(), Quadratic = character(), stringsAsFactors = FALSE)
  for (i in 2:length(terms)) {  
    term <- terms[i]
    coef <- coefficients[i]
    base_var <- sub("_poly[12]", "", term)
    if (grepl("_poly1$", term)) {
      linear_effect <- ifelse(coef > 0, "Increase", "Decrease")
      quadratic_effect <- effects[effects$Variable == base_var, "Quadratic"]
      if (length(quadratic_effect) == 0) {
        quadratic_effect <- ""
      }
    } else {
      quadratic_effect <- ifelse(coef > 0, "Accelerates Increase", "Accelerates Decrease")
      linear_effect <- effects[effects$Variable == base_var, "Linear"]
      if (length(linear_effect) == 0) {
        linear_effect <- ""
      }
    }
    effects <- effects[effects$Variable != base_var, ]
    effects <- rbind(effects, data.frame(Variable = base_var, Linear = linear_effect, Quadratic = quadratic_effect, stringsAsFactors = FALSE))
  }
  return(effects)
}
summary_table <- create_summary_table(poly_coefficients)
print(summary_table)
var_importance <- varImp(poly_cv, scale = FALSE)
print(var_importance)
importance_df <- as.data.frame(var_importance$importance)
importance_df$Variable <- rownames(importance_df)
summary_table <- merge(summary_table, importance_df, by = "Variable", all.x = TRUE)
summary_table <- summary_table[order(-summary_table$Overall), ]
print(summary_table)
var_importance_plot <- ggplot(importance_df, aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  ggtitle("Variable Importance") +
  ylab("Importance") +
  xlab("Variable")
print(var_importance_plot)


#random forest using k-fold cross validation.
features <- data_final[, !names(data_final) %in% 'Stream']
target <- data_final$Stream
train_control <- trainControl(method = "cv", number = 10, allowParallel = TRUE)
cores <- detectCores() - 1
cl <- makeCluster(cores)
registerDoParallel(cl)
tune_grid <- expand.grid(mtry = c(2, 4, 6, 8, 10))
set.seed(123)
rf_model <- train(
  x = features, y = target,
  method = "rf",
  tuneGrid = tune_grid,
  trControl = train_control,
  metric = "RMSE"
)
stopCluster(cl)
print(rf_model$bestTune)
print(rf_model$results)
grid_search_plot <- ggplot(rf_model$results, aes(x = mtry, y = RMSE)) +
  geom_point() +
  geom_line() +
  labs(title = "Grid Search Results", x = "Number of Variables Tried at Each Split (mtry)", y = "RMSE") +
  theme_minimal()
print(grid_search_plot)
final_model <- rf_model$finalModel
varImpPlot(final_model, main = "Variable Importance")
final_model
#Making the PDPs
generate_pdp_data <- function(model, var_name) {
  pd <- partial(model, pred.var = var_name)
  pd_df <- as.data.frame(pd)
  pd_df$Variable <- var_name
  names(pd_df)[names(pd_df) == var_name] <- "Value"
  return(pd_df)
}
pdp_data_list <- lapply(important_vars[1:4], function(var) {
  generate_pdp_data(rf_model, var)
})
ggplot(pdp_data, aes(x = Value, y = yhat)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "red", size = 2) +
  labs(title = "Partial Dependence Plots for Top 4 Important Variables",
       x = "Predictor Value",
       y = "Partial Dependence") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.title.x = element_text(size = 12, face = "bold"),
    axis.title.y = element_text(size = 12, face = "bold"),
    strip.text = element_text(size = 12, face = "bold")
  ) +
  scale_color_manual(values = c("blue", "red")) +
  facet_wrap(~ Variable, scales = "free_x")


#Neural network with k-fold cross validation
target <- "Stream"
predictors <- setdiff(names(data_final), target)
train_control <- trainControl(method = "cv", number = 10, allowParallel = TRUE)
hyper_grid <- expand.grid(
  size = c(5, 10, 15),
  decay = c(0, 0.001, 0.01)
)
set.seed(123)
model <- train(
  form = as.formula(paste(target, "~ .")),
  data = data_final,
  method = "nnet",
  trControl = train_control,
  tuneGrid = hyper_grid,
  linout = TRUE
)
stopCluster(cl)
print(model)
plot(model)
var_imp <- varImp(model, scale = FALSE)
print(var_imp)
plot(var_imp)
for (var in predictors) {
  pdp <- partial(model, pred.var = var, grid.resolution = 50)
  plot(pdp, main = paste("PDP for", var))
}
plot(model)
dev.off()
plot(var_imp)
dev.off()
best_model <- model$finalModel
best_hyperparameters <- model$bestTune
best_model
best_hyperparameters
predictions <- predict(model, data_final)
mse <- mean((data_final[[target]] - predictions)^2)
rmse <- sqrt(mse)
r_squared <- 1 - sum((data_final[[target]] - predictions)^2) / sum((data_final[[target]] - mean(data_final[[target]]))^2)
mse
rmse
r_squared


#Support Vector Machines model with k-fold cross validation

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
train_control <- trainControl(method = "cv", number = 10, allowParallel = TRUE)
tune_grid <- expand.grid(C = 2^(-2:2), 
                         sigma = 2^(-2:2))
set.seed(123)
svm_model <- train(Stream ~ ., data = data_final, 
                   method = "svmRadial",
                   trControl = train_control,
                   tuneGrid = tune_grid,
                   metric = "RMSE")
best_model <- svm_model$finalModel
best_params <- svm_model$bestTune
best_rmse <- min(svm_model$results$RMSE)
best_r_squared <- svm_model$results[which.min(svm_model$results$RMSE), "Rsquared"]
print(best_model)
print(best_params)
print(paste("Best RMSE:", best_rmse))
print(paste("Best R-squared:", best_r_squared))
#grid search results
ggplot(svm_model$results, aes(x = log2(C), y = log2(sigma), color = RMSE)) + 
  geom_point(size = 5) + 
  scale_color_gradient(low = "blue", high = "red") +
  labs(title = "Grid Search Results", x = "log2(C)", y = "log2(Sigma)")
stopCluster(cl)


#XGboost model model with k-fold cross validation
target <- "Stream"
numeric_features <- sapply(data_final, is.numeric)
numeric_features <- names(numeric_features[numeric_features])
features <- setdiff(numeric_features, target)
dtrain <- xgb.DMatrix(data = as.matrix(data_final[, features]), label = data_final[[target]])
cv_folds <- 5
train_control <- trainControl(method = "cv", number = cv_folds, allowParallel = TRUE)
tune_grid <- expand.grid(
  nrounds = c(100, 200, 300),
  max_depth = c(3, 6, 9),
  eta = c(0.01, 0.1, 0.3),
  gamma = c(0, 0.1, 0.2),
  colsample_bytree = c(0.7, 0.8, 0.9),
  min_child_weight = c(1, 5, 10),
  subsample = c(0.7, 0.8, 0.9)
)
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
results <- data.frame()
total_combinations <- nrow(tune_grid)
for (i in 1:total_combinations) {
  params <- tune_grid[i, ]
  print(paste(i, "/", total_combinations))
  xgb_model <- train(
    x = data_final[, features],
    y = data_final[[target]],
    method = "xgbTree",
    trControl = train_control,
    tuneGrid = params,
    metric = "RMSE"
  )
  result <- cbind(params, xgb_model$results)
  results <- rbind(results, result)
}
stopCluster(cl)
registerDoSEQ()
best_model_index <- which.min(results$RMSE)
best_model <- results[best_model_index, ]
best_params <- best_model[, names(tune_grid)]
best_mse <- best_model$RMSE
best_r2 <- best_model$Rsquared
print(best_model)
print(best_params)
print(paste("Best MSE:", best_mse))
print(paste("Best R-squared:", best_r2))
best_xgb_model <- xgb.train(
  data = dtrain,
  params = list(
    max_depth = best_params$max_depth,
    eta = best_params$eta,
    gamma = best_params$gamma,
    colsample_bytree = best_params$colsample_bytree,
    min_child_weight = best_params$min_child_weight,
    subsample = best_params$subsample
  ),
  nrounds = best_params$nrounds
)
importance_matrix <- xgb.importance(feature_names = features, model = best_xgb_model)
xgb.plot.importance(importance_matrix)





















