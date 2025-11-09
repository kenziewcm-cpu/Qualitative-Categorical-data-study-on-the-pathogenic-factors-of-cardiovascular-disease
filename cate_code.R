# Load necessary packages
library(readxl)
library(dplyr)
library(base)
library(ISLR)
library(pROC)

# Read Excel file
heart <- read_excel("heart.xlsx")

# Summarize the dataset
summary(heart)

# Set plotting area (2x2 layout)
par(mfrow = c(2, 2))

# Plot histogram (specify variable later)
hist(heart$)

# Check the structure of the dataset
str(heart)

# Create a frequency table (specify variable later)
table()

# View help documentation for sapply
?sapply

# Identify qualitative (factor) variables
qualitative_vars <- sapply(heart, is.factor)

# Convert qualitative variables to character and then back to factor
heart <- heart %>%
  mutate(across(all_of(names(heart)[qualitative_vars]), as.character)) %>%
  mutate(across(all_of(names(heart)[qualitative_vars]), as.factor))

# Perform one-hot encoding on qualitative variables
encoded_data <- model.matrix(~ . - 1, data)

# Export the one-hot encoded dataset to Excel
write.xlsx(encoded_data, "output_file_one_hot_encoded.xlsx", row.names = FALSE)

# View help documentation for write function
?write

# Fit a full logistic regression model using all predictors
myglm.full <- glm(
  HeartDisease ~ ., 
  data = encoded_data,
  family = binomial(link = "logit")
)

# Display model summary
summary(myglm.full)

# Suppose 'encoded_data' is the one-hot encoded dataframe 
# where 'HeartDisease' is the binary response variable (0 or 1)

# Select columns to be standardized
cols_to_standardize <- c(1, 7, 8, 12, 14)

# Standardize the selected columns
encoded_data[, cols_to_standardize] <- scale(encoded_data[, cols_to_standardize])

# Print or inspect the standardized dataset
print(data)

# Fit logistic regression model using glm function
logistic_model <- glm(
  HeartDisease ~ ., 
  data = as.data.frame(encoded_data), 
  family = binomial(link = "logit")
)

# Display the summary of the logistic regression model
summary(logistic_model)


# Qualitative midterm assignment - second part of the code

summary(heart1)


# At this point the dataset has already been one-hot encoded
logistic_model <- glm(
  HeartDisease ~ .,
  data = as.data.frame(heart3),
  family = binomial(link = "logit")
)
summary(logistic_model)

par(mfrow = c(1, 1))
set.seed(5)
heart4 = as.data.frame(heart3)  # dataset used for prediction/evaluation
y = heart4$HeartDisease         # true labels
yprob = myresult$fitted.values  # predicted probabilities
r = roc(y, yprob)
plot.roc(r, print.auc = TRUE, print.thres = TRUE)  # draw ROC curve

yhat = ifelse(yprob > 0.5, 1, 0)    # classify using 0.5 as threshold

# Using 0.5 as the cutoff to predict Yes/No
sum(y != yhat)
sum(y == 0)
sum(y == 1)
sum(y == yhat)
mean(y != yhat)
FN = sum(y == 1 & y != yhat)
TP = sum(y == 1 & y == yhat)
FP = sum(y == 0 & y != yhat)
TN = sum(y == 0 & y == yhat)
# Recall / Sensitivity
TPR = (TP) / (TP + FN)
TPR
# Precision
Precision = (TP) / (TP + FP)
Precision
# Misclassification rate
(FP + FN) / (FP + FN + TP + TN)
# False positive rate (can be computed as FP / (FP + TN))


# Example using simulated data – replace with your actual data
set.seed(123)
y <- sample(c(0, 1), 100, replace = TRUE)
yhat <- runif(100)

# Function to compute F-score at different thresholds
calculate_f_score <- function(actual, predicted, threshold) {
  predicted_labels <- ifelse(predicted >= threshold, 1, 0)
  confusion_matrix <- table(Actual = actual, Predicted = predicted_labels)
  
  if (length(confusion_matrix) == 4) {
    true_positive <- confusion_matrix[4]
    false_positive <- confusion_matrix[3]
    false_negative <- confusion_matrix[2]
    
    precision <- true_positive / (true_positive + false_positive)
    recall <- true_positive / (true_positive + false_negative)
    
    f_score <- 2 * (precision * recall) / (precision + recall)
    
    return(f_score)
  } else {
    # If the confusion matrix is not 2x2, return NA
    return(NA)
  }
}

# Define a grid of thresholds
yprobs <- seq(0, 1, by = 0.01)

# Compute F-score for each threshold
f_scores <- sapply(yprobs, function(threshold) {
  calculate_f_score(y, yhat, threshold)
})

# Remove NA values
f_scores <- na.omit(f_scores)

# Plot F-score vs threshold
plot(
  yprobs[1:length(f_scores)],
  f_scores,
  type = "l",
  col = "darkgrey",
  lwd = 2,
  xlab = "Threshold (yprob)",
  ylab = "F-score",
  main = "F-score vs. Threshold"
)

# Find the threshold that maximizes the F-score
optimal_threshold <- yprobs[which.max(f_scores)]
optimal_f_score <- max(f_scores)

cat("Optimal Threshold (yprob):", optimal_threshold, "\n")
cat("Max F-score:", optimal_f_score, "\n")


# Variable selection

# Method 1: best subset selection
install.packages("leaps")
library(leaps)
lm.full = regsubsets(HeartDisease ~ ., data = heart, nvmax = 19)
summary(lm.full)
# This warning means regsubsets() detected linear dependency among predictors.
# This usually happens when predictors are highly correlated.
# It can lead to unstable or unreliable models.
lm.full.sum = summary(lm.full)
names(lm.full.sum)
par(mfrow = c(2, 2))

plot(lm.full.sum$rsq)
plot(lm.full.sum$adjr2)
plot(lm.full.sum$cp)
plot(lm.full.sum$bic)


vif(lm.full)
cor(heart4)

# Variable selection using AIC
logistic_model.aic = step(
  logistic_model,
  direction = "both",
  trace = FALSE,
  k = 2
)
summary(logistic_model.aic)
yprob.aic = logistic_model.aic$fitted.values  # predicted probabilities
r.aic = roc(y, yprob.aic)
plot.roc(r.aic, print.auc = TRUE, print.thres = TRUE)


# Variable selection using BIC
n = dim(heart3)[1]
logistic_model.bic = step(
  logistic_model,
  direction = "both",
  trace = FALSE,
  k = log(n)
)

summary(logistic_model.bic)
yprob.bic = logistic_model.bic$fitted.values
r.bic = roc(y, yprob.bic)
plot.roc(r.bic, print.auc = TRUE, print.thres = TRUE)  # draw ROC curve

# Try a new model (drop some predictors)
logistic_model2 <- glm(
  HeartDisease ~ . - RestingBP - Cholesterol - Age,
  data = as.data.frame(heart3),
  family = binomial(link = "logit")
)
summary(logistic_model2)

# Try another model (drop different predictors)
logistic_model3 <- glm(
  HeartDisease ~ . - RestingBP - Cholesterol - MaxHR,
  data = as.data.frame(heart3),
  family = binomial(link = "logit")
)
summary(logistic_model3)


# Assume r.bic, r.aic, r are your three ROC objects

# Plot the first ROC curve
plot.roc(r.bic, col = "red", print.auc = TRUE, print.thres = TRUE, main = "ROC Curves")
# Add the second ROC curve
plot.roc(r.aic, col = "yellow", add = TRUE, print.auc = TRUE, print.thres = TRUE)
# Add the third ROC curve
plot.roc(r, col = "blue", add = TRUE, print.auc = TRUE, print.thres = TRUE)

# Add legend
legend(
  "bottomright",
  legend = c("r.bic", "r.aic", "r"),
  col = c("red", "yellow", "blue"),
  lty = 1,
  cex = 0.8
)


# Same three ROC curves with nicer styles
plot.roc(r.bic, col = "darkred", lty = 1, lwd = 2, print.auc = FALSE, print.thres = TRUE, main = "ROC Curves")
plot.roc(r.aic, col = "darkorange", add = TRUE, lty = 2, lwd = 2, print.auc = FALSE, print.thres = TRUE)
plot.roc(r, col = "darkblue", add = TRUE, lty = 3, lwd = 2, print.auc = FALSE, print.thres = TRUE)

legend(
  "bottomright",
  legend = c("r.bic", "r.aic", "r"),
  col = c("darkred", "darkorange", "darkblue"),
  lty = 1:3,
  lwd = 2,
  cex = 0.8
)


# Draw ROC curves again but without threshold labels, and add AUC manually
plot.roc(r.bic, col = "darkred", lty = 1, lwd = 2, print.auc = FALSE, print.thres = FALSE, main = "ROC Curves")
plot.roc(r.aic, col = "darkorange", add = TRUE, lty = 2, lwd = 2, print.auc = FALSE, print.thres = FALSE)
plot.roc(r, col = "darkblue", add = TRUE, lty = 3, lwd = 2, print.auc = FALSE, print.thres = FALSE)

# Manually add AUC labels
text(0.3, 0.5, paste("AUC = ", round(auc(r.bic), 3)), col = "darkred")
text(0.35, 0.45, paste("AUC = ", round(auc(r.aic), 3)), col = "darkorange")
text(0.4, 0.4, paste("AUC = ", round(auc(r), 3)), col = "darkblue")
auc(r.bic)
# Manually add threshold labels
text(0.6, 0.93, "0.455(0.827,0.911)", col = "darkred")
text(0.65, 0.87, "0.466(0.827,0.909)", col = "darkorange")
text(0.7, 0.80, "0.455(0.827,0.911)", col = "darkblue")

# Add legend
legend(
  "bottomright",
  legend = c("r.bic", "r.aic", "r"),
  col = c("darkred", "darkorange", "darkblue"),
  lty = 1:3,
  lwd = 2,
  cex = 0.8
)


# Install and load required packages
install.packages("ROCR")
library(ROCR)
install.packages("caret")
library(caret)


library(ROCR)
library(pROC)

# Suppose you have four models: modelA, modelB, modelC, modelD
# Suppose 'data' is your input dataset containing features and labels

# Set 10-fold cross-validation
set.seed(42)
folds <- createFolds(heart4$HeartDisease, k = 10, list = TRUE)

# Create an empty plot for ROC curves
plot(
  0, 0,
  type = "n",
  xlim = c(0, 1),
  ylim = c(0, 1),
  xlab = "False Positive Rate",
  ylab = "True Positive Rate",
  main = "ROC Curve"
)

# Perform 10-fold CV and plot ROC for four models
for (i in 1:10) {
  # Training and test indices
  train_index <- unlist(folds[i])
  test_index <- setdiff(1:length(heart4$HeartDisease), train_index)
  
  # Get training and test data
  train_data <- heart4[train_index, ]
  test_data <- heart4[test_index, ]
  
  # Fit model A
  predictionA <- predict(logistic_model, newdata = test_data, type = "response")
  rocA <- roc(test_data$HeartDisease, predictionA)
  aucA <- auc(rocA)
  
  # Fit model B
  predictionB <- predict(logistic_model.aic, newdata = test_data, type = "response")
  rocB <- roc(test_data$HeartDisease, predictionB)
  aucB <- auc(rocB)
  
  # Fit model C
  predictionC <- predict(logistic_model.bic, newdata = test_data, type = "response")
  rocC <- roc(test_data$HeartDisease, predictionC)
  aucC <- auc(rocC)
  
  # Fit model D
  predictionD <- predict(logistic_model2, newdata = test_data, type = "response")
  rocD <- roc(test_data$HeartDisease, predictionD)
  aucD <- auc(rocD)
  
  # Add each model’s ROC curve to the plot
  lines(rocA, col = "red", lwd = 2, lty = i)
  lines(rocB, col = "blue", lwd = 2, lty = i)
  lines(rocC, col = "green", lwd = 2, lty = i)
  lines(rocD, col = "purple", lwd = 2, lty = i)
  
  # Print AUCs for each fold
  cat(sprintf(
    "Fold %d - Model A AUC: %.3f, Model B AUC: %.3f, Model C AUC: %.3f, Model D AUC: %.3f\n",
    i, aucA, aucB, aucC, aucD
  ))
  # Here we could also store all ROC curves for later averaging
}

# Add legend
legend(
  "bottomright",
  legend = c("Model A", "Model B", "Model C", "Model D"),
  col = c("red", "blue", "green", "purple"),
  lty = 1:2,
  lwd = 2
)


# Install and load required packages (again, if not installed)
install.packages("ROCR")
library(ROCR)
install.packages("caret")
library(caret)
# Install pROC and caret if not installed
# install.packages("pROC")
# install.packages("caret")

library(pROC)
library(caret)

# Assume 'data' is the input dataset
# Assume your model objects have already been created

# 10-fold cross-validation
set.seed(42)
folds <- createFolds(heart4$HeartDisease, k = 10, list = TRUE)

# Empty plot
plot(
  0, 0,
  type = "n",
  xlim = c(0, 1),
  ylim = c(0, 1),
  xlab = "False Positive Rate",
  ylab = "True Positive Rate",
  main = "ROC Curve"
)

# Loop through each model
models <- c("logistic_model", "logistic_model.aic", "logistic_model.bic", "logistic_model2")

for (model_name in models) {
  auc_values <- numeric()  # store AUC for each fold
  
  # 10-fold CV
  for (i in 1:10) {
    train_index <- unlist(folds[i])
    test_index <- setdiff(1:length(heart4$HeartDisease), train_index)
    
    train_data <- heart4[train_index, ]
    test_data <- heart4[test_index, ]
    
    # Get current model by name
    current_model <- get(model_name)
    
    # Predict on test set
    prediction <- predict(current_model, newdata = test_data, type = "response")
    
    # Compute ROC
    roc_curve <- roc(test_data$HeartDisease, prediction)
    
    # Save AUC
    auc_values <- c(auc_values, auc(roc_curve))
    
    # Add to plot
    lines(roc_curve, col = rainbow(length(folds))[i], lwd = 2)
  }
  
  # Print mean AUC for current model
  cat(sprintf("Model %s Mean AUC: %.3f\n", model_name, mean(auc_values)))
}

# Add legend
legend(
  "bottomright",
  legend = c(models, "Random"),
  col = c(rainbow(length(folds)), "gray"),
  lty = 1:2,
  lwd = 2
)


# 10-fold cross-validation again, this time trying to store ROC curves
set.seed(42)
folds <- createFolds(heart4$HeartDisease, k = 10, list = TRUE)

# Empty plot
plot(
  0, 0,
  type = "n",
  xlim = c(0, 1),
  ylim = c(0, 1),
  xlab = "False Positive Rate",
  ylab = "True Positive Rate",
  main = "ROC Curve"
)

# Models to compare
models <- c("logistic_model", "logistic_model.aic", "logistic_model.bic", "logistic_model2")

roc_curves <- list()

for (model_name in models) {
  auc_values <- numeric()  # store AUC for each fold
  
  # 10-fold CV
  for (i in 1:10) {
    train_index <- unlist(folds[i])
    test_index <- setdiff(1:length(heart4$HeartDisease), train_index)
    
    train_data <- heart4[train_index, ]
    test_data <- heart4[test_index, ]
    
    # Get model
    current_model <- get(model_name)
    
    # Predict
    prediction <- predict(current_model, newdata = test_data, type = "response")
    
    # ROC for this fold
    roc_curve <- roc(test_data$HeartDisease, prediction)
    
    # Store ROC curve
    roc_curves[[length(roc_curves) + 1]] <- roc_curve
    
    # Draw ROC for the first fold of each model
    if (i == 1) {
      lines(
        roc_curve,
        col = rainbow(length(models))[which(models == model_name)],
        lwd = 2,
        main = "ROC Curve"
      )
    }
  }
  
  # Print mean AUC for current model
  cat(sprintf("Model %s Mean AUC: %.3f\n", model_name, mean(auc_values)))
}

# (The following part tries to create an “average” ROC curve from the stored ROC objects)

# Add random classifier ROC curve
# (Note: the line below uses 'data' which is not defined in the original script)
# You should replace 'data' with your real dataset name
random_roc <- roc(
  rep(heart4$HeartDisease, length.out = nrow(heart4)),
  rep(runif(nrow(heart4)), each = nrow(heart4))
)
lines(random_roc, col = "gray", lwd = 2, lty = 2)

# Add legend
legend(
  "bottomright",
  legend = c(models, "Mean", "Random"),
  col = c(rainbow(length(models)), "black", "gray"),
  lty = 1:2,
  lwd = 2
)
