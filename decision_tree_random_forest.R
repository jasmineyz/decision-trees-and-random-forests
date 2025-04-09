# --------------------------------------------
# Machine Learning with Decision Trees and Random Forests
# --------------------------------------------
# This script demonstrates training and evaluating 
# classification trees and random forests in R.

# Load required libraries
library(caret)        # For training models with cross-validation
library(tidyverse)    # For data manipulation and visualization
library(rpart)        # For building decision trees
library(dslabs)       # For access to sample datasets
library(randomForest) # For random forest modeling

# --------------------------------------------
# Part 1: Decision Tree on polls_2008 dataset
# --------------------------------------------

# Fit a simple decision tree
# Predict 'margin' using all available predictors (only 'day' here)
fit <- rpart(margin ~ ., data = polls_2008)

# Visualize the fitted decision tree
plot(fit, margin = 0.1)
text(fit, cex = 0.75)

# Evaluate model fit visually
# Add tree predictions to the dataset
polls_2008 %>%  
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(day, margin)) +          # Actual data points
  geom_step(aes(day, y_hat), col = "red") # Tree-based predictions

# --------------------------------------------
# Part 2: Classification Tree with Cross-Validation on mnist_27
# --------------------------------------------

# Train a classification tree with cross-validation
# 'cp' (complexity parameter) is tuned from 0.0 to 0.1 over 25 values
train_rpart <- train(y ~ .,
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0.0, 0.1, length.out = 25)),
                     data = mnist_27$train)

# Visualize cross-validation results
plot(train_rpart)

# Evaluate model performance on the test set
rpart_accuracy <- confusionMatrix(
  predict(train_rpart, mnist_27$test),
  mnist_27$test$y
)$overall["Accuracy"]
print(rpart_accuracy)

# Visualize the final decision tree structure
plot(train_rpart$finalModel, margin = 0.1)
text(train_rpart$finalModel)

# --------------------------------------------
# Part 3: Random Forest on mnist_27
# --------------------------------------------

# Train a random forest model using default settings
train_rf <- randomForest(y ~ ., data = mnist_27$train)

# Evaluate random forest performance
rf_accuracy <- confusionMatrix(
  predict(train_rf, mnist_27$test),
  mnist_27$test$y
)$overall["Accuracy"]
print(rf_accuracy)

# --------------------------------------------
# Part 4: Random Forest with Cross-Validation (Rborist)
# --------------------------------------------

# Train a random forest model (using Rborist) with cross-validation
# Tune 'predFixed' and 'minNode' parameters
train_rf_2 <- train(y ~ .,
                    method = "Rborist",
                    tuneGrid = data.frame(predFixed = 2, minNode = c(3, 50)),
                    data = mnist_27$train)

# Evaluate the cross-validated random forest performance
rf2_accuracy <- confusionMatrix(
  predict(train_rf_2, mnist_27$test),
  mnist_27$test$y
)$overall["Accuracy"]
print(rf2_accuracy)
