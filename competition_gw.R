# Classification problem (k = 3)
# functional, needs repair, nonfunctional
# use KNN, multiclass logistic regression (228), LDA (246), QDA, Naive Bayes

library(tidyverse)
library(mice)
library(MLmetrics) # For F1 score.
library(caret) # Lots of stuff. See https://cran.r-project.org/web/packages/caret/vignettes/caret.html.
library(caretEnsemble)
library(AER)
library(janitor)
library(randomForest)
library(FNN)
library(nnet)

set.seed(242)

###############################
### Read in the training data.
###############################

df <- read_csv("training.csv") %>% 

# Drop id and recorded_by because they provide no predictive information.

  select(-id, -recorded_by) %>% 

# Also drop variables with too many levels.
# They cause problems for the random forest later on.

  select(-ward, -lga)
  
# Change the missing construction_year to NAs so that mice() can impute them later.
# Use a loop because mutate() will not assign NA values.

new_year <- c()

for (n in 1:nrow(df)) {
  if (df$construction_year[n] == 0) {
    new_year <- c(new_year, NA)
  }
  else {
    new_year <- c(new_year, df$construction_year[n])
  }
}

df$construction_year <- new_year

###############################
### Imput the missing values with mice.
###############################

# See the missing data in permit and public_meeting with md.pattern(raw).
# Use the default method.

imputed <- mice(df, m = 1)
filled <- complete(imputed)

# Assign the new values to the original df to keep it as a tibble.

df$construction_year <- (filled[, "construction_year"])
df$permit <- as.logical(filled[, "permit"])
df$public_meeting <- as.logical(filled[, "public_meeting"])

###############################
### Change character variables to factors.
###############################

# Cycle through variable names and save only those that are characters.

factors <- c()

for (var in names(df)) {
  if (class(df[[var]]) == "character") {
    factors <- c(factors, var)
  }
}

# Change the character variables to factors.

df <- df %>%
  mutate_at(.vars = factors, as.factor)

###############################
### Pre-process the data to ignore some values, scale others, etc.
###############################

# From the caret package.
# pre <- preProcess(df, method = c("center", "scale"))
# df <- predict(pre, df)

###############################
### Prepare the model.
###############################

# Define the predictors and outcome variable.

predictors <- df %>%
  select(-status_group) %>%
  names()

outcome <- "status_group"

# Make the outcome level names so that train() can use them.

df <- df %>% 
  mutate(status_group = case_when(
    status_group == "functional" ~ "functional",
    status_group == "functional needs repair" ~ "repair",
    status_group == "non functional" ~ "nonfunctional"
  )) %>% 
  mutate(status_group = as.factor(status_group))

# Split the data into training and testing sets.

partition <- createDataPartition(df$status_group, p = 0.8, list = FALSE)

training <- df[partition, ]
testing <- df[-partition, ]

###############################
### Prediction.
###############################

###
# Run a random forest model.
###
# NOTE: With ntree = 500 (default), this takes about 7 minutes to run on my machine.

model_rf <- randomForest(status_group ~ .,
                         data = training,
                         ntree = 100)

###
# Run a KNN model.
###
# First use data matrices, since knn.cv() only uses that.
# Run through k values from 1 through 12 to pick the best model.

cl <- data.matrix(training[,38])
test <- data.matrix(testing[,1:37])
train <- data.matrix(training[,1:37])

# Uncomment the below code to check which value of k is the best.
# I find k = 5 produces the highest F1 score.

# scores <- tibble(k = seq(1:13), f1 = NA)

# for (n in seq(3,13)) {
#   
#   model_knn <- knn.cv(train, cl, k = n)
# 
#   # Change the numeric results to characters.
#     
#   knn_results <- tibble(model_knn) %>% 
#     mutate(status_group = case_when(
#       knn_results[1] == 1 ~ "functional",
#       knn_results[1] == 2 ~ "nonfunctional",
#       knn_results[1] == 3 ~ "repair"))
#   
#   # Save the F1 score.
#   
#   scores$f1[n] <- F1_Score(training$status_group, knn_results$status_group)
# }

# Run the final model with the highest F1 score.

model_knn <- knn(train, test, cl, k = 5)

# Change the numeric results to characters.

knn_results <- tibble(model_knn) %>% 
  mutate(status_group = case_when(
    model_knn == 1 ~ "functional",
    model_knn == 2 ~ "nonfunctional",
    model_knn == 3 ~ "repair"))

###
# Run a multinomial logistic regression model.
###

model_log <- multinom(status_group ~ .,
                      data = training)

# Generate predicted values.

preds_rf <- predict(model_rf, testing[,-38])
preds_knn <- knn_results$status_group
preds_log <- predict(model_log, testing[,-38])

# Calculate the macro F1 score using the MLmetrics package.

f1_rf <- F1_Score(testing$status_group, preds_rf) # 80% test fold F1 is .843
f1_knn <- F1_Score(testing$status_group, preds_knn) # F1 is .767
f1_log <- F1_Score(testing$status_group, preds_log) # F1 is .797

###############################
### Voting.
###############################

voting <- tibble(rf = preds_rf, knn = preds_knn, log = preds_log, vote = NA)
