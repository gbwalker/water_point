library(tidyverse)
library(mice)
library(caret)
library(randomForest)
library(FNN)
library(nnet)

set.seed(242)

###############################
### Read in the training and test data.
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

### Do the same for the test data.

test <- read_csv("test.csv")

# Save the ids for the final prediction.

walker_prediction <- tibble(Id = test$id, Prediction = NA)

# Now remove the extra variables and add a column so it matches with df.

test <- test %>% 
  select(-id, -recorded_by, -ward, -lga, -num_private) %>% 
  mutate(status_group = factor("unknown"))

new_year <- c()

for (n in 1:nrow(test)) {
  if (test$construction_year[n] == 0) {
    new_year <- c(new_year, NA)
  }
  else {
    new_year <- c(new_year, test$construction_year[n])
  }
}

test$construction_year <- new_year

###############################
### Imput the missing values with mice.
###############################

# See the missing data in permit and public_meeting with md.pattern(raw).
# Use the default method.

imputed <- mice(df, m = 3)
filled <- complete(imputed)

# Assign the new values to the original df to keep it as a tibble.

df$construction_year <- (filled[, "construction_year"])
df$permit <- as.logical(filled[, "permit"])
df$public_meeting <- as.logical(filled[, "public_meeting"])

### Do the same for the test data.

imputed <- mice(test, m = 3)
filled <- complete(imputed)

test$construction_year <- (filled[, "construction_year"])
test$permit <- as.logical(filled[, "permit"])
test$public_meeting <- as.logical(filled[, "public_meeting"])

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

### Do the same for the test data.

factors <- c()

for (var in names(test)) {
  if (class(test[[var]]) == "character") {
    factors <- c(factors, var)
  }
}

test <- test %>%
  mutate_at(.vars = factors, as.factor)

###############################
### Prediction.
###############################

###
# Run a random forest model.
###

model_rf <- randomForest(status_group ~ .,
                         data = df,
                         ntree = 100)

###
# Run a KNN model.
###
# k = 5 was determined to have the highest F1 score. See competition_gw.R.
# Note that this model is not actually used in the final stacked version,
# which uses knn.cv with k = 5.

training <- data.matrix(df[,1:37])
testing <- data.matrix(test[,1:37])
cl <- data.matrix(df[,38])

model_knn <- knn(training, testing, cl, k = 5)

# Change the numeric results to characters.

knn_results <- tibble(model_knn) %>% 
  mutate(status_group = case_when(
    model_knn == 1 ~ "functional",
    model_knn == 2 ~ "non functional",
    model_knn == 3 ~ "functional needs repair")) %>% 
  mutate(status_group = factor(status_group))

###
# Run a multinomial logistic regression model.
###

model_log <- multinom(status_group ~ .,
                      data = df)

# Generate predicted values.

# First do a weird workaround to get the random forest to work.
# See https://stackoverflow.com/questions/24829674/r-random-forest-error-type-of-predictors-in-new-data-do-not-match.

test <- rbind(df[1, ], test)
test <- test[-1,]

preds_rf <- predict(model_rf, test[,-38])
preds_knn <- knn_results$status_group
preds_log <- predict(model_log, test[,-38])

###############################
### Voting.
###############################

voting <- tibble(rf = factor(preds_rf),
                 knn = factor(preds_knn),
                 log = factor(preds_log)) 

# Uncomment this to find the consensus between the three models.
# Note that preds_knn has only two factor levels, so a third must be added for the code to run.
# %>% 
# mutate(vote = case_when(
#   rf == knn ~ rf,
#   knn == log ~ knn,
#   rf == log ~ rf
# ))

###############################
### Stacking.
###############################

# Run another KNN for the training data.

model_knn_stacked <- knn.cv(training, cl, k = 5)

knn_results_stacked <- tibble(model_knn_stacked) %>% 
  mutate(status_group = case_when(
    model_knn_stacked == 1 ~ "functional",
    model_knn_stacked == 2 ~ "non functional",
    model_knn_stacked == 3 ~ "functional needs repair")) %>% 
  mutate(status_group = factor(status_group))

# Calculate training predictions again.

preds_rf_stacked <- predict(model_rf, df[,-38])
preds_knn_stacked <- knn_results_stacked$status_group
preds_log_stacked <- predict(model_log, df[,-38])

# Make a new meta-model.

training_ensemble <- tibble(rf = preds_rf_stacked,
                            knn = preds_knn_stacked,
                            log = preds_log_stacked,
                            actual = df[[38]])

# Generate the stacked multinomial model, i.e., predict the 
# actual training data based on the three models.

model_stacked <- multinom(actual ~ .,
                          data = training_ensemble)

# Use the meta-model to predict values of the test data.

preds_stacked <- predict(model_stacked, voting[,1:3])

# Save and export the predictions.

walker_prediction$Prediction <- as.character(preds_stacked)

write_csv(walker_prediction, "walker_prediction.csv")