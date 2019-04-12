library(tidyverse)
library(mice)
library(caret)
library(randomForest)
library(kknn)
library(nnet)

set.seed(242)

###############################
### Read in the training and test data.
###############################

df <- read_csv("training.csv") %>% 
  
  # Drop id and recorded_by because they provide no predictive information.
  
  select(-id, -recorded_by)

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
  select(-id, -recorded_by, -num_private) %>% 
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

# Eliminate two variables that have too many levels for computation.

df_rf <- df %>% 
  select(-lga, -ward)

model_rf <- randomForest(status_group ~ .,
                         data = df_rf,
                         ntree = 100)

###
# Run a KNN model.
###

model_knn <- train.kknn(status_group ~ ., df)

### 
# Run a multinomial logistic regression model.
###

# Used the reduced rf df to save time.

model_log <- multinom(status_group ~ .,
                      data = df_rf)

###############################
### Stacking.
###############################

# Calculate training predictions again.

preds_rf_stacked <- predict(model_rf, df_rf[,-38])
preds_knn_stacked <- predict(model_knn, df[, -40])
preds_log_stacked <- predict(model_log, df_rf[,-38])

# Make a new meta-model.

training_ensemble <- tibble(rf = preds_rf_stacked,
                            knn = preds_knn_stacked,
                            log = preds_log_stacked,
                            actual = df[[40]])

# Generate the stacked multinomial model, i.e., predict the 
# actual training data based on the three models.

model_stacked <- multinom(actual ~ .,
                          data = training_ensemble)

# Use the models to predict the test data.

test_rf <- test %>% 
  select(-lga, -ward)

# First do a weird workaround to get the random forest and knn to work.
# See https://stackoverflow.com/questions/24829674/r-random-forest-error-type-of-predictors-in-new-data-do-not-match.

test_rf <- rbind(df_rf[1, ], test_rf)
test_rf <- test_rf[-1,]

test <- rbind(df[1, ], test)
test <- test[-1,]

preds_rf_test <- predict(model_rf, test_rf[,-38])
preds_knn_test <- predict(model_knn, test[,-40])
preds_log_test <- predict(model_rf, test_rf[,-38])

# Combine the three results.

combined <- tibble(rf = factor(preds_rf_test),
                 knn = factor(preds_knn_test),
                 log = factor(preds_log_test))

# Use the meta-model to predict values of the test data.

preds_stacked <- predict(model_stacked, combined)

# Save and export the predictions.

walker_prediction$Prediction <- as.character(preds_stacked)

write_csv(walker_prediction, "walker_prediction_new.csv")
