# Classification problem (k = 3)
# functional, needs repair, nonfunctional
# use KNN, multiclass logistic regression (228), LDA (246), QDA, Naive Bayes

# Judged by macro F1 score. Use F1_score() from the MLmetrics package.

library(tidyverse)
library(mice)
library(MLmetrics) # For F1 score.
library(caret) # Lots of stuff. See https://cran.r-project.org/web/packages/caret/vignettes/caret.html.
library(caretEnsemble)
library(AER)
library(janitor)
library(randomForest)

set.seed(242)

### Read in the training data.

df <- read_csv("training.csv") %>% 

# Drop id and recorded_by because they provide no predictive information.

  select(-id, -recorded_by)

# Change the missing construction_year to NAs so that mice()
# can impute them later.
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

### Imput the missing values in raw with mice.
# See the missing data in permit and public_meeting with md.pattern(raw).
# Use the default method.

imputed <- mice(df, m = 1)
filled <- complete(imputed)

# Assign the new values to the original df to keep it as a tibble.

df$construction_year <- as.factor(filled[, "construction_year"])
df$permit <- as.logical(filled[, "permit"])
df$public_meeting <- as.logical(filled[, "public_meeting"])

### Change most variables to factors.
# Make a list of the ones to keep as doubles.

doubles <- c("id", "population", "amount_tsh", "population")
factors <- c()

# Cycle through variable names and save only those not in the doubles list.

for (var in names(df)) {
  if (! var %in% doubles) {
    factors <- c(factors, var)
  }
}

# Change the factor variables to factors.

df <- df %>%
  mutate_at(.vars = factors, as.factor)

### Pre-process the data to fill in NAs with median value, ignore some values, scale others, etc.
# From the caret package.
pre <- preProcess(df, method = c("center", "scale"))
df <- predict(pre, df)

# Define the predictors and outcome variable.
predictors <- df %>%
  select(-status_group) %>%
  names()

outcome <- "status_group"

# Make the outcome levels usable variable names so that train()
# can use them.

df <- df %>% 
  mutate(status_group = case_when(
    status_group == "functional" ~ "functional",
    status_group == "functional needs repair" ~ "repair",
    status_group == "non functional" ~ "nonfunctional"
  )) %>% 
  mutate(status_group = as.factor(status_group))

# Split the data into training (80%) and testing data (20%).

partition <- createDataPartition(df$status_group, p = 0.75, list = FALSE)

training <- df[partition, ]
testing <- df[-partition, ]

### Renamed the F1 function from the MLmetrics F1_Score() function
# to make sure it works in-code.
# See MLmetrics:::F1_Score for source code.

macro_f1 <- function (y_true, y_pred, positive = NULL) {
  Confusion_DF <- ConfusionDF(y_pred, y_true)
  if (is.null(positive) == TRUE) 
    positive <- as.character(Confusion_DF[1, 1])
  
  Precision <- Precision(y_true, y_pred, positive)
  Recall <- Recall(y_true, y_pred, positive)
  
  F1_Score <- 2 * (Precision * Recall)/(Precision + Recall)
  
  return(F1_Score)
}

# Set the training parameters for multiple models.
# Default method is "boot".
# Number of folds or resampling iterations is 10 or 25.
# savePredictions = "final" only saves the optimal tuning parameters.
# classProbs = TRUE calculates the class probabilities along with the predicted classification.

parameters <- trainControl(
  method = "cv", 
  number = 1,
  savePredictions = "final",
  # classProbs = TRUE,
  verboseIter = TRUE)
  # summaryFunction = macro_f1

### Random forest model.

testmod <- train(as.factor(training$status_group) ~ .,
                 data = training,
                 method = "ranger",
                 trControl = parameters)

### TEST
library(e1071)
nbmod <- naiveBayes(training$status_group ~ ., training)

nb_predict <- predict(nbmod, testing[1:40])


#####
### NOTES
#####

# Use this to collect all the values with NAs in them to look at them.
test <- bind_rows(NULL, NULL)
for (n in 1:nrow(df)) {
  if (anyNA(df[n, ])) {
    test <- bind_rows(test, df[n, ])
  }
}
