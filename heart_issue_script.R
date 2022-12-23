# standard imports
library(tidyverse)
library(tidymodels)
# install.packages(“doParallel”) # if not installed already
# load in training data, turn columns from character into
# numeric, turn response into factor
heart_train <- read_csv("heart_train.csv")
heart_train <- heart_train %>%
  mutate_if(is.character, as.numeric) %>%
  suppressWarnings() %>%
  mutate(num = as.factor(num))
# remove id, trestbps, chol, fbs, and restecg columns
# call the reduced dataframe heart_train2
heart_train2 <- heart_train %>% select(-c(id, trestbps, chol,
                                          fbs, restecg))
# load in testing data, turn columns from character into
# numeric
heart_test <- read_csv("heart_test.csv")
heart_test2 <- heart_test %>%
  mutate_if(is.character, as.numeric) %>%
  suppressWarnings()
# create initial xgboost model
xgb_model <- boost_tree(mode = "classification",
                        trees=tune(), mtry=tune())
# create recipe
xgb_rec <- recipe(num~., data=heart_train2)
# create cross validation folds
set.seed(100)
heart_train2_folds <- vfold_cv(heart_train2, 10)
# workflow for tuning model
xgb_wf <- workflow() %>%
  add_model(xgb_model) %>%
  add_recipe(xgb_rec)
# create parameter grid for tuning
param_grid <- grid_latin_hypercube(
  mtry(range(c(1,50))),
  trees(),
  size = 10
)
# fit models from grid
doParallel::registerDoParallel()
set.seed(100)
xgb_res <- xgb_wf %>%
  tune_grid(
    resamples = heart_train2_folds,
    grid = param_grid,
    control = control_grid(verbose = T, save_pred = T)
  )
# find model with best accuracy and use in workflow
best_xgb <- xgb_res %>% select_best("accuracy")
final_xgb_wf <- xgb_wf %>% finalize_workflow(best_xgb)
# fit model to training data and apply to testing data
set.seed(100)
xgb_1 <- fit(final_xgb_wf, data = heart_train2)
final_test_results <- heart_train2 %>% select(num) %>%
  bind_cols(predict(xgb_1, new_data = heart_train2))
pred <- predict(xgb_1, heart_test2)
# write predictions to csv
newpred <- data.frame(id = heart_test2$id, Predicted = pred)
names(newpred)[2] <- "Predicted"
write.csv(newpred, row.names= FALSE, file =
            "xgb_1_classification.csv")