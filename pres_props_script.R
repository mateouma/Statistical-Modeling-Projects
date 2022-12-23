library(tidyverse)
library(tidymodels)
library(xgboost)
# Load in data
dem_data <- read_csv("train.csv")
test <- read_csv("test.csv")
# select percentages, total population, housing units
dem_data4 <- dem_data %>% select(
  'percent_dem', '0001E', '0086E',
  contains("P"), contains("C02")
)
# model
xgb_model <- boost_tree(mode = "regression",
                        engine = "xgboost",
                        trees = 1000,
                        mtry = 48,
                        min_n = 21,
                        tree_depth = 8,
                        learn_rate = 4.191037e-02,
                        loss_reduction = 3.710132e-03,
                        sample_size = 0.5318252
)
# recipe
dem_recipe <-
  recipe(percent_dem ~ ., data = dem_data4) %>%
  step_interact( ~ `0037PE`:C02_008E) %>% # white * 9-12 edu
  step_interact( ~ `0037PE`:C02_010E) %>% # white * some
  college
step_interact( ~ `0037PE`:C02_021E) %>% # white * bachelors+
  step_interact( ~ `0037PE`:`0002PE`) %>% # white * male
  step_interact( ~ `0038PE`:C02_010E) %>% # black * above
  step_interact( ~ `0038PE`:C02_021E) %>%
  step_interact( ~ `0038PE`:`0002PE`) %>%
  step_interact( ~ `0071PE`:C02_010E) %>% # hispanic * above
  step_interact( ~ `0071PE`:C02_021E) %>%
  step_interact( ~ `0071PE`:`0002PE`)
xgb_submitted <- read_csv("xgb_mu.csv")
# create xgb workflow
set.seed(1)
xgb_test_wflow <-
  workflow() %>%
  add_model(xgb_model) %>%
  add_recipe(dem_recipe)
# fit model to test data using workflow
set.seed(1)
xgb_test_fit <- fit(xgb_test_wflow, dem_data)
# predict values from test data
xgb_test_results <-
  test %>%
  select(id) %>%
  bind_cols(predict(xgb_test_fit, new_data = test))
# create output file
names(xgb_test_results) <- c("Id", "Predicted")
write.csv(