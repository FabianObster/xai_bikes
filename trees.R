library(readxl)
library(tidyverse)
library(tree)
library(randomForest)
library(caret)
library(gbm)
library(e1071)
library(rstanarm)
library(FNN)
library(neuralnet)
library(tidymodels)
library(rpart)
library(kknn)
library(xgboost)
library(kernlab)
library(nnet)
set.seed(3)
for(data in c('Rad_München_Master_28_09_2021','Rad_Berlin_2017_2020', 
              'Rad_Wien_2017_2020', 'Rad_Düsseldorf_2017_2020')){

train_df <- read.csv(paste0('df/',data,'_train_df.csv')) %>%
  select(-X) %>% 
  mutate_if(str_detect(colnames(.), 'Dummy'), as.character) %>%
  mutate_if(is.character, as.factor)
test_df <- read.csv(paste0('df/',data,'_test_df.csv')) %>%
  select(-X) %>% 
  mutate_if(str_detect(colnames(.), 'Dummy'), as.character) %>%
  mutate_if(is.character, as.factor) 
# Tree Model
tree3_model <- tree(data = train_df %>% select(-date), formula = Bike_total ~.,
                   control = tree.control(
                     nobs = nrow(train_df),
                     mincut = 3,
                     minsize = 6,
                     mindev = 0.005
                   ))
tree1_model <- tree(data = train_df %>% select(-date), formula = Bike_total ~.,
                   control = tree.control(
                     nobs = nrow(train_df),
                     mincut = 1,
                     minsize = 2,
                     mindev = 0.0005
                   ))

saveRDS(tree1_model, paste0('Models/',data,'/','tree_model1.RDS'))
saveRDS(tree3_model, paste0('Models/',data,'/','tree_model2.RDS'))

# knn
knn1_model <- knn.reg(train = train_df %>% select(-date, -Bike_total) %>% 
                  mutate_all(function(x){scale(as.numeric(x))}),
                  test = test_df %>% select(-date, -Bike_total) %>% 
                    mutate_all(function(x){scale(as.numeric(x))}),
                y = train_df$Bike_total, k = 1)
saveRDS(knn1_model, paste0('Models/',data,'/','knn1_model.RDS'))


knn27_model <- knn.reg(train = train_df %>% select(-date, -Bike_total) %>% 
                        mutate_all(function(x){scale(as.numeric(x))}),
                      test = test_df %>% select(-date, -Bike_total) %>% 
                        mutate_all(function(x){scale(as.numeric(x))}),
                      y = train_df$Bike_total, k = 27)
saveRDS(knn27_model, paste0('Models/',data,'/','knn27_model.RDS'))
# rf model
rf_model <- randomForest(Bike_total ~.,data = train_df %>% select(-date), nodes = T)
summary(rf_model)
saveRDS(rf_model, paste0('Models/',data,'/','rf1_model.RDS'))
# rf 
rf_model <- randomForest(Bike_total ~.,data = train_df %>% select(-date), 
                         nodes = T, ntree = 1000, mtry = 6)
summary(rf_model)
saveRDS(rf_model, paste0('Models/',data,'/','rf2_model.RDS'))
# gbm

gbm7_model <- gbm(
  formula = Bike_total ~ ., n.trees = 100, interaction.depth = 7, n.minobsinnode = 1,
  distribution = "gaussian",
  data = train_df %>% select(-date),
)  
saveRDS(gbm7_model, paste0('Models/',data,'/','gbm7_model.RDS'))

gbm3_model <- gbm(
  formula = Bike_total ~ ., n.trees = 100, interaction.depth = 3, n.minobsinnode = 1,
  distribution = "gaussian",
  data = train_df %>% select(-date),
)  
saveRDS(gbm3_model, paste0('Models/',data,'/','gbm3_model.RDS'))

gbm_grid <- expand.grid(n.trees = c(50, 75, 100,150, 200),
                      interaction.depth = c(5, 6, 7, 8, 9))
train_cv <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
gbm_tune <- train(Bike_total ~ .,data = train_df %>% select(-date) %>% mutate_all(as.numeric), method = "gbm",
                  distribution = "gaussian", trControl = train_cv, verbose = FALSE)
saveRDS(gbm_tune$finalModel, paste0('Models/',data,'/','gbmtune_model.RDS'))
# svm

svmp_model <- e1071::svm(formula = Bike_total ~ ., data = train_df %>% select(-date), 
                        kernel = 'polynomial', degree = 3)
saveRDS(svmp_model, paste0('Models/',data,'/','svmp_model.RDS'))

svmr_model <- e1071::svm(formula = Bike_total ~ ., data = train_df %>% select(-date), 
                        kernel = 'radial', degree = 3)
saveRDS(svmr_model, paste0('Models/',data,'/', 'svmr_model.RDS'))

# mean model

mean_model <- lm(formula = Bike_total ~ 1, data = train_df %>% select(-date))
saveRDS(mean_model, paste0('Models/',data,'/','mean_model.RDS'))

# bayes

bayes_model <- stan_glm(formula = Bike_total ~ ., data = train_df %>% select(-date))
saveRDS(bayes_model, paste0('Models/',data,'/','bayes_model.RDS'))

# NN


nn1=neuralnet(Bike_total~., data= train_df %>% select(-date) %>%
                mutate_all(function(x){scale(as.numeric(x))}), err.fct = 'sse',
              hidden=c(5), linear.output=T, act.fct = "logistic", stepmax=1e7)
#plot(nn1)
saveRDS(nn1, paste0('Models/',data,'/','nn1_model.RDS'))


nn2=neuralnet(Bike_total~., data= train_df %>% select(-date) %>%
                mutate_all(function(x){scale(as.numeric(x))}), err.fct = 'sse',
              hidden=c(5,3), linear.output=T, act.fct = "logistic", stepmax=1e7)
saveRDS(nn2, paste0('Models/',data,'/','nn2_model.RDS'))


### with hyperparameter tuning
data_folds <- vfold_cv(train_df %>% select(-date))
tune_me <- function(tune_spec, model_grid){
  model_wf <- workflow() %>%
    add_model(tune_spec) %>%
    add_formula(Bike_total~.)
  model_res <- model_wf %>%
    tune_grid(resamples = data_folds, grid = model_grid)
  best_model <- model_res %>%
    select_best('rmse') 
  final_wf <- model_wf %>%
    finalize_workflow(best_model)
  model_fit <- final_wf %>%
    fit(data = train_df %>% select(-date)) %>%
    extract_fit_engine() 
  return(model_fit)
}

# tree
tune_spec <- 
  decision_tree(cost_complexity = tune(), tree_depth = tune()) %>%
  set_engine('rpart') %>%
  set_mode(mode = 'regression')

tree_grid <- grid_regular(cost_complexity(), tree_depth(), levels = 5)
tree_tune_model <- tune_me(tune_spec = tune_spec, model_grid = tree_grid)
saveRDS(tree_tune_model, paste0('Models/',data,'/','tree_tune_model.RDS'))

# knn
tune_spec <- 
  nearest_neighbor(neighbors = tune()) %>%
  set_engine('kknn') %>%
  set_mode(mode = 'regression')

knn_grid <- grid_regular(neighbors(), levels = 25)
knn_tune_model <- tune_me(tune_spec = tune_spec, model_grid = knn_grid)
best_k <- knn_tune_model$best.parameters$k

knn_tune_model <- knn.reg(train = train_df %>% select(-date, -Bike_total) %>% 
                         mutate_all(function(x){scale(as.numeric(x))}),
                       test = test_df %>% select(-date, -Bike_total) %>% 
                         mutate_all(function(x){scale(as.numeric(x))}),
                       y = train_df$Bike_total, k = best_k)
saveRDS(knn_tune_model, paste0('Models/',data,'/','knn_tune_model.RDS'))
# RF
tune_spec <- 
  rand_forest(mtry = tune(), trees = tune()) %>%
  set_engine('randomForest') %>%
  set_mode(mode = 'regression')

rf_grid <- expand.grid(
  mtry = c(1, 3, 4, 5, 7), 
  trees = c(250, 500, 750, 1000, 2000)
)
rf_tune_model <- tune_me(tune_spec = tune_spec, model_grid = rf_grid)
saveRDS(rf_tune_model, paste0('Models/',data,'/','rf_tune_model.RDS'))


# # xgboost
# tune_spec <- 
# parsnip::boost_tree(min_n = tune(), tree_depth = tune()) %>%
#   set_engine('xgboost') %>%
#   set_mode(mode = 'regression')
# 
# xgb_folds <- vfold_cv(train_df %>% select(-date) %>% mutate_all(as.numeric))
# 
# xgb_grid <- grid_regular(min_n(), tree_depth(), levels = 5)
# model_wf <- workflow() %>%
#   add_model(tune_spec) %>%
#   add_formula(Bike_total~.)
# model_res <- model_wf %>%
#   tune_grid(resamples = xgb_folds, grid = xgb_grid)
# best_model <- model_res %>%
#   select_best('rmse') 
# final_wf <- model_wf %>%
#   finalize_workflow(best_model)
# model_fit <- final_wf %>%
#   fit(data = train_df %>% select(-date) %>%
#         mutate_all(as.numeric)) %>%
#   extract_fit_engine() 
# saveRDS(model_fit, paste0('Models/',data,'/','xgb_tune_model.RDS'))
####################
# svm
tune_spec <- 
  parsnip::svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
  set_engine('kernlab') %>%
  set_mode(mode = 'regression')

svm_grid <- grid_regular(cost(), rbf_sigma(), levels = 5)
svm_tune_model <- tune_me(tune_spec = tune_spec, svm_grid)
saveRDS(svm_tune_model, paste0('Models/',data,'/','svm_tune_model.RDS'))

# nn
covars <- train_df %>% select(-date, -Bike_total) %>% colnames()
data_folds <- vfold_cv(train_df %>% select(-date) %>%
                         mutate_at(covars,as.numeric) %>%
                         mutate_at(covars,function(x){(x-min(x))/(max(x)-min(x))}) %>%
                         mutate(Bike_total = as.numeric(scale(Bike_total))))
tune_spec <- 
  parsnip::mlp(hidden_units = tune(), penalty = tune(), activation = "relu") %>%
  set_engine('nnet') %>%
  set_mode(mode = 'regression')

nn_grid <- grid_regular(hidden_units(), penalty(), levels = 5)
model_wf <- workflow() %>%
  add_model(tune_spec) %>%
  add_formula(Bike_total~.)
model_res <- model_wf %>%
  tune_grid(resamples = data_folds, grid = nn_grid)
best_model <- model_res %>%
  select_best('rmse') 
final_wf <- model_wf %>%
  finalize_workflow(best_model)
model_fit <- final_wf %>%
  fit(data = train_df %>% select(-date) %>%
        mutate_at(covars,as.numeric) %>%
        mutate_at(covars,function(x){(x-min(x))/(max(x)-min(x))}) %>%
        mutate(Bike_total = as.numeric(scale(Bike_total)))) %>%
  extract_fit_engine() 
saveRDS(model_fit, paste0('Models/',data,'/','nn_tune_model.RDS'))
print(data)
}
