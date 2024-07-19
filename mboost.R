library(readxl)
library(tidyverse)
library(mboost)
library(tictoc)

for(data in c('Rad_München_Master_28_09_2021',
              'Rad_Berlin_2017_2020', 
              'Rad_Wien_2017_2020', 'Rad_Düsseldorf_2017_2020')){

  train_df <- read.csv(paste0('df/',data,'_train_df.csv')) %>%
  select(-X) %>% 
  mutate_if(str_detect(colnames(.), 'Dummy'), as.factor)
  train_df <- read.csv(paste0('df/',data,'_train_df.csv')) %>%
    select(-X) %>% 
    mutate_if(str_detect(colnames(.), 'Dummy'), as.character) %>%
    mutate_if(is.character, as.factor)
# Linear Model
lm_model <- lm(data = train_df %>% select(-date), formula = Bike_total ~.)
summary(lm_model)
saveRDS(lm_model, paste0('Models/',data,'/','lm_model.RDS'))

# boosted splines

# Define formula
mboost_formula_numeric <-
  train_df %>%
  select(-date, -Bike_total) %>%
  select_if(is.numeric) %>%
  colnames() %>%
  paste0('bols(', ., ', df = 1) + ', 'bbs(', ., ', df = 1, center = TRUE)',
         collapse = ' + ')

mboost_formula_categorical <-
  train_df %>%
  select(-date) %>%
  select_if(is.factor) %>%
  colnames() %>%
  paste0('bols(', ., ', df = 1)', collapse = ' + ')

mboost_formula_char <-
  paste0('Bike_total ~ ', mboost_formula_numeric, ' + ', mboost_formula_categorical)
mboost_formula <- mboost_formula_char %>%
  as.formula()

set.seed(403)
tic()
gam_mboost <- mboost(mboost_formula, data = train_df,
                        control = boost_control(mstop = 20000, nu = 0.1))
gam_mboost_small <- mboost(mboost_formula, data = train_df,
                           control = boost_control(mstop = 2000, nu = 0.1))
toc()
tic()
cl <- makeCluster(8)
myApply <- function(X, FUN, ...) {
    myFun <- function(...) {
        library("mboost") # load mboost on nodes
        FUN(...)
    }
    ## further set up steps as required
    parLapply(cl = cl, X, myFun, ...)
}
saveRDS(object =  gam_mboost, file = paste0('Models/',data,'/','mboost.RDS'))
saveRDS(object =  gam_mboost_small, file = paste0('Models/',data,'/','mboost_small.RDS'))
cv_gam_mboost <- cvrisk(gam_mboost, papply = myApply, grid = round(seq(1,20000, length.out = 25)),
                        folds = cv(model.weights(gam_mboost), type = 'kfold', B = 10))
stopCluster(cl)
gam_mboost <- gam_mboost[mstop(cv_gam_mboost)]
saveRDS(object =  gam_mboost, file = paste0('Models/',data,'/','mboost_tune.RDS'))
toc()
numeric_vars  <-
  train_df %>%
  select(-date, -Bike_total) %>%
  select_if(is.numeric) %>%
  colnames() 

mboost_formula_spatial <- vector()
for(var_1 in numeric_vars){
  for(var_2 in numeric_vars){
    if(var_2 != var_1){
      mboost_formula_spatial[paste0(var_1,var_2)] <- paste0('bspatial(', var_1,',', var_2, ', df = 1, center = TRUE)')
    }
  }
}
mboost_formula_spatial <- mboost_formula_spatial %>% paste0(collapse = ' + ')

character_vars  <-
  train_df %>%
  select(-date) %>%
  select_if(is.factor) %>%
  colnames() 
  
mboost_formula_interact <- vector()
for(var_1 in numeric_vars){
  for(var_2 in character_vars){
    if(var_2 != var_1){
      mboost_formula_interact[paste0(var_1,var_2)] <- paste0('bbs(', var_1,', by =', var_2, ', df = 1, center = TRUE)')
    }
  }
}
# mboost_formula_interact <- mboost_formula_interact %>% paste0(collapse = ' + ')
# 
# mboost_formula_int <- paste0(mboost_formula_char,' + ', mboost_formula_interact) %>%
#   as.formula()
# tic()
# interact_mboost <- mboost(mboost_formula_int, data = train_df,
#                      control = boost_control(mstop = 20000, nu = 0.1))
# toc()
# saveRDS(object = interact_mboost , file = paste0('Models/',data,'/','mboost_interact.RDS'))
# tic()
# cl <- makeCluster(8)
# cv_interact_mboost <- cvrisk(interact_mboost, papply = myApply,
#                              folds = cv(model.weights(interact_mboost), type = 'kfold', B = 10))
# toc()
# stopCluster(cl)
# mstop(cv_interact_mboost)
# interact_mboost <- interact_mboost[mstop(cv_interact_mboost)]
# saveRDS(object = interact_mboost , file = paste0('Models/',data,'/','mboost_interact_tune.RDS'))
mboost_formula_spatial <- paste0(mboost_formula_char,' + ', mboost_formula_interact, '+', mboost_formula_spatial) %>%
  as.formula()

tictoc::tic()
spatial_mboost <- mboost(formula = mboost_formula_spatial, data = train_df,
                        control = boost_control(mstop = 15000, nu = 0.1))
spatial_mboost_small <- mboost(formula = mboost_formula_spatial, data = train_df,
                         control = boost_control(mstop = 1500, nu = 0.1))
tictoc::toc()
saveRDS(object = spatial_mboost , file = paste0('Models/',data,'/','mboost_spatial.RDS'))
tic()
cl <- makeCluster(8)
cv_spatial_mboost <- cvrisk(spatial_mboost, papply = myApply, grid = round(seq(1,15000, length.out = 25)),
                            folds = cv(model.weights(spatial_mboost), type = 'kfold', B = 10))
stopCluster(cl)
tictoc::toc()
spatial_mboost <- spatial_mboost[mstop(cv_spatial_mboost)]
saveRDS(object = spatial_mboost , file = paste0('Models/',data,'/','mboost_spatial_tune.RDS'))
saveRDS(object = spatial_mboost_small , file = paste0('Models/',data,'/','mboost_spatial_small.RDS'))
print(1)
}

