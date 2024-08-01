library(ggpubr)
library(tidyverse)
library(tree)
library(randomForest)
library(caret)
library(gbm)
library(e1071)
library(rstanarm)
library(neuralnet)
library(mboost)
library(FNN)
library(GGally)
library(mgcv)
library(lme4)
library(tidymodels)
library(rpart)
library(kknn)
library(xgboost)
library(kernlab)
library(nnet)
library(readxl)
library(writexl)

for(data in c('Rad_München_Master_28_09_2021','Rad_Berlin_2017_2020', 
              'Rad_Wien_2017_2020', 'Rad_Düsseldorf_2017_2020')){

  test_df <- read.csv(paste0('df/',data,'_test_df.csv')) %>%
  select(-X) %>% 
  mutate_if(str_detect(colnames(.), 'Dummy'), as.character) %>%
  mutate_if(is.character, as.factor)
train_df <- read.csv(paste0('df/',data,'_train_df.csv')) %>%
  select(-X) %>% 
  mutate_if(str_detect(colnames(.), 'Dummy'), as.character) %>%
  mutate_if(is.character, as.factor)
covars <- train_df %>% select(-date, -Bike_total) %>% colnames()
lm.model <- readRDS(paste0('Models/',data,'/','lm_model.RDS')) #
mb1.model <- readRDS(paste0('Models/',data,'/','mboost.RDS')) #
mb2.model <- readRDS(paste0('Models/',data,'/','mboost_small.RDS'))
mbtune.model <- readRDS(paste0('Models/',data,'/','mboost_tune.RDS'))
#mb.int.model <- readRDS(paste0('Models/',data,'/','mboost_interact.RDS')) #
mbsp1.model <- readRDS(paste0('Models/',data,'/','mboost_spatial.RDS')) #
mbsp2.model <- readRDS(paste0('Models/',data,'/','mboost_spatial_small.RDS'))
mbsptune.model <- readRDS(paste0('Models/',data,'/','mboost_spatial_tune.RDS'))
rf1.model <- readRDS(paste0('Models/',data,'/','rf1_model.RDS')) #
rf2.model <- readRDS(paste0('Models/',data,'/','rf2_model.RDS')) #
rftune.model <- readRDS(paste0('Models/',data,'/','rf_tune_model.RDS'))
svmp.model <- readRDS(paste0('Models/',data,'/','svmp_model.RDS')) #
svmr.model <- readRDS(paste0('Models/',data,'/','svmr_model.RDS')) #
svmtune.model <- readRDS(paste0('Models/',data,'/','svm_tune_model.RDS'))
mean.model <- readRDS(paste0('Models/',data,'/','mean_model.RDS')) #
tree1.model <- readRDS(paste0('Models/',data,'/','tree_model1.RDS')) #
tree2.model <- readRDS(paste0('Models/',data,'/','tree_model2.RDS'))
treetune.model <- readRDS(paste0('Models/',data,'/','tree_tune_model.RDS'))
bay.model <- readRDS(paste0('Models/',data,'/','bayes_model.RDS'))
gbm1.model <- readRDS(paste0('Models/',data,'/','gbm3_model.RDS'))
gbm2.model <- readRDS(paste0('Models/',data,'/','gbm7_model.RDS'))
gbmtune.model <- readRDS( paste0('Models/',data,'/','gbmtune_model.RDS'))
nn1.model <- readRDS(paste0('Models/',data,'/','nn1_model.RDS'))
nn2.model <- readRDS(paste0('Models/',data,'/','nn2_model.RDS'))
nntune.model <- readRDS(paste0('Models/',data,'/','nn_tune_model.RDS'))
knn2.model <- readRDS(paste0('Models/',data,'/','knn1_model.RDS'))
knn1.model <- readRDS(paste0('Models/',data,'/','knn27_model.RDS'))
knntune.model <- readRDS(paste0('Models/',data,'/','knn_tune_model.RDS'))
get_mb_params <- function(model){
   coef_vec <- model$coef() %>% lapply(length) %>% unlist() 
   coef_df <- tibble(learner = as.character(names(coef_vec)), number = coef_vec)
   coef_df <- coef_df %>%
     mutate(
       learner = str_replace_all(coef_df$learner, '\\(.*', '')) %>%
     mutate(number = case_when(learner == 'bols' ~ number-1L, T ~ number))
   return(sum(coef_df$number, na.rm = T))
}


pred_df <- test_df %>%
  dplyr::mutate(lm = as.numeric(predict(object =  lm.model, newdata = test_df)),
                mb1 = as.numeric(predict(mb1.model, test_df)),
                mb2 = as.numeric(predict(mb2.model, test_df)),
                mbtune = as.numeric(predict(mbtune.model, test_df)),
                mbsp1 = as.numeric(predict(mbsp1.model, test_df)),
                mbsp2 = as.numeric(predict(mbsp2.model, test_df)),
                mbsptune = as.numeric(predict(mbsptune.model, test_df)),
                mb1_params = get_mb_params(mb1.model),
                mb2_params = get_mb_params(mb2.model),
                mbtune_params = get_mb_params(mbtune.model),
                mbsp1_params = get_mb_params(mbsp1.model),
                mbsp2_params = get_mb_params(mbsp2.model),
                mbsptune_params = get_mb_params(mbsptune.model),
                lm_params = length(lm.model$coefficients),
                tree1_params = summary(tree1.model)$size,
                tree1 = as.numeric(predict(tree1.model, newdata = test_df)),
                tree2_params = summary(tree2.model)$size,
                tree2 = as.numeric(predict(tree2.model, newdata = test_df)),
                treetune_params = max(treetune.model$cptable[,2]),
                treetune = predict(treetune.model, newdata = test_df),
                mean = mean(train_df$Bike_total),
                mean_params = 1,
                rf1_params = sum(rf1.model$forest$ndbigtree),
                rf1 = as.numeric(predict(rf1.model, newdata = test_df %>% 
                                          select(-date,- Bike_total))),
                rf2_params = sum(rf2.model$forest$ndbigtree),
                rf2 = as.numeric(predict(rf2.model, newdata = test_df %>% 
                                          select(-date,- Bike_total))),
                rftune = predict(rftune.model, newdata = test_df),
                rftune_params = sum(rftune.model$forest$ndbigtree),
                svmp = as.numeric(predict(svmp.model, newdata = test_df)),
                svmp_params = length(svmp.model$coefs)*3,
                svmr = as.numeric(predict(svmr.model, newdata = test_df)),
                svmr_params = length(svmr.model$coefs)*3,
                svmtune = predict(svmtune.model, test_df),
                svmtune_params = nSV(svmtune.model),
                bay = as.numeric(predict(bay.model, newdata = test_df)),
                bay_params = length(bay.model$coefficients),
                gbm1 = as.numeric(predict(gbm1.model, newdata = test_df)),
                gbm1_params = (gbm1.model$interaction.depth+1)*gbm1.model$n.trees,
                gbm2 = as.numeric(predict(gbm2.model, newdata = test_df)),
                gbm2_params = (gbm2.model$interaction.depth+1)*gbm2.model$n.trees,
                gbmtune = as.numeric(predict(
                  gbmtune.model, newdata = test_df %>% select(-Bike_total, -date) %>%
                    mutate_all(as.numeric))),
                gbmtune_params = (gbmtune.model$interaction.depth+1)*gbmtune.model$n.trees,
                nn1 = as.numeric(predict(nn1.model, newdata = test_df %>%
                                           select(-date) %>% 
                                           mutate_all(function(x){scale(as.numeric(x))}),
                                                      linear.output=T)),
                nn1 = nn1*sd(train_df$Bike_total)+ mean(train_df$Bike_total),
                nn1_params = length(unlist(nn1.model$weights)),
                nn2 = as.numeric(predict(nn2.model, newdata = test_df %>%
                                           select(-date) %>%
                                           mutate_all(function(x){scale(as.numeric(x))}),
                                         linear.output=T)),
                nn2 = nn2*sd(train_df$Bike_total)+ mean(train_df$Bike_total),
                nn2_params = length(unlist(nn1.model$weights)),
                nntune = predict(nntune.model, test_df %>% select(-date) %>%
                  mutate_at(covars,as.numeric) %>%
                  mutate_at(covars,function(x){(x-min(x))/(max(x)-min(x))})),
                nntune = nntune*sd(train_df$Bike_total)+ mean(train_df$Bike_total),
                nntune_params = length(nntune.model$wts),
                knn2 = knn2.model$pred,
                knn2_params = round(knn2.model$n/knn1.model$k),
                knn1 = knn1.model$pred,
                knn1_params = round(knn1.model$n/knn1.model$k),
                knntune = knntune.model$pred,
                knntune_params = round(knntune.model$n/knntune.model$k),
                year = substring(date,1,4)
                )
summary_df <- pred_df %>% 
  dplyr::group_by(year) %>%
  summarize_at(c('lm', 'mb1','mb2', 'mbtune', 'mbsp1', 'mbsp2', 'mbsptune' , 'knn1', 'knn2', 'knntune',
                 'tree1', 'tree2','treetune', 'svmp', 'svmr', 'svmtune','gbm1', 'gbm2', 'gbmtune',
                 'rf1', 'rf2','rftune', 'nn1', 'nn2', 'nntune'
                 ), 
               list(
                 'rmse' = ~(sqrt(mean(abs(.-Bike_total)^2))),
                 'mse' = ~(mean(abs(.-Bike_total)^2)),
                 'mae' = ~(mean(abs(.-Bike_total))),
                 'cor' = ~(mean(cor(., Bike_total)))
               )) %>%
  cbind(pred_df %>% dplyr::slice(1) %>% select(contains('_params')))
summary_df <- 
  summary_df %>%
  pivot_longer(!year, names_to = c('model', '.value'), names_pattern = '(.*)_(.*)') %>%
  mutate(model_type = str_replace_all(model, '[0-9]|\\..*', ''),
         model_type = str_replace_all(model_type, 'tune', ''),
         model_type = case_when(model_type %in% c('svmp', 'svmr') ~ 'svm',
                                T ~ model_type),
         model_type_rank = case_when(model_type == 'mean' ~ 0,
                                     model_type == 'lm' ~ 1,
                                     model_type == 'tree' ~ 2,
                                     model_type == 'knn' ~ 3,
                                     model_type == 'mb' ~ 4,
                                     model_type == 'mbsp' ~ 5,
                                     model_type == 'svm' ~ 6,
                                     model_type == 'gbm' ~ 7,
                                     model_type == 'rf' ~ 8,
                                     model_type == 'nn' ~ 9))
summary_df <- summary_df %>%
  group_by(year) %>%
  arrange(model_type_rank, params) %>%
  mutate(rank = 1:n(), city = data)
# summary_dff <- summary_df %>%
#   mutate(rank = case_when(model == 'mean' ~ 0, model == 'lm' ~ 1, model == 'tree' ~ 2,
#                           model == 'knn1' ~ 3, model == 'knn27' ~ 4,
#                           model == 'mb' ~ 5, model == 'mb.int' ~ 6, 
#                           model == 'mb.sp' ~ 8, model == 'bay' ~ 9,
#                           model == 'svmp' ~ 10, model == 'svmr' ~ 11,
#                           model == 'gbm3' ~ 12, model == 'gbm7' ~ 13,
#                           model == 'rf1' ~ 14, model == 'rf2' ~ 15,
#                           model == 'nn1' ~ 16, model == 'nn2' ~ 17),
#          city = data)
write.csv(summary_df, paste0('Figures/_',data, '_test_metrics.csv'))

# Compute correlations
cor(summary_df$rank, summary_df$mse, method = 'spearman')
cor(summary_df$params, summary_df$rmse)
cor(log(summary_df$params), summary_df$rmse)
cor(summary_df$rank, summary_df$cor, method = 'spearman')
cor(summary_df$params, summary_df$cor)
cor(log(summary_df$params), summary_df$cor)


#
summary_df %>%
  ggplot(aes(y = cor, x = log(params), label = model)) + geom_point() + 
  geom_text(hjust = 0, nudge_x = 0.05) + theme_bw(base_size = 18)
  ggsave(paste0('Figures/',data, '/_cor.png'), width = 7, height = 5)
  
  summary_df %>% filter(!(model %in% c('mean'))) %>%
    ggplot(aes(y = rmse, x = log(params), label = model)) + geom_point() + 
    geom_text(hjust = 0, nudge_x = 0.05) + theme_bw(base_size = 18)
  ggsave(paste0('Figures/',data, '/_rmse.png'), width = 7, height = 5)

  summary_df %>% filter(!(model %in% c('mean'))) %>%
    ggplot(aes(y = mae, x = log(params), label = model)) + geom_point() + 
    geom_text(hjust = 0, nudge_x = 0.05) + theme_bw(base_size = 18)
  ggsave(paste0('Figures/',data, '/_mae.png'), width = 7, height = 5)
  
}

# all together

dat_list <- list.files("Figures", pattern="*.csv", full.names=TRUE)
dat_list <- dat_list
df <- lapply(dat_list[1:4], read.csv) %>% 
  bind_rows() 
df <- df %>%
  filter(model_type != 'mean', cor >0.3) %>%
  mutate(interpretability = model_type_rank < 6)

df %>% group_by(interpretability) %>%
  summarize(cor_params = round(cor(params, cor),3),
            log_cor_params = round(cor(log(params), cor),3),
            cor_rank = round(cor(rank, cor, method = 'spearman'),3)) %>%
  writexl::write_xlsx('Figures/rank.cor.all.xlsx')
df %>% 
  filter(str_detect(model, 'tune')) %>%
  group_by(interpretability) %>%
  summarize(cor_params = round(cor(params, cor),3),
            log_cor_params = round(cor(log(params), cor),3),
            cor_rank = round(cor(rank, cor, method = 'spearman'),3)) %>%
  writexl::write_xlsx('Figures/rank.cor.tune.xlsx')


log_cor_params_all <- round(cor(log(df$params), df[['cor']]),2)
cor_rank_all <- round(cor(df$rank, df[['cor']], method = 'spearman'), 2)

log_cor_params_tune <- round(cor(log(df[str_detect(df$model, 'tune'),]$params),
                                df[str_detect(df$model, 'tune'),][['cor']]),2)
cor_rank_tune <- round(cor(df[str_detect(df$model, 'tune'),]$rank,
                           df[str_detect(df$model, 'tune'),]$cor, method = 'spearman'), 2)

# scatterplot
df %>% 
  mutate(tuned = case_when(str_detect(model, 'tune')~ 'Tuned models', T ~'All models'),
         label= case_when(!str_detect(model, 'tune')~
                            paste0('cor: ', log_cor_params_all,'\n', 'rank cor: ', cor_rank_all),
                          T ~ paste0('cor: ', log_cor_params_tune,'\n', 'rank cor: ', cor_rank_tune))) %>%
  ggplot(aes(x = log(params), y = cor)) +
  geom_point(aes(color = model_type), size = 0.8) + 
  scale_color_brewer(palette="Paired") +
  theme_bw(base_size = 18) + xlab('Number parameters (log)') + ylab(bquote(R^2)) +
  geom_smooth(method='lm', se = F, color = 'black') + ylim(c(0.65,1)) + 
  geom_label(aes(label = label), x=11, y=0.66) +
  theme(legend.title = element_blank()) + facet_wrap(~tuned)
ggsave('../Figures/scatterplot.png', width = 7, height = 5)

df %>% 
  ggplot(aes(x = log(params), y = cor)) +
  geom_point(aes(color = model_type), size = 0.8) + 
  scale_color_brewer(palette="Paired") +
  theme_bw(base_size = 18) +
  geom_smooth(method='lm', se = F, color = 'black') + ylim(c(0.65,1)) + 
  geom_label(label= paste0('cor: ', log_cor_params_all,'\n', 'rank cor: ', cor_rank_all), x=11, y=0.66) +
  theme(legend.title = element_blank()) 
ggsave('Figures/Tables/cor_all.png', width = 7, height = 5)

df %>% 
  filter(str_detect(model, 'tune')) %>%
  ggplot(aes(x = log(params), y = cor)) +
  geom_point(aes(color = model_type), size = 0.8) + 
  scale_color_brewer(palette="Paired") +
  theme_bw(base_size = 18) +
  geom_smooth(method='lm', se = F, color = 'black') + ylim(c(0.65,1)) + 
  theme(legend.title = element_blank()) +
  geom_label(label= paste0('cor: ', log_cor_params_tune,'\n', 'rank cor: ', cor_rank_tune), x=11, y=0.66)
ggsave('Figures/Tables/cor_tune.png', width = 7, height = 5)


df %>% 
  ggplot(aes(x = log(params), y = cor, linetype = interpretability)) +
  geom_point(aes(color = model_type), size = 0.8) + 
  scale_color_brewer(palette="Paired") +
  theme_bw(base_size = 18) +
  geom_smooth(method='lm', se = F, color = 'black') + ylim(c(0.65,1)) + 
  geom_label(label= paste0('cor: ', log_cor_params_all,'\n', 'rank cor: ', cor_rank_all), x=11, y=0.66)
ggsave('Figures/cor_all.png', width = 7, height = 5)

df %>% 
  filter(str_detect(model, 'tune')) %>%
  ggplot(aes(x = log(params), y = cor, linetype = interpretability)) +
  geom_point(aes(color = model_type), size = 0.8) + 
  scale_color_brewer(palette="Paired") +
  theme_bw(base_size = 18) +
  geom_smooth(method='lm', se = F, color = 'black') + ylim(c(0.65,1)) + 
  geom_label(label= paste0('cor: ', log_cor_params_tune,'\n', 'rank cor: ', cor_rank_tune), x=11, y=0.66)
ggsave('Figures/cor_tune.png', width = 7, height = 5)


# boxplots
model_vars <- df %>% 
  arrange(rank) %>% 
  mutate(model = as.character(model)) %>% 
  magrittr::extract2('model') %>%
  unique()
model_type_vars <- df %>% 
  arrange(rank) %>% 
  mutate(model = as.character(model)) %>% 
  magrittr::extract2('model_type') %>%
  unique()

box_small <- df %>%
  mutate(model = factor(
    model, 
    levels = model_vars)
    ) %>%
  ggplot(aes_string(x = 'model', y = 'cor')) + geom_boxplot() + 
  ylim(c(0.65,1)) + ylab(bquote(R^2)) + xlab('Model') + theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust=1)) 
ggsave('Figures/cor_all_boxplot.png', width = 7, height = 5)

box_big <- df %>%
  mutate(model = factor(
    model_type, 
    levels = model_type_vars)
  ) %>%
  ggplot(aes_string(x = 'model', y = 'cor')) + geom_boxplot() + 
  ylim(c(0.65,1)) + theme_bw(base_size = 12) +  ylab(bquote(R^2)) + xlab('Model type') +
  theme(axis.text.x = element_text(angle = 45, hjust=1)) 
ggsave('Figures/cor_model_type_boxplot.png', width = 7, height = 5)


mean_big <- df %>%
  mutate(model = factor(
    model, 
    levels = model_vars),
    interpretability = case_when(interpretability~ 'interpretable', T~ 'not interpretable')
  ) %>%
  group_by(model, interpretability) %>%
  summarize(cor = mean(cor), rank = mean(rank)) %>%
  ggplot(aes_string(x = 'model', y = 'cor',  color = 'interpretability')) +
  ylab(bquote(R^2)) + xlab('Model') +
  geom_point() + geom_smooth(aes(x = rank, y = cor), method = 'lm', se = F) +
  ylim(c(0.65,1)) + theme_bw(base_size = 12) + scale_color_manual(values = c("#999999", "#E69F00")) +
  theme(axis.text.x = element_text(angle = 45, hjust=1), legend.title=element_blank(),
        legend.position = 'top') 
ggsave('Figures/cor_model_scatter.png', width = 7, height = 5)



mean_small <- df %>%
  mutate(model = factor(
    model_type, 
    levels = model_type_vars),
    interpretability = case_when(interpretability~ 'interpretable', T~ 'not interpretable')
  ) %>%
  group_by(model, interpretability) %>%
  summarize(cor = mean(cor), rank = mean(model_type_rank)) %>%
  ggplot(aes_string(x = 'model', y = 'cor',  color = 'interpretability')) +
  geom_point() + geom_smooth(aes(x = rank, y = cor), method = 'lm', se = F) +
  ylim(c(0.65,1)) + theme_bw(base_size = 12) + scale_color_manual(values = c("#999999", "#E69F00")) +
  theme(axis.text.x = element_text(angle = 45, hjust=1), legend.title=element_blank(),
        legend.position = 'top') + ylab(bquote(R^2)) + xlab('Model type')
ggsave('Figures/cor_model_type_scatter.png', width = 7, height = 5)
grid_plot <- ggarrange(mean_big, mean_small,box_small, box_big, ncol=2, nrow = 2, 
                          labels = c("a)","b)","c)","d)"))
ggsave( '../Figures/rank_figure.png',grid_plot, width = 9, height = 8)


df %>% 
  group_by(model) %>%
  summarize_at(c('rmse', 'mse', 'mae', 'cor', 'params',  'rank'), 
               list('mean' = function(x){round(mean(x), 3)}, 
                    'sd' = function(x){round(sd(x), 3)}, 
                    'median' = function(x){round(median(x), 3)})) %>%
  write_xlsx('Figures/all_metrics_models.xlsx')
  
df %>% 
  group_by(model_type) %>%
  summarize_at(c('rmse', 'mse', 'mae', 'cor', 'params',  'rank'), 
               list('mean' = function(x){round(mean(x), 3)}, 
                    'sd' = function(x){round(sd(x), 3)}, 
                    'median' = function(x){round(median(x), 3)})) %>%
  write_xlsx('Figures/all_metrics_model_type.xlsx')
  

# parallel plot

# par_df <- df %>% select(model, city, year, cor) %>%
#   spread(value = cor, key = model) %>%
#   select(model_vars, city,year) %>%
#   mutate(year = as.character(year))
# par_df %>%
#   ggparcoord(
#              columns = 1:(length(colnames(par_df))-3),
#              groupColumn = "year",
#              showPoints = F,
#              scale = 'globalminmax'
#              ) +
#     scale_color_brewer(palette = "Set2") +
#   theme_bw(base_size = 18) + ylim(c(0.65,1)) +
#   theme(axis.text.x = element_text(angle = 45, hjust=1))
# ggsave('Figures/cor_parallel.png', width = 7, height = 5)

# models
# within model (same results)

interpretability_mod_1 <- gam(data = df %>% filter(model != 'lm', interpretability) %>% mutate(model_type = as.factor(model_type)), 
                  formula = cor ~ rank + s(model_type, bs = 're'),
                  method="REML", family = 'gaussian')
summary(interpretability_mod_1)
interpretability_mod_2 <- gam(data = df %>% filter(model != 'lm', !interpretability) %>% mutate(model_type = as.factor(model_type)), 
                              formula = cor ~ rank + s(model_type, bs = 're'),
                              method="REML", family = 'gaussian')
summary(interpretability_mod_2)


global_mod_1 <- gam(data = df %>% mutate(model_type = as.factor(model_type)) %>% filter(interpretability), 
                  formula = cor ~  rank,
                  method="REML", family = 'gaussian')
summary(global_mod_1)
global_mod_2 <- gam(data = df %>% mutate(model_type = as.factor(model_type)) %>% filter(!interpretability), 
                    formula = cor ~  rank,
                    method="REML", family = 'gaussian')
summary(global_mod_2)


params_mod_1 <- gam(data = df%>% mutate(model_type = as.factor(model_type)) %>% filter(interpretability), 
                  formula = cor ~  log(params) ,
                  method="REML", family = 'gaussian')

params_mod_1 %>% summary()

params_mod_2 <- gam(data = df %>% mutate(model_type = as.factor(model_type)) %>% filter(!interpretability), 
                    formula = cor ~  log(params) ,
                    method="REML", family = 'gaussian')

params_mod_2 %>% summary()

#### Interactions
gam(data = df %>% filter(model != 'lm') %>% mutate(model_type = as.factor(model_type)), 
    formula = cor ~ rank*interpretability + s(model_type, bs = 're'),
    method="REML", family = 'gaussian') %>% summary()

gam(data = df %>% mutate(model_type = as.factor(model_type)), 
    formula = cor ~  rank*interpretability,
    method="REML", family = 'gaussian') %>% summary()


gam(data = df%>% mutate(model_type = as.factor(model_type)), 
    formula = cor ~  log(params)*interpretability ,
    method="REML", family = 'gaussian') %>% summary()
####


# models figure

plotdata <- summary(interpretability_mod_1)$p.table %>% as.data.frame() %>% rownames_to_column() %>% mutate(model = 'Within-model Interpretability scale', type = 'Interpretable') %>%
  bind_rows(
    summary(interpretability_mod_2)$p.table %>% as.data.frame() %>% rownames_to_column() %>% mutate(model = 'Within-model Interpretability scale', type = 'Not interpretable')
  ) %>% 
  bind_rows(
    summary(global_mod_1)$p.table %>% as.data.frame() %>% rownames_to_column() %>% mutate(model = 'Interpretability scale', type = 'Interpretable')
  ) %>% 
  bind_rows(
    summary(global_mod_2)$p.table %>% as.data.frame() %>% rownames_to_column() %>% mutate(model = 'Interpretability scale', type = 'Not interpretable')
  ) %>% 
  bind_rows(
    summary(params_mod_1)$p.table %>% as.data.frame() %>% rownames_to_column() %>% mutate(model = 'Parameters (log)', type = 'Interpretable')
  ) %>%
  bind_rows(
    summary(params_mod_2)$p.table %>% as.data.frame() %>% rownames_to_column() %>% mutate(model = 'Parameters (log)', type = 'Not interpretable')
  ) %>%
  mutate_if(is.numeric, function(x){round(x,4)}) %>%
  View()
  
plotdata %>%
  mutate(lower = Estimate-1.96*`Std. Error`, upper = Estimate+1.96*`Std. Error`) %>%
  filter(rowname != '(Intercept)') %>%
  ggplot(aes(x = factor(model,levels = c('Interpretability scale', 'Within-model Interpretability scale','Parameters (log)')), color = as.factor(type), fill = as.factor(type), y = Estimate,ymin=lower, ymax=upper)) +
  geom_errorbar(aes(color = as.factor(type)), position=position_dodge(width=0.6), width = 0.2) + 
  geom_point(position=position_dodge(width=0.6), shape=21, size=2) +
  geom_hline(yintercept = 0, linetype='dashed', size = 0.1,
             col = 'black') +
  theme_bw(base_size = 12) + #ylim(c(-2.6,2.6)) + 
  xlab('') + ylab('Estimate') + ylim(c(-0.015,0.015)) + 
  theme(legend.title = element_blank(),axis.text.x = element_text(angle = 30, vjust = 1, hjust=1)) + scale_color_manual(values = c("#999999", "#E69F00")) + scale_fill_manual(values = c("#999999", "#E69F00"))
ggsave('Figures/estimates.png', dpi = 900, width = 7, height = 5)

