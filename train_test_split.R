library(readxl)
library(splitstackshape)
library(tidyverse)
for(data in c('Rad_Berlin_2017_2020', 'Rad_München_Master_28_09_2021', 
              'Rad_Wien_2017_2020', 'Rad_Düsseldorf_2017_2020')){
df_raw <- readxl::read_xlsx(paste0('../DATA/',data,'.xlsx'))
df <- df_raw %>%
  mutate(year = substring(date, 1, 4)) 
if(data %in% 'Rad_München_Master_28_09_2021'){
  df <- df %>%
    select(-Arnulf, -Erhardt, -Hirsch, -Kreuther, -Margareten, -Olympia)
}
set.seed(1)
train_df <- stratified(df, 'year', 0.5)
test_df <- df %>% 
  filter(!(date %in% train_df$date))


# test if sampling worked
train_df %>% 
  group_by(year) %>% 
  tally()
test_df %>%
  group_by(year) %>%
  tally()
print(ggplot(data = train_df, aes(x = Bike_total)) + geom_histogram() + xlab(data))
print(ggplot(data = test_df, aes(x = Bike_total)) + geom_histogram() + xlab(data))
write.csv(train_df %>% select(-year), paste0('df/',data, '_train_df.csv'))
write.csv(test_df %>% select(-year), paste0('df/',data, '_test_df.csv'))

}
