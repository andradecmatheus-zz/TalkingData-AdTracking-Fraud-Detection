##### Stage 0: Collecting the data
## Setting the working directory
setwd("~/Development/DataScienceAcademy/FCD/BigDataRAzure/ProjetoFinal/TalkingData-AdTracking-Fraud-Detection/EDA")
#getwd()

## Libraries
library(data.table)
#library(Amelia)
library(dplyr)
library(ggplot2)
library(treemap)
library(gridExtra)
library(corrplot)


## Loading dataset
df_original <- fread("../Datasets/train_sample.csv", header=T)
# df <- fread(file.choose(), header=T)
df <- df_original

##### Stage 1: knowing the data as they are
str(df) #glipmse(df)
dim(df)
summary(df)

## Duplicated rows analysis
any(duplicated(df)) 
# what are these rows?
df[duplicated(df), ] #df %>% !distinct() # There is one
# removing duplicated rows
df <- df[!duplicated(df), ]
any(duplicated(df))


## quantity of null values in each columns
any(is.na(df))
# missmap(df, main = "Missing Values Map", col = c("red", "black"), legend = FALSE)
sapply(df, function(x) sum(is.na(x)))
# just 'attributed_time' has null values: 99773
sum(is.na(df[,"attributed_time"]))/length(df$attributed_time)*100
# as for the users who didn't download the app, the time has not been recorded 
# and the column wasn't filled with any value

## quantity of unique values in each columns
sapply(df, function(x) length(unique(x)))

# the label column is a factor with two levels
df$is_attributed <- as.factor(df$is_attributed)

# "ip", "app", "device", "os", "channel" are also factor type
df[,1:5] <- lapply(df[,1:5], factor)#######
#str(df)

## High class imbalance problem 
summary(df$is_attributed) #table(df$is_attributed)
prop.table(table(df$is_attributed))*100
# it means that the models will be overfitted about no-downloads

## when "attributed_time" is NULL "is_attributed" is 0.
# there isn't attributed_time when there wasn't made download

## Generating weekday and hour from click_time intending to explore days and hours
# 'weekdays' to names, 'wday' to numbers (it starts with 0 = Sunday)
df$weekday <- weekdays(df$click_time)
df$hour <- hour(df$click_time)
#glimpse(df)
# quantity of unique values in the new columns
sapply(df[, 9:10], function(x) length(unique(x)))

## changing the columns order
# names(df)
df <- df[, c(6,9,10,1,2,3,4,5,7,8)]

# changing for factor the new columns
df$weekday <- as.factor(df$weekday)
df$hour <- as.factor(df$hour)

# after data munging, just confirming whether the data integrity is as before
sapply(df, function(x) sum(is.na(x)))
sapply(df, function(x) length(unique(x)))


## Creating subsets from is_attributed classes
df_IsAttributed0 <- df %>%
  filter(is_attributed == '0')

df_IsAttributed1 <- df %>%
  filter(is_attributed == '1') %>%
  mutate(wday_IsAttributed1 = weekdays(attributed_time), 
         hour_IsAttributed1 = hour(attributed_time)) 

# as attributed_time represents the event time we want predict, it's deleted
df$attributed_time = NULL

summary(df)



##### Stage 2: Exploratory Data Analysis
# For categorical variables (or grouping variables). 
# the count of categories is visualized using a bar plot, pie chart or dot charts to show the proportion of each category.

# creating a list with the same dataset rows length 
df_subsets <- list(1:nrow(df))
#df_subsets[[1]][1]

# creting a aux dataset with only columns in df_subsets
df_aux <- df[, 2:8] # without click_time, which is represented by hour and weekday columns


### 2.1 Creating dataset for each variable counting its frequency    
for(i in 1:7){
  #print(names(df_dataSets[i]) )
  df_subsets[[i]] <- df_aux %>%
    group_by_at(i) %>%
    summarise(counts = n())
}
View(df_subsets)

### 2.2 Creating bar plots for each variable counting its frequency from its datasets
# automatizing visualization for bar plots 
for (i in 1:length(df_subsets)){
  
  if(i==3) 
    next # as the IP bar plot is heavy for load, this bar is omitted 
  
  x_column <- unlist(df_subsets[[i]][,1])
  title_x <- names(df_subsets[[i]][,1])
  title_y <- names(df_subsets[[i]][,2])
  
  
  barplot <- ggplot(df_subsets[[i]],aes(x = x_column , y = counts)) +
    geom_bar(fill = "#00A4DEF7", stat = "identity") + 
    #geom_text(aes(label = counts), vjust = -0.3, size = 3) +
    ggtitle(paste("Bar Plot", i, title_x,"x", title_y)) +
    theme(plot.title = element_text(hjust = 0.5),
          #axis.title.x=element_blank(),
          axis.title.y=element_blank()) +
    xlab(title_x)
  
  print(barplot +
          if (i == 3)
            theme(axis.text.x = element_text(angle=-90, size=0))
        else if (i == 4)
          theme(axis.text.x = element_text(angle=-90, size=3))
        else if (i == 5)
          theme(axis.text.x = element_text(angle=-90, size=5))
        else if (i == 6)
          theme(axis.text.x = element_text(angle=-90, size=4))
        else if (i == 7)
          theme(axis.text.x = element_text(angle=-90, size=2.5)))
}

ip = as.data.frame(df_subsets[[3]])
ip <- ip %>% arrange(desc(counts)) %>%
  mutate(percentOcur = round(prop.table(counts),4)*100) 
treemap(ip[1:15,],
  index="ip", vSize="counts", vColor="percentOcur",
  palette=c("White","White", "#499894"), # "YlOrRd"
  type="value", title.legend="percentOcur", title="The 15 Most Frequent IPs")

#### Insights from bar plots

## Day:
# wednesday > tuesday > thursday > monday (a lot less)

## Hour
# between 17 and 22 hour is made less download
# great peak: 0h to 15h;
# peak 1 is 0h to 7h; peak 2 is 9h to 15h
# curiously 8 has the minor value compared with its proximities;
# @ that's interesting to relate day and hour together

## IP
# there are 5 great used IP. They are 5348, 5314, 73487, 73516, 5345
# there are also more +-20 less used than this 5 and more than others

## App
# in general, the firsts are more downloaded
# but there is some whose highlight a tiny
# and one who highlight a lot more

## Device  
# there is one who highlights in a huge way

## OS  
# there are 2 who highlite a lot

## Channel  
# there is one that stands out a lot
# and others who highlights less


### 2.3 is_attributed frequency: comparison between download and no-download
# is_attributed0 weekday
df_IsAttributed0Day <- df_IsAttributed0 %>%
  group_by(weekday) %>%
  summarise(counts = n())

p1_IsA_Day <- ggplot(df_IsAttributed0Day, aes(x = weekday, y = counts)) +
  geom_bar(fill = "#00A4DEF7", stat = "identity") +
  geom_text(aes(label = counts), vjust = 1) +
  ggtitle(" Hour when df_IsAttributed is 0") +
  theme(plot.title = element_text(hjust = 0.5),
        axis.title.x=element_blank(),
        axis.title.y=element_blank()) 

# is_attributed1 weekday
df_IsAttributed1Day <- df_IsAttributed1 %>%
  group_by(weekday) %>%
  summarise(counts = n())

p2_IsA_Day <- ggplot(df_IsAttributed1Day, aes(x = weekday, y = counts)) +
  geom_bar(fill = "#00A4DEF7", stat = "identity") +
  geom_text(aes(label = counts), vjust = 1) +
  ggtitle(" Hour when df_IsAttributed is 1") +
  theme(plot.title = element_text(hjust = 0.5),
        axis.title.x=element_blank(),
        axis.title.y=element_blank()) 

grid.arrange(p1_IsA_Day, p2_IsA_Day, ncol = 1)
# Apparently there isn't no significant difference

# is_attributed0 hour
df_IsAttributed0Hour <- df_IsAttributed0 %>%
  group_by(hour) %>%
  summarise(counts = n())

p1_IsA_Hour <- ggplot(df_IsAttributed0Hour, aes(x = hour, y = counts)) +
  geom_bar(fill = "#00A4DEF7", stat = "identity") +
  geom_text(aes(label = counts), vjust = 1, size = 3) +
  ggtitle(" Hour when df_IsAttributed is 0") +
  theme(plot.title = element_text(hjust = 0.5),
        axis.title.x=element_blank(),
        axis.title.y=element_blank(),
        axis.text.x = element_text(size=5)) +
  geom_line(aes(x = hour, y = mean(counts)), size = 0.2, color="#8B4513", group = 1)

# is_attributed1 hour
df_IsAttributed1Hour <- df_IsAttributed1 %>%
  group_by(hour) %>%
  summarise(counts = n())

p2_IsA_Hour <- ggplot(df_IsAttributed1Hour, aes(x = hour, y = counts)) +
  geom_bar(fill = "#00A4DEF7", stat = "identity") +
  geom_text(aes(label = counts), vjust = 1, size = 3) +
  ggtitle(" Hour when df_IsAttributed is 1") +
  theme(plot.title = element_text(hjust = 0.5),
        axis.title.x=element_blank(),
        axis.title.y=element_blank(),
        axis.text.x = element_text(size=5)) +
  geom_line(aes(x = hour, y = mean(counts)), size = 0.2, color="#8B4513", group = 1)

grid.arrange(p1_IsA_Hour, p2_IsA_Hour, ncol = 1)
# There is a significative difference
# It's interesting to use statistics to know more about this difference
# but basically, the no-downloads happened more between 0 to 15h

# in this sample, there isn't download at the 16h, but there is a lot no-download.
grid.arrange(p1_IsA_Day, p2_IsA_Day, p1_IsA_Hour, p2_IsA_Hour, ncol = 2)


### 2.4 investigating more the relations between day and time variables
# is_attributed0 hour x day
df_IsAttributed0HourDay <- df_IsAttributed0 %>%
  group_by(weekday,hour) %>%
  summarise(counts = n())

a <- ggplot(df_IsAttributed0HourDay, aes(x = hour, y = counts)) +
  geom_bar(fill = "#00A4DEF7", stat = "identity") + 
  facet_grid(weekday ~ .) +
  geom_text(aes(label = counts), vjust = 1, size = 2.5) +
  ggtitle("is_attributed0 - Hour x weekday") +
  theme(#axis.text.x = element_text(size=5),
        plot.title = element_text(hjust = 0.5))

# is_attributed1 hour x day
df_IsAttributed1HourDay <- df_IsAttributed1 %>%
  group_by(weekday,hour) %>%
  summarise(counts = n())

b <- ggplot(df_IsAttributed1HourDay, aes(x = hour, y = counts)) +
  geom_bar(fill = "#00A4DEF7", stat = "identity") + 
  facet_grid(weekday ~ .) +
  geom_text(aes(label = counts), vjust = 1, size = 3.5) +
  ggtitle("is_attributed1 - Hour x weekday") +
  theme(#axis.text.x = element_text(size=5),
        plot.title = element_text(hjust = 0.5))

# there wasn't downloads at Thursday in 17 to 03h
grid.arrange(a, b, ncol = 2)



##### Stage 3: Correlation
names(df)

# Generating new columns from 'IP' relating it with another columns
data <- df %>%  
  add_count(ip, weekday, hour) %>% rename("ipDayHour"     = n) %>%
  add_count(ip, hour, channel) %>% rename("ipHourChannel" = n) %>%
  add_count(ip, hour, os)      %>% rename("ipHourOs"      = n) %>%
  add_count(ip, hour, app)     %>% rename("ipHourApp"     = n) %>%
  add_count(ip, hour, device)  %>% rename("ipHourDevice"  = n)%>%
  select(-click_time) 
# 'click_time' isn't necessary as we already have created weekday and hour columns

## Creating a dataset for doesn't alter the original
data_num <- data
# to observe how much each variable correlates depending if it was made download or no
data_0 <- data %>%
  filter(is_attributed == '0') %>%
  select(-is_attributed)
data_1 <- data %>%
  filter(is_attributed == '1') %>%
  select(-is_attributed)

# converting to numeric to correlate variables
data_num[,1:ncol(data_num)] = lapply(data_num[,1:ncol(data_num)], as.numeric)
data_0[,1:ncol(data_0)] <- lapply(data_0[,1:ncol(data_0)], as.numeric)
data_1[,1:ncol(data_1)] <- lapply(data_1[,1:ncol(data_1)], as.numeric)

## correlations
corrplot(cor(data_num, method="pearson"), method = 'number',
         number.cex= 7/ncol(data_num), type="lower")
mtext("data_num", at=12, line=2, cex=1)
corrplot(cor(data_0, method="pearson"), method = 'number', 
         number.cex= 7/ncol(data_0), type="lower")
mtext("data_0", at=12, line=2, cex=1)
corrplot(cor(data_1, method="pearson"), method = 'number', 
         number.cex= 7/ncol(data_1), type="lower")
mtext("data_1", at=12, line=2, cex=1)
# data_num: only 'ip' and 'app' are very weakly and positively correlated with 
#   the target variable; 'device' and 'os' correlate positively with 'app' and with each other.
 
# data_0: particularly, for no-downloads 'OS' is high positive correlated with 
#   'device', and weakly is 'app' with them.
 
# data_1: there are some weakly and very weakly correlations; 'ip', 'app', and 
#   'os' aren't correlated as 'data_num' and 'data_0'.