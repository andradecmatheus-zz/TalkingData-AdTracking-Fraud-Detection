##### Data munging
# Now that the data is knew let's automatize its processing and transformations
# aux = 0 means for modelling
# aux = 1 means for submitting

featureEngineering <- function(data=df[], aux = aux){
  
  # the same manipulate made in EDA stage
  #data <- data[!duplicated(data), ] 
  data$weekday <- as.factor(weekdays(data$click_time))
  data$hour <- as.factor(hour(data$click_time))
  
  if(aux == 0) {
    #data <- data[!duplicated(data), ] 
    data$is_attributed <- as.factor(data$is_attributed)
    data <- data[, c(6,9,10,1,2,3,4,5,7,8)]
    # preparing data for to use in ML algorithms further
    data[,1:9] <- lapply(data[,1:9], as.numeric)
  }
  else if(aux == 1) {
    data <- data[, c(6,8,9,1,2,3,4,5,7)]
    # data[,1:11] <- lapply(data[,1:11], as.numeric)
  }
  #sapply(data_factor, function(x) length(unique(x)))
  
  # Generating new columns from 'IP' relating it with another columns
  data <- data %>%  
    add_count(ip, weekday, hour) %>% rename("ipDayHour"     = n) %>%
    add_count(ip, hour, channel) %>% rename("ipHourChannel" = n) %>%
    add_count(ip, hour, os)      %>% rename("ipHourOs"      = n) %>%
    add_count(ip, hour, app)     %>% rename("ipHourApp"     = n) %>%
    add_count(ip, hour, device)  %>% rename("ipHourDevice"  = n) %>%
    select(-attributed_time, -click_time, -ip) 
  # 'IP' can be dynamic or fake, so it was deleted as well
  # 'attributed_time' and 'click_time' were deleted as in EDA stage;
  
  # preparing data for to use in ML algorithms further
  if(aux == 0) {
    data[,8:12] <- lapply(data[,8:12], as.numeric)
  }
  else if(aux == 1) {
    data[,1:11] <- lapply(data[,1:11], as.numeric)
  }
  
  #str(data) #glimpse(data)
  return(data)
}