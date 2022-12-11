library(tidyverse) # loading packages
library(tidymodels)
library(janitor)
library(ggplot2)
library(rpart.plot)
library(randomForest)
library(vip)
library(ranger)
library(xgboost)
library(corrplot)
library(corrr)
library(ggcorrplot)
library(dplyr)
library(klaR)
tidymodels_prefer()

pho<-read.csv("C:/Users/lisha/Downloads/nasa.csv") # read the data

set.seed(403) # will be useful later in modelling 

dim(pho) # 4687 40
pho %>%
  head(1)

# feature extraction
pho<-pho %>%
  clean_names() # clean the names of variables
pho<-pho%>%
  select(-est_dia_in_feet_max,-est_dia_in_feet_min,
         -est_dia_in_miles_max,-est_dia_in_miles_min,-est_dia_in_m_max,-est_dia_in_m_min,
         -close_approach_date,-relative_velocity_km_per_sec,-miss_dist_lunar,-miss_dist_miles,
         -miss_dist_kilometers,-perihelion_arg,-name,-neo_reference_id,-orbit_determination_date,
         -orbit_id,-orbiting_body,-equinox) # exclude unimportant variables or variables measure the same parameter in different units

pho_cor<-pho%>%
  select(-hazardous)%>%
  cor()%>%
  corrplot(tl.cex = 0.7) # check covariance plot to see if we need to further delete variables

pho_cor_new<-pho%>%
  select(-est_dia_in_km_min,-miles_per_hour,
         -jupiter_tisserand_invariant,-orbital_period,
         -aphelion_dist,-perihelion_time,-mean_motion,-hazardous)%>% # exlude variables have strong correlation
  cor() %>%
  corrplot(tl.cex = 0.7) # check covariance plot again

pho<-pho%>%
  select(-est_dia_in_km_min,-miles_per_hour,
         -jupiter_tisserand_invariant,
         -orbital_period,-aphelion_dist,
         -perihelion_time,-mean_motion) # finalize our dataset

# data visualization
ggplot(pho,aes(hazardous))+
  geom_bar(color="red")+
  labs(x="Potentially hazardous asteroid")+
  labs(title="Distribution of output variable") # distribution of output variable

hist(pho$est_dia_in_km_max,xlim = c(0,10),
     xlab="Distribution of estimated maximum diameter in kilometer",main="",
     col = "#ffeff1") # distribution of estimated maximum diameter
ggplot(pho,aes(x=est_dia_in_km_max,y=hazardous))+
  geom_boxplot(color="blue",fill="lightgreen")+labs(
  title = "Boxplot of estimated diameter by being hazardous or not")+ theme_minimal() # relationship between this variable and response

ggplot(pho,aes(x=minimum_orbit_intersection,group=factor(hazardous),fill=factor(hazardous)))+
  geom_histogram(color="aquamarine")+labs(title="Histogram of minimum orbit intersection")+
  scale_fill_discrete(labels=c('TRUE', 'FALSE')) + guides(fill=guide_legend(title="Hazardous")) 
+ theme_minimal() # distribution of minimum orbit intersection
ggplot(pho,aes(x=minimum_orbit_intersection,y=hazardous))+
  geom_boxplot(color="firebrick",fill="gold1")+
  labs(title = "Boxplot of minimum orbit intersection 
       by being hazardous or not")+ theme_minimal() # relationship between this variable and response
  
ggplot(pho,aes(x=inclination,group=factor(hazardous),fill=factor(hazardous)))+
  geom_histogram(color="yellow")+labs(title="Histogram of inclination")+
  scale_fill_discrete(labels=c('TRUE', 'FALSE')) + 
  guides(fill=guide_legend(title="Hazardous")) + theme_minimal()  # distribution of inclination
ggplot(pho,aes(x=inclination,y=hazardous))+
  geom_boxplot(color="dark blue",fill="deeppink1")+
  labs(title = "Boxplot of inclination by being hazardous or not")+ theme_minimal() # relationship between this variable and response

ggplot(pho,aes(x=relative_velocity_km_per_hr,group=factor(hazardous),fill=factor(hazardous)))+
  geom_histogram(color="white")+
  labs(title="Histogram of relative velocity in km per hour")+
  theme_minimal()  # distribution of relative velocity in km per hour
ggplot(pho,aes(x=relative_velocity_km_per_hr,y=hazardous))+
  geom_boxplot(color=" brown",fill="lightcyan")+
  labs(title = "Boxplot of relative velocity by being hazardous or not")+theme_minimal() # relationship between this variable and response

