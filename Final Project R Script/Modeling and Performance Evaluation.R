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

#data cleaning
pho$hazardous<-factor(pho$hazardous,levels=c("True","False")) 
# initially hazardous is considered as a character type, so we want to convert it to factor type
pho_split<-pho%>%
  initial_split(pro=0.7,strata = hazardous) # stratified sampling for imbalance data
pho_train<-training(pho_split)
pho_test<-testing(pho_split)
dim(pho_train)
dim(pho_test)

# preparation for modeling
pho_recipe <-recipe(
  hazardous~.,pho_train) %>%
  step_dummy(all_nominal_predictors())%>% # dummy predictor on categorical variables
  step_normalize() # this step is important because some of our variables are extremely small, while others are quite large.

pho_folds<-vfold_cv(pho_train,v=5,strate=hazardous) # we stratify our data for the same reason

# Logistic Regression
log_reg <-logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

log_wkflow<-workflow() %>%
  add_model(log_reg)%>%
  add_recipe(pho_recipe)

log_fit_res<-tune_grid(
  object=log_wkflow,
  resamples=pho_folds # for cross-validation
)

log_fit<-fit(log_wkflow,pho_train)

augment(log_fit,new_data=pho_train) %>%
  conf_mat(truth=hazardous,estimate=.pred_class)%>%
  autoplot(type="heatmap")

log_acc<-augment(log_fit,new_data=pho_train)%>%
  accuracy(truth=hazardous,estimate=.pred_class)
print(log_acc) # Accuracy is 0.9579

log_precision<-augment(log_fit,pho_train)%>%
  precision(hazardous,.pred_class) # Precision is 0.872137
print(log_precision)

# Quadratic Discriminant Analysis
library(MASS)
library(discrim) # for quadratic discriminant analysis
qda_mod<-discrim_quad()%>%
  set_mode("classification")%>%
  set_engine("MASS")

qda_wkflow<-workflow()%>%
  add_model(qda_mod)%>%
  add_recipe(pho_recipe)
qda_fit<-fit(qda_wkflow,pho_train)

augment(qda_fit,new_data=pho_train) %>%
  conf_mat(truth=hazardous,estimate=.pred_class)%>%
  autoplot(type="heatmap")

qda_acc<-augment(qda_fit,new_data=pho_train)%>%
  accuracy(truth=hazardous,estimate=.pred_class)
print(qda_acc) # Accuracy is 0.9646341	

qda_precision<-augment(qda_fit,pho_train)%>%
  precision(hazardous,.pred_class) # Precision is 0.8503401	
print(qda_precision)

# Support Vector Machine
library(kernlab)
svm_rbf_spec <- svm_rbf() %>%
  set_mode("classification") %>%
  set_engine("kernlab", scaled = FALSE)

svm_rbf_fit <- svm_rbf_spec %>% 
  fit(hazardous ~ ., data=pho_train)

mtx<-augment(svm_rbf_fit, new_data = pho_train) %>%
  conf_mat(truth = hazardous, estimate = .pred_class)%>%
  autoplot(type="heatmap")

svm_acc<-augment(svm_rbf_fit,new_data=pho_train)%>%
  accuracy(truth=hazardous,estimate=.pred_class)
print(svm_acc) # Accuracy is 0.8390244	

svm_precision<-augment(svm_rbf_fit,pho_train)%>%
  precision(hazardous,.pred_class) # Precision is NA
print(svm_precision)

# Classification Tree Model
pho_tree_spec <-decision_tree() %>%
  set_engine("rpart")

pho_class_tree_spec <- pho_tree_spec %>%
  set_mode("classification")

pho_class_tree_wkflow <- workflow() %>%
  add_model(pho_class_tree_spec %>% set_args(cost_complexity=tune())) %>% # for hyperparameter-tuning
  add_formula(hazardous~.)

pho_param_grid <-grid_regular(cost_complexity(range=c(-3,-1)),levels=10)

pho_class_tree_res <-tune_grid(
  pho_class_tree_wkflow,
  resamples=pho_folds,
  grid=pho_param_grid,
  metrics=metric_set(roc_auc)) # for cross-validation

autoplot(pho_class_tree_res)

best_class_tree_complexity <-select_best(pho_class_tree_res)
collect_metrics(pho_class_tree_res) %>% arrange(-mean)

class_tree_final<-finalize_workflow(pho_class_tree_wkflow,best_class_tree_complexity)
class_tree_final_fit<-fit(class_tree_final,data=pho_train)

augment(class_tree_final_fit,new_data=pho_train) %>%
  conf_mat(truth=hazardous,estimate=.pred_class)%>%
  autoplot(type="heatmap")

class_tree_acc<-augment(class_tree_final_fit,new_data=pho_train)%>%
  accuracy(truth=hazardous,estimate=.pred_class)
print(class_tree_acc) # Accuracy is 0.9960366	

class_tree_precision<-augment(class_tree_final_fit,pho_train)%>%
  precision(hazardous,.pred_class) # Precision is 0.9923518
print(class_tree_precision)

class_tree_final_fit %>%
  extract_fit_engine()%>%
  rpart.plot()# for tree visualization

# Random Forest Model
pho_rf_spec<-rand_forest() %>%
  set_engine("ranger",importance="impurity")%>%
  set_mode("classification")

pho_rf_Wkflow<-workflow()%>%
  add_model(pho_rf_spec %>% set_args(mtry=tune(),trees=tune(),min_n=tune()))%>%
  add_formula(hazardous~.)

param_grid_rf<-grid_regular(mtry(range= c(1,8)),trees(range = c(10,200)),min_n(range = c(2,20)),levels = 8)

pho_res_rf <-tune_grid(
  pho_rf_Wkflow,
  resample=pho_folds,
  grid=param_grid_rf,
  metrics=metric_set(roc_auc)
)

autoplot(pho_res_rf)

library(vip)
collect_metrics(pho_res_rf) %>% arrange(-mean)
best_complexity_rf<-select_best(pho_res_rf)

rf_final<-finalize_workflow(pho_rf_Wkflow,best_complexity_rf)
rf_final_fit<-fit(rf_final,data=pho_train)

rf_final_fit %>%
  extract_fit_engine()%>%
  vip() # for variable importance plot

augment(rf_final_fit,new_data=pho_train) %>%
  conf_mat(truth=hazardous,estimate=.pred_class)%>%
  autoplot(type="heatmap")

rf_acc<-augment(rf_final_fit,new_data=pho_train)%>%
  accuracy(truth=hazardous,estimate=.pred_class)
print(rf_acc) # Accuracy is 0.9993902	

rf_precision<-augment(rf_final_fit,pho_train)%>%
  precision(hazardous,.pred_class) # Precision is 0.9924386	
print(rf_precision)

# Boosted Tree Model
pho_boost_spec<-boost_tree()%>%
  set_engine("xgboost")%>%
  set_mode("classification")

pho_boost_Wkflow<-workflow()%>%
  add_model(pho_boost_spec %>% set_args(trees=tune()))%>%
  add_formula(hazardous~.)

param_grid_boost<-grid_regular(trees(range = c(10,2000)),levels = 10)

tune_res_boost <-tune_grid(
  pho_boost_Wkflow,
  resample=pho_folds,
  grid=param_grid_boost,
  metrics=metric_set(roc_auc)
)

autoplot(tune_res_boost)

collect_metrics(tune_res_boost)%>%arrange(-mean)
best_tree_boost<-select_best(tune_res_boost)

boost_final<-finalize_workflow(pho_boost_Wkflow,best_tree_boost)
boost_final_fit<-fit(boost_final,data=pho_train)

augment(boost_final_fit,new_data=pho_train) %>%
  conf_mat(truth=hazardous,estimate=.pred_class)%>%
  autoplot(type="heatmap")

boost_acc<-augment(boost_final_fit,new_data=pho_train)%>%
  accuracy(truth=hazardous,estimate=.pred_class)
print(boost_acc) # Accuracy is 1

boost_precision<-augment(boost_final_fit,pho_train)%>%
  precision(hazardous,.pred_class) # Precision is 1
print(boost_precision)

# Overall Performance of Each Model
# Accuracy
library(dbplyr)
acc_tab<-matrix(c(0.8390244,0.9579268	,0.9646341, 0.9960366	,0.9978659,1),ncol=1,byrow=TRUE)
colnames(acc_tab)<-"accuracy"
rownames(acc_tab)<-c("support vector machine",
                     "quadratic discriminant analysis","logistic regression",
                     "classification tree","random forest","boosted tree")
print(acc_tab)

# Precision
precision_tab<-matrix(c(svm_precision$.estimate,qda_precision$.estimate,
                        log_precision$.estimate,class_tree_precision$.estimate,
                        rf_precision$.estimate,boost_precision$.estimate),
                      ncol=1,byrow=TRUE)
colnames(precision_tab)<-"precision"
rownames(precision_tab)<-c("support vector machine","quadratic discriminant analysis",
                           "logistic regression","classification tree",
                           "random forest","boosted tree")
print(precision_tab)

# Analysis of The Test Set Using Boosted Tree 
augment(boost_final_fit,new_data=pho_test) %>%
  conf_mat(truth=hazardous,estimate=.pred_class)

boost_acc<-augment(boost_final_fit,new_data=pho_test)%>%
  accuracy(truth=hazardous,estimate=.pred_class)
print(boost_acc) # Accuracy is 0.9978678	

boost_precision<-augment(boost_final_fit,pho_test)%>%
  precision(hazardous,.pred_class) # Precision is 0.9912281	
print(boost_precision)