library(caret)
library(pROC)



files <- read.csv("C:/Users/krishna/Desktop/ML/project/train.csv",header=TRUE, stringsAsFactors=FALSE, sep=",")
dframe <- data.frame(files)
dframe <-subset(dframe, select=-c(1,4))
colnames(dframe) <- c("f1", "f2","f3","label")
dframe$label <- factor(ifelse(dframe$label==0, "Zero", "One"))

file_test <- read.csv("C:/Users/krishna/Desktop/ML/project/test.csv",header=TRUE, stringsAsFactors=FALSE, sep=",")
test_df1 <- data.frame(file_test)
test_df <-subset(test_df1, select=-c(1,4))
colnames(test_df) <- c("f1", "f2","f3")



estimaterfunction <- function(data,lev = NULL, model = NULL){
  cf <- confusionMatrix(data$pred,data$obs)
  
  stats <- c(cf$overall[1],cf$byClass[c(1,5,6,7)])
  
  
  return(stats)
}
# define training control- i.e to train the model

train_control<- trainControl(method="cv", number=10, classProbs = TRUE ,summaryFunction =estimaterfunction)

######### RANDOM FOREST ##########
# Giving a range of parameters to build our model
rftune <- expand.grid(mtry = c(2,3,4,5,6,7))
# Building the model (both training and buiding the model happens with the same in built function)
rfmodel<- train(label~ ., data=dframe, trControl=train_control, method="cforest",tuneGrid=rftune,metric="Accuracy")
print(rfmodel) 
plot(rfmodel)

########## BOOSTING ##########
# Giving a range of parameters to build our model
boosttune <- expand.grid(nIter = c(50,100,150) ,method=c("Adaboost.M1","Real adaboost"))
# Building the model (both training and buiding the model happens with the same in built function)
 boostmodel<- train(label~ ., data=dframe, trControl=train_control, method="adaboost",tuneGrid=boosttune,  metric="Accuracy")
print(boostmodel)
plot(boostmodel)

#######ANN######
# Giving a range of parameters to build our model
anntune <- expand.grid(nhid = c(2,3,4,5,6,7,8) ,actfun=c("sin","radbas","tansig","purelin"))
# Building the model (both training and buiding the model happens with the same in built function)
annmodel<- train(label~ ., data=dframe, trControl=train_control, method="elm",tuneGrid=anntune , metric="Accuracy")
print(annmodel)
plot(annmodel)

#testing the code for submissions
predictions <- predict(object=rfmodel, test_df, type='prob')
head(predictions$One)
submit<-data.frame(test_df1$X,predictions$One)
colnames(submit)[2] <- "Made Donation in March 2007"
