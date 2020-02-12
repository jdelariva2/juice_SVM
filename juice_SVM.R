#Author: Jorge de la Riva


#Read in data
pacman::p_load(e1071, ggplot2, caret, rmarkdown, corrplot)
options(digits = 3)
set.seed(123)
juice <- read.csv('juice.csv')


#Problem 1: Create a training set with 80/20 split
juice.index <- createDataPartition(juice$Purchase, p =.8, list=FALSE)
juice.train <- juice[juice.index,]
juice.test <- juice[-juice.index,]

#Problem 2:Create svm with summary
svm1 <- svm(Purchase ~.,data=juice.train, method = "svmLinear",kernel = 'linear',
           trControl=trainControl(method = "repeatedcv", 
                                  number = 10, repeats = 3),  
           preProcess = c("center", "scale"),
           cost=.01,
           tuneLength = 10)
summary(svm1)


#Question 3: error rates for regular SVM
pred1 <- predict(svm1, juice.test)
confusionMatrix(table(pred1, juice.test$Purchase))
#model 1 description: linear, cost=.01, 445 support vectors, 84% accuracy with test data.



#Question 4: Optimize cost hyperparamter. 
grid <- expand.grid(C=seq(0.01,10,0.1))
svm_grid <- svm(Purchase ~.,data=juice.train, method = "svmLinear",kernel = 'linear',
           trControl=trainControl(method = "repeatedcv", 
                                  number = 10, repeats = 3),  
           preProcess = c("center", "scale"),
           tuneGrid = grid,
           tuneLength = 10)
summary(svm_grid)
#Question 5: error rates for grid svm
pred2 <- predict(svm_grid, juice.test)
confusionMatrix(table(pred2, juice.test$Purchase))
svm_grid$cost
#model 2 (grid) description: best cost was 1, 345 support vectors, 84.5% accuracy with test data.



#6 change svm to radial kernel, use the default gamma
svm_rad <- svm(Purchase ~.,data=juice.train,kernel = 'radial',cost=.01
)
summary(svm_rad)
pred3 <- predict(svm_rad, juice.test)
confusionMatrix(table(pred3, juice.test$Purchase))
#Radial SVM description: 626 support vectos, radial kernel, 61% accuracy


svm_radtune <- svm(Purchase ~.,data=juice.train, method = "svmLinear",kernel = 'radial',
                trControl=trainControl(method = "repeatedcv", 
                                       number = 10, repeats = 3),  
                preProcess = c("center", "scale"),
                tuneGrid = grid,
                tuneLength = 10)
predradtune <- predict(svm_radtune, juice.test)
confusionMatrix(table(predradtune, juice.test$Purchase))
#Radial SVM tuned description:radial kernel, 385 support vectors, 85% accuracy with test data.

svmpoly <- svm(Purchase ~.,data=juice.train,kernel = 'polynomial')
predpoly <- predict(svmpoly, juice.test)
confusionMatrix(table(predpoly, juice.test$Purchase))
#7 svm with polynomial degree 2 kernel
svm_poly <- svm(Purchase ~.,data=juice.train, method = "svmLinear",kernel = 'polynomial',
               trControl=trainControl(method = "repeatedcv", 
                                      number = 10, repeats = 3),  
               preProcess = c("center", "scale"),
               tuneGrid = grid,
               tuneLength = 10)
summary(svm_poly)
pred4 <- predict(svm_poly, juice.test)
confusionMatrix(table(pred4, juice.test$Purchase))

#Polynomial SVM description: polynomial kernel, 422 support vectors, 83.5% accuracy with test data

#8 Which approach gives the best results
summary(svm1)
summary(svm_grid)
summary(svm_rad)
summary(svm_radtune)
summary(svm_poly)
confusionMatrix(table(pred1, juice.test$Purchase))
confusionMatrix(table(pred2, juice.test$Purchase))
confusionMatrix(table(pred3, juice.test$Purchase))
confusionMatrix(table(predradtune, juice.test$Purchase))
confusionMatrix(table(pred4, juice.test$Purchase))

#All of the tuned models fall within 84% +- 1.5 percentage points. The best model was the tuned radial model at 85%. 2 of the models were very close to the original with hypertuning. It was only the radial model that benefitted the most from the tuning (an increase of 20 %points).

