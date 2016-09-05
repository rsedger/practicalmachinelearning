# PML - Project

# Parallel Processing stuff
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
# and later do this - need to place this code in the file ...
stopCluster(cluster)

library(caret)
whole_train <- read.csv("pml-training.csv")
str(whole_train) # 19622 obs. of  160 variables
names(whole_train)

inTrain <- createDataPartition(y=whole_train$classe, p=0.7, list=FALSE)
training <- whole_train[inTrain,]
validation <- whole_train[-inTrain,]
testing <- read.csv("pml-testing.csv")

# remove first 7 vars - the time stamp ones - not relevant til we do time series (if
# we do time series)
training <- training[,-c(1,2,3,4,5,6,7)]
validation <- validation[,-c(1,2,3,4,5,6,7)]

set.seed(1234) # set a seed for reproducibility

# remove the NAs
training <-training[,colSums(is.na(training))==0]
validation <- validation[,colSums(is.na(validation))==0]

# find the factor vars coz we want to remove any with >32 categories
col_names <- c()
n <- ncol(training)-1
for (i in 1:n) {
        if (is.factor(training[,i])){
                col_names <- c(col_names,i)
        }
}

col_namesV <- c()
n <- ncol(validation)-1
for (i in 1:n) {
        if (is.factor(validation[,i])){
                col_namesV <- c(col_namesV,i)
        }
}

# remove these coz they have >32 levels
training <- training[,-col_names]
validation <- validation[,-col_namesV]

str(training)
# 53 vars, the only factor var left is classe
# 'data.frame':	13737 obs. of  53 variables:
str(validation)
# 'data.frame':	5885 obs. of  53 variables:
str(testing)
# 'data.frame':	20 obs. of  53 variables:
# last column is called problem_id and is just the numbers 1 to 20

# Just out of curiosity let's see if a Linear Model could possibly tell us anything:
lmFit <- lm(classe ~ ., training)
# Warning messages:
# 1: In model.response(mf, "numeric") :
#         using type = "numeric" with a factor response will be ignored
# 2: In Ops.factor(y, z$residuals) : ‘-’ not meaningful for factors
summary(lmFit)
# Residuals:
# Error in quantile.default(resid) : factors are not allowed
# In addition: Warning message:
#         In Ops.factor(r, 2) : ‘^’ not meaningful for factors

# OK - total non-starter?
lmFit2 <- lm(classe ~ total_accel_dumbbell, training)
summary(lmFit2)
# same result - can't use a response var that's a factor.


# adabag?
# adabag example
library(adabag)
## rpart library should be loaded
library(rpart)

train.adaboost <- boosting(classe~., data=training, boos=TRUE, mfinal=5)
train.adaboost

validation.adaboost.pred <- predict.boosting(train.adaboost,newdata=validation)
validation.adaboost.pred$confusion
validation.adaboost.pred$error
#Observed Class
#Predicted Class    A    B    C    D    E
#               A 1577  170   15   53   14
#               B   36  808   66   27   54
#               C   29  109  924  101   89
#               D   26   26   19  759   65
#               E    6   26    2   24  860

# Obviously not using this correctly

## Make a correlation matrix plot
corMat <- cor(training[,-dim(training)[2]],)
corrplot(corMat, method = "color", type="lower", order="hclust", tl.cex = 0.75, tl.col="black", tl.srt = 45)

#selecting a few of the more promising predictors to be plotted
colSelection<- c("magnet_dumbbell_y","magnet_dumbbell_z", "gyros_dumbbell_y", "accel_dumbbell_y", "total_accel_dumbbell","gyros_dumbbell_z")

#creating a feature plot 
featurePlot(x=training[,colSelection],y = training$classe,plot="pairs")


# some separate plots of correlation
qplot(magnet_dumbbell_y, total_accel_dumbbell, colour=classe, data=training)
qplot(roll_belt, roll_forearm, colour=classe, data=training)

# gbm with caret
fitControl <- trainControl(## 10-fold CV
        method = "repeatedcv",
        number = 10,
        ## repeated ten times
        repeats = 10)

gbmFit1 <- train(classe ~ ., data = training,
                 method = "gbm",
                 trControl = fitControl,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE)
gbmFit1
#Stochastic Gradient Boosting 
#
#13737 samples
#52 predictors
#5 classes: 'A', 'B', 'C', 'D', 'E' 
#
#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 10 times) 
#Summary of sample sizes: 12365, 12363, 12364, 12364, 12362, 12362, ... 
#Resampling results across tuning parameters:
#        
#        interaction.depth  n.trees  Accuracy   Kappa    
#        1                   50      0.7546267  0.6888799
#        1                  100      0.8255082  0.7791819
#        1                  150      0.8570142  0.8190957
#        2                   50      0.8572618  0.8191773
#        2                  100      0.9067849  0.8820336
#        2                  150      0.9313169  0.9130897
#        3                   50      0.8964410  0.8689109
#        3                  100      0.9407659  0.9250455
#        3                  150      0.9603116  0.9497885
#
#Tuning parameter 'shrinkage' was held constant at a value of 0.1
#Tuning
#parameter 'n.minobsinnode' was held constant at a value of 10
#Accuracy was used to select the optimal model using  the largest value.
#The final values used for the model were n.trees = 150, interaction.depth = 3, shrinkage
#= 0.1 and n.minobsinnode = 10. 

gbmFit1$finalModel
# A gradient boosted model with multinomial loss function.
# 150 iterations were performed.
# There were 52 predictors of which 44 had non-zero influence.

summary(gbmFit1)

trellis.par.set(caretTheme())
plot(gbmFit1)
plot(gbmFit1, metric = "Kappa")

predict(gbmFit1, newdata = testing)
#[1] B A B A A E D B A A B C B A E E A B B B
#Levels: A B C D E

# first look with rf 
modelFit <-randomForest(classe ~., data = training, importance = TRUE)
print(modelFit)

#Call:
#        randomForest(formula = classe ~ ., data = training, importance = TRUE) 
#Type of random forest: classification
#Number of trees: 500
#No. of variables tried at each split: 7
#
#OOB estimate of  error rate: 0.57%
#Confusion matrix:
#        A    B    C    D    E  class.error
#   A 3904    1    0    0    1 0.0005120328
#   B   14 2634   10    0    0 0.0090293454
#   C    0   14 2380    2    0 0.0066777963
#   D    0    0   28 2222    2 0.0133214920
#   E    0    0    1    5 2519 0.0023762376

str(training) # only got 53 vars now - 52 predictors and the outcome

qplot(classe, total_accel_dumbbell, data = training)

str(modelFit$importance)
prediction <- predict(modelFit, validation)
validation$rightPred <- prediction == validation$classe
accuracy <- sum(validation$rightPred)/nrow(validation)
accuracy
# [1] 0.9937128

## variable importance plot - have to use caret's rf first - NB takes a while!
set.seed(1234)
rfFit1 <- train(classe~., method = "rf", data=training, trControl = trainControl(method = "cv"), importance=TRUE)
varImpPlot(rfFit1$finalModel, main = "Importance of Predictors in the Fit", 
           pch=19, col="blue",cex=0.75, sort=TRUE, type=1)

rfFit1$finalModel
rfFit1
#Random Forest 
#
#13737 samples
#52 predictors
#5 classes: 'A', 'B', 'C', 'D', 'E' 
#
#No pre-processing
#Resampling: Cross-Validated (10 fold) 
#Summary of sample sizes: 12365, 12363, 12364, 12364, 12362, 12362, ... 
#Resampling results across tuning parameters:
#        
#        mtry  Accuracy   Kappa    
#         2    0.9926470  0.9906983
#        27    0.9927199  0.9907910
#        52    0.9853673  0.9814880
#
#Accuracy was used to select the optimal model using  the largest value.
#The final value used for the model was mtry = 27. 

plot(rfFit1)

predict(rfFit1, newdata = testing)
# [1] B A B A A E D B A A B C B A E E A B B B
# Levels: A B C D E

validation <- validation[, names(validation) %in% names(training)]
predictedData <- predict(rfFit1, validation)
confusionMatrix(validation$classe, predictedData)
#Confusion Matrix and Statistics
#
#          Reference
#Prediction    A    B    C    D    E
#         A 1670    1    2    0    1
#         B    7 1123    9    0    0
#         C    0    3 1020    3    0
#         D    1    0   10  953    0
#         E    0    1    1    2 1078
#
#Overall Statistics
#
#               Accuracy : 0.993          
#                 95% CI : (0.9906, 0.995)
#    No Information Rate : 0.2851         
#    P-Value [Acc > NIR] : < 2.2e-16      
#
#Kappa : 0.9912         
#Mcnemar's Test P-Value : NA             
#
#Statistics by Class:
#
#                     Class: A Class: B Class: C Class: D Class: E
#Sensitivity            0.9952   0.9956   0.9789   0.9948   0.9991
#Specificity            0.9990   0.9966   0.9988   0.9978   0.9992
#Pos Pred Value         0.9976   0.9860   0.9942   0.9886   0.9963
#Neg Pred Value         0.9981   0.9989   0.9955   0.9990   0.9998
#Prevalence             0.2851   0.1917   0.1771   0.1628   0.1833
#Detection Rate         0.2838   0.1908   0.1733   0.1619   0.1832
#Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
#Balanced Accuracy      0.9971   0.9961   0.9888   0.9963   0.9991

# the following cannot be done due to differences in testing data set
testing <- testing[, names(testing) %in% names(training)]
predictedData <- predict(rfFit1, newdata = testing)
confusionMatrix(testing$classe, predictedData)
#Error in confusionMatrix.default(testing$classe, predictedData) : 
#The data must contain some levels that overlap the reference.


validation <- validation[, names(validation) %in% names(training)]
predictedData <- predict(rfFit1, validation)
confusionMatrix(validation$classe, predictedData)
#Confusion Matrix and Statistics
#
#Reference
#Prediction    A    B    C    D    E
#       A   1673    0    1    0    0
#       B     10 1127    2    0    0
#       C      0   12 1006    8    0
#       D      0    0   12  952    0
#       E      0    1    1    0 1080
#
#Overall Statistics
#
#Accuracy : 0.992           
#95% CI : (0.9894, 0.9941)
#No Information Rate : 0.286           
#P-Value [Acc > NIR] : < 2.2e-16       
#
#Kappa : 0.9899          
#Mcnemar's Test P-Value : NA              
#
#Statistics by Class:
#
#                     Class: A Class: B Class: C Class: D Class: E
#Sensitivity            0.9941   0.9886   0.9843   0.9917   1.0000
#Specificity            0.9998   0.9975   0.9959   0.9976   0.9996
#Pos Pred Value         0.9994   0.9895   0.9805   0.9876   0.9982
#Neg Pred Value         0.9976   0.9973   0.9967   0.9984   1.0000
#Prevalence             0.2860   0.1937   0.1737   0.1631   0.1835
#Detection Rate         0.2843   0.1915   0.1709   0.1618   0.1835
#Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
#Balanced Accuracy      0.9969   0.9930   0.9901   0.9946   0.9998

postResample(predictedData, validation$classe)
# Accuracy     Kappa 
#0.9930331 0.9911873 

plot(rfFit1)

# rf with absolutely no data preparation
set.seed(1234)
rfFit2 <- train(classe~., method = "rf", data=whole_train, trControl = trainControl(method = "cv"), importance=TRUE)
varImpPlot(rfFit2$finalModel, main = "Importance of Predictors in the Fit", 
           pch=19, col="blue",cex=0.75, sort=TRUE, type=1)

rfFit2$finalModel
#Call:
#        randomForest(x = x, y = y, mtry = param$mtry, importance = TRUE) 
#Type of random forest: classification
#Number of trees: 500
#No. of variables tried at each split: 6952
#
#OOB estimate of  error rate: 0.74%
#Confusion matrix:
#        A  B  C  D  E class.error
#    A 109  0  0  0  0  0.00000000
#    B   0 78  1  0  0  0.01265823
#    C   0  1 69  0  0  0.01428571
#    D   0  0  0 69  0  0.00000000
#    E   0  0  0  1 78  0.01265823


rfFit2
#Random Forest 
#
#19622 samples
#159 predictors
#5 classes: 'A', 'B', 'C', 'D', 'E' 
#
#No pre-processing
#Resampling: Cross-Validated (10 fold) 
#Summary of sample sizes: 366, 365, 366, 366, 365, 365, ... 
#Resampling results across tuning parameters:
#        
#        mtry  Accuracy   Kappa    
#           2  0.2684756  0.0000000
#         117  0.8967073  0.8694709
#        6952  0.9926220  0.9906985
#
#Accuracy was used to select the optimal model using  the largest value.
#The final value used for the model was mtry = 6952. 

plot(rfFit2)

predict(rfFit2, newdata = testing$problem_id, submodels = NULL)
testing <- read.csv("pml-testing.csv")
names(testing)

# can't do any of the next 2 blocks - error
testing <- testing[, names(testing) %in% names(whole_train)]
predictedData <- predict(rfFit2, newdata = testing)
confusionMatrix(testing$classe, predictedData)

predict(rfFit2, newdata = testing)

# BUT ... if we go back to creating a validation set then this will give us the OOB
# error rate.
prediction <- predict(rfFit1, testing)
missClass = function(values, prediction) {
        sum(prediction != values)/length(values)
}
errRate = missClass(testing$classe, prediction)

# rf without caret - have to get rid of na cols - Nah - don't bother with it
library(randomForest)
modelFit <-randomForest(classe ~., data = whole_train, importance = TRUE)
print(modelFit)

str(modelFit$importance)
prediction <- predict(modelFit, testing)
testing$rightPred <- prediction == testing$problem_id
accuracy <- sum(testing$rightPred)/nrow(testing)
accuracy

# try rpart
rpFit1 <- train(classe ~ ., method = "rpart", data = training)
print(rpFit1$finalModel)

library(rattle)
fancyRpartPlot(rpFit1$finalModel)
rpFit1
#CART 
#
#13737 samples
#52 predictors
#5 classes: 'A', 'B', 'C', 'D', 'E' 
#
#No pre-processing
#Resampling: Bootstrapped (25 reps) 
#Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
#Resampling results across tuning parameters:
#        
#        cp          Accuracy   Kappa     
#        0.02812532  0.5615127  0.43833088
#        0.02922727  0.5557112  0.43084502
#        0.11738379  0.3181264  0.04997556
#
#Accuracy was used to select the optimal model using  the largest value.
#The final value used for the model was cp = 0.02812532. 

predict(rpFit1, newdata = testing)
#  [1] C A C C C C C C A A A C B A C B C A C B
# Levels: A B C D E

# back to randomForest modelFit
plot(modelFit$oob.times, modelFit$y)

# plot below is just an example of the lack of further pattern that could be used to
# reduce the set of predictors
qplot(x = training[, "accel_belt_x"], y = training[, "accel_arm_x"], color = training$classe)

# dumbbell example:
qplot(x = training$accel_dumbbell_x, y = training$accel_dumbbell_y, color = training$classe)

# What about Boosting or Additive Models?
fitControl <- trainControl(method="repeatedcv", number=10, repeats=10)
gbmFit <- train(classe ~ ., data = training, method = "gbm", trControl=fitControl, verbose=FALSE)
summary(gbmFit)
#var     rel.inf
#roll_belt                       roll_belt 22.03115563
#pitch_forearm               pitch_forearm 12.06780094
#yaw_belt                         yaw_belt  9.04491389
#magnet_dumbbell_z       magnet_dumbbell_z  6.46462794
#magnet_dumbbell_y       magnet_dumbbell_y  5.62617367
#roll_forearm                 roll_forearm  4.79873882
#magnet_belt_z               magnet_belt_z  4.06533467
#gyros_belt_z                 gyros_belt_z  3.61332351
#roll_dumbbell               roll_dumbbell  3.25577321
#accel_forearm_x           accel_forearm_x  2.85380732
#pitch_belt                     pitch_belt  2.73404715
#accel_dumbbell_y         accel_dumbbell_y  2.43885264
#magnet_dumbbell_x       magnet_dumbbell_x  2.31600878
#gyros_dumbbell_y         gyros_dumbbell_y  2.28352357
#magnet_forearm_z         magnet_forearm_z  2.02830809
#yaw_arm                           yaw_arm  1.56219712
#magnet_arm_z                 magnet_arm_z  1.42343798
#magnet_belt_y               magnet_belt_y  1.39778139
#accel_dumbbell_x         accel_dumbbell_x  1.39728162
#accel_forearm_z           accel_forearm_z  1.12786945
#accel_dumbbell_z         accel_dumbbell_z  0.90779751
#magnet_forearm_x         magnet_forearm_x  0.86477501
#magnet_arm_x                 magnet_arm_x  0.80134221
#magnet_belt_x               magnet_belt_x  0.60469102
#gyros_belt_y                 gyros_belt_y  0.57665608
#gyros_arm_y                   gyros_arm_y  0.52177942
#roll_arm                         roll_arm  0.40648822
#accel_belt_z                 accel_belt_z  0.38081781
#accel_forearm_y           accel_forearm_y  0.29601886
#magnet_arm_y                 magnet_arm_y  0.29403794
#gyros_dumbbell_x         gyros_dumbbell_x  0.28732420
#total_accel_forearm   total_accel_forearm  0.25538185
#accel_arm_y                   accel_arm_y  0.24103895
#accel_arm_x                   accel_arm_x  0.19941087
#total_accel_dumbbell total_accel_dumbbell  0.18917259
#gyros_forearm_z           gyros_forearm_z  0.15076214
#total_accel_arm           total_accel_arm  0.13527406
#total_accel_belt         total_accel_belt  0.11786636
#gyros_dumbbell_z         gyros_dumbbell_z  0.09561717
#magnet_forearm_y         magnet_forearm_y  0.08594066
#accel_arm_z                   accel_arm_z  0.05684962
#gyros_belt_x                 gyros_belt_x  0.00000000
#accel_belt_x                 accel_belt_x  0.00000000
#accel_belt_y                 accel_belt_y  0.00000000
#pitch_arm                       pitch_arm  0.00000000
#gyros_arm_x                   gyros_arm_x  0.00000000
#gyros_arm_z                   gyros_arm_z  0.00000000
#pitch_dumbbell             pitch_dumbbell  0.00000000
#yaw_dumbbell                 yaw_dumbbell  0.00000000
#yaw_forearm                   yaw_forearm  0.00000000
#gyros_forearm_x           gyros_forearm_x  0.00000000
#gyros_forearm_y           gyros_forearm_y  0.00000000
plot(gbmFit)
gbmFit
#Stochastic Gradient Boosting 
#
#13737 samples
#52 predictors
#5 classes: 'A', 'B', 'C', 'D', 'E' 
#
#No pre-processing
#Resampling: Cross-Validated (10 fold, repeated 10 times) 
#Summary of sample sizes: 12362, 12364, 12365, 12363, 12363, 12363, ... 
#Resampling results across tuning parameters:
#        
#        interaction.depth  n.trees  Accuracy   Kappa    
#       1                   50      0.7522604  0.6858836
#       1                  100      0.8226765  0.7755426
#       1                  150      0.8553619  0.8169582
#       2                   50      0.8555155  0.8169582
#       2                  100      0.9057734  0.8807633
#       2                  150      0.9296575  0.9109913
#       3                   50      0.8948032  0.8668298
#       3                  100      0.9407157  0.9249844
#       3                  150      0.9605012  0.9500301
#
#Tuning parameter 'shrinkage' was held constant at a value of 0.1
#Tuning
#parameter 'n.minobsinnode' was held constant at a value of 10
#Accuracy was used to select the optimal model using  the largest value.
#The final values used for the model were n.trees = 150, interaction.depth = 3, shrinkage =
#        0.1 and n.minobsinnode = 10. 

# RCS: need to try gbm again with shrinkage (epsilon) set to 0.001 (ain't that the default?)
# Also, gbm doesn't need any data preprocessing so, even though we can see that the
# features that wse removed would have no effect on the outcome, maybe we should try it
# with the complete untouched data set.

# Also, we haven't done any prediction with the test set yet. Need to get that done.

