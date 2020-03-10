# Lab 3 part 1
# Author: Himanshu Nirmal

require(ISLR)
require(MASS)
require(class)
?Weekly

#(a) Produce some numerical and graphical summaries of the Weekly data. Do there appear to be any patterns?

data(Weekly)
summary(Weekly)
pairs(Weekly)
cor(Weekly[,-9])
attach(Weekly)
#There is a positive relationship between year and volume

#(b) Use the full data set to perform a logistic regression with Direction as the
#response and the five lag variables plus Volume as predictors. Use the summary
#function to print the results. Do any of the predictors appear to be statistically significant? If so, which ones?

glm1 = glm(Direction ~ .-Year -Today, data=Weekly, family="binomial")
summary(glm1)

# we can summarize that lag 2 has some significant relationship with our response

#(c) Compute the confusion matrix and overall fraction of correct predictions. Explain
#what the confusion matrix is telling you about the types of mistakes made by logistic regression.


glmfit = predict(glm1, type="response")
glmpredict = ifelse(glmfit>.5, "Up", "Down")
confmat = table(Weekly$Direction,glmpredict)
confmat

# When the market is up model performs well 557/(557+48)=92.1% and performs poorly when it is down 54/(54+430)=11.2%

#(d) Now fit the logistic regression model using a training data period from 1990 to
#2008, with Lag2 as the only predictor. Compute the confusion matrix and the
#overall fraction of correct predictions for the held out data (that is, the data from 2009 and 2010).

trains = Weekly[Weekly$Year<2009,]
tests = Weekly[Weekly$Year>2008,]
glm2 = glm(Direction ~ Lag2, data=trains, family="binomial")

testProbs =predict(glm2, type="response", newdata = tests)
testsDir = Weekly$Direction[Weekly$Year>2008]

testpredict = rep("Down", 104)
testpredict[testProbs>0.5] = "Up"
mean(testProbs)
matlog = table(testsDir, testpredict)
matlog

pred= (matlog["Down", "Down"] + matlog["Up", "Up"])/sum(matlog)
testerr = ((1-pred)*100)
testerr

#test error rate for logistic model is 37.5%

#(e) Repeat (d) using LDA.
lda2 = lda(Direction~Lag2, data= trains)

lda2pred = predict(lda2, newdata=tests, type="response")
ldaclass = lda2pred$class
ldamat = table(tests$Direction,ldaclass)
ldamat

# correct predictions
predlda<- (ldamat["Down", "Down"] + ldamat["Up", "Up"])/sum(ldamat)


# Test error rate

testerrlda <- ((1-predlda)*100)
testerrlda

#test error rate for lda model is same as logistic model i.e. 37.5%

#(f) Repeat (d) using QDA.
qda1=qda(Direction~Lag2, data=trains)

qpred = predict(qda1, newdata=tests, type="response")
QDAclass = qpred$class
matqda = table(tests$Direction, QDAclass)
matqda

# correct predictions
qdapred= (matqda["Down", "Down"] + matqda["Up", "Up"])/sum(matqda)
qdapred

# Test error rate

qdaerr = ((1-qdapred)*100)
qdaerr

# test error rate for qda model is 41.3%


#Repeat (d) using KNN with K = 1.

set.seed(1)
training.X = cbind(trains$Lag2)
testing.X = cbind(tests$Lag2)
training.Y = cbind(trains$Direction)
KNNPred = knn(training.X, testing.X, training.Y, k=1)
matknn=table(tests$Direction, KNNPred)
matknn

# correct predictions
knnpred= (matknn["Down", "1"] + matknn["Up", "2"])/sum(matknn)
knnpred

# Test error rate

terKNN = ((1-knnpred)*100)
terKNN



# test error rate for knn model is 50%

#(h) Which of these methods appears to provide the best results on this data?
# Logistic regression and Lda appear to work best followed by qda and then knn for this data

#(i) Experiment with different combinations of predictors, including possible
#transformations and interactions, for each of the methods. Report the variables,
#method, and associated confusion matrix that appears to provide the best results
#on the held out data. Note that you should also experiment with values for K in the KNN classifier.

#logistic model with interaction between lag 2 and lag 1


attach(Weekly)
train = (Year < 2009)
test = Weekly[!train, ]
glm1 = glm(Direction ~ Lag2:Lag1, data = Weekly, family = binomial, subset = train)
glmprob = predict(glm1, test, type = "response")
glmpredict = rep("Down", length(glmprob))
glmpredict[glmprob > 0.5] = "Up"
test1 = Direction[!train]
table(glmpredict, test1)

val=mean(glmpredict == test1)
#val

errorrate=(1-val)*100
errorrate

#error rate for logistic model increases from 37.5% to 41.3% when we include the Lag1 predictor

# lda with all Lag predictors

lda.fit = lda(Direction ~ Lag1+Lag2+Lag3+Lag4+Lag5, data = Weekly, subset = train)
lda.pred = predict(lda.fit, test)
val1=mean(lda.pred$class == test1)
errorrate1=(1-val1)*100
errorrate1

#error rate for lda model increases from 37.5% to 45.19% when we include all predictors



# qda with interaction between Lag1 and Lag2

qda.fit = qda(Direction ~ Lag2:Lag1, data = Weekly, subset = train)
qda.class = predict(qda.fit, test)
val1=mean(qda.class$class == test1)
errorrate1=(1-val1)*100
errorrate1

#error rate for qda model increases from 41.3% to 56.7% when we include the interaction between Lag2 and Lag1


#knn model with k=5

knn5 = knn(training.X, testing.X, training.Y, k=5)
confMatrixKNN5=table(tests$Direction, knn5)

# correct predictions
correctPred5=(confMatrixKNN5["Down", "1"] + confMatrixKNN5["Up", "2"])/sum(confMatrixKNN5)
correctPred5

# Test error rate

terKNN5 =((1-correctPred5)*100)
terKNN5

#as k value is increased from 1 to 5, the error rate decreased

#knn model with k=100

knn100 = knn(training.X, testing.X, training.Y, k=100)
confMatrixKNN=table(tests$Direction, knn100)

# correct predictions
correctPred=(confMatrixKNN["Down", "1"] + confMatrixKNN["Up", "2"])/sum(confMatrixKNN)
correctPred

# Test error rate

terKNN =((1-correctPred)*100)
terKNN


#as k is again increased from 5 to 100, the error rate has decreased farther







