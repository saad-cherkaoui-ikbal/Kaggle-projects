
library("readxl")

train_data <- read_excel('train_data.xlsx')
CV_data <- read_excel('CV_data.xlsx')
test_data <- read_excel('test_data.xlsx')

## A standard logistic model was also estimated for the sake of comparison

#standard logit
mylogit <- glm(Survived ~ Sex + Age + SibSp + Parch + Pclass_2 + Pclass_3 + Embarked_1 + Embarked_2
               , data=train_data, family = "binomial")
summary(mylogit)

#weighted logit
mylogit1 <- glm(Survived ~ Sex + Age + SibSp + Parch + Pclass_2 + Pclass_3 + Embarked_1 + Embarked_2
                , data=train_data, weights = train_data$Weight, family = "binomial")
summary(mylogit1)


#predicting CV results
CV_pred <- predict(mylogit, type='response', newdata=CV_data[c(-1,-11,-12,-13)])
CV_pred1 <- predict(mylogit1, type='response', newdata=CV_data[c(-1,-11,-12,-13)])

survive_pred <- ifelse(CV_pred >0.5, 1, 0)
survive_pred1 <- ifelse(CV_pred1 >0.5, 1, 0)

survive_pred
survive_pred1

cm <- table(survive_pred, CV_data[[1]])
cm1 <- table(survive_pred1, CV_data[[1]])
cm
cm1

#predicting test results
test_pred <- predict(mylogit, type='response', newdata=test_data[c(-10,-11,-12)])
test_pred1 <- predict(mylogit1, type='response', newdata=test_data[c(-10,-11,-12)])

survive_pred <- ifelse(test_pred >0.5, 1, 0)
survive_pred1 <- ifelse(test_pred1 >0.5, 1, 0)

cm <- table(survive_pred)
cm1 <- table(survive_pred1)
cm
cm1

# After comparing the results of both logistic models, it turns out that the weights don't affect the results as we would've hoped for
