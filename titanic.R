library(e1071); library(xgboost); library(caret); library(doSNOW); library(ipred)

train <- read.csv('data/train.csv')
head(train)

table(train$Embarked)
train$Embarked[train$Embarked == ""]  <- 'S' # Replacing missing values with mode
table(train$Embarked)

summary(train$Age)
train$MissingAge <- ifelse(is.na(train$Age), 'Y', 'N')
train$FamilySize <- 1 + train$SibSp + train$Parch

train$Survived <- as.factor(train$Survived)
train$Pclass <- as.factor(train$Pclass)
train$Sex <- as.factor(train$Sex)
train$Embarked <- as.factor(train$Embarked)

features <- c("Survived", "Pclass", "Sex", "Age", 
              "SibSp", "Parch", "Fare", "Embarked", 
              "MissingAge", "FamilySize")
train <- train[, features]
str(train)

dummy.vars <- dummyVars(~., data = train[, -1])
train.dummy <- predict(dummy.vars, train[, -1])
head(train.dummy)

pre.process <- preProcess(train.dummy, method = 'bagImpute')
imputed.data <- predict(pre.process, train.dummy)
head(imputed.data)

train$Age <- imputed.data[, 6]
head(train, 10)

indexes <- createDataPartition(train$Survived, 
                               times = 1,
                               p = 0.7,
                               list = FALSE)
titanic.train <- train[indexes, ]
titanic.test <- train[-indexes, ]

prop.table(table(train$Survived))
prop.table(table(titanic.train$Survived))
prop.table(table(titanic.test$Survived))

train.control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 3,
                              search = 'grid')

tune.grid <- expand.grid(eta = c(0.05, 0.075, 0.1),
                         nrounds = c(50, 75, 100),
                         max_depth = 6:8,
                         min_child_weight = c(2.0, 2.25, 2.5),
                         colsample_bytree = c(0.3, 0.4, 0.5),
                         gamma = 0,
                         subsample = 1)
head(tune.grid)

cl <- makeCluster(10, type = "SOCK")

registerDoSNOW(cl)

caret.cv <- caret::train(Survived ~.,
                         data = titanic.train,
                         method = "xgbTree",
                         tuneGrid = tune.grid,
                         trControl = train.control)

stopCluster(cl)

caret.cv

class(caret.cv)
