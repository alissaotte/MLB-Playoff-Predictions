#### READING IN THE DATA ####
MLB.df <- read.csv("Regular Season MLB Team Stats.csv", stringsAsFactors = TRUE)
t(t(names(MLB.df)))
summary(MLB.df)


#### CREATING DUMMY VARIABLES ####
library(fastDummies)
MLB.df <- dummy_cols(MLB.df,
                         select_columns = "Playoffs", 
                         remove_first_dummy = TRUE,
                         remove_selected_columns = TRUE)
t(t(names(MLB.df)))
summary(MLB.df)

#### EDA ####

## Correlation Matrix with Heatmap
library(gplots)
colfun <- colorRampPalette(c("red", "white", "green"))
heatmap.2(round(cor(MLB.df[, c(2:4, 6:22, 25)]), 2), Rowv = FALSE, Colv = FALSE,
          dendrogram = "none", col = colfun(5), lwid=c(0.1,4), lhei=c(0.1,4),
          cellnote = round(cor(MLB.df[, c(2:4, 6:22, 25)]), 2), 
          notecol = "black", key = FALSE, trace = "none", margins = c(10, 10))

## Updated Heatmap: DROP R, AVG, OBP, ER, H_BATTING, OBS_BATTING
colfun <- colorRampPalette(c("red", "white", "green"))
heatmap.2(round(cor(MLB.df[, c(2, 6,  9:12, 14:15, 17:22, 25)]), 2), Rowv = FALSE, Colv = FALSE,
          dendrogram = "none", col = colfun(5), lwid=c(0.1,4), lhei=c(0.1,4),
          cellnote = round(cor(MLB.df[, c(2, 6, 9:12, 14:15, 17:22, 25)]), 2), 
          notecol = "black", key = FALSE, trace = "none", margins = c(10, 10))

## scatterplot of Playoffs_Yes by R_BATTING
plot(MLB.df$Playoffs_Yes ~ MLB.df$W, xlab = "Runs (Batting)", 
     ylab = "Playoffs_Yes")

## scatterplot of Wins by Pitching ERA
plot(MLB.df$W ~ MLB.df$ERA, xlab = "ERA", 
     ylab = "Wins")

## scatterplot of Wins by Batting Runs
plot(MLB.df$W ~ MLB.df$R_BATTING, xlab = "Runs (Batting)", 
     ylab = "Wins")

## boxplot of Playoffs distribution compared to ERA
boxplot(MLB.df$ERA ~ MLB.df$Playoffs_Yes, xlab = "Playoffs", ylab = "ERA")

## boxplot of Playoffs distribution compared to R_BATTING
boxplot(MLB.df$R_BATTING ~ MLB.df$Playoffs_Yes, xlab = "Playoffs", ylab = "Runs (Batting)")

## create new data frame containing only the variables to be used for analysis
MLBsub.df <- MLB.df[, c(2, 6,  9:12, 14:15, 17:21, 25)]
head(MLBsub.df)


#### PARTITIONING THE DATA ####
# use set.seed() to get the same partitions when re-running the R code
set.seed(1)

## partitioning into training (70%) and validation (30%) 
# randomly sample 70% of the row IDs for training; the remaining 30% serve as
# validation
train.rows <- sample(rownames(MLBsub.df),       # sampling from the row IDs
                     nrow(MLBsub.df)*0.7)       # using 70% for training data

# collect all the columns with training row ID into training set:
train.data <- MLBsub.df[train.rows, ]

# assign row IDs that are not already in the training set into validation
valid.rows <- setdiff(rownames(MLBsub.df), train.rows)
valid.data <- MLBsub.df[valid.rows, ]

head(train.data)
head(valid.data)


#### Full Logistic Regression Model ####
playoffs.glm <- glm(Playoffs_Yes ~ ., family = "binomial", data = train.data)
options(scipen=999, digits=7)
summary(playoffs.glm)
dim(summary(playoffs.glm)$coefficients)[1]-1
AIC(playoffs.glm)
BIC(playoffs.glm)
deviance(playoffs.glm)

# out-of-sample prediction
library(forecast)
pred.glm.valid <- predict(playoffs.glm, newdata = valid.data, type = "response")
pred.glm.valid
# create the confusion matrix
library(caret)
confusionMatrix(as.factor(ifelse(pred.glm.valid >= 0.3, "1", "0")), as.factor(valid.data$Playoffs_Yes), 
                positive = "1")

#### odds coefficients ####
options(scipen=5, digits=7)
data.frame(summary(playoffs.glm)$coefficient, odds = exp(coef(playoffs.glm)))



#### RANDOM FOREST FOR CLASSIFICATION ####
library(randomForest)

MLB.rf <- randomForest(as.factor(Playoffs_Yes) ~ .,
                       data = train.data,
                       ntree = 500,
                       mtry = 4,
                       nodesize = 5,
                       importance = TRUE)

#### VARIABLE IMPORTANT PLOT ####
varImpPlot(MLB.rf, type = 1)

#### CONFUSION MATRIX ####
MLB.rf.pred <- predict(MLB.rf, valid.data)
confusionMatrix(MLB.rf.pred, as.factor(valid.data$Playoffs_Yes), positive = "1")



#### K-NEAREST NEIGHBORS FOR CLASSIFICATION ####

colnames(MLBsub.df)

#### STANDARDIZATION ####
train.norm <- train.data
valid.norm <- valid.data

cols <- colnames(train.norm[ , c(1:14)])
for (i in cols) {
  valid.norm[[i]] <- (valid.data[[i]] - min(train.data[[i]])) / (max(train.data[[i]]) - min(train.data[[i]]))
  train.norm[[i]] <- (train.data[[i]] - min(train.data[[i]])) / (max(train.data[[i]]) - min(train.data[[i]]))
}

summary(train.norm)
summary(valid.norm)


#### USING K=1 ####
library(FNN)
## Excluding outcome var Playoffs_Yes (15)
MLB.nn <- knn(train = train.norm[, c(1:13)], 
              test = valid.norm[, c(1:13)],
              cl = train.norm$Playoffs_Yes,
              k = 1)

# look at the confusion matrix
library(caret)
confusionMatrix(MLB.nn, as.factor(valid.norm$Playoffs_Yes), positive = "1")


#### DETERMINE BEST K FOR CLASSIFICATION ####

# initialize a data frame with two columns: k and accuracy
accuracy.df <- data.frame(k = seq(1, 30, 1), accuracy = rep(0, 30))

# compute knn for different k on validation set
for (i in 1:30) {
  MLB.knn.pred <- knn(train = train.norm[, c(1:13)], 
                      test = valid.norm[, c(1:13)],
                      cl = train.norm$Playoffs_Yes, 
                      k = i)
  accuracy.df[i, 2] <- confusionMatrix(MLB.knn.pred, as.factor(valid.norm$Playoffs_Yes), positive = "1")$overall[1]
}
accuracy.df

#### Confusion Matrix for k = 5 ####
MLB.nn5 <- knn(train = train.norm[, c(1:13)], 
               test = valid.norm[, c(1:13)],
               cl = train.norm$Playoffs_Yes,
               k = 5)
confusionMatrix(MLB.nn5, as.factor(valid.norm$Playoffs_Yes), positive = "1")


