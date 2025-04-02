library(tidyverse)
library(caret)
library(e1071)
library(class)
library(cluster)
data <- read.csv("insurance.csv")
data <- na.omit(data)  # Remove missing values
data$charges_bin <- ifelse(data$charges > median(data$charges), 1, 0)
data$charges_bin <- as.factor(data$charges_bin)

# Encoding categorical variables
data <- data %>%
  mutate(
    sex = as.numeric(as.factor(sex)),
    smoker = as.numeric(as.factor(smoker)),
    region = as.numeric(as.factor(region))
  )

# Split dataset into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data$charges_bin, p = 0.7, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]

# Separate features and target
X_train <- train %>% select(-charges, -charges_bin)
y_train <- train$charges_bin
X_test <- test %>% select(-charges, -charges_bin)
y_test <- test$charges_bin

# Supervised Learning Models
# 1. KNN
knn_pred <- knn(X_train, X_test, y_train, k = 5)

# 2. Naive Bayes
nb_model <- naiveBayes(charges_bin ~ ., data = train)
nb_pred <- predict(nb_model, test)

# 3. SVM
svm_model <- svm(charges_bin ~ ., data = train, kernel = "linear")
svm_pred <- predict(svm_model, test)

# Evaluate and visualize
evaluate_model <- function(y_true, y_pred, model_name) {
  confusion <- confusionMatrix(as.factor(y_pred), as.factor(y_true))
  accuracy <- confusion$overall['Accuracy']
  cat(paste("Accuracy for", model_name, ":", round(accuracy, 4), "\n"))
  print(confusion)
  
  # Actual vs Predicted Scatter Plot
  df <- data.frame(Actual = y_true, Predicted = y_pred)
  ggplot(df, aes(x = Actual, y = Predicted)) +
    geom_jitter(width = 0.2, alpha = 0.6) +
    labs(title = paste(model_name, ": Actual vs Predicted"), x = "Actual", y = "Predicted") +
    theme_minimal()
}

# Evaluate models
evaluate_model(y_test, knn_pred, "KNN")
evaluate_model(y_test, nb_pred, "Naive Bayes")
evaluate_model(y_test, svm_pred, "SVM")

# Unsupervised Learning Models
# 1. K-Means Clustering
set.seed(123)
kmeans_res <- kmeans(X_train, centers = 3, nstart = 25)
clusplot( X_train, 
  kmeans_res$cluster, 
  lines=0,
  shade=TRUE,color=TRUE,
  labels=1,plotchar=TRUE,
  span=TRUE,
  main = "K-Means Clustering Visualization"
)
# 2. Hierarchical Clustering
hclust_res <- hclust(dist(X_train,method="euclidean"))
plot(hclust_res)
rect.hclust(hclust_res,k=3,border="blue")
