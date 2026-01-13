library(dplyr)
library(caret)

# 1. Load the dataset
data <- read.csv("Customer-Churn-Records.csv")

# 2. Preprocessing: Remove identifiers and the leakage-causing 'Complain' column
data_clean <- data %>% 
  select(-RowNumber, -CustomerId, -Surname, -Complain)

# 3. Convert text data to factors
data_clean$Geography <- as.factor(data_clean$Geography)
data_clean$Gender <- as.factor(data_clean$Gender)
data_clean$Card.Type <- as.factor(data_clean$Card.Type)
data_clean$Exited <- as.factor(data_clean$Exited)

# 4. Custom Sampling to get exactly 2,400 rows per class
set.seed(123)

# Separate the classes
class_0 <- data_clean %>% filter(Exited == 0)
class_1 <- data_clean %>% filter(Exited == 1)

# Down-sample Class 0 to 2400
sampled_class_0 <- class_0 %>% sample_n(2400)

# Up-sample Class 1 to 2400 (using replace = TRUE since there are only 2038 rows)
sampled_class_1 <- class_1 %>% sample_n(2400, replace = TRUE)

# Combine into a final balanced dataset
data_balanced <- rbind(sampled_class_0, sampled_class_1)

print("New Class Distribution:")
print(table(data_balanced$Exited))

# 5. Split into Training (80%) and Testing (20%)
trainIndex <- createDataPartition(data_balanced$Exited, p = 0.8, list = FALSE)
train_data <- data_balanced[trainIndex, ]
test_data  <- data_balanced[-trainIndex, ]

# 6. Build the Logistic Regression Model
model <- glm(Exited ~ ., data = train_data, family = "binomial")

# 7. Model Summary and Evaluation
summary(model)

# Predictions
predictions_prob <- predict(model, newdata = test_data, type = "response")
predictions_class <- ifelse(predictions_prob > 0.5, 1, 0)

# Confusion Matrix
conf_matrix <- confusionMatrix(as.factor(predictions_class), test_data$Exited)
print(conf_matrix)