# library(dplyr)
# library(caret)

# # 1. Load the dataset
# data <- read.csv("Customer-Churn-Records.csv")

# # 2. Preprocessing: Remove identifiers and the leakage-causing 'Complain' column
# data_clean <- data %>% 
#   select(-RowNumber, -CustomerId, -Surname, -Complain)

# # 3. Convert text data to factors
# data_clean$Geography <- as.factor(data_clean$Geography)
# data_clean$Gender <- as.factor(data_clean$Gender)
# data_clean$Card.Type <- as.factor(data_clean$Card.Type)
# data_clean$Exited <- as.factor(data_clean$Exited)

# # 4. Custom Sampling to get exactly 2,400 rows per class
# set.seed(123)

# # Separate the classes
# class_0 <- data_clean %>% filter(Exited == 0)
# class_1 <- data_clean %>% filter(Exited == 1)

# # Down-sample Class 0 to 2400
# sampled_class_0 <- class_0 %>% sample_n(2400)

# # Up-sample Class 1 to 2400 (using replace = TRUE since there are only 2038 rows)
# sampled_class_1 <- class_1 %>% sample_n(2400, replace = TRUE)

# # Combine into a final balanced dataset
# data_balanced <- rbind(sampled_class_0, sampled_class_1)

# print("New Class Distribution:")
# print(table(data_balanced$Exited))

# # 5. Split into Training (80%) and Testing (20%)
# trainIndex <- createDataPartition(data_balanced$Exited, p = 0.8, list = FALSE)
# train_data <- data_balanced[trainIndex, ]
# test_data  <- data_balanced[-trainIndex, ]

# # 6. Build the Logistic Regression Model
# model <- glm(Exited ~ ., data = train_data, family = "binomial")

# # 7. Model Summary and Evaluation
# summary(model)

# # Predictions
# predictions_prob <- predict(model, newdata = test_data, type = "response")
# predictions_class <- ifelse(predictions_prob > 0.5, 1, 0)

# # Confusion Matrix
# conf_matrix <- confusionMatrix(as.factor(predictions_class), test_data$Exited)
# print(conf_matrix)









# ===============================
# 0. Load required libraries
# ===============================
library(dplyr)
library(caret)
library(ggplot2)
library(pROC)
library(randomForest)
library(themis)
library(recipes)

# ===============================
# 1. Load the dataset
# ===============================
data <- read.csv("Customer-Churn-Records.csv")

# ===============================
# 2. Original Class Distribution (PNG)
# ===============================
orig_dist <- as.data.frame(table(data$Exited))
colnames(orig_dist) <- c("Class", "Count")

p1 <- ggplot(orig_dist, aes(x = Class, y = Count, fill = Class)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = Count), vjust = -0.3, size = 5) +
  labs(title = "Original Class Distribution", x = "Exited", y = "Count") +
  theme_minimal()

ggsave("class_distribution_original.png", p1, width = 6, height = 5)

# ===============================
# 3. Data preprocessing
# ===============================
data_clean <- data %>%
  select(-RowNumber, -CustomerId, -Surname, -Complain)

data_clean$Geography <- as.factor(data_clean$Geography)
data_clean$Gender    <- as.factor(data_clean$Gender)
data_clean$Card.Type <- as.factor(data_clean$Card.Type)
data_clean$Exited    <- as.factor(data_clean$Exited)

# ===============================
# 4. Train-Test Split
# ===============================
set.seed(123)
trainIndex <- createDataPartition(data_clean$Exited, p = 0.8, list = FALSE)
train_data <- data_clean[trainIndex, ]
test_data  <- data_clean[-trainIndex, ]

# ===============================
# 5. SMOTE with proper preprocessing
# ===============================

set.seed(123)

rec <- recipe(Exited ~ ., data = train_data) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_smote(Exited)

rec_prep <- prep(rec)

train_smote <- bake(rec_prep, new_data = NULL)
test_processed <- bake(rec_prep, new_data = test_data)


# ===============================
# 6. Class Distribution After SMOTE (PNG)
# ===============================
smote_dist <- as.data.frame(table(train_smote$Exited))
colnames(smote_dist) <- c("Class", "Count")

p2 <- ggplot(smote_dist, aes(x = Class, y = Count, fill = Class)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = Count), vjust = -0.3, size = 5) +
  labs(title = "Class Distribution After SMOTE", x = "Exited", y = "Count") +
  theme_minimal()

ggsave("class_distribution_smote.png", p2, width = 6, height = 5)

# ===============================
# 7. Apply SAME preprocessing to test data
# ===============================
test_processed <- bake(rec, new_data = test_data)

# ===============================
# 8. Random Forest Model
# ===============================
set.seed(123)

rf_model <- randomForest(
  Exited ~ .,
  data = train_smote,
  ntree = 300,
  importance = TRUE
)

print(rf_model)

# ===============================
# 9. Predictions
# ===============================
pred_prob <- predict(rf_model, newdata = test_processed, type = "prob")[, "1"]
pred_class <- predict(rf_model, newdata = test_processed, type = "response")

# ===============================
# 10. Confusion Matrix
# ===============================
conf_matrix <- confusionMatrix(
  data = pred_class,
  reference = test_processed$Exited,
  positive = "1"
)

print(conf_matrix)

# ===============================
# 11. Confusion Matrix Heatmap (PNG)
# ===============================
cm_df <- as.data.frame(conf_matrix$table)
colnames(cm_df) <- c("Prediction", "Reference", "Count")

p3 <- ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), size = 6) +
  scale_fill_gradient(low = "#E3F2FD", high = "#0D47A1") +
  labs(
    title = "Confusion Matrix Heatmap (Random Forest)",
    x = "Actual Class",
    y = "Predicted Class"
  ) +
  theme_minimal()

ggsave("confusion_matrix_heatmap_rf.png", p3, width = 6, height = 5)

# ===============================
# 12. ROC Curve & AUC (PNG)
# ===============================
roc_obj <- roc(
  response = as.numeric(as.character(test_processed$Exited)),
  predictor = pred_prob
)

auc_score <- auc(roc_obj)

p4 <- ggroc(roc_obj, color = "darkgreen", size = 1.2) +
  ggtitle(paste("ROC Curve – Random Forest (AUC =", round(auc_score, 3), ")")) +
  theme_minimal()

ggsave("roc_auc_curve_rf.png", p4, width = 6, height = 5)

cat("AUC Score (Random Forest):", auc_score, "\n")
