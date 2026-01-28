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

ggsave(
  "class_distribution_original.png",
  ggplot(orig_dist, aes(x = Class, y = Count, fill = Class)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = Count), vjust = -0.3) +
    theme_minimal(),
  width = 6, height = 5
)

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
idx <- createDataPartition(data_clean$Exited, p = 0.8, list = FALSE)
train_data <- data_clean[idx, ]
test_data  <- data_clean[-idx, ]

# ===============================
# 5. Recipe: Dummy + SMOTE (CORRECT)
# ===============================
rec <- recipe(Exited ~ ., data = train_data) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_smote(Exited)

rec_prep <- prep(rec, training = train_data)

train_smote <- bake(rec_prep, new_data = NULL)
test_processed <- bake(rec_prep, new_data = test_data)

# ===============================
# 6. Class Distribution After SMOTE (PNG)
# ===============================
smote_dist <- as.data.frame(table(train_smote$Exited))
colnames(smote_dist) <- c("Class", "Count")

ggsave(
  "class_distribution_smote.png",
  ggplot(smote_dist, aes(x = Class, y = Count, fill = Class)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = Count), vjust = -0.3) +
    theme_minimal(),
  width = 6, height = 5
)

# ===============================
# 7. Random Forest Model
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
# 8. Predictions
# ===============================
pred_prob <- predict(rf_model, test_processed, type = "prob")[, "1"]
pred_class <- predict(rf_model, test_processed, type = "response")

# ===============================
# 9. Confusion Matrix
# ===============================
cm <- confusionMatrix(pred_class, test_processed$Exited, positive = "1")
print(cm)

# ===============================
# 10. Confusion Matrix Heatmap (PNG)
# ===============================
cm_df <- as.data.frame(cm$table)
colnames(cm_df) <- c("Prediction", "Reference", "Count")

ggsave(
  "confusion_matrix_heatmap_rf.png",
  ggplot(cm_df, aes(Reference, Prediction, fill = Count)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Count), size = 6) +
    scale_fill_gradient(low = "#E3F2FD", high = "#0D47A1") +
    theme_minimal(),
  width = 6, height = 5
)

# ===============================
# 11. ROC Curve & AUC (PNG)
# ===============================
roc_obj <- roc(
  as.numeric(as.character(test_processed$Exited)),
  pred_prob
)

auc_score <- auc(roc_obj)

ggsave(
  "roc_auc_curve_rf.png",
  ggroc(roc_obj) +
    ggtitle(paste("ROC Curve (AUC =", round(auc_score, 3), ")")) +
    theme_minimal(),
  width = 6, height = 5
)

cat("AUC Score:", auc_score, "\n")
