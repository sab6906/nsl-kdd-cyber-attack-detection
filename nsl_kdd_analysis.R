#############################################################
# Predictive Modelling for Cyber Attack Detection
# NSL-KDD Dataset — Logistic Regression, Random Forest,
# XGBoost, and SVM comparison with cost-based threshold
# optimisation for auto-blocking simulation.
#############################################################

# ============================================================
# 1. DATA CLEANING
# ============================================================
library(tidyverse)
library(janitor)
library(readr)
library(dplyr)

# Load raw NSL-KDD train/test CSVs
train <- read.csv(file.choose())
test  <- read.csv(file.choose())

# Remove duplicate records
train <- distinct(train)
test  <- distinct(test)

# Identify and standardise the target column
candidates <- c("class","label","attack","attack_type","target","classification")
found <- intersect(candidates, names(train))
if (length(found) == 0) {
  stop(paste0(
    "Couldn't find a target column. Available columns are:\n",
    paste(names(train), collapse = ", "),
    "\nSet target_col manually, e.g. target_col <- 'label'"
  ))
}
target_col <- found[1]

normalize_target <- function(x) {
  x <- tolower(as.character(x))
  case_when(
    x %in% c("normal", "benign", "0") ~ "normal",
    x %in% c("attack", "anomaly", "malicious", "1") ~ "attack",
    TRUE ~ x
  )
}

train[[target_col]] <- factor(normalize_target(train[[target_col]]), levels = c("normal", "attack"))
test[[target_col]]  <- factor(normalize_target(test[[target_col]]),  levels = c("normal", "attack"))

# Align test columns to train
test <- test %>% select(all_of(names(train)))

table(train[[target_col]])
table(test[[target_col]])

# Save cleaned datasets
write_csv(train, "train_clean_simple.csv")
write_csv(test, "test_clean_simple.csv")
cat("Cleaning done.\n")


# ============================================================
# 2. DATA VISUALISATION
# ============================================================
library(ggplot2)
library(dplyr)
library(reshape2)

target_col <- "label"
stopifnot(target_col %in% names(train))

train[[target_col]] <- tolower(as.character(train[[target_col]]))
train[[target_col]] <- dplyr::recode(
  train[[target_col]],
  "benign" = "normal", "0" = "normal",
  "1" = "attack", "anomaly" = "attack", "malicious" = "attack",
  .default = train[[target_col]]
)
train[[target_col]] <- factor(train[[target_col]], levels = c("normal","attack"))

# Bar chart — class counts
ggplot(train, aes(x = .data[[target_col]], fill = .data[[target_col]])) +
  geom_bar() +
  labs(title = "Normal vs Attack — Count", x = "Class", y = "Count") +
  scale_fill_manual(values = c("normal" = "steelblue", "attack" = "firebrick")) +
  theme_minimal()

# Pie chart — class proportion
train %>%
  count(.data[[target_col]]) %>%
  ggplot(aes(x = "", y = n, fill = .data[[target_col]])) +
  geom_col(width = 1) +
  coord_polar(theta = "y") +
  labs(title = "Normal vs Attack — Proportion", fill = "Class") +
  scale_fill_manual(values = c("normal" = "green", "attack" = "red")) +
  theme_void()

# Box plot — duration by class (log scale)
ggplot(train, aes(x = .data[[target_col]], y = duration, fill = .data[[target_col]])) +
  geom_boxplot(outlier.alpha = 0.3) +
  scale_y_log10() +
  labs(title = "Duration by Class (Log Scale)", x = "Class", y = "Duration (log scale)") +
  scale_fill_manual(values = c("normal" = "darkgreen", "attack" = "firebrick")) +
  theme_minimal()

# Correlation heatmap (numeric features)
num_vars <- train %>% select(where(is.numeric))
cor_mat <- cor(num_vars, use = "complete.obs")
cor_df <- melt(cor_mat)

ggplot(cor_df, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                        midpoint = 0, limit = c(-1,1), space = "Lab",
                        name = "Pearson\nCorrelation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  coord_fixed() +
  labs(title = "Feature Correlation Heatmap", x = "", y = "")


# ============================================================
# 3. LOGISTIC REGRESSION (BASELINE)
# ============================================================
library(readr)
library(dplyr)
library(caret)
library(lattice)
library(pROC)

target_col <- "label"
train[[target_col]] <- factor(train[[target_col]], levels = c("normal","attack"))
test[[target_col]]  <- factor(test[[target_col]],  levels = c("normal","attack"))

cat_cols <- names(train)[sapply(train, function(x) is.character(x) || is.factor(x))]
cat_cols <- setdiff(cat_cols, target_col)
for (col in cat_cols) {
  lvls <- union(levels(factor(train[[col]])), levels(factor(test[[col]])))
  train[[col]] <- factor(train[[col]], levels = lvls)
  test[[col]]  <- factor(test[[col]],  levels = lvls)
}

ctrl <- trainControl(method = "none", classProbs = TRUE)
logit_model <- train(
  reformulate(termlabels = setdiff(names(train), target_col), response = target_col),
  data = train,
  method = "glm",
  family = "binomial",
  trControl = ctrl
)

pred_probs <- predict(logit_model, newdata = test, type = "prob")[, "attack"]
pred_class <- predict(logit_model, newdata = test, type = "raw")

cm <- confusionMatrix(pred_class, test[[target_col]], positive = "attack")
print(cm)

roc_obj <- roc(response = test[[target_col]], predictor = pred_probs, levels = c("normal","attack"))
cat("ROC-AUC:", auc(roc_obj), "\n")
plot(roc_obj, col = "blue", main = "ROC Curve - Logistic Regression")


# ============================================================
# 4. RANDOM FOREST
# ============================================================
library(randomForest)

y_train <- factor(train[[target_col]], levels = c("normal","attack"))
y_test  <- factor(test[[target_col]],  levels = c("normal","attack"))
X_train <- train %>% select(-all_of(target_col))
X_test  <- test %>% select(-all_of(target_col))

dmy <- dummyVars(~ ., data = X_train, fullRank = TRUE)
train_X <- as.data.frame(predict(dmy, newdata = X_train))
test_X  <- as.data.frame(predict(dmy, newdata = X_test))

missing_in_test <- setdiff(names(train_X), names(test_X))
if (length(missing_in_test)) for (nm in missing_in_test) test_X[[nm]] <- 0
extra_in_test <- setdiff(names(test_X), names(train_X))
if (length(extra_in_test)) test_X <- test_X[, setdiff(names(test_X), extra_in_test)]
test_X <- test_X[, names(train_X)]

set.seed(123)
rf_model <- randomForest(
  x = train_X, y = y_train,
  ntree = 300, mtry = floor(sqrt(ncol(train_X))),
  importance = TRUE
)

pred_probs <- predict(rf_model, newdata = test_X, type = "prob")[, "attack"]
pred_class <- predict(rf_model, newdata = test_X, type = "class")

cm <- caret::confusionMatrix(pred_class, y_test, positive = "attack")
print(cm)

roc_obj <- pROC::roc(response = y_test, predictor = pred_probs, levels = c("normal","attack"))
auc_val <- auc(roc_obj)
cat("ROC-AUC:", auc_val, "\n")
plot(roc_obj, col = "darkgreen", main = paste0("ROC Curve - Random Forest (AUC = ", round(auc_val, 3), ")"))


# ============================================================
# 5. XGBOOST
# ============================================================
library(xgboost)

y_train <- factor(train[[target_col]], levels = c("normal", "attack"))
y_test  <- factor(test[[target_col]],  levels = c("normal", "attack"))
X_train <- train %>% select(-all_of(target_col))
X_test  <- test %>% select(-all_of(target_col))

dmy <- dummyVars(~ ., data = X_train, fullRank = TRUE)
train_X <- as.data.frame(predict(dmy, newdata = X_train))
test_X  <- as.data.frame(predict(dmy, newdata = X_test))

missing_in_test <- setdiff(names(train_X), names(test_X))
if (length(missing_in_test)) for (nm in missing_in_test) test_X[[nm]] <- 0
extra_in_test <- setdiff(names(test_X), names(train_X))
if (length(extra_in_test)) test_X <- test_X[, setdiff(names(test_X), extra_in_test)]
test_X <- test_X[, names(train_X)]

dtrain <- xgb.DMatrix(data = as.matrix(train_X), label = as.numeric(y_train) - 1)
dtest  <- xgb.DMatrix(data = as.matrix(test_X),  label = as.numeric(y_test) - 1)

set.seed(123)
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1, max_depth = 6,
  subsample = 0.8, colsample_bytree = 0.8
)

xgb_model <- xgb.train(
  params = params, data = dtrain, nrounds = 200,
  watchlist = list(train = dtrain, eval = dtest), verbose = 0
)

pred_probs <- predict(xgb_model, dtest)
pred_class <- ifelse(pred_probs >= 0.5, "attack", "normal") %>% factor(levels = c("normal", "attack"))

cm <- caret::confusionMatrix(pred_class, y_test, positive = "attack")
print(cm)

roc_obj <- roc(response = y_test, predictor = pred_probs, levels = c("normal", "attack"))
auc_val <- auc(roc_obj)
cat("ROC-AUC:", auc_val, "\n")
plot(roc_obj, col = "orange", main = paste0("ROC Curve - XGBoost (AUC = ", round(auc_val, 3), ")"))


# ============================================================
# 6. SUPPORT VECTOR MACHINE (LINEAR KERNEL)
# ============================================================
library(e1071)

y_train <- factor(train[[target_col]], levels = c("normal", "attack"))
y_test  <- factor(test[[target_col]],  levels = c("normal", "attack"))

dmy <- dummyVars(~ ., data = train %>% select(-all_of(target_col)), fullRank = TRUE)
train_X <- as.data.frame(predict(dmy, newdata = train %>% select(-all_of(target_col))))
test_X  <- as.data.frame(predict(dmy, newdata = test %>% select(-all_of(target_col))))

missing_in_test <- setdiff(names(train_X), names(test_X))
for (nm in missing_in_test) test_X[[nm]] <- 0
extra_in_test <- setdiff(names(test_X), names(train_X))
if (length(extra_in_test)) test_X <- test_X[, setdiff(names(test_X), extra_in_test)]
test_X <- test_X[, names(train_X)]

set.seed(123)
svm_model <- train(
  x = train_X, y = y_train, method = "svmLinear",
  trControl = trainControl(method = "none", classProbs = TRUE),
  preProcess = c("center", "scale")
)

pred_probs <- predict(svm_model, newdata = test_X, type = "prob")[, "attack"]
pred_class <- predict(svm_model, newdata = test_X, type = "raw")

cm <- confusionMatrix(pred_class, y_test, positive = "attack")
print(cm)

roc_obj <- roc(response = y_test, predictor = pred_probs, levels = c("normal", "attack"))
auc_val <- auc(roc_obj)
cat("ROC-AUC:", auc_val, "\n")
plot(roc_obj, col = "purple", main = paste0("ROC Curve - SVM (AUC = ", round(auc_val, 3), ")"))


# ============================================================
# 7. THRESHOLD OPTIMISATION FOR HIGH-PRECISION AUTO-BLOCKING
# ============================================================
metric_at <- function(th) {
  pred <- factor(ifelse(pred_probs >= th, "attack", "normal"), levels = c("normal","attack"))
  cm <- confusionMatrix(pred, y_test, positive = "attack")
  data.frame(
    threshold = th,
    precision = as.numeric(cm$byClass["Pos Pred Value"]),
    recall = as.numeric(cm$byClass["Sensitivity"]),
    f1 = 2 * (as.numeric(cm$byClass["Pos Pred Value"]) * as.numeric(cm$byClass["Sensitivity"])) /
      (as.numeric(cm$byClass["Pos Pred Value"]) + as.numeric(cm$byClass["Sensitivity"])),
    accuracy = as.numeric(cm$overall["Accuracy"])
  )
}

ths <- seq(0.05, 0.95, by = 0.01)
metrics <- do.call(rbind, lapply(ths, metric_at))

target_prec <- 0.95
candidates <- metrics %>% filter(precision >= target_prec)
best_row <- if (nrow(candidates) == 0) {
  metrics[which.max(metrics$precision), ]
} else {
  candidates[which.max(candidates$recall), ]
}
best_th <- best_row$threshold

final_pred <- factor(ifelse(pred_probs >= best_th, "attack", "normal"), levels = c("normal","attack"))
final_cm <- confusionMatrix(final_pred, y_test, positive = "attack")
print(final_cm)

cat(sprintf("Chosen threshold: %.2f | Precision: %.3f | Recall: %.3f | F1: %.3f | Accuracy: %.3f\n",
            best_th, best_row$precision, best_row$recall, best_row$f1, best_row$accuracy))

# Precision / Recall / F1 vs Threshold plot
library(tidyr)
m_long <- metrics %>%
  select(threshold, precision, recall, f1) %>%
  pivot_longer(-threshold, names_to = "metric", values_to = "value")

ggplot(m_long, aes(threshold, value, color = metric)) +
  geom_line(size = 1) +
  theme_minimal() +
  labs(title = "Precision / Recall / F1 vs Threshold", y = "Score", x = "Threshold")


# ============================================================
# 8. SIMULATE STREAMING & AUTO-BLOCK DECISION LOG
# ============================================================
simulate_autoblock <- function(probs, labels, th = 0.95) {
  decision <- ifelse(probs >= th, "block", "allow")
  truth <- ifelse(labels == "attack", "attack", "normal")

  ok_block <- decision == "block" & truth == "attack"
  fp_block <- decision == "block" & truth == "normal"
  miss     <- decision == "allow" & truth == "attack"
  pass_ok  <- decision == "allow" & truth == "normal"

  log <- tibble(
    id = seq_along(probs),
    prob_attack = probs,
    decision = factor(decision, levels = c("allow","block")),
    truth = factor(truth, levels = c("normal","attack")),
    correct = ok_block | pass_ok,
    error_type = case_when(
      fp_block ~ "false_block",
      miss ~ "missed_attack",
      TRUE ~ "none"
    )
  )

  cm <- confusionMatrix(
    reference = labels,
    data = factor(ifelse(decision == "block","attack","normal"), levels = c("normal","attack")),
    positive = "attack"
  )

  list(log = log, cm = cm,
       metrics = list(
         threshold = th,
         block_rate = mean(decision == "block"),
         false_block_rate = mean(fp_block),
         attack_pass_rate = mean(miss),
         precision = cm$byClass["Pos Pred Value"],
         recall = cm$byClass["Sensitivity"],
         f1 = 2*(cm$byClass["Pos Pred Value"]*cm$byClass["Sensitivity"]) /
           (cm$byClass["Pos Pred Value"]+cm$byClass["Sensitivity"]),
         accuracy = cm$overall["Accuracy"]
       ))
}

sim <- simulate_autoblock(pred_probs, y_test, th = best_th)
sim$cm
sim$metrics
write_csv(sim$log, "autoblock_decision_log.csv")

# Cumulative operational impact plot
cum <- sim$log %>%
  mutate(
    tp = (decision=="block" & truth=="attack")*1,
    fp = (decision=="block" & truth=="normal")*1,
    fn = (decision=="allow" & truth=="attack")*1
  ) %>%
  mutate(
    cum_tp = cumsum(tp), cum_fp = cumsum(fp), cum_fn = cumsum(fn),
    seen = row_number()
  )

ggplot(cum, aes(seen)) +
  geom_line(aes(y = cum_tp), size = 1) +
  geom_line(aes(y = cum_fp)) +
  geom_line(aes(y = cum_fn)) +
  labs(title = paste0("Cumulative Auto-Block Impact (threshold = ", best_th, ")"),
       x = "Flows seen (test stream order)", y = "Cumulative count") +
  theme_minimal()


# ============================================================
# 9. MODEL COMPARISON IN PREVENTION MODE
# ============================================================
get_probs <- function(model_name){
  switch(model_name,
    "LogReg" = as.numeric(predict(logit_model, newdata = test, type = "prob")[, "attack"]),
    "RandomForest" = as.numeric(predict(rf_model, newdata = test_X, type = "prob")[, "attack"]),
    "XGBoost" = as.numeric(predict(xgb_model, dtest)),
    "SVM" = as.numeric(predict(svm_model, newdata = test_X, type = "prob")[, "attack"]),
    stop("Unknown model_name")
  )
}

simulate_autoblock_metrics <- function(probs, labels, th){
  decision <- ifelse(probs >= th, "block", "allow")
  cm <- confusionMatrix(
    data = factor(ifelse(decision=="block","attack","normal"), levels=c("normal","attack")),
    reference = labels, positive = "attack"
  )
  tibble(
    threshold = th,
    precision = as.numeric(cm$byClass["Pos Pred Value"]),
    recall = as.numeric(cm$byClass["Sensitivity"]),
    f1 = 2*(as.numeric(cm$byClass["Pos Pred Value"])*as.numeric(cm$byClass["Sensitivity"])) /
      (as.numeric(cm$byClass["Pos Pred Value"])+as.numeric(cm$byClass["Sensitivity"])),
    accuracy = as.numeric(cm$overall["Accuracy"]),
    block_rate = mean(decision=="block"),
    false_block_rate = mean(decision=="block" & labels=="normal"),
    attack_pass_rate = mean(decision=="allow" & labels=="attack")
  )
}

find_threshold_for_precision <- function(probs, labels, target_prec = 0.95){
  ths <- seq(0.05, 0.95, by = 0.01)
  rows <- lapply(ths, function(th){
    pred <- factor(ifelse(probs >= th, "attack","normal"), levels=c("normal","attack"))
    cm <- confusionMatrix(pred, labels, positive="attack")
    tibble(threshold = th,
           precision = as.numeric(cm$byClass["Pos Pred Value"]),
           recall = as.numeric(cm$byClass["Sensitivity"]))
  }) %>% bind_rows()
  cand <- dplyr::filter(rows, precision >= target_prec)
  if (nrow(cand) == 0) rows$threshold[which.max(rows$precision)] else cand$threshold[which.max(cand$recall)]
}

models <- c("LogReg","RandomForest","XGBoost","SVM")

# Fixed threshold (0.95) comparison
fixed_tbl <- lapply(models, function(m){
  probs <- get_probs(m)
  simulate_autoblock_metrics(probs, y_test, th = 0.95) %>% mutate(model = m, .before = 1)
}) %>% bind_rows() %>% arrange(desc(precision), desc(recall))
write_csv(fixed_tbl, "prevention_comparison_fixed095.csv")

# Per-model tuned threshold (>=95% precision) comparison
tuned_tbl <- lapply(models, function(m){
  probs <- get_probs(m)
  th <- find_threshold_for_precision(probs, y_test, target_prec = 0.95)
  simulate_autoblock_metrics(probs, y_test, th = th) %>% mutate(model = m, .before = 1)
}) %>% bind_rows() %>% arrange(desc(precision), desc(recall))
write_csv(tuned_tbl, "prevention_comparison_tuned95prec.csv")


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================
rf_imp <- randomForest::importance(rf_model) %>%
  as.data.frame() %>%
  tibble::rownames_to_column("feature") %>%
  as_tibble() %>%
  arrange(desc(MeanDecreaseGini)) %>%
  slice_head(n = 15)

ggplot(rf_imp, aes(x = reorder(feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_col() + coord_flip() +
  labs(title = "Random Forest - Top 15 Features", x = "Feature", y = "Mean Decrease Gini") +
  theme_minimal()

xgb_imp <- xgb.importance(model = xgb_model) %>%
  as_tibble() %>% arrange(desc(Gain)) %>% slice_head(n = 15)

ggplot(xgb_imp, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col() + coord_flip() +
  labs(title = "XGBoost - Top 15 Features by Gain", x = "Feature", y = "Gain") +
  theme_minimal()


# ============================================================
# 11. COST-BASED THRESHOLD OPTIMISATION (ALL MODELS)
# ============================================================
cost_missed_attack <- 5000   # cost of an undetected attack (FN)
cost_false_block   <- 50     # cost of blocking a legitimate flow (FP)

cost_eval <- function(probs, labels, th, model){
  decision <- ifelse(probs >= th, "attack", "normal")
  false_blocks   <- sum(decision == "attack" & labels == "normal")
  missed_attacks <- sum(decision == "normal" & labels == "attack")
  total_cost <- (missed_attacks * cost_missed_attack) + (false_blocks * cost_false_block)
  tibble(model = model, threshold = th, false_blocks = false_blocks,
         missed_attacks = missed_attacks, total_cost = total_cost)
}

sweep_costs <- function(model){
  probs <- get_probs(model)
  ths <- seq(0.05, 0.95, by = 0.01)
  bind_rows(lapply(ths, function(th) cost_eval(probs, y_test, th, model)))
}

per_threshold <- bind_rows(lapply(models, sweep_costs))

best_per_model <- per_threshold %>%
  group_by(model) %>%
  slice_min(total_cost, n = 1, with_ties = FALSE) %>%
  ungroup()

print(best_per_model)

write_csv(per_threshold, "cost_per_threshold_all_models.csv")
write_csv(best_per_model, "cost_best_threshold_by_model.csv")

# Cost curves per model, with optimal threshold highlighted
ggplot(per_threshold, aes(threshold, total_cost)) +
  geom_line() +
  facet_wrap(~ model, scales = "free_y") +
  geom_point(data = best_per_model, aes(threshold, total_cost), color = "red", size = 2) +
  labs(title = "Cost-Based Threshold Optimization", x = "Decision Threshold", y = "Total Cost (£)") +
  theme_minimal()

# Minimum cost comparison bar chart
ggplot(best_per_model, aes(x = model, y = total_cost)) +
  geom_col() +
  labs(title = "Minimum Expected Cost by Model", x = "Model", y = "Total Cost (£)") +
  theme_minimal()

cat("Analysis complete.\n")
