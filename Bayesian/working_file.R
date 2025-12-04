# Bayesian Breast Cancer Classification
# 1. Loads and explores the Wisconsin Breast Cancer data
# 2. Define features
# 3. EDA
# 4. Correlation
# 5. Data prep for modeling
# 6. Feature selection
# 7. Baseline logistic regression 
# 8. Bayesian model with one predictor (bare_nuclei)
# 9. Bayesian model with normal priors
# 10. Try horseshoe prior model
# 11. Diagnostics for main Bayesian model
# 12. Feature effect viz
# 13. Posterior prediction
# 14. Uncertainty & entropy
# 15. Model eval
# 16. Final model performance comparison
# 17. High level summary
# 18. Save fitted models


## Prep
# import libs and set theme
library(tidyverse)
library(brms)
library(corrplot)
library(pROC)
library(gridExtra)
library(ggplot2)

# set a global plotting theme with a pink aesthetic
theme_set(
  theme_minimal() +
    theme(
      plot.title    = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 12),
      panel.grid.minor = element_blank()
    )
)

# Make lines and points pink by default to keep visuals consistent
update_geom_defaults("line",   list(color = "hotpink"))
update_geom_defaults("point",  list(color = "hotpink"))
update_geom_defaults("bar",    list(fill  = "hotpink"))
update_geom_defaults("col",    list(fill  = "hotpink"))
update_geom_defaults("smooth", list(color = "hotpink", fill = "pink"))



## 1. Load Data And Basic EDA
bc <- read.csv("/Users/ellawiser/Desktop/DS-4420-final-project/Data/bayes_breast_cancer_clean.csv")

# look at data
head(bc)

# data size
cat("Dataset dimensions:", dim(bc), "\n")

# class distribution
cat("Class distribution:\n")
print(table(bc$class))

cat("\nClass proportions:\n")
print(prop.table(table(bc$class)))
# Class proportions:
# B         M 
# 0.6500732 0.3499268 
# Class distribution:
# B  M 
# 444 239 


# counts + proportions
class_df <- bc |>
  dplyr::count(class) |>
  dplyr::mutate(proportion = n / sum(n))

# plot distribution
p1 <- ggplot(class_df, aes(x = class, y = n, fill = class)) +
  geom_col(width = 0.6, alpha = 0.8, color = "white") +
  scale_fill_manual(values = c("B" = "blue", "M" = "hotpink")) +
  labs(
    title = "Class Distribution: Benign vs Malignant",
    x = "Tumor Class",
    y = "Count"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

print(p1)

## 2. Define Features
# numeric features
features <- c(
  "clump_thickness",
  "uniformity_cell_size",
  "uniformity_cell_shape",
  "marginal_adhesion",
  "single_epithelial_cell_size",
  "bare_nuclei",
  "bland_chromatin",
  "normal_nucleoli",
  "mitoses"
)


## 3. Exploratory Data Analysis
# convert to long format 
bc_long <- bc |>
  mutate(class = factor(class, levels = c("B", "M"))) |>
  pivot_longer(
    cols = all_of(features),
    names_to  = "feature",
    values_to = "value"
  )

# histograms for each feature
ggplot(bc_long, aes(x = value, fill = class)) +
  geom_histogram(
    position = "identity",  
    alpha    = 0.5,
    bins     = 10,
    color    = "white"
  ) +
  facet_wrap(~ feature, scales = "free_y", ncol = 3) +
  scale_fill_manual(
    values = c("B" = "blue", "M" = "hotpink"),
    name   = "Tumor class",
    labels = c("Benign", "Malignant")
  ) +
  labs(
    title = "Distribution of Features by Tumor Class",
    x     = "Feature value",
    y     = "Count"
  ) +
  theme(
    plot.title      = element_text(size = 20),
    axis.title.x    = element_text(size = 15),
    axis.title.y    = element_text(size = 15),
    axis.text.x     = element_text(size = 10),
    axis.text.y     = element_text(size = 10),
    legend.title    = element_text(size = 15),
    legend.text     = element_text(size = 15),
    strip.text      = element_text(face = "bold", size = 15)
  )
# box plots of raw features
ggplot(bc_long, aes(x = class, y = value, fill = class)) +
  geom_boxplot(alpha = 0.7, outlier.color = "black", outlier.size = 2) +
  facet_wrap(~ feature, scales = "free_y", ncol = 3) +
  scale_fill_manual(
    values = c("B" = "blue", "M" = "hotpink"),
    name   = "Tumor class",
    labels = c("Benign", "Malignant")
  ) +
  labs(
    title    = "Feature Distributions by Class (Raw Values)",
    subtitle = "Box plots show median, quartiles, and outliers",
    x        = "Tumor class",
    y        = "Feature value"
  ) +
  theme(
    strip.text  = element_text(face = "bold", size = 11),
    axis.text.x = element_text(size  = 12, face = "bold")
  )


# Box plots of scaled features
# scaled copy to show standardized distributions
bc_scaled <- bc
bc_scaled[features] <- scale(bc[features])

bc_scaled_long <- bc_scaled |>
  mutate(class = factor(class, levels = c("B", "M"))) |>
  pivot_longer(
    cols = all_of(features),
    names_to  = "feature",
    values_to = "value"
  )

ggplot(bc_scaled_long, aes(x = class, y = value, fill = class)) +
  geom_boxplot(alpha = 0.7, outlier.color = "black", outlier.size = 2) +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
  facet_wrap(~ feature, ncol = 3) +
  scale_fill_manual(
    values = c("B" = "blue", "M" = "hotpink"),
    name   = "Tumor class",
    labels = c("Benign", "Malignant")
  ) +
  labs(
    title    = "Feature Distributions by Class (Scaled Values)",
    subtitle = "Standardized features (mean = 0, standard deviation = 1)",
    x        = "Tumor class",
    y        = "Standardized feature value"
  ) +
  theme(
    strip.text  = element_text(face = "bold", size = 11),
    axis.text.x = element_text(size  = 12, face = "bold")
  )


## 4. Correlation
# create a numeric outcome for correlation 0 for benign, 1 for malignant
bc$malignant <- ifelse(bc$class == "M", 1, 0)

# correlation of each feature
cor_with_outcome <- cor(bc[, features], bc$malignant)

# plot correlation
par(mfrow = c(1, 2))

barplot(
  sort(cor_with_outcome[, 1], decreasing = TRUE),
  las  = 2,
  main = "Feature Correlation with Malignancy",
  ylab = "Correlation",
  col  = colorRampPalette(c("lightpink", "hotpink", "black"))(length(features)),
  ylim = c(0, 0.85),
  cex.names = 0.9
)

corrplot(
  cor(bc[, features]),
  method = "circle",
  type   = "upper",
  tl.cex = 0.8,
  tl.col = "black",
  col    = colorRampPalette(c("white", "pink", "hotpink", "darkred"))(100),
  title  = "Feature Correlations",
  mar    = c(0, 0, 2, 0)
)

par(mfrow = c(1, 1))


# summary stats
summary_stats <- bc |>
  select(all_of(features), class) |>
  group_by(class) |>
  summarise(across(
    everything(),
    list(
      mean   = ~ mean(.),
      sd     = ~ sd(.),
      median = ~ median(.),
      min    = ~ min(.),
      max    = ~ max(.)
    ),
    .names = "{.col}_{.fn}"
  ))

# compare means for each feature 
mean_comparison <- bc |>
  select(all_of(features), class) |>
  group_by(class) |>
  summarise(across(everything(), mean)) |>
  pivot_longer(-class, names_to = "feature", values_to = "mean") |>
  pivot_wider(names_from = class, values_from = mean) |>
  mutate(
    Difference = M - B,
    Ratio      = M / B
  ) |>
  arrange(desc(Ratio))

print(mean_comparison)

"feature              B     M Difference Ratio
<chr>            <dbl> <dbl>      <dbl> <dbl>
  1 bare_nuclei       1.35  7.63       6.28  5.66
2 uniformity_cell…  1.31  6.58       5.27  5.04
3 normal_nucleoli   1.26  5.86       4.60  4.64
4 uniformity_cell…  1.41  6.56       5.15  4.64
5 marginal_adhesi…  1.35  5.59       4.24  4.15
6 bland_chromatin   2.08  5.97       3.89  2.87
7 single_epitheli…  2.11  5.33       3.22  2.53
8 mitoses           1.07  2.60       1.54  2.44
9 clump_thickness   2.96  7.19       4.22  2.43"


# Visualization with one predictor
# using bare_nuclei as a single predictor
ggplot(bc, aes(x = bare_nuclei, y = as.numeric(class) - 1)) +
  geom_jitter(aes(color = class), height = 0.05, alpha = 0.6, size = 2) +
  geom_smooth(
    method      = "glm",
    method.args = list(family = "binomial"),
    se          = TRUE,
    color       = "black",
    fill        = "gray80"
  ) +
  scale_color_manual(values = c("B" = "blue", "M" = "hotpink")) +
  labs(
    title    = "Bare Nuclei as Single Predictor of Malignancy",
    subtitle = "Logistic regression fit with 95 percent confidence band",
    x        = "Bare nuclei score",
    y        = "P(Malignant)"
  ) +
  theme(legend.position = "none")



## 5. Data preprocessing for modeling
# create dataset, remove sample_code, remove malignant 
bc_model <- bc |>
  select(-sample_code, -malignant)
bc$class <- factor(bc$class, levels = c("B", "M"))

bc_model <- bc |>
  select(-sample_code, -malignant) |>
  mutate(class = factor(class, levels = c("B", "M")))
# identify numeric columns
num_cols <- features

# scale predictors
bc_model[num_cols] <- scale(bc_model[num_cols])

# train and test split 80% train, 20% test
set.seed(42)
n         <- nrow(bc_model)
train_idx <- sample(1:n, size = floor(0.8 * n))

bc_train <- bc_model[train_idx, ]
bc_test  <- bc_model[-train_idx, ]

cat("Training set:", dim(bc_train), "\n")
cat("Test set:", dim(bc_test), "\n")
# Training set: 546 10 
# Test set: 137 10 


## 6. Feature selection
# combiniing correlation, selection decision, and medical importance
feature_importance <- data.frame(
  Feature         = features,
  Correlation     = round(cor_with_outcome[, 1], 3),
  Selected_Normal = features %in% c(
    "clump_thickness",
    "bare_nuclei",
    "uniformity_cell_shape",
    "marginal_adhesion",
    "normal_nucleoli"
  ),
  Medical_Importance = c(
    "Cluster thickness and density",
    "Irregular nuclei indicate malignancy",
    "Shape irregularity is a strong signal",
    "Adhesion relates to spread potential",
    "Nucleoli changes relate to activity",
    "Exposed nuclei suggest cancer",
    "Chromatin texture changes in cancer",
    "Nucleoli prominence reflects activity",
    "Mitotic rate reflects growth"
  )
)

print(feature_importance[order(-feature_importance$Correlation), ])
"bare_nuclei                         Exposed nuclei suggest cancer
uniformity_cell_shape       Shape irregularity is a strong signal
uniformity_cell_size         Irregular nuclei indicate malignancy
bland_chromatin               Chromatin texture changes in cancer
normal_nucleoli             Nucleoli prominence reflects activity
clump_thickness                     Cluster thickness and density
marginal_adhesion            Adhesion relates to spread potential
single_epithelial_cell_size   Nucleoli changes relate to activity
mitoses                              Mitotic rate reflects growth"



## 7. Baseline logistic regression 
# GLM using the selected five features
glm_fit <- glm(
  class ~ clump_thickness + bare_nuclei + uniformity_cell_shape +
    marginal_adhesion + normal_nucleoli,
  data   = bc_train,
  family = binomial(link = "logit")
)

summary(glm_fit)
"Coefficients:
  Estimate Std. Error z value Pr(>|z|)    
(Intercept)            -1.2571     0.3353  -3.749 0.000178 ***
  clump_thickness         1.5159     0.4094   3.703 0.000213 ***
  bare_nuclei             1.4690     0.3609   4.071 4.68e-05 ***
  uniformity_cell_shape   2.3445     0.6819   3.438 0.000586 ***
  marginal_adhesion       0.8447     0.3392   2.490 0.012759 *  
  normal_nucleoli         0.8504     0.3741   2.273 0.023022 *  
  ---
  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

Null deviance: 703.096  on 545  degrees of freedom
Residual deviance:  80.437  on 540  degrees of freedom
AIC: 92.437

Number of Fisher Scoring iterations: 8"

# predict on the test set
glm_probs <- predict(glm_fit, newdata = bc_test, type = "response")
glm_pred  <- ifelse(glm_probs > 0.5, "M", "B")
glm_pred  <- factor(glm_pred, levels = levels(bc_test$class))

glm_acc <- mean(glm_pred == bc_test$class)

cat("\nBaseline GLM Performance:\n")
cat("Accuracy:", round(glm_acc, 3), "\n\n")
print(table(True = bc_test$class, Pred = glm_pred))

"Baseline GLM Performance:
Accuracy: 0.949 
Pred
True  B  M
B 84  2
M  5 46"


## 8. Bayesian model with one predictor
# using bare_nuclei as a predictor
simple_bayes <- brm(
  class ~ bare_nuclei,
  family = bernoulli("logit"),
  data   = bc_train,
  prior  = c(
    # Intercept prior
    # is a wide Student t around zero
    set_prior("normal(0, 5)", class = "Intercept"),
    # Slope prior
    # Positive mean since malignant tends to have higher bare_nuclei
    set_prior("normal(1, 2)", class = "b")
  ),
  iter   = 2000,
  warmup = 500,
  chains = 4,
  seed   = 123
)

summary(simple_bayes)
"Family: bernoulli 
Links: mu = logit 
Formula: class ~ bare_nuclei 
Data: bc_train (Number of observations: 546) 
Draws: 4 chains, each with iter = 2000; warmup = 500; thin = 1;
total post-warmup draws = 6000

Regression Coefficients:
  Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
Intercept      -0.61      0.18    -0.96    -0.25 1.00     3637     3575
bare_nuclei     3.08      0.27     2.58     3.66 1.00     3227     2927

Draws were sampled using sampling(NUTS). For each parameter, Bulk_ESS
and Tail_ESS are effective sample size measures, and Rhat is the potential
scale reduction factor on split chains (at convergence, Rhat = 1)."

posterior_simple <- as.data.frame(simple_bayes)

hist(
  posterior_simple$b_bare_nuclei,
  col    = "hotpink",
  main   = "Posterior Distribution: Bare Nuclei Effect",
  xlab   = "Coefficient",
  breaks = 30
)
abline(v = 0, col = "red", lty = 2, lwd = 2)



## 9. Bayesian model with normal priors
# uses five predictors with informative Normal priors based on correlation and medical information

manual_prior_normal <- c(
  # Intercept is a Student t prior
  set_prior("student_t(4, 0, 5)", class = "Intercept"),
  # slopes for the five selected features
  # stronger prior on bare_nuclei and clump_thickness because they correlate more with malignant tumors
  set_prior("normal(1.5, 1)",  class = "b", coef = "bare_nuclei"),
  set_prior("normal(1, 1)",    class = "b", coef = "clump_thickness"),
  set_prior("normal(1, 1)",    class = "b", coef = "uniformity_cell_shape"),
  set_prior("normal(0.5, 1.5)", class = "b", coef = "marginal_adhesion"),
  set_prior("normal(0.5, 1.5)", class = "b", coef = "normal_nucleoli")
)

bc_brm_normal <- brm(
  class ~ clump_thickness + bare_nuclei + uniformity_cell_shape +
    marginal_adhesion + normal_nucleoli,
  family  = bernoulli("logit"),
  data    = bc_train,
  # more iterations and warmup for a more stable posterior
  iter    = 4000,
  warmup  = 1000,
  chains  = 4,
  # higher adapt_delta to reduce divergences
  control = list(adapt_delta = 0.99),  
  prior   = manual_prior_normal,
  seed    = 123
)

summary(bc_brm_normal)
"Family: bernoulli 
Links: mu = logit 
Formula: class ~ clump_thickness + bare_nuclei + uniformity_cell_shape + marginal_adhesion + normal_nucleoli 
Data: bc_train (Number of observations: 546) 
Draws: 4 chains, each with iter = 4000; warmup = 1000; thin = 1;
total post-warmup draws = 12000

Regression Coefficients:
  Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
Intercept                -1.24      0.33    -1.91    -0.62 1.00    10703     8123
clump_thickness           1.56      0.38     0.86     2.33 1.00    10008     8420
bare_nuclei               1.60      0.34     0.96     2.30 1.00    10936     8953
uniformity_cell_shape     2.05      0.54     1.02     3.13 1.00     9305     7674
marginal_adhesion         0.90      0.34     0.25     1.60 1.00    11546     8192
normal_nucleoli           0.98      0.37     0.29     1.73 1.00    11146     7447

Draws were sampled using sampling(NUTS). For each parameter, Bulk_ESS
and Tail_ESS are effective sample size measures, and Rhat is the potential
scale reduction factor on split chains (at convergence, Rhat = 1)."



## 10. Trying horseshoe prior model
# using all nine features and a horseshoe prior on b
bc_brm_horseshoe <- brm(
  class ~ clump_thickness + uniformity_cell_size + uniformity_cell_shape +
    marginal_adhesion + single_epithelial_cell_size + bare_nuclei +
    bland_chromatin + normal_nucleoli + mitoses,
  family = bernoulli("logit"),
  data   = bc_train,
  prior  = c(
    set_prior("normal(-1, 2)", class = "Intercept"),
    # horseshoe prior for regression coefficients
    set_prior("horseshoe(df = 1, scale_global = 0.05)", class = "b")
  ),
  chains  = 4,
  iter    = 4000,
  warmup  = 2000,
  control = list(adapt_delta = 0.95),
  seed    = 123
)

summary(bc_brm_horseshoe)
" Family: bernoulli 
  Links: mu = logit 
Formula: class ~ clump_thickness + uniformity_cell_size + uniformity_cell_shape + marginal_adhesion + single_epithelial_cell_size + bare_nuclei + bland_chromatin + normal_nucleoli + mitoses 
   Data: bc_train (Number of observations: 546) 
  Draws: 4 chains, each with iter = 4000; warmup = 2000; thin = 1;
         total post-warmup draws = 8000

Regression Coefficients:
                            Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
Intercept                      -1.10      0.36    -1.83    -0.43 1.00     7260     5281
clump_thickness                 1.35      0.42     0.57     2.20 1.00     8345     6789
uniformity_cell_size            0.74      0.70    -0.33     2.31 1.00     5993     6384
uniformity_cell_shape           1.43      0.78    -0.01     2.95 1.00     6483     4058
marginal_adhesion               0.60      0.39    -0.07     1.39 1.00     6513     3896
single_epithelial_cell_size     0.33      0.38    -0.29     1.13 1.00     7473     7692
bare_nuclei                     1.35      0.38     0.63     2.11 1.00     8922     6540
bland_chromatin                 0.54      0.45    -0.19     1.50 1.00     6488     5863
normal_nucleoli                 0.59      0.39    -0.07     1.38 1.00     6461     3817
mitoses                         0.43      0.47    -0.28     1.51 1.00     7552     6977

Draws were sampled using sampling(NUTS). For each parameter, Bulk_ESS
and Tail_ESS are effective sample size measures, and Rhat is the potential
scale reduction factor on split chains (at convergence, Rhat = 1)."

## 11. Diagnostics for main Bayesian model
# visual diagnostics
plot(bc_brm_normal, N = 6)

# bumerical diagnostics, Rhat values 
model_summary <- summary(bc_brm_normal)
rhats <- model_summary$fixed[, "Rhat"]

cat("\nRhat values (should be close to 1.00):\n")
print(rhats)
# Rhat values (should be close to 1.00):
#[1] 1.0008953 1.0002715 1.0003332 0.9999807 1.0008690 1.0002429  


## 12. Feature effect viz
# conditional effects show how each predictor affects P(malignant)
par(mfrow = c(2, 3))
plot(conditional_effects(bc_brm_normal, "clump_thickness"))
plot(conditional_effects(bc_brm_normal, "bare_nuclei"))
plot(conditional_effects(bc_brm_normal, "uniformity_cell_shape"))
plot(conditional_effects(bc_brm_normal, "marginal_adhesion"))
plot(conditional_effects(bc_brm_normal, "normal_nucleoli"))
plot(conditional_effects(bc_brm_normal, "bare_nuclei:uniformity_cell_shape"))
par(mfrow = c(1, 1))

# extract posterior draws for coefficients
post_draws <- as.data.frame(bc_brm_normal)

coef_names <- c(
  "clump_thickness",
  "bare_nuclei",
  "uniformity_cell_shape",
  "marginal_adhesion",
  "normal_nucleoli"
)

effects <- data.frame(
  Feature = coef_names,
  Mean    = sapply(paste0("b_", coef_names), function(x) mean(post_draws[[x]])),
  SD      = sapply(paste0("b_", coef_names), function(x) sd(post_draws[[x]])),
  Prob_Positive = sapply(
    paste0("b_", coef_names),
    function(x) mean(post_draws[[x]] > 0)
  )
) |>
  arrange(desc(Mean))

print(effects)
"                                      Feature      Mean        SD Prob_Positive
b_uniformity_cell_shape uniformity_cell_shape 2.0471050 0.5405022     1.0000000
b_bare_nuclei                     bare_nuclei 1.6042420 0.3447258     1.0000000
b_clump_thickness             clump_thickness 1.5623167 0.3757770     1.0000000
b_normal_nucleoli             normal_nucleoli 0.9757299 0.3699164     0.9983333
b_marginal_adhesion         marginal_adhesion 0.9011194 0.3420305     0.9972500"

# plot posterior means with around 95 percent of credible intervals
ggplot(effects, aes(x = reorder(Feature, Mean), y = Mean)) +
  geom_point(size = 5, color = "hotpink") +
  geom_errorbar(
    aes(ymin = Mean - 2 * SD, ymax = Mean + 2 * SD),
    width = 0.2,
    color = "hotpink",
    size  = 1
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
  coord_flip() +
  labs(
    title    = "Feature Effects on Cancer Risk",
    subtitle = "Posterior means with approximate 95 percent credible intervals",
    x        = "",
    y        = "Standardized effect"
  ) +
  theme(axis.text.y = element_text(size = 12))



# 13. Posterior predictions
# posterior predictive probabilities on test data
post_prob_normal    <- posterior_epred(bc_brm_normal,    newdata = bc_test)
post_prob_horseshoe <- posterior_epred(bc_brm_horseshoe, newdata = bc_test)

# looking at the first patient as a concrete example
patient1_prob_normal    <- post_prob_normal[, 1]
patient1_prob_horseshoe <- post_prob_horseshoe[, 1]

par(mfrow = c(1, 2))
hist(
  patient1_prob_normal,
  col    = "lightpink",
  breaks = 30,
  main   = "Patient 1: Normal Prior",
  xlab   = "P(Malignant)"
)
abline(v = mean(patient1_prob_normal), col = "red", lwd = 2)

hist(
  patient1_prob_horseshoe,
  col    = "hotpink",
  breaks = 30,
  main   = "Patient 1: Horseshoe Prior",
  xlab   = "P(Malignant)"
)
abline(v = mean(patient1_prob_horseshoe), col = "red", lwd = 2)
par(mfrow = c(1, 1))

cat("Patient 1 True class:", as.character(bc_test$class[1]), "\n")
# Patient 1 True class: B 
cat("Normal prior Mean P(Malgnant):", round(mean(patient1_prob_normal), 3), "\n")
# Normal prior Mean P(Malgnant): 0.01 
cat("Horseshoe prior Mean P(Malignant):", round(mean(patient1_prob_horseshoe), 3), "\n")
# Horseshoe prior Mean P(Malignant): 0.013 

# summaries for all patients under the normal prior model
prob_mean <- colMeans(post_prob_normal)
prob_ci   <- apply(post_prob_normal, 2, quantile, probs = c(0.025, 0.975))

bc_test_pred <- bc_test |>
  mutate(
    prob_mean         = prob_mean,
    prob_low          = prob_ci[1, ],
    prob_high         = prob_ci[2, ],
    uncertainty_width = prob_high - prob_low
  )



## 14. Uncertainty & entropy
# the entropy function for Bernoulli predictive probability
# low entropy = high confidence, high entropy = near 0.5 probability
entropy_fun <- function(p) {
  p <- pmin(pmax(p, 1e-6), 1 - 1e-6)
  -(p * log(p) + (1 - p) * log(1 - p))
}

bc_test_pred <- bc_test_pred |>
  mutate(
    entropy = entropy_fun(prob_mean),
    # uncertainty aware classification:
    # label as malignant if p > 0.7
    # label as benign if p < 0.3
    # otherwise call it uncertain and defer
    pred_class = case_when(
      prob_mean > 0.7 ~ "M",
      prob_mean < 0.3 ~ "B",
      TRUE            ~ "uncertain"
    )
  )

# visualize distribution of predicted probabilities and entropy
par(mfrow = c(1, 2))

hist(
  bc_test_pred$prob_mean,
  breaks = 30,
  col    = "pink",
  main   = "Distribution of Predicted Probabilities",
  xlab   = "P(Malignant)"
)
abline(v = c(0.3, 0.7), col = "red", lty = 2, lwd = 2)

plot(
  bc_test_pred$prob_mean,
  bc_test_pred$entropy,
  col  = ifelse(bc_test_pred$class == "M", "#C2185B", "#F9A1C2"),
  pch  = 19,
  cex  = 1.2,
  main = "Prediction Entropy vs Probability",
  xlab = "P(Malignant)",
  ylab = "Entropy"
)

par(mfrow = c(1, 1))

# evaluate only the confident subset
confident <- bc_test_pred |>
  filter(pred_class != "uncertain") |>
  mutate(pred_class = factor(pred_class, levels = levels(class)))

cat("\nUncertainty aware classification summary:\n")
cat("Total predictions:", nrow(bc_test_pred), "\n")
cat(
  "Confident predictions:",
  nrow(confident), "(",
  round(100 * nrow(confident) / nrow(bc_test_pred), 1),
  "%)\n"
)
cat(
  "Uncertain predictions:",
  sum(bc_test_pred$pred_class == "uncertain"),
  "\n"
)
cat(
  "Accuracy on confident subset:",
  round(mean(confident$pred_class == confident$class), 3),
  "\n"
)
"Uncertainty aware classification summary:
Total predictions: 137 
Confident predictions: 133 ( 97.1 %)
Uncertain predictions: 4 
Accuracy on confident subset: 0.962 "


# 15. Model eval
# ROC curves for: GLM, Bayesian normal prior model, Bayesian horseshoe prior model

roc_glm       <- roc(bc_test$class, glm_probs, levels = c("B", "M"))
roc_normal    <- roc(bc_test$class, prob_mean, levels = c("B", "M"))
roc_horseshoe <- roc(
  bc_test$class,
  colMeans(post_prob_horseshoe),
  levels = c("B", "M")
)

plot(roc_glm, col = "gray50", lwd = 2, main = "ROC Curve Comparison")
plot(roc_normal, add = TRUE, col = "hotpink", lwd = 2)
plot(roc_horseshoe, add = TRUE, col = "darkred", lwd = 2)

legend(
  "bottomright",
  c(
    paste("GLM (AUC:", round(auc(roc_glm), 3), ")"),
    paste("Bayes Normal (AUC:", round(auc(roc_normal), 3), ")"),
    paste("Bayes Horseshoe (AUC:", round(auc(roc_horseshoe), 3), ")")
  ),
  col = c("gray50", "hotpink", "darkred"),
  lwd = 2
)

# cost based threshold selection: Assuming a false negative is 10 times as costly as a false positive
costs      <- numeric(99)
thresholds <- seq(0.01, 0.99, 0.01)

for (i in 1:99) {
  pred <- ifelse(prob_mean > thresholds[i], "M", "B")
  pred <- factor(pred, levels = c("B", "M"))
  conf_mat <- table(True = bc_test$class, Pred = pred)
  
  # conf_mat["B","M"] is false positive count
  # conf_mat["M","B"] is false negative count
  costs[i] <- conf_mat["B", "M"] * 1 + conf_mat["M", "B"] * 10
}

optimal_threshold <- thresholds[which.min(costs)]

plot(
  thresholds,
  costs,
  type = "l",
  col  = "hotpink",
  lwd  = 3,
  main = "Cost Sensitive Threshold Selection",
  xlab = "Classification threshold",
  ylab = "Total cost (FN cost = 10 x FP cost)"
)
abline(v = optimal_threshold, col = "red",  lty = 2, lwd = 2)
abline(v = 0.5,              col = "blue", lty = 2, lwd = 2)

legend(
  "topright",
  c("Optimal threshold", "Default (0.5)"),
  col = c("red", "blue"),
  lty = 2,
  lwd = 2
)



# 16. Final model performance comparison
# helper to compute metrics
calc_metrics <- function(probs, true_class, threshold = 0.5) {
  pred <- factor(ifelse(probs > threshold, "M", "B"), levels = c("B", "M"))
  cm   <- table(True = true_class, Pred = pred)
  
  list(
    Accuracy    = mean(pred == true_class),
    Sensitivity = cm["M", "M"] / sum(cm["M", ]),  # recall for malignant
    Specificity = cm["B", "B"] / sum(cm["B", ]),  # recall for benign
    PPV         = cm["M", "M"] / sum(cm[, "M"]),  # precision for malignant
    NPV         = cm["B", "B"] / sum(cm[, "B"])   # precision for benign
  )
}

# metrics with default threshold 0.5
metrics_glm        <- calc_metrics(glm_probs, bc_test$class)
metrics_normal     <- calc_metrics(prob_mean, bc_test$class)
metrics_horseshoe  <- calc_metrics(colMeans(post_prob_horseshoe), bc_test$class)

# metrics at cost based optimal threshold
metrics_optimal    <- calc_metrics(prob_mean, bc_test$class, optimal_threshold)

results <- bind_rows(
  data.frame(Model = "GLM",                     metrics_glm),
  data.frame(Model = "Bayes Normal",            metrics_normal),
  data.frame(Model = "Bayes Horseshoe",         metrics_horseshoe),
  data.frame(Model = "Bayes Optimal Threshold", metrics_optimal)
)

print(results)

"                    Model  Accuracy Sensitivity Specificity       PPV       NPV
1                     GLM 0.9489051   0.9019608   0.9767442 0.9583333 0.9438202
2            Bayes Normal 0.9489051   0.9019608   0.9767442 0.9583333 0.9438202
3         Bayes Horseshoe 0.9562044   0.9215686   0.9767442 0.9591837 0.9545455
4 Bayes Optimal Threshold 0.9635036   1.0000000   0.9418605 0.9107143 1.0000000
"


# 17. High level summary
cat("Clinical style summary\n\n")
cat("1. Strongest predictors:\n")
cat("   Bare nuclei (correlation =", round(cor_with_outcome["bare_nuclei", ], 3), ")\n")
#Bare nuclei (correlation = 0.823 )
cat("   Cell shape uniformity (correlation =", round(cor_with_outcome["uniformity_cell_shape", ], 3), ")\n")
# Cell shape uniformity (correlation = 0.822 )
cat("   Cell size uniformity (correlation =", round(cor_with_outcome["uniformity_cell_size", ], 3), ")\n\n")
#Cell size uniformity (correlation = 0.821 )

cat("2. Model performance overview:\n")
cat("   Best AUC among Bayesian models:",
    round(max(auc(roc_normal), auc(roc_horseshoe)), 3),
    "\n"
)
# Best AUC among Bayesian models: 0.992 
cat("   Cost sensitive optimal threshold for P(Malignant):",
    round(optimal_threshold, 3),
    "\n"
)
# Cost sensitive optimal threshold for P(Malignant): 0.08 
cat("   Number of uncertain cases that could be flagged for extra screening:",
    sum(bc_test_pred$pred_class == "uncertain"),
    "\n\n"
)
# Number of uncertain cases that could be flagged for extra screening: 4

# 18. Save fitted models
saveRDS(
  bc_brm_normal,
  "/Users/ellawiser/Desktop/DS-4420-final-project/Data/bc_brm_normal_final.rds"
)

saveRDS(
  bc_brm_horseshoe,
  "/Users/ellawiser/Desktop/DS-4420-final-project/Data/bc_brm_horseshoe_final.rds"
)


