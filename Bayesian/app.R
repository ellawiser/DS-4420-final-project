# app.R

library(shiny)
library(tidyverse)
library(brms)
library(bslib)

options(mc.cores = parallel::detectCores())

# -------------------------------------------------
# Data loading and preprocessing
# -------------------------------------------------
bc <- read.csv("/Users/ellawiser/Desktop/DS-4420-final-project/Data/bayes_breast_cancer_clean.csv")

# class as factor, B as reference
bc$class <- factor(bc$class, levels = c("B", "M"))

# Keep clean version for modeling
bc_model <- bc |>
  select(-sample_code)

num_cols <- colnames(bc_model)[colnames(bc_model) != "class"]

# scale numeric predictors, store center and scale
bc_model[num_cols] <- scale(bc_model[num_cols])
scale_center <- attr(bc_model[num_cols], "scaled:center")
scale_scale  <- attr(bc_model[num_cols], "scaled:scale")

set.seed(42)
n <- nrow(bc_model)
train_idx <- sample(1:n, size = floor(0.8 * n))
bc_train <- bc_model[train_idx, ]
bc_test  <- bc_model[-train_idx, ]

# -------------------------------------------------
# Bayesian logistic regression model
# -------------------------------------------------
manual_prior_bc <- c(
  set_prior("normal(0, 2)", class = "b")
)

bc_brm_logit <- brm(
  class ~ clump_thickness + bare_nuclei + uniformity_cell_shape +
    marginal_adhesion + normal_nucleoli,
  family = bernoulli("logit"),
  data   = bc_train,
  iter   = 6000,
  warmup = 2000,
  chains = 4,
  control = list(adapt_delta = 0.99, max_treedepth = 15),
  prior  = manual_prior_bc,
  refresh = 0
)

# -------------------------------------------------
# Posterior predictive on test set and entropy
# -------------------------------------------------
entropy_fun <- function(p) {
  p <- pmin(pmax(p, 1e-6), 1 - 1e-6)
  -(p * log(p) + (1 - p) * log(1 - p))
}

post_prob_test <- posterior_epred(bc_brm_logit, newdata = bc_test)
prob_mean_test <- colMeans(post_prob_test)
entropy_test   <- entropy_fun(prob_mean_test)

bc_test_pred <- bc_test |>
  mutate(
    prob_mean = prob_mean_test,
    entropy   = entropy_test
  )

# -------------------------------------------------
# Long-format data for histograms
# -------------------------------------------------
bc_long <- bc |>
  mutate(class = factor(class, levels = c("B", "M"))) |>
  pivot_longer(
    cols = c(
      clump_thickness,
      uniformity_cell_size,
      uniformity_cell_shape,
      marginal_adhesion,
      single_epithelial_cell_size,
      bare_nuclei,
      bland_chromatin,
      normal_nucleoli,
      mitoses
    ),
    names_to  = "feature",
    values_to = "value"
  )

# features used in the Bayesian model
selected_features <- c(
  "clump_thickness",
  "bare_nuclei",
  "uniformity_cell_shape",
  "marginal_adhesion",
  "normal_nucleoli"
)

# -------------------------------------------------
# UI
# -------------------------------------------------
ui <- fluidPage(
  theme = bs_theme(
    version = 5,
    bootswatch = "flatly",
    base_font = font_google("Nunito")
  ),
  tags$head(
    tags$style(HTML("
      body {
        background-color: #ffeef6;
      }
      .card {
        border-radius: 16px;
        border: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
      }
      .card-header {
        background-color: #fbd1e5;
        border-bottom: none;
        font-weight: 700;
      }
      .btn-pink {
        background-color: #f48fb1;
        border-color: #f48fb1;
        color: white;
      }
      .btn-pink:hover {
        background-color: #ec407a;
        border-color: #ec407a;
      }
      .nav-pills .nav-link.active,
      .nav-tabs .nav-link.active {
        background-color: #f48fb1;
        color: white !important;
      }
    "))
  ),
  
  titlePanel("Bayesian Breast Cancer Explorer"),
  
  tabsetPanel(
    id = "main_tabs",
    
    # ---------------- Tab 1: Data and Background ----------------
    tabPanel(
      "Data and Background",
      br(),
      fluidRow(
        column(
          width = 12,
          div(
            class = "card",
            div(class = "card-header", "Dataset Overview"),
            div(
              class = "card-body",
              p("This app uses the Wisconsin Breast Cancer dataset. 
                Each row is a tumor sample with features scored from 1 to 10 
                and a class label: Benign (B) or Malignant (M)."),
              tableOutput("data_head")
            )
          )
        )
      ),
      br(),
      fluidRow(
        column(
          width = 12,
          div(
            class = "card",
            div(class = "card-header", "Feature Distributions by Tumor Class"),
            div(
              class = "card-body",
              p("These histograms show how each feature is distributed 
                for benign and malignant tumors. Light pink bars are benign, 
                darker pink bars are malignant."),
              plotOutput("feature_hist", height = "700px")
            )
          )
        )
      )
    ),
    
    # ---------------- Tab 2: Bayesian Predictor ----------------
    tabPanel(
      "Bayesian Breast Cancer Predictor",
      br(),
      fluidRow(
        # Input side
        column(
          width = 4,
          div(
            class = "card",
            div(class = "card-header", "Input Tumor Features"),
            div(
              class = "card-body",
              sliderInput(
                "clump_thickness", "Clump thickness", 
                min = 1, max = 10, value = 5, step = 1
              ),
              sliderInput(
                "bare_nuclei", "Bare nuclei", 
                min = 1, max = 10, value = 3, step = 1
              ),
              sliderInput(
                "uniformity_cell_shape", "Uniformity of cell shape", 
                min = 1, max = 10, value = 3, step = 1
              ),
              sliderInput(
                "marginal_adhesion", "Marginal adhesion", 
                min = 1, max = 10, value = 2, step = 1
              ),
              sliderInput(
                "normal_nucleoli", "Normal nucleoli", 
                min = 1, max = 10, value = 2, step = 1
              ),
              actionButton("predict_btn", "Predict", class = "btn btn-pink btn-lg mt-3")
            )
          )
        ),
        
        # Output side
        column(
          width = 8,
          div(
            class = "card",
            div(class = "card-header", "Prediction Output"),
            div(
              class = "card-body",
              h4("Predicted class"),
              verbatimTextOutput("predicted_class_text"),
              h4("Probability of malignancy"),
              verbatimTextOutput("posterior_prob_text"),
              h4("Predictive entropy"),
              verbatimTextOutput("entropy_text"),
              br(),
              h5("Posterior distribution of P(malignant) for this tumor"),
              plotOutput("post_prob_hist", height = "300px"),
              br(),
              h5("Where this tumor sits in the overall entropy distribution"),
              plotOutput("entropy_hist", height = "300px")
            )
          )
        )
      ),
      br(),
      fluidRow(
        column(
          width = 12,
          div(
            class = "card",
            div(class = "card-header", "Histograms of model features"),
            div(
              class = "card-body",
              p("These histograms focus on the features used in the Bayesian model. 
                They help you see how the model inputs differ for benign versus malignant tumors."),
              plotOutput("selected_feature_hists", height = "500px")
            )
          )
        )
      )
    ),
    
    # ---------------- Tab 3: Findings ----------------
    tabPanel(
      "Findings",
      br(),
      fluidRow(
        column(
          width = 12,
          div(
            class = "card",
            div(class = "card-header", "Summary of Findings"),
            div(
              class = "card-body",
              h4("Key Takeaways"),
              p("Here you can summarize:"),
              tags$ul(
                tags$li("Overall accuracy of logistic regression versus the Bayesian model"),
                tags$li("How often the Bayesian model abstains as uncertain"),
                tags$li("Which features matter most for predicting malignancy"),
                tags$li("How predictive entropy highlights borderline cases")
              ),
              p("You can paste your final writeup from the report in this section.")
            )
          )
        )
      )
    )
  )
)

# -------------------------------------------------
# Server
# -------------------------------------------------
server <- function(input, output, session) {
  
  # ---------------- Data tab outputs ----------------
  
  output$data_head <- renderTable({
    head(bc, 8)
  })
  
  # All feature histograms by class
  output$feature_hist <- renderPlot({
    ggplot(bc_long, aes(x = value, fill = class)) +
      geom_histogram(
        position = "identity",
        alpha = 0.55,
        bins = 15,
        color = "white"
      ) +
      facet_wrap(~ feature, scales = "free_x", ncol = 3) +
      scale_fill_manual(
        values = c("B" = "#F7A6C1", "M" = "#C2185B"),
        name = "Tumor class",
        labels = c("Benign", "Malignant")
      ) +
      labs(
        x = NULL,
        y = "Count"
      ) +
      theme_minimal(base_size = 14) +
      theme(
        strip.text = element_text(size = 13, face = "bold"),
        axis.text  = element_text(size = 11),
        legend.position = "bottom",
        panel.spacing = unit(1, "lines")
      )
  })
  
  # ---------------- Predictor tab logic ----------------
  
  # Build scaled newdata based on input sliders
  new_patient_scaled <- eventReactive(input$predict_btn, {
    # construct raw row matching bc_model columns, fill non modeled features with means
    x_raw <- tibble(
      clump_thickness             = input$clump_thickness,
      uniformity_cell_size        = mean(bc$uniformity_cell_size),
      uniformity_cell_shape       = input$uniformity_cell_shape,
      marginal_adhesion           = input$marginal_adhesion,
      single_epithelial_cell_size = mean(bc$single_epithelial_cell_size),
      bare_nuclei                 = input$bare_nuclei,
      bland_chromatin             = mean(bc$bland_chromatin),
      normal_nucleoli             = input$normal_nucleoli,
      mitoses                     = mean(bc$mitoses)
    )
    
    # scale using training means and sds
    x_scaled <- x_raw
    x_scaled[num_cols] <- sweep(
      sweep(as.matrix(x_raw[num_cols]), 2, scale_center, "-"),
      2, scale_scale, "/"
    )
    
    as.data.frame(x_scaled)
  })
  
  # Posterior draws of P(M) for this tumor as a numeric vector
  pred_draws <- reactive({
    req(new_patient_scaled())
    draws_mat <- posterior_epred(bc_brm_logit, newdata = new_patient_scaled())
    as.numeric(draws_mat)
  })
  
  output$predicted_class_text <- renderText({
    req(pred_draws())
    p_hat <- mean(pred_draws())
    if (p_hat > 0.5) {
      "Malignant"
    } else {
      "Benign"
    }
  })
  
  output$posterior_prob_text <- renderText({
    req(pred_draws())
    p_hat <- mean(pred_draws())
    paste0(round(p_hat * 100, 1), "% chance of malignancy")
  })
  
  output$entropy_text <- renderText({
    req(pred_draws())
    p_hat <- mean(pred_draws())
    h     <- entropy_fun(p_hat)
    paste0("Entropy: ", round(h, 3), 
           " (lower values mean a more confident prediction)")
  })
  
  # Histogram of posterior P(M) for this tumor
  output$post_prob_hist <- renderPlot({
    req(pred_draws())
    probs <- pred_draws()
    mean_p <- mean(probs)
    ci95   <- quantile(probs, c(0.025, 0.975))
    
    df_plot <- data.frame(p_malignant = probs)
    
    ggplot(df_plot, aes(x = p_malignant)) +
      geom_histogram(
        bins  = 30,
        fill  = "#ffb3d9",
        color = "white",
        alpha = 0.9
      ) +
      geom_vline(
        xintercept = mean_p,
        color = "#0033cc",
        linewidth = 1.2
      ) +
      geom_vline(
        xintercept = ci95,
        color = "#ff3366",
        linetype = "dashed",
        linewidth = 1
      ) +
      labs(
        x = "Posterior samples of P(malignant)",
        y = "Frequency"
      ) +
      theme_minimal(base_size = 13)
  })
  
  # Entropy histogram with a line for the current tumor
  output$entropy_hist <- renderPlot({
    req(pred_draws())
    p_hat <- mean(pred_draws())
    h_new <- entropy_fun(p_hat)
    
    ggplot(
      data.frame(entropy = entropy_test),
      aes(x = entropy)
    ) +
      geom_histogram(
        bins = 30,
        fill = "#F7A6C1",
        color = "white",
        alpha = 0.8
      ) +
      geom_vline(xintercept = h_new, color = "#C2185B", linewidth = 1.2) +
      labs(
        x = "Predictive entropy for test patients",
        y = "Count"
      ) +
      theme_minimal(base_size = 13)
  })
  
  # Histograms of selected model features, layered by class
  output$selected_feature_hists <- renderPlot({
    bc_long_selected <- bc_long |>
      filter(feature %in% selected_features)
    
    ggplot(bc_long_selected, aes(x = value, fill = class)) +
      geom_histogram(
        position = "identity",
        alpha = 0.55,
        bins = 15,
        color = "white"
      ) +
      facet_wrap(~ feature, scales = "free_x") +
      scale_fill_manual(
        values = c("B" = "#F7A6C1", "M" = "#C2185B"),
        name = "Tumor class",
        labels = c("Benign", "Malignant")
      ) +
      labs(
        x = NULL,
        y = "Count"
      ) +
      theme_minimal(base_size = 14) +
      theme(
        strip.text = element_text(size = 13, face = "bold"),
        axis.text  = element_text(size = 11),
        legend.position = "bottom"
      )
  })
}

shinyApp(ui, server)