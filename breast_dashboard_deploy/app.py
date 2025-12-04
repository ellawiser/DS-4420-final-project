from pathlib import Path
from shiny import App, ui, render

OUT_DIR = Path("Bayesian/outputs")


# helper wrapper for clearly separated chart sections
def chart_block(title, output_id, description):
    return ui.div(
        # separator and space from the previous block
        ui.hr(style="margin-top:60px; margin-bottom:30px;"),

        ui.h3(title, style="margin-bottom:20px; text-align:center;"),

        # Image container with max height constraint
        ui.div(
            ui.output_image(output_id),
            style="text-align:center; max-height:400px; overflow:hidden;",
        ),

        ui.p(description, style="margin-top:20px; text-align:center;"),

        # overall spacing after this whole section
        style="text-align:center; margin-bottom:60px;",
    )

def two_chart_row(block1, block2):
    return ui.div(
        ui.div(
            block1,
            style=(
                "flex:1; padding:10px;"
            ),
        ),
        ui.div(
            block2,
            style=(
                "flex:1; padding:10px;"
            ),
        ),
        style=(
            "display:flex;"
            "flex-direction:row;"
            "flex-wrap:wrap;"
            "gap:20px;"
            "align-items:flex-start;"
            "justify-content:center;"
            "margin-top:30px;"
            "margin-bottom:30px;"
        ),
    )

# Three chart row helper
def three_chart_row(block1, block2, block3):
    return ui.div(
        ui.div(
            block1,
            style="flex:1; padding:10px;",
        ),
        ui.div(
            block2,
            style="flex:1; padding:10px;",
        ),
        ui.div(
            block3,
            style="flex:1; padding:10px;",
        ),
        style=(
            "display:flex;"
            "flex-direction:row;"
            "flex-wrap:wrap;"
            "gap:20px;"
            "align-items:flex-start;"
            "justify-content:center;"
            "margin-top:30px;"
            "margin-bottom:30px;"
        ),
    )

app_ui = ui.page_fluid(
    # Add CSS to constrain image sizes
    ui.tags.style("""
        .shiny-image-output img {
            max-height: 400px;
            width: auto;
            max-width: 80%;
            display: block;
            margin: 0 auto;
        }
    """),

    ui.h2("Breast Cancer Classification Dashboard"),

    ui.navset_tab(

        # 1. DATA OVERVIEW TAB
        ui.nav_panel(
            "Data overview",
            ui.div(

                # ---- ABSTRACT ----
                ui.h3("Abstract", style="margin-top:10px;"),
                ui.markdown("""
        This project explores different approaches for predicting breast cancer malignancy using both tabular cell-level features and medical imaging. I started by analyzing the Breast Cancer Wisconsin dataset, which includes nine cytological measurements extracted from fine needle aspirate (FNA) samples. These features help capture structural differences between benign and malignant cells. I fit a baseline logistic regression model, a Bayesian model with informative priors, and later compare these results with a CNN trained on ultrasound images. The goal is to understand which modeling strategies offer the strongest accuracy, interpretability, and reliability for medical decision-making.
                """),

                # ---- PURPOSE ----
                ui.h3("Purpose"),
                ui.markdown("""
        Early and accurate identification of malignant tumors is critical, and different modeling approaches offer different strengths. Logistic regression provides a simple baseline, Bayesian models help quantify uncertainty (which is important in medical contexts), and CNNs can uncover image-based patterns that don’t appear in structured data. This dashboard pulls everything together so we can compare performance, visualize patterns in the data, and understand where each method works best.
                """),

                # ---- DATA SOURCES ----
                ui.h3("Datasets Used"),
                ui.markdown("""
        I use two publicly available datasets in this project:

        - **Breast Cancer Wisconsin (Diagnostic) Dataset** – tabular cytology measurements  
          Source: <https://www.kaggle.com/datasets/saurabhbadole/breast-cancer-wisconsin-state/data>  

        - **BUSI Ultrasound Breast Cancer Dataset** – ultrasound images (for CNN modeling)  
          Source: <https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset>  

        Together, these datasets allow me to analyze breast cancer prediction from both *feature-based* and *image-based* perspectives.
                """),

                # ---- FEATURE DESCRIPTION ----
                ui.h3("About the Features"),
                ui.markdown("""
        The diagnostic dataset includes nine numeric features that describe cell morphology, such as:

        - clump thickness  
        - uniformity of cell size  
        - uniformity of cell shape  
        - uniformity of cell shape  
        - marginal adhesion  
        - bare nuclei  
        - bland chromatin  
        - normal nucleoli  
        - mitoses  

        Each sample is labeled as **B (Benign)** or **M (Malignant)**.  
        These features help characterize structural differences in cell nuclei that are often associated with cancer.
                """),

                # ---- CHARTS ----
                # ---- FIRST ROW: CNN + TABULAR CLASS + FEATURE HISTS ----

                three_chart_row(
                    chart_block(
                        "CNN image class distribution",
                        "cnn_count",
                        "Image counts for benign and normal cases in the ultrasound dataset."
                    ),
                    chart_block(
                        "Bayesian class distribution",
                        "class_dist",
                        "Counts of benign and malignant cases in the tabular diagnostic dataset."
                    ),
                    chart_block(
                        "Bayesian Feature distributions by class",
                        "feature_hist",
                        "Histograms for each feature, split by benign vs malignant tumors."
                    ),
                ),

                two_chart_row(
                    chart_block(
                        "Bayesian Feature boxplots",
                        "box_plots",
                        "Standardized boxplots comparing feature medians and spread across classes."
                    ),
                    chart_block(
                        "Bayesian Feature effects on cancer risk",
                        "feature_effects",
                        "Posterior means and credible intervals from the Bayesian model for each feature."
                    )
                ),

                style="margin:30px;",
            ),
        ),

        # 2. CNN TAB
        ui.nav_panel(
            "CNN",
            ui.div(
                ui.h3("Convolutional Neural Network (CNN) model"),
                ui.markdown(
                    """
We built a convolutional neural network (CNN) from scratch in Python to classify breast ultrasound images as benign versus normal.
                    """
                ),

                # ---- DATASET + PREPROCESSING ----
                ui.h4("Dataset and preprocessing"),
                ui.markdown(
                    """
The CNN uses the BUSI breast ultrasound dataset. Images are labeled as benign, malignant, or normal. For this project we focused on a binary task: benign versus normal. Early experiments with three class prediction could not reach stable accuracy, so malignant images were removed from the CNN pipeline.

For the final setup:
- Training set: 353 benign images and 109 normal images  
- Test set: 84 benign images and 24 normal images  
- All images were converted to grayscale NumPy arrays, resized to 112 × 112 pixels, and standardized before training.
                    """
                ),

                # ---- ARCHITECTURE MOVED UP ----
                ui.h4("CNN architecture overview"),
                ui.markdown(
                    """
The CNN architecture includes:
- Three convolutional layers, each followed by ReLU activation  
- One max pooling layer after the final convolution  
- A flattening step  
- A fully connected hidden layer with ReLU  
- A Sigmoid output layer that returns the probability an image is benign

Every component was implemented manually in NumPy, including convolution operations, ReLU, max pooling, and dense layers. We used binary cross entropy loss and hand-written gradients for backpropagation. Training used:
- Learning rate: 0.005  
- Batch size: 16  
- 20 epochs with shuffled mini batches  
                    """
                ),

                # ---- EXAMPLE IMAGES MOVED UP ----
                ui.h4("Example ultrasound images"),
                ui.markdown(
                    """
We inspected raw scans from each class to confirm the labels looked correct and to build intuition for what the CNN needs to learn. Normal scans tend to show smoother, more uniform textures. Benign scans often contain darker, well defined circular or oval masses against a brighter background. These structural differences are what the CNN tries to pick up.
                    """
                ),

                chart_block(
                    "Sample images from each class",
                    "sample_images",
                    "Top row shows normal scans and the bottom row shows benign scans used during training."
                ),

                # ---- PIXEL LEVEL STRUCTURE ----
                ui.h4("Pixel level structure"),
                ui.markdown(
                    """
Before training the CNN we analyzed pixel intensities to see how benign and normal images differ. While there is overlap, benign images often have slightly higher median intensity and a wider upper tail. These plots summarize those patterns.
                    """
                ),

                two_chart_row(
                    chart_block(
                        "Pixel intensity distribution",
                        "pixel_dist",
                        "Histograms of normalized pixel intensities for normal and benign images."
                    ),
                    chart_block(
                        "Pixel intensity box plot",
                        "pixel_box",
                        "Box plots comparing the spread and central tendencies for each class."
                    ),
                ),

                # ---- BRIGHTNESS ----
                ui.h4("Brightness per image"),
                ui.markdown(
                    """
We also summarized each image using its mean pixel intensity. This gives a simple global measure of brightness that helps show how individual scans vary across classes.
                    """
                ),

                two_chart_row(
                    chart_block(
                        "Mean brightness per image",
                        "mean_brightness",
                        "Violin plot of mean pixel intensity for normal and benign images."
                    ),
                    chart_block(
                        "Brightness distribution by class",
                        "brightness_scatter",
                        "Scatter plot of mean brightness values for each image."
                    ),
                ),

                style="margin:30px;",
            ),
        ),
        # 3. BAYESIAN TAB
        ui.nav_panel(
            "Bayesian",
            ui.div(
                ui.h3("Bayesian modeling"),

                ui.markdown(
                    """
Our Bayesian analysis models the probability that a tumor is malignant using the same cell level features shown on the Data Overview tab. We started with a simple one predictor model, moved to a five predictor Bayesian logistic regression with informative priors, and then fit a horseshoe prior model with all nine features to see how regularization affects the coefficients and predictions.
                    """
                ),

                ui.h4("Model structure"),
                ui.markdown(
                    """
**Outcome**
- Binary label: malignant vs benign

**Baseline Bayesian model**
- Predictor: `bare_nuclei` (strongest single feature from the EDA)
- Likelihood: logistic regression
- Priors:
  - Intercept: Normal(0, 5)
  - Slope for `bare_nuclei`: Normal(1, 2), reflecting a prior belief that higher bare nuclei scores increase malignancy risk

**Extended Bayesian model**
- Five predictor model with informative Normal priors on:
  - `clump_thickness`
  - `bare_nuclei`
  - `uniformity_cell_shape`
  - `marginal_adhesion`
  - `normal_nucleoli`
- Horseshoe prior variant that includes all nine features and shrinks weaker coefficients toward zero while allowing strong coefficients to stay large
                    """
                ),

                ui.h4("Key predictors and effect sizes"),
                ui.markdown(
                    """
From the earlier EDA, three features stood out as especially strong signals of malignancy based on both correlation with the outcome and how far apart benign and malignant means were:
- **bare_nuclei**
  - Correlation with malignancy: 0.823
  - Mean (benign): 1.35, mean (malignant): 7.63
  - Malignant to benign mean ratio: 5.66

- **uniformity_cell_shape**
  - Correlation with malignancy: 0.822
  - Malignant to benign mean ratio: 4.64

- **uniformity_cell_size**
  - Correlation with malignancy: 0.821
  - Malignant to benign mean ratio: 5.04
                    """
                ),

                ui.h4("Convergence and reliability"),
                ui.markdown(
                    """
For the five predictor Normal prior model:
- All Rhat values were essentially 1.00, showing convergence:
  - `clump_thickness`: 1.00027
  - `bare_nuclei`: 1.00033
  - `uniformity_cell_shape`: 0.99998
  - `marginal_adhesion`: 1.00087
  - `normal_nucleoli`: 1.00024
                    """
                ),

                ui.h4("Posterior predictive behavior"),
                ui.markdown(
                    """
For a test patient with a true benign diagnosis:
- Normal prior model: mean P(malignant) **0.01**
- Horseshoe prior model: mean P(malignant) **0.013**
Both models assign extremely low malignancy probabilities with tight posterior uncertainty, which aligns with the patient’s actual benign label and shows that the models are behaving as expected at the individual level.
                    """
                ),

                ui.h4("Performance and tradeoffs"),
                ui.markdown(
                    """
We compared the baseline GLM and the Bayesian models on the same 137 patient test set.
**Standard threshold (0.5)**
- **GLM baseline**
  - Accuracy: 0.949
  - Recall for malignant: 0.902
  - Recall for benign: 0.977
  - Precision for malignant: 0.958
  - Precision for benign: 0.944
- **Bayesian Normal prior model**
  - Accuracy: 0.949
  - Recall for malignant: 0.902
  - Recall for benign: 0.977
- **Bayesian Horseshoe model (nine predictors)**
  - Accuracy: 0.956
  - Recall for malignant: 0.922
  - Recall for benign: 0.977
  - Precision for malignant: 0.959
  - Precision for benign: 0.955

**Area Under The Curve**
- GLM: 0.990
- Bayesian Normal prior: 0.992
- Bayesian Horseshoe: 0.992
All three models have extremely high area under the curve, with the Bayesian models doing better than the base logistic regression.
                    """
                ),
                ui.markdown(
                    """
We also encoded clinical preferences directly into the Bayesian predictions.

**Cost sensitive thresholding**
- False negative cost: 10
- False positive cost: 1
- Optimal threshold for P(malignant) under this loss: 0.08
At this threshold for the Bayesian Normal model:
- Accuracy: 0.964
- Recall for malignant: 1.00 (we catch all malignant cases)
- Recall for benign: 0.942
- Precision for malignant: 0.911
- Precision for benign: 1.00
This trades a small increase in false positives for virtually no missed cancers.

**Uncertainty aware classification**
Using:
- P(malignant) less than 0.3 -> classify as benign
- P(malignant) greater than 0.7 -> classify as malignant
- Otherwise -> flag as uncertain and defer

On the 137 test cases:
- Total predictions: 137
- 133 confident predictions 
- 4 uncertain predictions
- Accuracy on the confident subset: 0.962

This showed that the Bayesian model can be used to confidently classify most patients while flagging a small subset for additional review.
                    """
                ),

                ui.h4("Horseshoe prior behavior"),
                ui.markdown(
                    """
In the horseshoe model that includes all nine predictors:
- Strongest retained signals:
  - `clump_thickness`: mean = 1.35
  - `bare_nuclei`: mean = 1.35
  - `uniformity_cell_shape`: mean = 1.43
- Weaker features like mitoses and epithelial cell size are shrunk closer to zero.
The horseshoe model matches the Normal prior model in AUC (around 0.992) and slightly improves precision for malignant and precision for benign, while reducing the influence of weaker features and making the model easier to interpret.
                    """
                ),

                ui.h4("Conclusion and key findings"),
                ui.markdown(
                    """
Overall, our Bayesian models provided highly accurate and interpretable predictions of breast cancer malignancy.  
Key findings:
- The strongest predictors across all models were **bare nuclei**, **uniformity of cell shape**, and **uniformity of cell size**.  
- All Bayesian models achieved extremely high AUC values (around 0.992), surpassing the GLM baseline.  
- The horseshoe prior down weighted weaker predictors while keeping the a strong predictive performance.
- Cost sensitive thresholding allowed us to prioritize catching malignant cases, reaching 100 percent sensitivity at a threshold of 0.08.  
- The uncertainty‑aware framework produced confident predictions for most patients while correctly flagging a few ambiguous cases.  
Together, these results show that Bayesian modeling adds value through interpretability, uncertainty quantification, and principled regularization, while maintaining excellent predictive performance.
                    """
                ),

                # ROW 1: Posterior + MCMC diagnostics
                two_chart_row(
                    chart_block(
                        "Posterior for bare nuclei coefficient",
                        "posterior_bare_nuclei",
                        "Posterior distribution for the bare nuclei coefficient in the simple Bayesian logistic model. The mass far above zero shows a strong positive association with malignancy."
                    ),
                    chart_block(
                        "MCMC diagnostic plots",
                        "b_plots",
                        "Trace and density plots used to check that chains have mixed well and converged. Rhat values near 1 and stable traces support reliable inference."
                    ),
                ),

                # ROW 2: Predictive probabilities + single patient example
                two_chart_row(
                    chart_block(
                        "Predicted probabilities on the test set",
                        "predicted_probs",
                        "Posterior predictive probabilities of malignancy for each test patient. This highlights which cases are clearly benign, clearly malignant, or near the decision boundary."
                    ),
                    chart_block(
                        "Patient 1 predictive distribution",
                        "patient_one",
                        "Predictive distribution for one example patient, showing the full posterior over P(malignant) instead of a single point estimate. This illustrates how the Bayesian model represents uncertainty for individual cases."
                    ),
                ),

                style="margin:30px;",
            ),
        ),
    ),
)


def server(input, output, session):
    def image_dict(filename: str, alt: str):
        path = OUT_DIR / filename
        if not path.exists():
            return {"src": "", "alt": f"{filename} not found"}
        return {
            "src": str(path),
            "alt": alt,
            # Remove the width parameter, let CSS handle sizing
        }

    # DATA OVERVIEW IMAGES
    @output
    @render.image
    def cnn_count():
        return image_dict(
            "cnn_counts.png",
            "CNN dataset class distribution",
        )

    @output
    @render.image
    def class_dist():
        return image_dict(
            "class_distribution.png",
            "Class distribution: benign vs malignant",
        )

    @output
    @render.image
    def feature_hist():
        return image_dict(
            "distribution_of_features.png",
            "Distribution of features by tumor class",
        )

    @output
    @render.image
    def box_plots():
        return image_dict(
            "box_plots.png",
            "Feature boxplots",
        )

    @output
    @render.image
    def feature_effects():
        return image_dict(
            "feature_effects_on_cancer.png",
            "Posterior feature effects on cancer risk",
        )

    # CNN IMAGE EDA
    @output
    @render.image
    def pixel_dist():
        return image_dict(
            "picel_distribution.png",
            "Pixel intensity distribution for normal and benign images",
        )

    @output
    @render.image
    def pixel_box():
        return image_dict(
            "pixel_box_plot.png",
            "Pixel intensity box plot for normal and benign images",
        )

    @output
    @render.image
    def mean_brightness():
        return image_dict(
            "mean_brightness.png",
            "Mean brightness per image for each class",
        )

    @output
    @render.image
    def brightness_scatter():
        return image_dict(
            "brightness_scatter.png",
            "Brightness distribution scatter plot by class",
        )

    @output
    @render.image
    def sample_images():
        return image_dict(
            "sample_images.png",
            "Sample ultrasound images from normal and benign classes",
        )

    # BAYESIAN IMAGES
    @output
    @render.image
    def posterior_bare_nuclei():
        return image_dict(
            "posterior_bare_nuclei.png",
            "Posterior distribution for bare nuclei coefficient",
        )

    @output
    @render.image
    def b_plots():
        return image_dict(
            "b_plots.png",
            "Bayesian model diagnostic plots",
        )

    @output
    @render.image
    def predicted_probs():
        return image_dict(
            "predicted_probabilities.png",
            "Distribution of predicted probabilities for test patients",
        )

    @output
    @render.image
    def patient_one():
        return image_dict(
            "patient_1_distribution.png",
            "Predictive distribution for example patient",
        )


app = App(app_ui, server)
