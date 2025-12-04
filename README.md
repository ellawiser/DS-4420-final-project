DS4300 Final Project — Ella Wiser & Lauren Montion
Breast Cancer Classification Project

Overview

This project explores two very different modeling approaches for predicting breast cancer outcomes: a Convolutional Neural Network (CNN) built from scratch in Python, and a set of Bayesian logistic regression models implemented in R. The goal is to compare image-based and feature-based methods for understanding tumor characteristics and predicting whether a case is benign or malignant.

The main objectives are:
	1.	Build and train a CNN from scratch using ultrasound images to classify benign versus normal scans.
	2.	Fit Bayesian logistic regression models that use cytology measurements to estimate the probability that a tumor is malignant.
	3.	Compare performance, interpretability, and uncertainty across both modeling approaches.
	4.	Present the results in an interactive dashboard built with Python Shiny.


Data Sources

This project uses two publicly available datasets:

1. BUSI Breast Ultrasound Images (for CNN modeling)

Contains ultrasound scans labeled as benign, malignant, or normal.
	•	Used for binary classification: benign vs normal
	•	Includes raw .png images
	•	Used to study pixel-level patterns and build the CNN pipeline
Source: [Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset/data)

2. Breast Cancer Wisconsin Diagnostic Dataset (for Bayesian modeling)

Tabular dataset containing 683 cases with nine cytological features such as:
	•	clump thickness
	•	uniformity of cell size
	•	uniformity of cell shape
	•	bare nuclei
	•	marginal adhesion
	•	and others

Each case is labeled as benign or malignant.
Source: [Kaggle](https://www.kaggle.com/datasets/saurabhbadole/breast-cancer-wisconsin-state/data)


Methods

1. Convolutional Neural Network (CNN) in Python
	•	Implemented entirely from scratch in NumPy without deep learning libraries
	•	Includes custom implementations of:
	•	Convolution layers
	•	Max pooling
	•	Fully connected layers
	•	ReLU and Sigmoid activations
	•	Binary cross entropy loss
	•	Backpropagation
	•	Input images were preprocessed by converting to grayscale, resizing to 112 x 112, and standardizing pixel values
	•	Final architecture:
	•	Three convolution blocks with ReLU
	•	One max pooling layer
	•	Flatten
	•	Fully connected hidden layer + Sigmoid output
	•	Trained with shuffled mini batches (batch size 16) for stability and generalization


2. Bayesian Logistic Regression in R (brms)

We built three Bayesian models to understand how cell-level features predict malignancy:
	1.	A single predictor model using bare nuclei
	2.	A five predictor model with informative Normal priors based on domain intuition
	3.	A full nine-predictor model using a horseshoe prior to shrink weaker predictors

Key components:
	•	Likelihood: logistic regression
	•	Priors reflect reasonable expectations about how features relate to malignancy
	•	MCMC sampling used to obtain posterior distributions
	•	Models validated using posterior predictive checks, convergence diagnostics (Rhat), and accuracy metrics
	•	Cost-sensitive thresholding included to reflect medical priorities (reducing missed cancers)


Dashboard

A Python Shiny dashboard presents:
	•	Data exploration
	•	Feature distributions
	•	CNN preprocessing visualizations
	•	Bayesian posterior plots
	•	Predictive probabilities
	•	Model comparison metrics

-	Ella Wiser
-	Lauren Montion
