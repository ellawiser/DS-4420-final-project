Wildfire Modeling Project

DS4300 Final Project — Ella Wiser & Lauren Montion

Overview

This project explores how machine learning can be used to analyze and predict wildfire behavior across North maerica. We apply two distinct different methods, Convolutional Neural Networks (CNNs) and Time Series Analysis. Our goal is to investigate patterns of wildfire ignition, spread, and severity using both spatial and temporal data.

The main objectives are:
1.	Detect and classify wildfire severity from satellite imagery using a CNN.
2.	Examine temporal patterns in wildfire activity using time series analysis to identify seasonal trends, long-term changes, and potential links to temperature or drought conditions.
3.	Visualize results through a severit by region map that highlights variation in fire intensity across Canada.

Data Sources

We are collecting open wildfire data from several public sources:
-	Canadian National Fire Database (CNFDB): ignition points, perimeters, burn area, and dates.
- Canadian Wildland Fire Information System (CWFIS): Fire Weather Index (FWI), Daily Severity Rating (DSR), and regional climatology.
- Satellite Imagery (Landsat 8, MODIS, VIIRS): used for CNN input and for computing burn severity metrics such as NDVI or dNBR.
- ERA5-Land or Daymet Weather Data: provides temperature, humidity, precipitation, and wind speed for time-series modeling.

Methods

1. Convolutional Neural Network (CNN) in Python
	-	Built manually using NumPy without deep learning libraries.
	-	Inputs: small image sections from satellite fire imagery.
	-	Target: classification of burn severity (for example, low, moderate, or high).
	-	Output: predicted severity class and heatmap overlay for visualization.

2. Time Series Analysis in R
	-	Models annual and seasonal trends in fire occurrence and burned area.
	-	Examines relationships between weather variables and fire severity.
	-	Produces smoothed trend plots, autocorrelation functions, and predictive intervals.
	-	May include a Bayesian-style time series to capture uncertainty if data quality allows.

Contributors
-	Ella Wiser
-	Lauren Montion
