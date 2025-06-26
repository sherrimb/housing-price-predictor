#CS 113 Assignment #3 Sherri Killough June 26, 2025
# Housing Price Prediction using Linear Regression

## Overview
This project is a Python-based machine learning application that predicts housing prices based on house size (in square feet) using linear regression. It is designed for an introductory course on Python, data analysis, and machine learning.

## Features
- Reads a dataset from `housing_data.csv`
- Displays basic statistics (mean, min, max, etc.)
- Visualizes the relationship between house size and price
- Trains a linear regression model using scikit-learn
- Displays model parameters: slope, intercept, and R² score
- Allows user to input a house size and get an estimated price
- Handles invalid inputs gracefully

## Dataset
The dataset `housing_data.csv` includes 20 rows of housing sizes and prices based on estimated market data in zip code 33570 (Ruskin, Florida). Prices were estimated using Zillow’s reported housing values for the Ruskin area as of June 2024.

## How to Run
1. Make sure you have Python 3 installed along with the following packages:
   - pandas
   - matplotlib
   - scikit-learn
   - numpy

2. Place `housing_data.csv` in the same folder as `housing_price_predictor.py`.

3. Run the script:
   ```bash
   python housing_price_predictor.py
   ```

4. Follow the on-screen prompt to enter a house size and receive a predicted price.

## Citation (APA Style)
Zillow. (2024). *Ruskin FL Home Prices & Home Values*. Zillow. https://www.zillow.com/ruskin-fl/home-values/

## Author Note
This project was developed as part of a college-level Python programming course. It demonstrates foundational skills in data handling, machine learning, and program design.
