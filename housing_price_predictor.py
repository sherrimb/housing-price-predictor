# housing_price_predictor.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

def load_data(filename):
    try:
        print(f"Trying to load: {filename}")
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        print("Error: The file was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        return None

def explore_data(df):
    print("\nFirst few rows of the dataset:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())

def train_model(df):
    X = df[['Size (sqft)']].values
    y = df['Price ($)'].values
    model = LinearRegression()
    model.fit(X, y)
    print(f"\nModel coefficient (slope): {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"R^2 score: {model.score(X, y):.4f}")
    return model

def predict_price_from_input(model):
    try:
        user_input = input("\nEnter the house size in square feet: ").strip()
        sqft = float(user_input)
        if sqft <= 0:
            print("Please enter a positive number for square footage.")
            return
        predicted_price = model.predict([[sqft]])[0]
        print(f"\nEstimated house price for {sqft} sqft is: ${predicted_price:,.2f}")
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
def plot_data_and_regression(df, model):
    X = df[['Size (sqft)']].values
    y = df['Price ($)'].values
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(X_range)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual Data')
    plt.plot(X_range, y_pred, color='red', linewidth=2, label='Regression Line')
    plt.title('House Size vs Price')
    plt.xlabel('Size (sqft)')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    filename = "housing_data.csv"
    df = load_data(filename)
    if df is not None:
        explore_data(df)
        model = train_model(df)
        predict_price_from_input(model)
        plot_data_and_regression(df, model)
        

if __name__ == "__main__":
    main()
