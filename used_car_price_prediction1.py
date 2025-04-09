# -*- coding: utf-8 -*-
"""Used Car Price Prediction - Save All Plots"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression, SelectKBest

# Setup
warnings.simplefilter(action='ignore')
pd.set_option("display.max_rows", None, "display.max_columns", None)

# Create folder for plots
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# Load data
df_main = pd.read_csv(r'C:\Users\hp\Desktop\Used_Car_Price_Prediction\used_cars_data.csv')
df_main.drop(columns=['S.No.'], inplace=True)

# Feature Engineering
df_main['Age'] = 2025 - df_main['Year']
df_main.drop('Year', axis=1, inplace=True)

df_main.rename(columns={
    'Mileage': 'Mileage(kmpl)',
    'Engine': 'Engine(cc)',
    'Power': 'Power(bhp)',
    'Price': 'Selling_Price'
}, inplace=True)

df_main.drop(columns=['New_Price'], inplace=True)

# Missing Values Handling
df_main['Seats'].fillna(df_main['Seats'].mode()[0], inplace=True)
df_main['Mileage(kmpl)'] = df_main['Mileage(kmpl)'].str.extract('(\d+.\d+)').astype(float)
df_main['Mileage(kmpl)'].fillna(df_main['Mileage(kmpl)'].median(), inplace=True)

df_main['Engine(cc)'] = df_main['Engine(cc)'].str.extract('(\d+)').astype(float)
df_main['Engine(cc)'].fillna(df_main['Engine(cc)'].median(), inplace=True)

df_main['Power(bhp)'] = df_main['Power(bhp)'].str.extract('(\d+.\d+)').astype(float)
df_main['Power(bhp)'].fillna(df_main['Power(bhp)'].median(), inplace=True)

df_main.dropna(subset=['Selling_Price'], inplace=True)

df_main['Name'].nunique()  # Count unique names
df_main['Name'].value_counts().head(10) 

# Extract Brand from Name
df_main['Brand'] = df_main['Name'].apply(lambda x: x.split()[0])
df_main.drop(columns=['Name'], inplace=True)

# Boxplot of Selling Price vs Brand
plt.figure(figsize=(10, 4))
sns.boxplot(x='Brand', y='Selling_Price', data=df_main)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{plot_dir}/boxplot_brand_vs_price.png")
plt.close()

# Selling Price Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df_main['Selling_Price'], bins=30, kde=True)
plt.title("Distribution of Selling Price")
plt.xlabel("Price")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{plot_dir}/selling_price_distribution.png")
plt.close()

# Log Transform
df_main['Selling_Price_Log'] = np.log1p(df_main['Selling_Price'])
plt.figure(figsize=(8, 4))
sns.histplot(df_main['Selling_Price_Log'], bins=30, kde=True)
plt.title("Log Transformed Distribution of Selling Price")
plt.tight_layout()
plt.savefig(f"{plot_dir}/log_transformed_price.png")
plt.close()

# Boxplot of Kilometers Driven
plt.figure(figsize=(8, 4))
sns.boxplot(x=df_main['Kilometers_Driven'])
plt.title("Boxplot of Kilometers Driven")
plt.tight_layout()
plt.savefig(f"{plot_dir}/boxplot_kms_before.png")
plt.close()

# Remove Outliers Function
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

# Remove outliers from all columns
columns_with_outliers = ['Kilometers_Driven', 'Mileage(kmpl)', 'Engine(cc)', 'Power(bhp)']
for col in columns_with_outliers:
    df_main = remove_outliers_iqr(df_main, col)

# Boxplots after removing outliers
for col in columns_with_outliers:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df_main[col])
    plt.title(f"Boxplot of {col} after Outlier Removal")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/boxplot_{col}_after.png")
    plt.close()

# Correlation Heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(df_main.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{plot_dir}/correlation_heatmap.png")
plt.close()

# Scatterplots
for col in ['Kilometers_Driven', 'Mileage(kmpl)', 'Engine(cc)', 'Power(bhp)', 'Age']:
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x=df_main[col], y=df_main['Selling_Price'])
    plt.title(f"{col} vs Selling Price")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/scatter_{col}_vs_price.png")
    plt.close()

# Histograms of all numerical features
df_main.select_dtypes(include=['int64', 'float64']).hist(figsize=(12, 8), bins=30, edgecolor='black')
plt.tight_layout()
plt.savefig(f"{plot_dir}/histograms_numerical_features.png")
plt.close()

# Categorical plots
categorical_cols = df_main.select_dtypes(include='object').columns
plt.figure(figsize=(12, 8))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(2, 3, i)
    sns.countplot(data=df_main, x=col, palette="viridis")
    plt.xticks(rotation=45)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig(f"{plot_dir}/categorical_distribution.png")
plt.close()

# Encoding
le = LabelEncoder()
df_main['Transmission'] = le.fit_transform(df_main['Transmission'])
df_main['Owner_Type'] = df_main['Owner_Type'].map({'First': 0, 'Second': 1, 'Third': 2, 'Fourth & Above': 3})

# Define threshold (e.g., 1% of total dataset)
threshold = 0.01 * len(df_main)
# Count fuel types
fuel_counts = df_main['Fuel_Type'].value_counts()
# Identify rare fuel types below threshold
rare_fuels = fuel_counts[fuel_counts < threshold].index.tolist()
print(f"Rare fuel types (to be dropped): {rare_fuels}")
# Filter them out
df_main = df_main[~df_main['Fuel_Type'].isin(rare_fuels)]
# One-Hot Encode Fuel_Type
df_main = pd.get_dummies(df_main, columns=['Fuel_Type'], drop_first=True)

from sklearn.model_selection import train_test_split

# Load your dataframe (assuming it's already loaded as df)
df = df_main.copy()  # if already defined

# Drop unnecessary columns
df.drop(['Selling_Price_Log'], axis=1, inplace=True, errors='ignore')

# Define features and target
X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']

# Split BEFORE any target encoding to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add target temporarily for encoding
train_temp = X_train.copy()
train_temp['Selling_Price'] = y_train

# Target Encoding for Brand
brand_price_map = train_temp.groupby('Brand')['Selling_Price'].mean()
X_train['Brand_encoded'] = X_train['Brand'].map(brand_price_map)
X_test['Brand_encoded'] = X_test['Brand'].map(brand_price_map)

# Target Encoding for Location
location_price_map = train_temp.groupby('Location')['Selling_Price'].mean()
X_train['Location_encoded'] = X_train['Location'].map(location_price_map)
X_test['Location_encoded'] = X_test['Location'].map(location_price_map)

# Drop original Brand and Location columns
X_train.drop(['Brand', 'Location'], axis=1, inplace=True)
X_test.drop(['Brand', 'Location'], axis=1, inplace=True)

import pickle
# Save brand_price_map
with open('brand_encoding.pkl', 'wb') as f:
    pickle.dump(brand_price_map.to_dict(), f)
# Save location_price_map
with open('location_encoding.pkl', 'wb') as f:
    pickle.dump(location_price_map.to_dict(), f)


# Final Check
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Features to preprocess
num_features = ['Kilometers_Driven', 'Mileage(kmpl)', 'Engine(cc)', 'Power(bhp)', 'Seats', 'Age']

# Create a pipeline for numerical preprocessing
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Fit and transform on training data
X_train[num_features] = num_pipeline.fit_transform(X_train[num_features])

# Only transform on test data
X_test[num_features] = num_pipeline.transform(X_test[num_features])

print(X_train.shape)
print(X_test.shape)

# Define candidate models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

# Container to hold evaluation results
results = []

# Train, predict, and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)             # Train model
    y_pred = model.predict(X_test)          # Predict on test data

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Store results
    results.append([name, mae, mse, rmse, r2])

# Create a DataFrame to display results
results_df = pd.DataFrame(results, columns=["Model", "MAE", "MSE", "RMSE", "R² Score"])
results_df = results_df.sort_values(by="R² Score", ascending=False)

# Show performance comparison
print("\n✅ Model Performance Comparison:\n")
print(results_df)

# Select and use the best model
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
print(f"\n🏆 Best Model Selected: {best_model_name}")

# Predict using best model (no need to reverse log transform)
y_pred = best_model.predict(X_test)

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# Define the base model
xgb = XGBRegressor(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='r2',         # since we're doing regression
    cv=3,                 # 3-fold cross-validation
    verbose=2,
    n_jobs=-1             # Use all available cores
)

# Fit on training data
grid_search.fit(X_train, y_train)

# Best estimator and score
print("Best R² Score:", grid_search.best_score_)
print("Best Parameters:", grid_search.best_params_)

# Use the best model
best_xgb_model = grid_search.best_estimator_

y_pred = best_xgb_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n🔍 Tuned XGBoost Performance:")
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R² Score: {r2:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Selling Price")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plot_dir}/Actual vs Predicted Selling Price.png")


import joblib
# Save the best model
joblib.dump(best_xgb_model, 'best_model.pkl')
print("📦 Best model saved as 'best_model.pkl'")


# # Splitting
# X = df_main.drop(columns=['Selling_Price', 'Selling_Price_Log'])
# y = df_main['Selling_Price_Log']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Preprocessing
# num_pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])
# num_features = ['Kilometers_Driven', 'Mileage(kmpl)', 'Engine(cc)', 'Power(bhp)', 'Seats', 'Age']
# X_train[num_features] = num_pipeline.fit_transform(X_train[num_features])
# X_test[num_features] = num_pipeline.transform(X_test[num_features])

# # Feature Selection
# selector = SelectKBest(score_func=mutual_info_regression, k=20)
# X_train_selected = selector.fit_transform(X_train, y_train)
# X_test_selected = selector.transform(X_test)
# selected_features = X_train.columns[selector.get_support()]
# print("Selected Features:", selected_features.tolist())

# # Model Training & Evaluation
# models = {
#     "Linear Regression": LinearRegression(),
#     "Decision Tree": DecisionTreeRegressor(random_state=42),
#     "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
#     "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
#     "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
# }

# results = []

# for name, model in models.items():
#     model.fit(X_train_selected, y_train)
#     y_pred = model.predict(X_test_selected)
#     results.append({
#         "Model": name,
#         "MAE": mean_absolute_error(y_test, y_pred),
#         "MSE": mean_squared_error(y_test, y_pred),
#         "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
#         "R2": r2_score(y_test, y_pred)
#     })

# # Display Results
# results_df = pd.DataFrame(results)
# print("\nModel Evaluation Results:")
# print(results_df)

# # Save evaluation as CSV
# results_df.to_csv(f"{plot_dir}/model_evaluation_results.csv", index=False)

# import joblib

# # Find best model by R2 score
# best_model = max(results, key=lambda x: x['R2'])['Model']
# print(f"\nBest Model: {best_model}")

# # Save the best model
# final_model = models[best_model]
# model_path = "final_model.pkl"
# joblib.dump(final_model, model_path)
# print(f"Model saved as {model_path}")

