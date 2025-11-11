import pandas as pd
import pymysql
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import sklearn
import pickle
import warnings

warnings.filterwarnings("ignore")

# === 1. Connect to MySQL and Load Data ===
connection = pymysql.connect(
    host='localhost',
    user='root',
    password='aditya524',
    database='BigMart'
)

df_item = pd.read_sql("SELECT * FROM item_info", connection)
df_outlet = pd.read_sql("SELECT * FROM outlet_info", connection)
df_sales = pd.read_sql("SELECT * FROM sales_info", connection)
connection.close()

# === 2. Merge DataFrames ===
df = df_item.merge(df_outlet, on='ID').merge(df_sales, on='ID')
df.drop('ID', axis=1, inplace=True)

# === 3. Feature Engineering ===
df['Outlet_Age'] = 2025 - df['Outlet_Establishment_Year']
df.drop('Outlet_Establishment_Year', axis=1, inplace=True)

df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
    'low fat': 'Low Fat',
    'LF': 'Low Fat',
    'reg': 'Regular'
})

df['Item_Visibility'] = np.where(df['Item_Visibility'] > 0.3, 0.3, df['Item_Visibility'])

# === 4. Prepare X, y ===
X = df.drop('Item_Outlet_Sales', axis=1)
y = df['Item_Outlet_Sales']

# === 5. Categorical Columns ===
categorical_cols = X.select_dtypes(include='object').columns.tolist()

# === 6. Preprocessing Pipeline ===
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# === 7. Define Models to Compare ===
models = {
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "LinearRegression": LinearRegression()
}

# === 8. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 9. Evaluate All Models ===
best_model_name = None
best_score = -np.inf
best_pipeline = None

for name, reg in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', reg)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    print(f"\n📊 {name} Results:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    
    if r2 > best_score:
        best_score = r2
        best_model_name = name
        best_pipeline = pipeline

# === 10. Save Best Model using Pickle ===
with open("bigmart_best_model.pkl", "wb") as f:
    pickle.dump((best_pipeline, sklearn.__version__), f)

print(f"\n✅ Best Model: {best_model_name} (R² = {best_score:.4f}) saved successfully as bigmart_best_model.pkl")