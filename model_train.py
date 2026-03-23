import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# Load dataset
df = pd.read_csv('data.csv')

# Features & Target
X = df[['Distance', 'Weight', 'Fuel_Price']]
y = df['cost']

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)

# Linear Regression Model (Comparison)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# Evaluation
print("Random Forest MAE:", mean_absolute_error(y_test, rf_pred))
print("Random Forest R2:", r2_score(y_test, rf_pred))

print("Linear Regression MAE:", mean_absolute_error(y_test, lr_pred))
print("Linear Regression R2:", r2_score(y_test, lr_pred))

# Save best model
pickle.dump(rf_model, open('model.pkl', 'wb'))

print("Model saved successfully 🚀")