# Individual Activity #3 (alternative version using load_wine)
# Multiple Linear Regression on the Wine dataset from sklearn.datasets.
# pip install scikit-learn pandas
#
# Note: sklearn's load_wine() has 178 wines and 13 chemical features (alcohol,
# malic_acid, ash, ...). Its target is the wine's origin class (0, 1, 2), not a
# quality score, so that is what the regression predicts here. For real quality
# scores, see 09_activity3_wine_quality_regression.py.
from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load the Wine dataset using sklearn.datasets
wine = load_wine(as_frame=True)
df = wine.frame

# 2. Use multiple features to predict the target
X = df.drop(columns=["target"])
y = df["target"]

# 3. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train a Multiple Linear Regression model and print the coefficients
model = LinearRegression()
model.fit(X_train, y_train)

print("Model Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# 5. Report the MSE and R^2 score of the model on the test set
y_pred = model.predict(X_test)
print(f"\nMean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R-squared: {r2_score(y_test, y_pred)}")