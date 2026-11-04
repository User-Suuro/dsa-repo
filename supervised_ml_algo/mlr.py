# Multiple Linear Regression (seaborn 'mpg' dataset)
# pip install scikit-learn matplotlib pandas seaborn
# Note: seaborn downloads the mpg dataset on first use, so an internet connection is needed.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the dataset
data = sns.load_dataset('mpg').dropna()  # Load mpg dataset and drop any missing values

# Select features and target variable
X = data[['horsepower', 'weight', 'displacement']]
y = data['mpg']

# Split data into training and testing sets
# (The slide text says "80-20 split", but the code uses test_size=0.3, i.e. 70-30.)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the multiple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Display model coefficients
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")

# Visualize predictions vs actual values
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2)
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Actual vs Predicted MPG")
plt.show()