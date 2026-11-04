# Simple Linear Regression (Diabetes dataset)
# pip install scikit-learn matplotlib numpy
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

diabetes = datasets.load_diabetes()

# Use only one feature (BMI, column index 2)
X = diabetes.data[:, np.newaxis, 2]

# Split: last 30 samples are used for testing
X_train = X[:-30]
X_test = X[-30:]
y_train = diabetes.target[:-30]
y_test = diabetes.target[-30:]

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()