# Individual Activity #1
# Use SVM to classify breast cancer tumors as malignant or benign.
# pip install scikit-learn
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 1. Load the Breast Cancer Wisconsin dataset
data = load_breast_cancer()
X, y = data.data, data.target  # target: 0 = malignant, 1 = benign

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Train an SVM model with an RBF kernel and make predictions
model = SVC(kernel='rbf')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 5. Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))