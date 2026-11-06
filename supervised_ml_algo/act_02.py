# Individual Activity #2
# Titanic survival prediction with a Decision Tree classifier.
# pip install scikit-learn matplotlib pandas
#
# Uses train.csv from the Kaggle Titanic dataset (included in this folder).
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Load the Titanic dataset from a CSV file
df = pd.read_csv("./titanic_dataset/train.csv")

# 2. Preprocess the data
# Drop identifier / free-text columns, and Cabin (about 77% missing)
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())            # 177 missing
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])  # 2 missing

# Encode categorical variables
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], dtype=int)

X = df.drop(columns=["Survived"])
y = df["Survived"]

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train a Decision Tree classifier and make predictions on the test set
# max_depth=4 keeps the tree small enough to read when plotted
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. Print the accuracy of the model and visualize the decision tree
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Did not survive", "Survived"],
    filled=True,
)
plt.title("Decision Tree for Titanic Survival")
plt.show()