import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load your dataframe here
try:
    df = pd.read_excel("tf_idf_score.xlsx")
except FileNotFoundError as e:
    raise Exception(e)

# Drop the first 3 columns
df = df.iloc[:, 3:]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, -14:],  # Select the last 14 columns as features
    df.iloc[:, 0:14],  # Select the first 14 columns as labels
    test_size=0.2, random_state=42)

# For each label, train a logistic regression model using binary relevance
for col in y_train.columns:
    # Train a logistic regression model on the current label
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train[col])

    # Make predictions on the test set
    y_pred = lr.predict(X_test)

    # Compute and print the accuracy score for the current label
    acc_score = accuracy_score(y_test[col], y_pred)
    print(f"Accuracy for label {col}: {acc_score:.4f}")
