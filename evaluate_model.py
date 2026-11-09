
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load cleaned dataset
data = pd.read_csv("data/cleaned_sms.csv")

# 2. Separate input and output
X = data["message"]
y = data["label"]

# 3. Create the same train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Load the trained model
model = joblib.load("spam_model.pkl")

# 5. Predict test messages
y_pred = model.predict(X_test)

# 6. Calculate evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label="spam"))
print("Recall:", recall_score(y_test, y_pred, pos_label="spam"))
print("F1 Score:", f1_score(y_test, y_pred, pos_label="spam"))

# 7. Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=["ham", "spam"]))

# 8. Display detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
