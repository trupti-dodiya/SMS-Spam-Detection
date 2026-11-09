
import pandas as pd

from sklearn.model_selection import train_test_split

# 1. Load cleaned dataset
data = pd.read_csv("data/cleaned_sms.csv")

# 2. Separate input and output
X = data["message"]
y = data["label"]

# 3. Divide data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Display dataset sizes
print("Original dataset:", len(data))

print("\nTraining data:", len(X_train))
print("Testing data:", len(X_test))

# 5. Display training label distribution
print("\nTraining labels:")
print(y_train.value_counts())

# 6. Display testing label distribution
print("\nTesting labels:")
print(y_test.value_counts())
