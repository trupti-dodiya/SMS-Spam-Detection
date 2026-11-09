
import pandas as pd

# 1. Load the original dataset
data = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("Original dataset size:", data.shape)

# 2. Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# 3. Check duplicate records
print("\nDuplicate records:")
print(data.duplicated().sum())

# 4. Remove duplicate records
data = data.drop_duplicates()

# 5. Convert messages to lowercase
data["message"] = data["message"].str.lower()

# 6. Remove extra spaces
data["message"] = data["message"].str.strip()

# 7. Normalize labels
data["label"] = data["label"].str.lower().str.strip()

# 8. Display cleaned dataset information
print("\nCleaned dataset size:", data.shape)

print("\nMessage distribution:")
print(data["label"].value_counts())

# 9. Save cleaned dataset
data.to_csv("data/cleaned_sms.csv", index=False)

print("\nCleaned dataset saved successfully!")
