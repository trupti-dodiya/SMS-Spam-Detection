import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Load dataset
data = pd.read_csv("data/cleaned_sms.csv")

# Step 2: Separate messages and labels
X = data["message"]
y = data["label"]

# Step 3: Divide dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 4: Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Step 5: Learn vocabulary and transform training messages
X_train_tfidf = vectorizer.fit_transform(X_train)

# Step 6: Transform testing messages
X_test_tfidf = vectorizer.transform(X_test)

# Step 7: Display results
print("Training data shape:", X_train_tfidf.shape)

print("Testing data shape:", X_test_tfidf.shape)

print("\nSome vocabulary words:")
print(vectorizer.get_feature_names_out()[:20])

print("\nFirst message numerical representation:")
print(X_train_tfidf[0].toarray())
