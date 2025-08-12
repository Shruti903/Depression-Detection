import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load datasets
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")

# Clean data
train_df = train_df.dropna(subset=["tweet"])
val_df = val_df.dropna(subset=["tweet"])
train_df["tweet"] = train_df["tweet"].astype(str)
val_df["tweet"] = val_df["tweet"].astype(str)

# Separate features and labels
X_train, y_train = train_df["tweet"], train_df["label"]
X_val, y_val = val_df["tweet"], val_df["label"]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Model
model = LogisticRegression(max_iter=300)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_val_tfidf)
print(classification_report(y_val, y_pred))

# Save
joblib.dump(model, "model/baseline_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
