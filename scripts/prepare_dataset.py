import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_label(path, label):
    df = pd.read_csv(path)
    # Automatically find a text column
    text_col = None
    for col in df.columns:
        if col.lower() in ["tweet", "text", "content", "message"]:
            text_col = col
            break
    if text_col is None:
        text_col = df.columns[0]  # fallback to first column
    
    df = df[[text_col]].copy()
    df.columns = ["tweet"]
    df["label"] = label
    return df

# Depressed datasets
df_dep1 = load_and_label("data/clean_d_tweets.csv", 1)
df_dep2 = load_and_label("data/d_tweets.csv", 1)

# Non-depressed datasets
df_non1 = load_and_label("data/clean_non_d_tweets.csv", 0)
df_non2 = load_and_label("data/non_d_tweets.csv", 0)

# Combine
df_all = pd.concat([df_dep1, df_dep2, df_non1, df_non2], ignore_index=True)

# Split
train_df, temp_df = train_test_split(df_all, test_size=0.3, stratify=df_all["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

# Save
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("✅ Dataset prepared!")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
