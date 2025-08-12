import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, pipeline
import numpy as np
from sklearn.metrics import accuracy_score

# Auto-detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# Dataset paths
data_files = {
    "train": "data/train.csv",
    "validation": "data/val.csv",
    "test": "data/test.csv"
}

# Load dataset
dataset = load_dataset("csv", data_files=data_files)

# Define columns
text_col = "tweet"
label_col = "label"
print(f"[INFO] Using '{text_col}' as text column and '{label_col}' as label column")

# Filter invalid rows
dataset = dataset.filter(lambda x: isinstance(x[text_col], str) and len(x[text_col].strip()) > 0)

# Use smaller dataset for CPU testing (comment this out for full training)
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(1000))
dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(200))
dataset["test"] = dataset["test"].shuffle(seed=42).select(range(200))

# Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples[text_col], padding="max_length", truncation=True, max_length=64)

dataset = dataset.map(tokenize_function, batched=True)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

# Compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# Training arguments
training_args = TrainingArguments(
    output_dir="./model",
    eval_strategy="epoch",  # updated for new transformers
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,  # just 2 for quick testing
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save model
trainer.save_model("./model")
tokenizer.save_pretrained("./model")
print("[INFO] Model saved to ./model")

# Test prediction
clf = pipeline("text-classification", model="./model", tokenizer="./model")
print(clf("I am feeling very sad today"))
print(clf("Life is going great and I am happy"))
