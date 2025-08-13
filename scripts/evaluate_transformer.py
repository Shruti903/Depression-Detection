from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer, Trainer

def main():
    model_path = "./model"
    test_file = "./data/test.csv"

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

    # Load test dataset
    dataset = load_dataset("csv", data_files={"test": test_file})["test"]

    # Detect text column automatically (first non-label column)
    possible_text_cols = [c for c in dataset.column_names if c.lower() not in ["label", "labels"]]
    if not possible_text_cols:
        raise ValueError("No text column found in dataset!")
    text_col = possible_text_cols[0]
    print(f"[INFO] Using '{text_col}' as text column")

    # Remove bad rows (empty, NaN, not a string)
    dataset = dataset.filter(lambda x: isinstance(x[text_col], str) and x[text_col].strip() != "")

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples[text_col], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Detect label column
    label_col = "label" if "label" in dataset.column_names else None
    if label_col:
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", label_col])
    else:
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

    # Create Trainer for evaluation
    trainer = Trainer(model=model)

    results = trainer.evaluate(eval_dataset=tokenized_dataset)
    print(results)

if __name__ == "__main__":
    main()
