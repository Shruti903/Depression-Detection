from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "models/transformer-final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()
    label = int(torch.argmax(outputs.logits, dim=1))
    label_name = "Depressed" if label == 1 else "Not Depressed"
    return label_name, probs

if __name__ == "__main__":
    sample_text = "I feel so empty and hopeless lately."
    label, probs = predict(sample_text)
    print(f"Prediction: {label} | Probabilities: {probs}")
