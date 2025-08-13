from transformers import pipeline

def main():
    model_path = "./model"

    # Create pipeline for text classification
    nlp = pipeline("text-classification", model=model_path, tokenizer=model_path)

    # Example tweets to test
    tweets = [
        "I feel really sad and hopeless today.",
        "Life is going great, I am so happy!"
    ]

    for t in tweets:
        result = nlp(t)[0]
        label = result['label']
        score = result['score']
        print(f"Tweet: {t}")
        print(f"Prediction: {label} | Confidence: {score:.4f}\n")

if __name__ == "__main__":
    main()
