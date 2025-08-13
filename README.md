# Depression Detection from Tweets using BERT

This project detects signs of depression in tweets using **BERT** from Hugging Face Transformers.  
It includes scripts for **training**, **inference**, and **evaluation**.  
Datasets are in `.csv` format.

---

## 📂 Folder Structure
depression/
│
├── data/
│ ├── train.csv # Training dataset
│ ├── validation.csv # Validation dataset
│ └── test.csv # Test dataset
│
├── model/ # Saved trained model
│
├── scripts/
│ ├── train_baseline.py # Baseline model (non-transformer)
│ ├── train_transformer.py # Train BERT model
│ ├── inference_transformer.py # Make predictions on custom text
│ └── evaluate_transformer.py # Evaluate model on test set
│
├── .venv/ # Virtual environment (optional, not included in repo)
│
└── README.md

yaml
Copy
Edit

---

## ⚙️ Installation

### 1. Clone this repository
```bash
git clone https://github.com/your-username/depression-detection.git
cd depression-detection
## 2. Create and activate a virtual environment
bash
Copy
Edit
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
### 3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
📊 Dataset
The project expects the datasets in .csv format inside the data/ folder.

Example format:

tweet	label
I feel hopeless and sad.	1
Today is a beautiful sunny day!	0

tweet → The text data (string)

label → 1 = Depressed, 0 = Not depressed

🚀 Training the Model
Baseline Model
bash
Copy
Edit
python scripts/train_baseline.py
Transformer Model (BERT)
bash
Copy
Edit
python scripts/train_transformer.py
The trained model will be saved in the ./model folder.

🔍 Inference (Custom Predictions)
Run the inference script to predict on custom text:

bash
Copy
Edit
python scripts/inference_transformer.py
Example Output:

yaml
Copy
Edit
Tweet: I feel really sad and hopeless today.
Prediction: LABEL_1 | Confidence: 0.9897

Tweet: Life is going great, I am so happy!
Prediction: LABEL_1 | Confidence: 0.9279
📈 Model Evaluation
Evaluate the model on the test.csv file:

bash
Copy
Edit
python scripts/evaluate_transformer.py
Example output:

pgsql
Copy
Edit
[INFO] Using 'tweet' as text column
{'eval_loss': 0.4563, 'eval_accuracy': 0.855}
🛠 Requirements
Create a requirements.txt with:

makefile
Copy
Edit
transformers==4.44.2
datasets
torch
pandas
scikit-learn
📌 Notes
Ensure all .csv datasets are in the data/ folder.

Column names should match:

tweet → text

label → target class (0/1)

Uses bert-base-uncased as the pretrained transformer model.

📜 License
This project is licensed under the Apache 2.0 License.

✨ Acknowledgements
Hugging Face Transformers

PyTorch

Dataset source (provide link if public)

yaml
Copy
Edit

---

If you want, I can now **add a sample training log from your actual run** to this README so it shows real results like the loss and accuracy improving across epochs. That will make your GitHub repo look more credible.

---
