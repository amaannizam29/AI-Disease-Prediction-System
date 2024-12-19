import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle

# Load model, tokenizer, and label encoder
def load_model_and_tokenizer():
    model = BertForSequenceClassification.from_pretrained("saved_model")
    tokenizer = BertTokenizer.from_pretrained("saved_model")
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

# Save new data
def save_new_data(user_input, prediction):
    file = "new_data.csv"
    new_entry = {"Symptoms": user_input, "Disease": prediction}
    pd.DataFrame([new_entry]).to_csv(file, mode='a', header=not os.path.exists(file), index=False)
