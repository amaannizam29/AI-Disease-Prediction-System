import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import os
import pickle

# File paths
DATASET_FILE = "dataset.csv"
SEVERITY_FILE = "Symptom-severity.csv"
DESCRIPTION_FILE = "symptom_Description.csv"
PRECAUTION_FILE = "symptom_precaution.csv"
COMBINED_DATA_FILE = "combined_data.csv"

# Combine all datasets into a single file
def combine_data():
    print("Combining all datasets into a single file...")

    # Load all CSV files
    dataset = pd.read_csv(DATASET_FILE)
    severity = pd.read_csv(SEVERITY_FILE)
    description = pd.read_csv(DESCRIPTION_FILE)
    precaution = pd.read_csv(PRECAUTION_FILE)

    # Merge all symptoms into a single Symptoms column
    symptom_columns = [col for col in dataset.columns if col.startswith("Symptom")]
    dataset["Symptoms"] = dataset[symptom_columns].fillna("").agg(", ".join, axis=1).str.strip(", ")

    # Drop the original symptom columns
    dataset = dataset[["Symptoms", "Disease"]]

    # Ensure column names are correct for other CSVs
    severity.columns = ["Symptom", "Severity"]
    description.columns = ["Disease", "Description"]
    precaution.columns = ["Disease", "Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]

    # Merge symptom severity
    severity_dict = severity.set_index("Symptom")["Severity"].to_dict()
    dataset["Severity"] = dataset["Symptoms"].map(
        lambda x: max([severity_dict.get(symptom.strip(), 0) for symptom in x.split(",")])
    )

    # Merge disease descriptions
    combined_data = dataset.merge(description, on="Disease", how="left")

    # Merge disease precautions
    combined_data = combined_data.merge(precaution, on="Disease", how="left")

    # Fill NaNs with empty strings
    combined_data = combined_data.fillna("")

    # Combine all precautions into a single field
    combined_data["Precautions"] = combined_data[["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]].agg(
        lambda x: ", ".join(filter(None, x)), axis=1
    )

    # Drop extra columns
    combined_data = combined_data[["Symptoms", "Disease", "Severity", "Description", "Precautions"]]
    combined_data.to_csv(COMBINED_DATA_FILE, index=False)

    print(f"Combined dataset saved to '{COMBINED_DATA_FILE}' with {len(combined_data)} rows.")

# Train the model
def train_model():
    print("Starting model training...")

    # Load combined data
    combined_data = pd.read_csv(COMBINED_DATA_FILE)

    # Encode labels
    label_encoder = LabelEncoder()
    combined_data["Disease_encoded"] = label_encoder.fit_transform(combined_data["Disease"])

    # Save the label encoder for future use
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    # Prepare inputs and labels
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encodings = tokenizer(combined_data["Symptoms"].tolist(), truncation=True, padding=True, max_length=64, return_tensors="pt")
    labels = torch.tensor(combined_data["Disease_encoded"].values, dtype=torch.long)

    class SymptomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)

    dataset = SymptomDataset(encodings, labels)

    # Load model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_strategy="no",
        logging_dir="./logs",
        logging_steps=10,
        report_to="none"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    # Train the model
    trainer.train()

    # Save the updated model
    model.save_pretrained("saved_model")
    tokenizer.save_pretrained("saved_model")

    print("Model training complete! Updated model saved to 'saved_model/'.")

# Main function
if __name__ == "__main__":
    # Step 1: Combine datasets
    combine_data()

    # Step 2: Train the model
    train_model()
