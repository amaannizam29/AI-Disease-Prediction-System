import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
import openai
import os
import pickle
import time

# GPT-4 API Key
openai.api_key = "your_openai_api_key_here"

# Step 1: Combine All Datasets
def create_combined_data():
    print("Combining all datasets to generate 'original_data.csv'...")

    # Load datasets
    dataset = pd.read_csv("dataset.csv")
    severity = pd.read_csv("Symptom-severity.csv")
    description = pd.read_csv("symptom_Description.csv")

    # Combine symptom columns into a single 'Symptoms' column
    symptom_columns = [col for col in dataset.columns if col.startswith("Symptom")]
    dataset["Symptoms"] = dataset[symptom_columns].apply(lambda x: ", ".join(x.dropna().astype(str)), axis=1)

    # Drop unnecessary columns and ensure Symptoms/Disease are intact
    dataset = dataset[["Symptoms", "Disease"]].drop_duplicates()

    # Add descriptions (if available)
    description_dict = dict(zip(description["Disease"], description["Description"]))
    dataset["Description"] = dataset["Disease"].map(description_dict)

    # Skip severity filtering (temporarily ensure all rows are included)
    print(f"'original_data.csv' created with {len(dataset)} rows (no filtering applied).")
    dataset.to_csv("original_data.csv", index=False)
    return dataset


# Step 2: Train the Model
def train_model(data):
    print("\nEncoding labels and tokenizing symptoms...")
    label_encoder = LabelEncoder()
    data["Disease_encoded"] = label_encoder.fit_transform(data["Disease"])

    # Save label encoder
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    # Tokenize symptoms
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encodings = tokenizer(data["Symptoms"].tolist(), truncation=True, padding=True, max_length=64, return_tensors="pt")
    labels = torch.tensor(data["Disease_encoded"].values)

    # Dataset class
    class SymptomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels.long()

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)

    dataset = SymptomDataset(encodings, labels)

    # Initialize model
    print("Initializing BERT model for training...")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_strategy="epoch",
        logging_dir="./logs",
    )

    # Trainer
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

    # Train the model
    print("Training the model...")
    trainer.train()

    # Save model and tokenizer
    if not os.path.exists("saved_model"):
        os.makedirs("saved_model")
    model.save_pretrained("saved_model")
    tokenizer.save_pretrained("saved_model")

    print("\nModel training complete and saved successfully!")
    return model, tokenizer, label_encoder

# Step 3: GPT-4 Conversation Loop for Refinement
def refine_model_with_gpt4(model, tokenizer, label_encoder, duration=1):
    print("\nStarting GPT-4 refinement loop...")

    start_time = time.time()
    refined_data = []

    while (time.time() - start_time) < duration * 3600:
        # GPT-4 generates synthetic symptoms
        gpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate random symptoms for a general physician-level disease."},
                {"role": "user", "content": "Provide a list of symptoms."}
            ]
        )
        symptoms = gpt_response['choices'][0]['message']['content']

        # Predict disease using the local model
        inputs = tokenizer(symptoms, return_tensors="pt", truncation=True, padding=True, max_length=64)
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()
        local_prediction = label_encoder.inverse_transform([predicted_label])[0]

        # GPT-4 refines the prediction
        refine_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert doctor refining disease predictions based on symptoms."},
                {"role": "user", "content": f"Symptoms: {symptoms}. The AI model predicts '{local_prediction}'. Please provide the correct diagnosis."}
            ]
        )
        correct_disease = refine_response['choices'][0]['message']['content']

        # Log refined data
        print(f"\nSymptoms: {symptoms}")
        print(f"Local Prediction: {local_prediction}")
        print(f"GPT-4 Refined Prediction: {correct_disease}")

        refined_data.append({"Symptoms": symptoms, "Disease": correct_disease})

    # Save refined data
    pd.DataFrame(refined_data).to_csv("refined_data.csv", mode='a', header=not os.path.exists("refined_data.csv"), index=False)
    print("\nGPT-4 refinement complete. Refined data saved.")

# Main Function
if __name__ == "__main__":
    print("Starting AI model setup...")

    # Step 1: Combine datasets
    data = create_combined_data()

    # Step 2: Train the initial model
    model, tokenizer, label_encoder = train_model(data)

    # Step 3: Run GPT-4 refinement loop for 1 hour
    refine_model_with_gpt4(model, tokenizer, label_encoder, duration=1)

    print("\nAI model setup and GPT-4 refinement completed successfully!")
