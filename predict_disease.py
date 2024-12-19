import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import openai
import os
import pickle

# GPT-4 API Key
openai.api_key = "your_openai_api_key_here"

# File Paths
REFINED_DATA_FILE = "refined_data.csv"

# Load model and tokenizer
def load_model():
    print("Loading AI model...")
    model = BertForSequenceClassification.from_pretrained("saved_model")
    tokenizer = BertTokenizer.from_pretrained("saved_model")
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

# GPT-4 Follow-Up Questions
def gpt4_followup(symptoms):
    prompt = f"""
The user described symptoms: '{symptoms}'.
Please generate up to max 10 concise follow-up questions to clarify the diagnosis further but not necessary to ask 10 you can ask less also.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a medical expert helping refine disease predictions."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

# GPT-4 Final Prediction
def gpt4_refine(symptoms, local_prediction):
    prompt = f"""
Given the symptoms: '{symptoms}' and the local AI's prediction: '{local_prediction}',
please provide a refined diagnosis, a brief explanation, and the symptoms associated with the disease.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a medical expert providing refined diagnoses."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

# Save data for future training
def save_training_data(symptoms, gpt4_prediction):
    disease, *details = gpt4_prediction.split("\n")
    new_data = pd.DataFrame([[symptoms, disease]], columns=["Symptoms", "Disease"])
    if not os.path.exists(REFINED_DATA_FILE):
        new_data.to_csv(REFINED_DATA_FILE, index=False)
    else:
        new_data.to_csv(REFINED_DATA_FILE, mode='a', header=False, index=False)
    print("Data saved for future retraining.")

# Predict Disease
def predict_disease():
    # Load model and tokenizer
    model, tokenizer, label_encoder = load_model()

    print("\nWelcome to the AI Disease Prediction System! (Type 'exit' to quit)\n")
    while True:
        symptoms = input("Describe your symptoms: ").strip()
        if symptoms.lower() in ['exit', 'quit']:
            print("Exiting the system. Goodbye!")
            break

        user_symptoms = symptoms

        # Local AI Prediction
        inputs = tokenizer(user_symptoms, return_tensors="pt", truncation=True, padding=True, max_length=64)
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()
        local_prediction = label_encoder.inverse_transform([predicted_label])[0]

        # GPT-4 Follow-Up
        print("\nRefining symptoms with GPT-4 follow-up questions...")
        gpt_questions = gpt4_followup(user_symptoms)
        print(f"GPT-4 Follow-Up Questions:\n{gpt_questions}")

        # Get user input for follow-up questions
        user_input = input("\nYour Answers (or type 'none' if no further info): ").strip()
        if user_input.lower() != "none":
            user_symptoms = f"{user_symptoms}, {user_input}"

        # Final Predictions
        print("\nFinalizing predictions...")
        gpt4_prediction = gpt4_refine(user_symptoms, local_prediction)

        # Show Results
        print("\n--- Final Results ---")
        print(f"Local AI Prediction: {local_prediction}")
        print(f"GPT-4 Refined Prediction: {gpt4_prediction}")

        # Save Data for Future Training
        save_training_data(user_symptoms, gpt4_prediction)

        # Evaluate Agreement
        evaluation_prompt = f"""
Local AI predicted: '{local_prediction}'.
GPT-4 refined diagnosis: '{gpt4_prediction}'.
Are these predictions the same? If not, explain why.
"""
        evaluation_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an evaluator comparing AI predictions."},
                {"role": "user", "content": evaluation_prompt}
            ]
        )
        evaluation_result = evaluation_response['choices'][0]['message']['content']
        print("\n--- Evaluation ---")
        print(evaluation_result)

if __name__ == "__main__":
    predict_disease()
