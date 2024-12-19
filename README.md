# AI-Disease-Prediction-System
AI Disease Prediction System
The AI Disease Prediction System is a robust tool that leverages AI to predict diseases based on symptoms provided by the user. It combines a locally trained BERT-based AI model with GPT-4 integration for accurate and refined diagnoses. The system supports continuous improvement by saving user input and predictions for future retraining.

Features
Local AI Model: A fine-tuned BERT model trained on medical datasets for disease prediction.

GPT-4 Integration: Enhances predictions by asking follow-up questions and refining results.

Data Saving for Retraining: Automatically stores new symptoms and diagnoses for improving the model.

Interactive Diagnoses: User-friendly interaction, with GPT-4 asking clarifying questions before finalizing predictions.

Installation
Follow these steps to set up the project:

Clone the repository:

-> git clone https://github.com/amaannizam29/AI-Disease-Prediction-System

-> cd AI-Disease-Prediction-System

Install the required dependencies:

-> pip install -r requirements.txt

Add your OpenAI API key in the predict_disease.py script:

-> openai.api_key = "your_openai_api_key_here"

Usage:
Predict Diseases

To run the disease prediction system, use:

-> python predict_disease.py

Enter symptoms in natural language (e.g., "I have a fever and sore throat").

GPT-4 will ask follow-up questions to refine the diagnosis.

Both the local AI model and GPT-4 will provide their predictions.

The system will indicate whether both predictions agree or differ.

Retrain the Model

To retrain the AI model with new or updated datasets:

-> python retrain_full_model.py

This script combines and processes all available datasets, then trains the local AI model.

Files and Directories:

saved_model/: Contains the trained BERT model for local predictions.

refined_data.csv: Stores user inputs and refined predictions for future retraining.

combined_data.csv: Combined dataset used for training.

predict_disease.py: Main script for disease prediction.

retrain_full_model.py: Script to retrain the local AI model on all datasets.

train_initial_model.py: Used for the initial training of the AI model.

utils.py: Utility functions for processing datasets and managing files.

Datasets:

The project uses multiple datasets to train the AI model:

dataset.csv: Base dataset with symptoms and diseases.

Symptom-severity.csv: Provides severity scores for symptoms.

symptom_Description.csv: Contains detailed descriptions of symptoms.

symptom_precaution.csv: Offers precautions for various diseases.

Additional data will be saved automatically during predictions to improve the model.

How It Works:

Symptom Input: Users describe their symptoms in natural language.

Local AI Prediction: The BERT-based model predicts a disease.

GPT-4 Refinement: GPT-4 asks follow-up questions to refine the prediction and make its own diagnosis.

Comparison: GPT-4 compares its prediction with the local AI's output:

If they match: The system confirms the prediction is accurate.

If they differ: The system saves the data for future retraining.

Output: Displays both predictions along with explanations.

Example Interaction:

Describe your symptoms: I have a fever and body ache.

Refining symptoms with GPT-4 follow-up questions...

GPT-4: Have you experienced any other symptoms like nausea or a cough?

Your Answers: Yes, I also have a mild cough.

Finalizing predictions...

--- Final Results ---

Local AI Prediction: Influenza (Flu)

GPT-4 Refined Prediction: Influenza (Flu)

Kudos! Local AI and GPT-4 agree on the diagnosis.

Future Enhancements:-

Expand the datasets to include more diseases and symptoms.

Incorporate real-time updates and feedback from users.

Improve model accuracy with periodic retraining.

Contributing:

Contributions are welcome! To contribute:

Fork the repository.

1. Create a new branch (git checkout -b feature-branch).

2. Commit your changes (git commit -m "Add new feature").

3. Push to the branch (git push origin feature-branch).

4. Open a pull request.

License:
This project is licensed under the MIT License.

Acknowledgments:
Transformers Library by Hugging Face for the BERT model.

OpenAI GPT-4 for refining predictions.

Community contributions for dataset enhancements and testing.
